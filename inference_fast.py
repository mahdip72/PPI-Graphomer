import os
import glob
import numpy as np
import esm as esm_fair
import torch
import torch.multiprocessing as mp  # Add this import
import tqdm  # Add this import
import torch.cuda.amp as amp  # Add for mixed precision
from esm import inverse_folding
import torch.nn.functional as F
import torch.utils.data as Data
import argparse
import warnings
import csv
import time
import traceback  # Add for detailed exception tracing
from utils import util_extract_protein_data, util_extract_protein_cpu_data, util_process_train_data

warnings.filterwarnings("ignore")

# Potentially define model paths or configurations here if needed
TRANSFORMER_MODEL_PATH = './model/model_0.pth'
HETATM_LIST_PATH = "./data/hetatm_list.npy"

# Global placeholders for models and data to avoid reloading per file
model_esm = None
alphabet = None
model_esmif = None  # Will be passed to MyDataSet
alphabet_if = None  # Will be passed to MyDataSet
transformer_model = None
hetatm_list_global = None


class MyDataSet(Data.Dataset):
    def __init__(self, pdb_path_list, model_esm, alphabet, pro_len, device_esm, device_data, hetatm_list,
                 use_mixed_precision):  # Added separate device for ESM and non-ESM
        super(MyDataSet, self).__init__()
        self.pdb_path_list = pdb_path_list
        self.model_esm = model_esm
        self.alphabet = alphabet
        self.device_esm = device_esm  # Device for ESM model
        self.device_data = device_data  # Device for non-ESM features
        self.pro_len = pro_len
        self.hetatm_list = hetatm_list  # Store for util_extract_protein_cpu_data and util_process_train_data
        self.use_mixed_precision = use_mixed_precision  # Store mixed precision flag

    def __len__(self):
        return len(self.pdb_path_list)

    def __getitem__(self, idx):
        pdb_path = self.pdb_path_list[idx]

        # Pass use_mixed_precision to util_extract_protein_data
        esm_data_full = util_extract_protein_data(pdb_path, self.model_esm, self.alphabet, self.device_esm,
                                                  self.use_mixed_precision)
        torch.cuda.empty_cache()

        cpu_data = util_extract_protein_cpu_data(pdb_path, self.hetatm_list, self.device_data)

        protein_name_key = cpu_data["protein_name"]
        esm_data_val_list = esm_data_full.get(protein_name_key)

        if esm_data_val_list is None:
            if len(esm_data_full) == 1:
                protein_name_key_esm = list(esm_data_full.keys())[0]
                esm_data_val_list = esm_data_full[protein_name_key_esm]
            else:
                print(
                    f"Warning: Protein key '{protein_name_key}' not found directly in esm_data output for {pdb_path}. Using available data if possible.")
                # Fallback: create empty/default esm_data_val_list if not found to prevent crash in util_process_train_data
                # util_process_train_data is designed to handle empty lists for ESM components.
                esm_data_val_list = [[], {}, [], [], torch.empty(0, 0, dtype=torch.long),
                                     torch.empty(0, dtype=torch.bool), []]

        # Pass separate devices: device_esm for ESM features, device_data for other tensors
        val_data_dict = util_process_train_data(cpu_data, esm_data_val_list, pro_len=self.pro_len,
                                                hetatm_list_len=len(self.hetatm_list),
                                                device_esm=self.device_esm, device_data=self.device_data)

        # Return processed data and raw coordinate info for inverse folding
        return (
            val_data_dict["protein_name"],
            val_data_dict["chain_id_res"],
            val_data_dict["enc_tokens"],
            val_data_dict["seq_features"],
            val_data_dict["interface_res_matrix"],
            val_data_dict["seq"],
            val_data_dict["interaction_type_matrix"],
            val_data_dict["interaction_matrix"],
            val_data_dict["res_mass_centor"],
            val_data_dict["hetatm_features"],
            val_data_dict["coor_dict_for_esmif"],
            val_data_dict["chain_ids_for_esmif"],
            val_data_dict["original_length"]
        )


def collate_fn(batch):
    protein_names = [item[0] for item in batch]
    chain_id_res = [item[1] for item in batch]
    enc_tokens = torch.stack([item[2] for item in batch])
    seq_features = torch.stack([item[3] for item in batch])
    interface_atoms = torch.stack([item[4] for item in batch])
    seqs = [item[5] for item in batch]
    interaction_type = torch.stack([item[6] for item in batch])
    interaction_matrix = torch.stack([item[7] for item in batch])
    res_mass_centor = torch.stack([item[8] for item in batch])
    hetatm_features = torch.stack([item[9] for item in batch])
    coor_dicts = [item[10] for item in batch]
    chain_ids_list = [item[11] for item in batch]
    original_lens = [item[12] for item in batch]
    return (protein_names, chain_id_res, enc_tokens, seq_features,
            interface_atoms, seqs, interaction_type,
            interaction_matrix, res_mass_centor, hetatm_features,
            coor_dicts, chain_ids_list, original_lens)


def evaluate(model, loader, device, model_esmif, alphabet_if, pro_len,
             use_mixed_precision):  # Added use_mixed_precision
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        output_list = []
        affinity_list = []
        protein_names_val_list = []

        device_esmif = next(model_esmif.parameters()).device  # Determine device of ESM-IF model

        # Wrap loader with tqdm for a progress bar
        for it, batch in enumerate(tqdm.tqdm(loader, desc="Evaluating")):
            (protein_names_val, chain_id_res_val, enc_tokens_val, seq_features_val,
             interface_atoms_val, seqs_val,
             interaction_type_val, interaction_matrix_val, res_mass_centor_val,
             hetatm_features_val, coor_dicts_val, chain_ids_list_val, original_lens_val) = batch
            # log time
            start = time.time()

            # Process ESM-IF1 features per item in the batch
            all_coor_features_padded_for_batch = []
            current_batch_size = len(protein_names_val)  # Actual batch size

            for item_idx in range(current_batch_size):
                current_item_coor_dict = coor_dicts_val[item_idx]  # Use numpy-based coordinate dict directly
                current_item_chain_ids = chain_ids_list_val[item_idx]
                # ESM-IF1 features for the current item
                item_specific_chain_feats = []
                with amp.autocast(enabled=use_mixed_precision and device.type == 'cuda'):
                    for chain_id in current_item_chain_ids:
                        if not current_item_coor_dict or not chain_id:  # Skip if coor_dict or chain_id is empty/None
                            continue
                        try:
                            # feat will be on device_esmif (e.g., cuda:1)
                            feat = inverse_folding.multichain_util.get_encoder_output_for_complex(
                                model_esmif, alphabet_if, current_item_coor_dict, chain_id
                            )
                            item_specific_chain_feats.append(feat)
                        except Exception as e:
                            print(
                                f"Error during ESM-IF1 processing for item {protein_names_val[item_idx]}, chain {chain_id}: {e}")
                            traceback.print_exc()  # Print full stack trace
                            # Optionally append a zero tensor or handle error
                            item_specific_chain_feats.append(torch.zeros((0, 512), device=device_esmif, dtype=torch.float32))

                if item_specific_chain_feats:
                    item_coor_features_concatenated = torch.cat(item_specific_chain_feats, dim=0)  # on device_esmif
                else:
                    # Create on device_esmif
                    item_coor_features_concatenated = torch.empty((0, 512), device=device_esmif, dtype=torch.float32)

                # Pad/truncate the concatenated features for the current item to pro_len
                # item_coor_features_concatenated is on device_esmif
                num_residues_for_item = item_coor_features_concatenated.size(0)
                if num_residues_for_item < pro_len:
                    padding_size = pro_len - num_residues_for_item
                    # Padding preserves device, so item_coor_features_padded will be on device_esmif
                    item_coor_features_padded = F.pad(item_coor_features_concatenated, (0, 0, 0, padding_size),
                                                      value=0.0)
                else:
                    # Slicing preserves device, so item_coor_features_padded will be on device_esmif
                    item_coor_features_padded = item_coor_features_concatenated[:pro_len, :]

                # Ensure consistent shape [pro_len, 512]
                if item_coor_features_padded.shape[0] != pro_len or item_coor_features_padded.shape[1] != 512:
                    # Create on device_esmif
                    item_coor_features_padded = torch.zeros((pro_len, 512), device=device_esmif, dtype=torch.float32)

                # Move the final item_coor_features_padded (which is on device_esmif) to the main evaluation device (e.g., cuda:0)
                all_coor_features_padded_for_batch.append(item_coor_features_padded.to(device))

            if all_coor_features_padded_for_batch:
                # All tensors in all_coor_features_padded_for_batch are now on 'device'
                coor_features_val = torch.stack(all_coor_features_padded_for_batch, dim=0)
                # .to(device) is not strictly necessary here as stack input is already on device, but harmless
                coor_features_val = coor_features_val.to(device)
            else:
                # Fallback for an empty batch (should ideally not occur if loader provides data)
                coor_features_val = torch.zeros((current_batch_size, pro_len, 512), device=device, dtype=torch.float32)

            # enc_tokens_val, seq_features_val etc. are already batched by collate_fn
            # Old ESM-IF1 processing logic removed.
            # coor_features_val is now shape [batch_size, pro_len, 512]

            # move inputs to device (other tensors that are directly from collate_fn)
            enc_tokens_val, seq_features_val = enc_tokens_val.type(torch.int64).to(device), seq_features_val.to(device)
            interface_atoms_val = interface_atoms_val.to(device)
            interaction_type_val, interaction_matrix_val, res_mass_centor_val = interaction_type_val.type(
                torch.int64).to(device), interaction_matrix_val.type(torch.int32).to(device), res_mass_centor_val.to(
                device)
            hetatm_features_val = hetatm_features_val.type(torch.float).to(device)

            # Apply mixed precision to transformer_model
            with amp.autocast(enabled=use_mixed_precision and device.type == 'cuda'):
                val_outputs = model(enc_tokens_val, seq_features_val, coor_features_val, hetatm_features_val,
                                    interface_atoms_val,
                                    interaction_type_val, interaction_matrix_val, res_mass_centor_val, seqs_val,
                                    protein_names_val, chain_id_res_val)
            end = time.time()
            # print each sample's prediction
            # for name, out in zip(protein_names_val, val_outputs.view(-1)):
            #     print(f"Sample {name} predicted affinity: {out.item()} (Time taken: {end - start:.4f} seconds)")
            output_list.append(val_outputs.view(-1))
            protein_names_val_list.extend(protein_names_val)

        output_all = torch.cat(output_list, dim=0)

    return epoch_loss / len(loader), output_all


if __name__ == '__main__':
    # Set the start method for multiprocessing
    # This is important for CUDA compatibility with multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        # This can happen if it's already set or not applicable in the current context
        print(f"Note: Multiprocessing start method: {e}")

    parser = argparse.ArgumentParser()

    parser.add_argument("--pdb_folder", '-p', type=str,
                        help="Path to the folder containing PDB files")
    parser.add_argument("--device", '--d', default="cuda", type=str,
                        help="Device to run the models on (e.g., 'cuda', 'cpu')")
    parser.add_argument("--mixed-precision", "--mp", default=True, action=argparse.BooleanOptionalAction,
                        help="Enable mixed precision for inference (CUDA only, default: enabled)")
    parser.add_argument("--batch_size", "-bs", default=8, type=int, help="Batch size for DataLoader")
    parser.add_argument("--num_workers", "-nw", default=4, type=int, help="Number of worker processes for DataLoader")
    parser.add_argument("--compile", "-c", default=True, action=argparse.BooleanOptionalAction,
                        help="Enable torch.compile for models (default: enabled)")
    parser.add_argument("--csv_dir", default="./result/default/", type=str,
                        help="Directory to save the prediction CSV file (default: ./result/default/)")

    args = parser.parse_args()

    pdb_folder_path = args.pdb_folder
    selected_device = args.device
    use_mixed_precision_arg = args.mixed_precision
    use_compile_arg = args.compile  # Control torch.compile calls
    batch_size_arg = args.batch_size
    num_workers_arg = args.num_workers
    csv_dir_arg = args.csv_dir

    # Determine devices for transformer, ESM, and ESM-IF
    device_transformer = torch.device(selected_device)
    device_esm = torch.device(selected_device)
    device_esmif = torch.device(selected_device)
    device_data = torch.device('cpu')

    print(f"Using transformer device: {device_transformer}")
    print(f"Using ESM device: {device_esm}")
    print(f"Using ESM-IF device: {device_esmif}")
    print(f"Using data device: {device_data}")
    print(f"Searching for PDB files in: {pdb_folder_path}")

    # Load models and data once
    print("Loading ESM2 model...")
    model_esm, alphabet = esm_fair.pretrained.esm2_t33_650M_UR50D()
    model_esm = model_esm.eval().to(device_esm)

    if use_compile_arg:
        print("Compiling ESM model...")
        model_esm = torch.compile(model_esm)

    print("Loading ESM-IF1 model...")
    model_esmif, alphabet_if = esm_fair.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model_esmif = model_esmif.eval().to(device_esmif)
    if use_compile_arg:
        print("Compiling ESM-IF model...")
        model_esmif = torch.compile(model_esmif)

    print("Loading Transformer model...")
    transformer_model = torch.load(TRANSFORMER_MODEL_PATH, map_location=device_transformer)
    transformer_model = transformer_model.eval().to(device_transformer)
    if use_compile_arg:
        print("Compiling Transformer model...")
        transformer_model = torch.compile(transformer_model)

        # Does not work with torch.compile
        # transformer_model = torch.compile(transformer_model, dynamic=True, fullgraph=True)

    print("Loading HETATM list...")
    hetatm_list_global = np.load(HETATM_LIST_PATH, allow_pickle=True)
    print("Models and data loaded.")

    # Batch inference over all PDB files
    pdb_files = glob.glob(os.path.join(pdb_folder_path, '**', '*.pdb'), recursive=True)
    print(f"Found {len(pdb_files)} PDB files. Running batch inference...")
    pro_len_const = 2000

    dataset = MyDataSet(pdb_files, model_esm, alphabet, pro_len_const, device_esm, device_data,
                        hetatm_list_global, use_mixed_precision_arg)
    val_loader = Data.DataLoader(dataset, batch_size=batch_size_arg, shuffle=False, collate_fn=collate_fn,
                                 num_workers=num_workers_arg)

    valid_loss, output_all = evaluate(transformer_model, val_loader, device_transformer, model_esmif, alphabet_if,
                                      pro_len_const, use_mixed_precision_arg)

    # Ensure the CSV directory exists
    if not os.path.exists(csv_dir_arg):
        os.makedirs(csv_dir_arg)

    csv_file_path = os.path.join(csv_dir_arg, "predictions.csv")

    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['pdb_name', 'binding_affinity'])  # Write header
        for pdb_path, aff in zip(pdb_files, output_all):
            filename_without_ext = os.path.splitext(os.path.basename(pdb_path))[0]
            csv_writer.writerow([filename_without_ext, aff.item()])
            # print(f"Predicted affinity for {filename_without_ext}.pdb: {aff.item()}") # Keep print statement if

    print(f"Inference completed. Predictions saved to {csv_file_path}")
    exit(0)
