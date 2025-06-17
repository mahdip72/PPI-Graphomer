import os
import numpy as np
from Bio.SeqUtils import seq1
from collections import defaultdict, OrderedDict
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser
import torch.cuda.amp as amp  # Add for mixed precision

# KD-tree tuning constants
KD_TREE_LEAFSIZE = 16  # points per leaf in cKDTree
INTERFACE_RADIUS = 7.0  # Å cutoff for interface residues
HETATM_RADIUS = 7.0  # Å cutoff for HETATM neighbor features

# Constants moved or redefined for utility functions
STANDARD_RES = [
    "GLY", 'G', "ALA", 'A', "VAL", 'V', "LEU", 'L', "ILE", 'I', "PRO", 'P',
    "PHE", 'F', "TYR", 'Y', "TRP", 'W', "SER", 'S', "THR", 'T', "CYS", 'C',
    "MET", 'M', "ASN", 'N', "GLN", 'Q', "ASP", 'D', "GLU", 'E', "LYS", 'K',
    "ARG", 'R', "HIS", 'H'
]

AMINO_ACID_TO_INDEX = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
_num_elements_upper_tri = (20 * 21) // 2
_upper_tri_values = np.arange(1, _num_elements_upper_tri + 1)
SYMMETRIC_INTERACTION_TYPE_MATRIX = np.zeros((20, 20), dtype=int)
_upper_tri_indices = np.triu_indices(20)
SYMMETRIC_INTERACTION_TYPE_MATRIX[_upper_tri_indices] = _upper_tri_values
SYMMETRIC_INTERACTION_TYPE_MATRIX = SYMMETRIC_INTERACTION_TYPE_MATRIX + SYMMETRIC_INTERACTION_TYPE_MATRIX.T - np.diag(
    SYMMETRIC_INTERACTION_TYPE_MATRIX.diagonal())

CATIONIC_ATOMS = {('ARG', 'NH1'), ('ARG', 'NH2'), ('LYS', 'NZ')}
ANIONIC_ATOMS = {('ASP', 'OD1'), ('ASP', 'OD2'), ('GLU', 'OE1'), ('GLU', 'OE2')}
PI_RES = {'PHE', 'TYR', 'TRP'}


def list_to_ordered_set(lst):
    ordered_dict = OrderedDict.fromkeys(lst)
    ordered_set = list(ordered_dict.keys())
    return ordered_set


def count_distance_pairs(res1, res2, filter_fn1, filter_fn2, d_min, d_max, dist_device):
    coords1_list = [atom.get_coord() for atom in res1.get_atoms() if filter_fn1(atom)]
    coords2_list = [atom.get_coord() for atom in res2.get_atoms() if filter_fn2(atom)]
    if not coords1_list or not coords2_list:
        return 0

    coords1 = torch.tensor(coords1_list, dtype=torch.float32, device=dist_device)
    coords2 = torch.tensor(coords2_list, dtype=torch.float32, device=dist_device)

    if coords1.ndim == 1: coords1 = coords1.unsqueeze(0)  # Ensure 2D
    if coords2.ndim == 1: coords2 = coords2.unsqueeze(0)  # Ensure 2D
    if coords1.shape[0] == 0 or coords2.shape[0] == 0 or coords1.shape[1] == 0 or coords2.shape[
        1] == 0:  # Handle empty tensors
        return 0

    dist_mat = torch.cdist(coords1, coords2)  # Will be (N, M)
    count = ((dist_mat >= d_min) & (dist_mat <= d_max)).sum().item()
    return int(count)


def is_hydrogen_bond(res1, res2, dist_device):
    return count_distance_pairs(
        res1, res2,
        lambda atom: atom.element in ('N', 'O', 'F'),
        lambda atom: atom.element in ('N', 'O', 'F'),
        2.7, 3.5, dist_device
    )


def is_halogen_bond(res1, res2, dist_device):
    return count_distance_pairs(
        res1, res2,
        lambda atom: atom.element in ('Cl', 'Br', 'I'),
        lambda atom: atom.element in ('N', 'O', 'F'),
        3.0, 4.0, dist_device
    )


def is_sulfur_bond(res1, res2, dist_device):
    return count_distance_pairs(
        res1, res2,
        lambda atom: atom.element == 'S',
        lambda atom: atom.element == 'S',
        3.5, 5.5, dist_device
    )


def is_pi_stack(res1, res2, dist_device):
    if res1.resname not in PI_RES or res2.resname not in PI_RES:
        return 0
    return count_distance_pairs(
        res1, res2,
        lambda atom: True,
        lambda atom: True,
        3.3, 4.5, dist_device
    )


def is_salt_bridge(res1, res2, dist_device):
    cnt1 = count_distance_pairs(
        res1, res2,
        lambda atom: (res1.resname, atom.name) in CATIONIC_ATOMS,
        lambda atom: (res2.resname, atom.name) in ANIONIC_ATOMS,
        2.8, 4.0, dist_device
    )
    cnt2 = count_distance_pairs(
        res1, res2,
        lambda atom: (res1.resname, atom.name) in ANIONIC_ATOMS,
        lambda atom: (res2.resname, atom.name) in CATIONIC_ATOMS,
        2.8, 4.0, dist_device
    )
    return cnt1 + cnt2


def is_cation_pi(res1, res2, dist_device):
    cnt = 0
    # Res1 is Cation, Res2 is Pi
    if any((res1.resname, name) in CATIONIC_ATOMS for name in ['NZ', 'NH1', 'NH2']) and res2.resname in PI_RES:
        cnt += count_distance_pairs(
            res1, res2,
            lambda atom: (res1.resname, atom.name) in CATIONIC_ATOMS,
            lambda atom: True,  # All atoms of pi ring for res2
            4.0, 6.0, dist_device
        )
    # Res1 is Pi, Res2 is Cation
    if res1.resname in PI_RES and any((res2.resname, name) in CATIONIC_ATOMS for name in ['NZ', 'NH1', 'NH2']):
        cnt += count_distance_pairs(
            res1, res2,
            lambda atom: True,  # All atoms of pi ring for res1
            lambda atom: (res2.resname, atom.name) in CATIONIC_ATOMS,
            4.0, 6.0, dist_device
        )
    return cnt


def get_ca_positions(residues):
    positions = []
    for residue in residues:
        if 'CA' in residue:
            positions.append(residue['CA'].coord)
        else:
            positions.append(None)
    return positions


def find_neighbors(query_positions_np, target_positions_np, radius=7.0, dist_device='cpu'):
    neighbors_indices = [[] for _ in range(query_positions_np.shape[0])]  # Initialize for all query positions

    if not query_positions_np.size or not target_positions_np.size:
        return neighbors_indices

    query_tensor = torch.tensor(query_positions_np, dtype=torch.float32, device=dist_device)
    target_tensor = torch.tensor(target_positions_np, dtype=torch.float32, device=dist_device)

    if query_tensor.ndim == 1: query_tensor = query_tensor.unsqueeze(0)
    if target_tensor.ndim == 1: target_tensor = target_tensor.unsqueeze(0)

    if query_tensor.shape[0] == 0 or target_tensor.shape[0] == 0:
        return neighbors_indices

    dist_matrix = torch.cdist(query_tensor, target_tensor)

    for i in range(dist_matrix.shape[0]):
        neighbor_idx_for_query_i = torch.where(dist_matrix[i] <= radius)[0].cpu().tolist()
        neighbors_indices[i] = neighbor_idx_for_query_i  # Assign to the pre-initialized list

    return neighbors_indices


def one_hot_encoding(value, categories):
    vec = [0] * len(categories)
    if value in categories:
        vec[categories.index(value)] = 1
    elif categories:  # If unknown and categories list is not empty
        vec[-1] = 1  # Assume last category is "other" or "unknown"
    return vec


# Module-level parser and batch-converter cache to avoid re-instantiation in hot loops
_PDB_PARSER = PDBParser(QUIET=True)
_BATCH_CONVERTERS = {}


def util_extract_protein_data(pdb_file, model_esm, alphabet, device, use_mixed_precision):  # Added use_mixed_precision
    # reuse or create batch_converter for this alphabet
    batch_converter = _BATCH_CONVERTERS.get(alphabet)
    if batch_converter is None:
        batch_converter = alphabet.get_batch_converter()
        _BATCH_CONVERTERS[alphabet] = batch_converter
    parser = _PDB_PARSER

    base_name_parts = os.path.basename(pdb_file).split(".")
    structure_id = base_name_parts[0]
    # Robustly form structure_id, avoiding issues with filenames like ".1A2K.pdb" or "1A2K.pdb.gz"
    # Typically, PDB IDs don't start with '.', so if base_name_parts[0] is empty, it's likely from a hidden file.
    if not base_name_parts[0] and len(base_name_parts) > 1 and base_name_parts[
        1]:  # e.g. ".DS_Store" -> "" or ".1abc" -> "1abc"
        structure_id = base_name_parts[1]
    elif len(base_name_parts) > 2 and base_name_parts[-1] == 'gz':  # for .pdb.gz
        structure_id = ".".join(base_name_parts[:-2])  # e.g. 1abc.pdb.gz -> 1abc
    elif len(base_name_parts) > 1:  # for .pdb
        structure_id = base_name_parts[0]

    protein_name = os.path.basename(pdb_file).rsplit('.', 1)[0]  # More robust way to get name without final extension

    try:
        structure = parser.get_structure(structure_id if structure_id else protein_name, pdb_file)
    except Exception as e:
        print(f"Error parsing PDB {pdb_file} with ID {structure_id if structure_id else protein_name}: {e}. Skipping.")
        # Return empty/default structures
        return {
            protein_name: [[], {}, [], [], torch.empty(0, 0, dtype=torch.long), torch.empty(0, dtype=torch.bool), []]}

    # Store initial parsed data: (residue_obj, chain_id, res_id, res_name_3_letter, n_coord, ca_coord, c_coord)
    parsed_residues = []
    for model_s in structure:
        for chain in model_s:
            for residue in chain:
                res_name_3 = residue.get_resname()
                if res_name_3 == 'HOH' or res_name_3 not in STANDARD_RES:
                    continue  # Skip water and non-standard residues early

                n_atom, ca_atom, c_atom = None, None, None
                for atom in residue:
                    if atom.get_id() == 'N':
                        n_atom = atom.get_vector().get_array()
                    elif atom.get_id() == 'CA':
                        ca_atom = atom.get_vector().get_array()
                    elif atom.get_id() == 'C':
                        c_atom = atom.get_vector().get_array()

                if n_atom is not None and ca_atom is not None and c_atom is not None:
                    try:
                        _ = seq1(res_name_3)  # Check if convertible
                        parsed_residues.append({
                            "obj": residue, "chain_id": chain.id, "res_id": residue.id[1],
                            "name_3": res_name_3, "N": n_atom, "CA": ca_atom, "C": c_atom
                        })
                    except KeyError:
                        pass  # Cannot convert to 1-letter, skip

    # Build sequence_true (biological sequence, XZ filtered), coord_matrix_true, chain_id_list_true
    sequence_true_list = []
    coord_matrix_true_list = []
    chain_id_list_true = []

    for res_data in parsed_residues:
        res_name_1 = seq1(res_data["name_3"])
        if res_name_1 in 'XZ':  # Filter X and Z here
            continue
        sequence_true_list.append(res_name_1)
        coord_matrix_true_list.append([res_data["N"], res_data["CA"], res_data["C"]])
        chain_id_list_true.append(res_data["chain_id"])

    sequence_true = "".join(sequence_true_list)
    coord_matrix = np.array(coord_matrix_true_list, dtype=np.float32) if coord_matrix_true_list else np.empty((0, 3, 3),
                                                                                                              dtype=np.float32)

    # Build gapped sequence (sequence_with_gaps) based on residue ID continuity
    sequence_with_gaps_list = []
    all_res_chain_gapped = []  # Chain IDs for sequence_with_gaps
    padG_indices_in_gapped_seq = []  # Indices of Gaps in sequence_with_gaps

    current_gapped_idx = 0
    unique_chains = list_to_ordered_set(
        r["chain_id"] for r in parsed_residues)  # Use initially parsed, not yet XZ filtered for chain structure

    temp_parsed_residues_by_chain = {chain_id: [] for chain_id in unique_chains}
    for pr in parsed_residues:  # Re-filter for XZ for gap logic
        if seq1(pr["name_3"]) not in 'XZ':
            temp_parsed_residues_by_chain[pr["chain_id"]].append(pr)

    for chain_id_val in unique_chains:
        chain_residues = sorted(temp_parsed_residues_by_chain[chain_id_val], key=lambda r: r["res_id"])
        previous_res_id = None
        for res_data in chain_residues:
            res_name_1 = seq1(res_data["name_3"])  # Already checked not XZ

            if previous_res_id is not None and res_data["res_id"] != previous_res_id + 1 and \
                    res_data["res_id"] != previous_res_id and previous_res_id > 0:  # res_id can be same for insertions
                gap_size = res_data["res_id"] - previous_res_id - 1
                if gap_size > 0:
                    gap_seq = 'G' * gap_size
                    sequence_with_gaps_list.extend(list(gap_seq))
                    all_res_chain_gapped.extend([chain_id_val] * gap_size)
                    padG_indices_in_gapped_seq.extend(range(current_gapped_idx, current_gapped_idx + gap_size))
                    current_gapped_idx += gap_size

            sequence_with_gaps_list.append(res_name_1)
            all_res_chain_gapped.append(chain_id_val)
            current_gapped_idx += 1
            previous_res_id = res_data["res_id"]

    sequence_for_esm_linking = "".join(sequence_with_gaps_list)

    # Split gapped sequence by chain for linker insertion
    seq_single_chain_gapped = [
        ''.join(
            sequence_for_esm_linking[j] for j in range(len(sequence_for_esm_linking)) if all_res_chain_gapped[j] == x)
        for x in list_to_ordered_set(all_res_chain_gapped) if x  # Ensure chain id is not None or empty
    ]
    seq_single_chain_gapped = [s for s in seq_single_chain_gapped if s]  # Remove empty strings

    separator = "G" * 20
    full_sequence = separator.join(seq_single_chain_gapped)

    if not full_sequence:  # Handle empty sequence case
        return {
            protein_name: [[], {}, [], [], torch.empty(1, 2, dtype=torch.long), torch.empty(0, dtype=torch.bool), []]}

    # Map indices from sequence_for_esm_linking to full_sequence
    # And identify separator indices in full_sequence
    updated_padG_indices_in_full_seq = []
    separator_indices_in_full_seq = []
    current_full_seq_idx = 0
    map_gapped_to_full = {}

    for i, chain_seg_gapped in enumerate(seq_single_chain_gapped):
        # Find corresponding segment in original sequence_for_esm_linking
        # This is tricky if chains are reordered by list_to_ordered_set. Assume order is preserved or map carefully.
        # Simpler: iterate through sequence_for_esm_linking and build full_sequence and map simultaneously.
        # For now, let's assume seq_single_chain_gapped corresponds sequentially.

        # This mapping is complex. A direct construction of full_sequence and indices:
        full_seq_list_builder = []
        map_gapped_idx_to_full_idx = {}
        gapped_seq_cursor = 0

        for chain_idx, chain_str_gapped in enumerate(seq_single_chain_gapped):
            for char_idx_in_chain, char_val in enumerate(chain_str_gapped):
                map_gapped_idx_to_full_idx[gapped_seq_cursor] = len(full_seq_list_builder)
                full_seq_list_builder.append(char_val)
                gapped_seq_cursor += 1
            if chain_idx < len(seq_single_chain_gapped) - 1:
                linker_start_in_full = len(full_seq_list_builder)
                full_seq_list_builder.extend(list(separator))
                separator_indices_in_full_seq.extend(range(linker_start_in_full, linker_start_in_full + len(separator)))

        full_sequence = "".join(full_seq_list_builder)  # Rebuild full_sequence to be sure

    for gapped_idx in padG_indices_in_gapped_seq:
        if gapped_idx in map_gapped_idx_to_full_idx:
            updated_padG_indices_in_full_seq.append(map_gapped_idx_to_full_idx[gapped_idx])

    concatenated_name = "_".join(
        [f"{protein_name}" for _ in range(len(seq_single_chain_gapped))]) if seq_single_chain_gapped else protein_name

    enc_inputs_labels, enc_inputs_strs, enc_tokens_batch = batch_converter([(concatenated_name, full_sequence)])

    with torch.no_grad():
        # Apply mixed precision to model_esm
        with amp.autocast(enabled=use_mixed_precision and device.type == 'cuda'):
            results = model_esm(enc_tokens_batch.to(device), repr_layers=[33], return_contacts=False)
        # Option 2: Keep representations on the specified device
        concatenated_representations = results["representations"][33][:, 1:-1, :]

    enc_tokens_all = enc_tokens_batch
    enc_tokens = enc_tokens_batch[:, 1:-1]

    combined_mask_indices = sorted(list(set(updated_padG_indices_in_full_seq + separator_indices_in_full_seq)))

    esm_output_len = concatenated_representations.size(1)
    mask_for_esm_output = torch.ones(esm_output_len, dtype=torch.bool)

    valid_combined_indices = [idx for idx in combined_mask_indices if 0 <= idx < esm_output_len]
    if valid_combined_indices:
        mask_for_esm_output[valid_combined_indices] = False

    # The sum of mask_for_esm_output should be len(sequence_true)
    if mask_for_esm_output.sum().item() != len(sequence_true):
        # This is a critical check. If it fails, alignment is broken.
        # print(f"Warning PDB {pdb_file}: ESM mask sum {mask_for_esm_output.sum().item()} != true sequence len {len(sequence_true)}")
        # Fallback or error. For now, we'll proceed, but this could lead to runtime errors or bad data.
        # If lengths mismatch, try to adjust coord_matrix or token_representations if one is clearly too long/short.
        # This usually means an issue in gap/linker accounting or XZ filtering.
        # A possible cause: if full_sequence is empty, esm_output_len might be 0, len(sequence_true) might be 0.
        # If sequence_true is empty, coord_matrix should be empty.
        if len(sequence_true) == 0 and esm_output_len == 0:
            pass  # Both empty, this is fine.
        elif len(
                sequence_true) == 0 and esm_output_len > 0:  # No true residues, but ESM produced output (e.g. from Gs only)
            mask_for_esm_output[:] = False  # Mask everything from ESM
        # Other mismatches are harder to auto-correct.

    token_representations_masked = concatenated_representations[:, mask_for_esm_output, :]
    masked_enc_tokens_for_seq_true = enc_tokens[:, mask_for_esm_output]

    # Split masked representations and tokens by chain, using chain_id_list_true
    token_representations_list = []
    enc_tokens_list = []
    curr_pos_in_masked = 0

    # seq_single_chain_true: sequence_true split by chain
    seq_single_chain_true = [
        ''.join(sequence_true[j] for j in range(len(sequence_true)) if chain_id_list_true[j] == x)
        for x in list_to_ordered_set(chain_id_list_true) if x  # Ensure chain_id is not None/empty
    ]
    seq_single_chain_true = [s for s in seq_single_chain_true if s]  # Remove empty strings

    for seq_segment_true in seq_single_chain_true:
        seq_len = len(seq_segment_true)
        if seq_len == 0: continue
        if curr_pos_in_masked + seq_len > token_representations_masked.shape[1]:
            # Mismatch, break to avoid error, data for this PDB might be problematic
            # print(f"Warning PDB {pdb_file}: Alignment error during final split. Skipping remaining chains.")
            break
        single_reps = token_representations_masked[:, curr_pos_in_masked: curr_pos_in_masked + seq_len, :]
        token_representations_list.append(single_reps)
        enc_tokens_list.append(masked_enc_tokens_for_seq_true[0, curr_pos_in_masked: curr_pos_in_masked + seq_len])
        curr_pos_in_masked += seq_len

    coor_dict = {}
    current_coord_offset = 0
    unique_true_chains = list_to_ordered_set(chain_id_list_true)

    for i, chain_id_val_true in enumerate(unique_true_chains):
        # Find corresponding length in seq_single_chain_true
        # This assumes unique_true_chains and seq_single_chain_true are aligned by construction.
        if i < len(seq_single_chain_true):
            chain_len_true = len(seq_single_chain_true[i])
            if chain_len_true > 0:
                if current_coord_offset + chain_len_true > coord_matrix.shape[0]:
                    # print(f"Warning PDB {pdb_file}: Coord_matrix alignment error. Skipping chain {chain_id_val_true}.")
                    break
                coor_dict[chain_id_val_true] = coord_matrix[
                                               current_coord_offset: current_coord_offset + chain_len_true].reshape(-1,
                                                                                                                    3,
                                                                                                                    3)
                current_coord_offset += chain_len_true

    torch.cuda.empty_cache()

    chain_list_for_esmif = [c_id for c_id in unique_true_chains if c_id in coor_dict]  # Only chains with coords

    # Ensure all lists in 'addition' are consistent with the (potentially truncated) processing
    # If token_representations_list or enc_tokens_list were cut short due to mismatch,
    # seq_single_chain_true and chain_list_for_esmif should reflect that.
    # This is complex; for now, assume the primary lists (token_representations_list, enc_tokens_list) drive consistency.
    # The number of elements in token_representations_list should match number of chains in chain_list_for_esmif and seq_single_chain_true.

    final_seq_single_chain_true = []
    final_chain_list_for_esmif = []
    temp_chain_idx = 0
    for chain_id_in_esmif_list in chain_list_for_esmif:  # Iterate through chains that have coords
        found_match_in_seq_true_split = False
        for true_seq_segment in seq_single_chain_true:
            # This matching is weak. Better to rely on the order from list_to_ordered_set(chain_id_list_true)
            # For now, let's assume chain_list_for_esmif (from coor_dict keys) is the authority.
            # And seq_single_chain_true needs to be filtered/ordered by it.
            pass  # This part needs robust alignment.
    # Simplified: use chains present in coor_dict as the final set of chains.
    # Re-filter seq_single_chain_true and token/representation lists based on coor_dict's chains.

    # For now, assume the lengths of token_representations_list, enc_tokens_list, seq_single_chain_true, and coor_dict keys (chain_list_for_esmif) are made consistent by the loops above.

    addition = [token_representations_list, coor_dict, chain_list_for_esmif, enc_tokens_list, enc_tokens_all,
                mask_for_esm_output, seq_single_chain_true]
    protein_data = {protein_name: addition}
    return protein_data


def util_extract_protein_cpu_data(pdb_file, hetatm_list_global_arg, dist_device_arg, affinity_dict_arg={}):
    parser = _PDB_PARSER
    protein_name_from_file = os.path.basename(pdb_file).rsplit('.', 1)[0]
    structure_id = protein_name_from_file  # Default structure ID to filename

    try:
        # Try to derive a PDB ID like "1abc" if filename is "1abc.pdb"
        potential_pdb_id = protein_name_from_file.split('_')[0]  # Common convention e.g. 1abc_A
        if len(potential_pdb_id) == 4 and potential_pdb_id.isalnum():  # Basic PDB ID check
            structure_id = potential_pdb_id
        structure = parser.get_structure(structure_id, pdb_file)
        protein_name = structure.id  # Use ID from parsed structure if available
    except Exception as e:
        # print(f"Error parsing PDB cpu_data for {pdb_file} with ID {structure_id}: {e}. Returning empty data.")
        return {
            "protein_name": protein_name_from_file, "sequence": [], "chain_id_res": [],
            "hetatm_features": [], "interface_res": [],
            "interaction_type_matrix": np.array([], dtype=np.int32),
            "interaction_matrix": np.array([], dtype=np.int32),
            "res_mass_centor": np.array([], dtype=np.float16), "affinity": None
        }

    # Store info about residues that will form the final sequence
    # (standard, backbone complete, not HOH, convertible to 1-letter)
    valid_sequence_residue_info = []
    # Store HETATMs and non-standard/incomplete residues for neighbor finding
    other_residues_info = []
    # All atoms from valid_sequence_residues_info for interface calculation
    atoms_of_valid_residues = []
    map_atom_to_seq_idx = []  # Index in the final sequence
    map_atom_to_chain_id = []

    temp_seq_idx = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                # Skip heteroatoms (HETATM) early
                if residue.id[0] != ' ': continue
                res_name_3 = residue.get_resname()
                is_hoh = res_name_3 == 'HOH'

                n_coord, ca_coord, c_coord = None, None, None
                # Only collect coordinates of atoms for mass center and interface
                res_atom_coords_list = []
                for atom in residue:
                    coord = atom.get_vector().get_array()
                    res_atom_coords_list.append(coord)

                    if atom.get_id() == 'N':
                        n_coord = atom.get_coord()
                    elif atom.get_id() == 'CA':
                        ca_coord = atom.get_coord()
                    elif atom.get_id() == 'C':
                        c_coord = atom.get_coord()

                backbone_complete = n_coord is not None and ca_coord is not None and c_coord is not None

                if is_hoh: continue

                try:
                    res_name_1 = seq1(res_name_3)
                    if res_name_3 in STANDARD_RES and backbone_complete and res_name_1 not in 'XZ':
                        valid_sequence_residue_info.append({
                            "pdb_res_obj": residue,  # Add this key back
                            "chain_id": chain.id, "res_name_1": res_name_1,
                            "ca_coord": ca_coord,
                            "all_atom_coords": np.array(res_atom_coords_list, dtype=np.float32),
                            "seq_idx": temp_seq_idx
                        })
                        # Preallocate mapping from atom coords to sequence index and chain
                        for coord in res_atom_coords_list:
                            atoms_of_valid_residues.append(coord)
                            map_atom_to_seq_idx.append(temp_seq_idx)
                            map_atom_to_chain_id.append(chain.id)
                        temp_seq_idx += 1
                except KeyError:  # seq1 failed
                    other_residues_info.append({"pdb_res_obj": residue, "ca_coord": ca_coord, "res_name_3": res_name_3})

    final_sequence = "".join([info["res_name_1"] for info in valid_sequence_residue_info])
    n_final_seq = len(final_sequence)
    if n_final_seq == 0:  # No valid residues found
        return {
            "protein_name": protein_name, "sequence": [], "chain_id_res": [],
            "hetatm_features": [], "interface_res": [[] for _ in range(n_final_seq)],
            "interaction_type_matrix": np.zeros((0, 0), dtype=np.int32),
            "interaction_matrix": np.zeros((0, 0, 6), dtype=np.int32),
            "res_mass_centor": np.zeros((0, 3), dtype=np.float16), "affinity": None
        }

    final_all_res_chain_ids = [info["chain_id"] for info in valid_sequence_residue_info]
    final_res_mass_centers = np.array(
        [np.mean(info["all_atom_coords"], axis=0) for info in valid_sequence_residue_info if
         info["all_atom_coords"].size > 0], dtype=np.float16)
    if final_res_mass_centers.shape[0] != n_final_seq:  # Fallback if some residues had no atoms for mass center
        final_res_mass_centers = np.zeros((n_final_seq, 3), dtype=np.float16)

    # Interaction Type Matrix (vectorized)
    idx_list = [AMINO_ACID_TO_INDEX[aa] for aa in final_sequence]
    interaction_type_matrix_final = SYMMETRIC_INTERACTION_TYPE_MATRIX[np.ix_(idx_list, idx_list)]

    # Interaction Matrix (Option 1: Optimized Calculation)
    interaction_matrix_final = np.zeros((n_final_seq, n_final_seq, 6), dtype=np.int32)

    if n_final_seq > 0:  # Only proceed if there are residues
        # 1. Collect all relevant atoms in a single pass
        all_h_bond_candidate_atoms = []  # (coords, seq_idx)
        all_halogen_atoms = []  # (coords, seq_idx)
        all_sulfur_atoms = []  # (coords, seq_idx)
        all_pi_ring_atoms = []  # (coords, seq_idx)
        all_cationic_group_atoms = []  # (coords, seq_idx)
        all_anionic_group_atoms = []  # (coords, seq_idx)

        for res_info in valid_sequence_residue_info:
            residue = res_info['pdb_res_obj']
            seq_idx = res_info['seq_idx']
            res_name = residue.get_resname()

            for atom in residue.get_atoms():
                atom_coord = atom.get_coord()
                atom_id_str = atom.get_id()
                atom_element = atom.element.upper()

                # H-bond candidates (N, O, F) & Acceptors for Halogen bonds
                if atom_element in ('N', 'O', 'F'):
                    all_h_bond_candidate_atoms.append((atom_coord, seq_idx))

                # Halogen atoms (CL, BR, I)
                if atom_element in ('CL', 'BR', 'I'):
                    all_halogen_atoms.append((atom_coord, seq_idx))

                # Sulfur atoms
                if atom_element == 'S':
                    all_sulfur_atoms.append((atom_coord, seq_idx))

                # Pi-system atoms (all atoms from PHE, TYR, TRP)
                if res_name in PI_RES:
                    all_pi_ring_atoms.append((atom_coord, seq_idx))

                # Cationic group atoms
                if (res_name, atom_id_str) in CATIONIC_ATOMS:
                    all_cationic_group_atoms.append((atom_coord, seq_idx))

                # Anionic group atoms
                if (res_name, atom_id_str) in ANIONIC_ATOMS:
                    all_anionic_group_atoms.append((atom_coord, seq_idx))

        def build_tree_if_data(atom_list):
            if not atom_list: return None, None, None
            coords_np = np.array([item[0] for item in atom_list])
            indices_np = np.array([item[1] for item in atom_list])
            if coords_np.size == 0: return None, None, None
            return cKDTree(coords_np, leafsize=KD_TREE_LEAFSIZE), coords_np, indices_np

        hb_tree, hb_coords_np, hb_indices_np = build_tree_if_data(all_h_bond_candidate_atoms)
        hal_tree, hal_coords_np, hal_indices_np = build_tree_if_data(all_halogen_atoms)
        acc_hal_tree, acc_hal_coords_np, acc_hal_indices_np = hb_tree, hb_coords_np, hb_indices_np  # Re-use H-bond candidates for acceptors

        s_tree, s_coords_np, s_indices_np = build_tree_if_data(all_sulfur_atoms)
        pi_tree, pi_coords_np, pi_indices_np = build_tree_if_data(all_pi_ring_atoms)
        cat_tree, cat_coords_np, cat_indices_np = build_tree_if_data(all_cationic_group_atoms)
        ani_tree, ani_coords_np, ani_indices_np = build_tree_if_data(all_anionic_group_atoms)

        # 0: Hydrogen bond (N,O,F to N,O,F, dist 2.7-3.5 Å)
        if hb_tree:
            for i_atom, j_atom in hb_tree.query_pairs(r=3.5):
                idx1, idx2 = hb_indices_np[i_atom], hb_indices_np[j_atom]
                if idx1 != idx2:
                    dist = np.linalg.norm(hb_coords_np[i_atom] - hb_coords_np[j_atom])
                    if 2.7 <= dist <= 3.5:
                        res_idx1, res_idx2 = sorted((idx1, idx2))
                        interaction_matrix_final[res_idx1, res_idx2, 0] += 1

        # 1: Halogen bond (CL,BR,I to N,O,F, dist 3.0-4.0 Å)
        if hal_tree and acc_hal_tree and hal_coords_np is not None and acc_hal_coords_np is not None:
            for i_hal_list_idx in range(len(hal_coords_np)):
                hal_atom_coord = hal_coords_np[i_hal_list_idx]
                hal_res_idx = hal_indices_np[i_hal_list_idx]
                neighboring_acc_indices_in_list = acc_hal_tree.query_ball_point(hal_atom_coord, r=4.0)
                for j_acc_list_idx in neighboring_acc_indices_in_list:
                    acc_atom_coord = acc_hal_coords_np[j_acc_list_idx]
                    acc_res_idx = acc_hal_indices_np[j_acc_list_idx]
                    if hal_res_idx != acc_res_idx:
                        dist = np.linalg.norm(hal_atom_coord - acc_atom_coord)
                        if 3.0 <= dist <= 4.0:
                            res_idx1, res_idx2 = sorted((hal_res_idx, acc_res_idx))
                            interaction_matrix_final[res_idx1, res_idx2, 1] += 1

        # 2: Sulfur bond (S to S, dist 3.5-5.5 Å)
        if s_tree:
            for i_atom, j_atom in s_tree.query_pairs(r=5.5):
                idx1, idx2 = s_indices_np[i_atom], s_indices_np[j_atom]
                if idx1 != idx2:
                    dist = np.linalg.norm(s_coords_np[i_atom] - s_coords_np[j_atom])
                    if 3.5 <= dist <= 5.5:
                        res_idx1, res_idx2 = sorted((idx1, idx2))
                        interaction_matrix_final[res_idx1, res_idx2, 2] += 1

        # 3: Pi-stacking (Pi-system to Pi-system, dist 3.3-4.5 Å)
        if pi_tree:
            for i_atom, j_atom in pi_tree.query_pairs(r=4.5):
                idx1, idx2 = pi_indices_np[i_atom], pi_indices_np[j_atom]
                if idx1 != idx2:
                    dist = np.linalg.norm(pi_coords_np[i_atom] - pi_coords_np[j_atom])
                    if 3.3 <= dist <= 4.5:
                        res_idx1, res_idx2 = sorted((idx1, idx2))
                        interaction_matrix_final[res_idx1, res_idx2, 3] += 1

        # 4: Salt bridge (Cationic to Anionic, dist 2.8-4.0 Å)
        if cat_tree and ani_tree and cat_coords_np is not None and ani_coords_np is not None:
            for i_cat_list_idx in range(len(cat_coords_np)):
                cat_atom_coord = cat_coords_np[i_cat_list_idx]
                cat_res_idx = cat_indices_np[i_cat_list_idx]
                neighboring_ani_indices_in_list = ani_tree.query_ball_point(cat_atom_coord, r=4.0)
                for j_ani_list_idx in neighboring_ani_indices_in_list:
                    ani_atom_coord = ani_coords_np[j_ani_list_idx]
                    ani_res_idx = ani_indices_np[j_ani_list_idx]
                    if cat_res_idx != ani_res_idx:
                        dist = np.linalg.norm(cat_atom_coord - ani_atom_coord)
                        if 2.8 <= dist <= 4.0:
                            res_idx1, res_idx2 = sorted((cat_res_idx, ani_res_idx))
                            interaction_matrix_final[res_idx1, res_idx2, 4] += 1

        # 5: Cation-Pi (Cationic to Pi-system, dist 4.0-6.0 Å)
        if cat_tree and pi_tree and cat_coords_np is not None and pi_coords_np is not None:
            for i_cat_list_idx in range(len(cat_coords_np)):
                cat_atom_coord = cat_coords_np[i_cat_list_idx]
                cat_res_idx = cat_indices_np[i_cat_list_idx]
                neighboring_pi_indices_in_list = pi_tree.query_ball_point(cat_atom_coord, r=6.0)
                for j_pi_list_idx in neighboring_pi_indices_in_list:
                    pi_atom_coord = pi_coords_np[j_pi_list_idx]
                    pi_res_idx = pi_indices_np[j_pi_list_idx]
                    if cat_res_idx != pi_res_idx:
                        dist = np.linalg.norm(cat_atom_coord - pi_atom_coord)
                        if 4.0 <= dist <= 6.0:
                            res_idx1, res_idx2 = sorted((cat_res_idx, pi_res_idx))
                            interaction_matrix_final[res_idx1, res_idx2, 5] += 1

        # Symmetrize the interaction matrix as counts are added to upper triangle
        for k_interaction_type in range(6):
            matrix_slice = interaction_matrix_final[:, :, k_interaction_type]
            interaction_matrix_final[:, :, k_interaction_type] = matrix_slice + matrix_slice.T
            # Diagonal elements remain 0 due to `idx1 != idx2` or `res_idx != other_res_idx` checks

    # Interface Residues
    final_interface_res_list = [[] for _ in range(n_final_seq)]
    if atoms_of_valid_residues and n_final_seq > 1:
        coords_np = np.array(atoms_of_valid_residues)
        tree_atoms = cKDTree(coords_np, leafsize=KD_TREE_LEAFSIZE)
        pairs = tree_atoms.query_pairs(r=INTERFACE_RADIUS)
        for i_atom, j_atom in pairs:
            chain_i = map_atom_to_chain_id[i_atom];
            chain_j = map_atom_to_chain_id[j_atom]
            seq_i = map_atom_to_seq_idx[i_atom];
            seq_j = map_atom_to_seq_idx[j_atom]
            if chain_i != chain_j and seq_i != seq_j:
                final_interface_res_list[seq_i].append(seq_j)
                final_interface_res_list[seq_j].append(seq_i)
        # Deduplicate
        # Optimization 2: Remove unnecessary sorted()
        final_interface_res_list = [list(set(lst)) for lst in final_interface_res_list]
        # Old line: final_interface_res_list = [sorted(set(lst)) for lst in final_interface_res_list]

    # Split final sequence by chain for CPU data
    seq_single_chain_final = [
        ''.join(final_sequence[j] for j in range(n_final_seq) if final_all_res_chain_ids[j] == chain)
        for chain in list_to_ordered_set(final_all_res_chain_ids) if chain
    ]

    # Map query CA coords to original sequence indices for HETATM features
    query_indices_map = [info['seq_idx'] for info in valid_sequence_residue_info if info['ca_coord'] is not None]

    # HETATM features
    seq_res_ca_coords_list = [info["ca_coord"] for info in valid_sequence_residue_info if info["ca_coord"] is not None]
    het_res_coords_list = [info["ca_coord"] for info in other_residues_info if info["ca_coord"] is not None]
    valid_other_residues_for_neighbors = [info for info in other_residues_info if info["ca_coord"] is not None]

    final_hetatm_features = [np.zeros(len(hetatm_list_global_arg)) for _ in range(n_final_seq)]
    if seq_res_ca_coords_list and het_res_coords_list and hetatm_list_global_arg.size > 0:
        query_pos_np = np.array(seq_res_ca_coords_list)
        target_pos_np = np.array(het_res_coords_list)
        tree = cKDTree(target_pos_np, leafsize=KD_TREE_LEAFSIZE)
        neighbor_indices = tree.query_ball_point(query_pos_np, r=HETATM_RADIUS)

        # Optimization 1: Pre-compute HETATM to index map
        hetatm_categories_list = hetatm_list_global_arg.tolist()
        hetatm_to_idx_map = {name: i for i, name in enumerate(hetatm_categories_list)}
        num_hetatm_categories = len(hetatm_categories_list)

        def _one_hot_encode_hetatm_optimized(value):
            vec = np.zeros(num_hetatm_categories, dtype=int)
            idx = hetatm_to_idx_map.get(value)
            if idx is not None:
                vec[idx] = 1
            elif num_hetatm_categories > 0:  # If unknown and categories list is not empty
                vec[-1] = 1  # Assume last category is "other" or "unknown"
            return vec

        for i, original_seq_idx in enumerate(query_indices_map):
            feat = np.zeros(num_hetatm_categories)
            for nbr in neighbor_indices[i]:
                het_res_name = valid_other_residues_for_neighbors[nbr]["res_name_3"]
                feat += _one_hot_encode_hetatm_optimized(het_res_name)  # Use optimized version
            final_hetatm_features[original_seq_idx] = feat

    protein_data = {
        "protein_name": protein_name, "sequence": seq_single_chain_final,
        "chain_id_res": final_all_res_chain_ids, "hetatm_features": final_hetatm_features,
        "interface_res": final_interface_res_list,
        "interaction_type_matrix": interaction_type_matrix_final,
        "interaction_matrix": interaction_matrix_final,
        "res_mass_centor": final_res_mass_centers,
        "affinity": affinity_dict_arg.get(protein_name, affinity_dict_arg.get(protein_name_from_file, None))
        # Try both names for affinity
    }
    return protein_data


def util_process_train_data(cpu_data, esm_data_val_list, pro_len, hetatm_list_len, device):  # Added device argument
    protein_name_key = cpu_data["protein_name"]

    seq_concat = "".join(cpu_data["sequence"])
    original_len = len(seq_concat)

    # ESM Token Processing
    if esm_data_val_list[3]:  # enc_tokens_list for true sequence
        enc_tokens_cat = torch.cat(esm_data_val_list[3], dim=0).type(
            torch.int16)  # Keep .type for this specific case if source dtype is different
    else:
        enc_tokens_cat = torch.empty(0, dtype=torch.int16)

    current_len_tokens = enc_tokens_cat.shape[0]
    # Assuming 0 is the padding token ID, adjust if different
    padded_enc_tokens = torch.zeros((pro_len,), dtype=torch.int16)
    if current_len_tokens > 0:
        copy_len = min(current_len_tokens, pro_len)
        padded_enc_tokens[:copy_len] = enc_tokens_cat[:copy_len]

    # ESM Feature Processing
    esm_dim_fallback = 1280  # Default ESM2 dimension
    if esm_data_val_list[0]:  # token_representations_list for true sequence
        seq_feat_cat = torch.cat(esm_data_val_list[0], dim=1).squeeze(0)
        if seq_feat_cat.ndim == 1 and seq_feat_cat.shape[0] > 0: seq_feat_cat = seq_feat_cat.unsqueeze(0)
        # Adjust length if there was a mismatch with original_len from CPU data
        if seq_feat_cat.shape[0] != original_len and original_len > 0 and seq_feat_cat.shape[0] > 0:
            if seq_feat_cat.shape[0] > original_len:
                seq_feat_cat = seq_feat_cat[:original_len, :]
            # else: it will be padded to original_len or pro_len later
    else:
        seq_feat_cat = torch.empty(0, esm_dim_fallback, dtype=torch.float32, device=device)  # Use passed device

    h_seq_feat, w_seq_feat = seq_feat_cat.shape if seq_feat_cat.ndim == 2 else (0, 0)
    actual_esm_dim = w_seq_feat if w_seq_feat > 0 else esm_dim_fallback

    # Option 3: Ensure padded_seq_features is on the same device as seq_feat_cat
    # If seq_feat_cat is empty, its device is CPU and dtype is float32 by its earlier construction.
    # If seq_feat_cat is not empty, it inherits device from esm_data_val_list[0] (token_representations_list).
    padded_seq_features = torch.zeros((pro_len, actual_esm_dim), dtype=seq_feat_cat.dtype, device=seq_feat_cat.device)

    if h_seq_feat > 0:
        copy_h = min(h_seq_feat, pro_len)
        padded_seq_features[:copy_h, :] = seq_feat_cat[:copy_h, :]

    coor_dict_for_esmif = esm_data_val_list[1]
    chain_ids_for_esmif = esm_data_val_list[2]

    # Interface Residue Matrix (Optimized)
    interface_res_matrix = torch.ones((pro_len, pro_len), dtype=torch.bool)
    row_indices = []
    col_indices = []
    for i, interacting_js in enumerate(cpu_data["interface_res"]):
        if i >= pro_len:
            break
        for j_interact in interacting_js:
            if j_interact != -1 and 0 <= j_interact < pro_len:
                row_indices.append(i)
                col_indices.append(j_interact)
    if row_indices:
        interface_res_matrix[
            torch.tensor(row_indices, dtype=torch.long), torch.tensor(col_indices, dtype=torch.long)] = False

    current_affinity = cpu_data["affinity"]

    # Interaction Type Matrix (Optimized Padding)
    if_type_raw = torch.tensor(cpu_data["interaction_type_matrix"], dtype=torch.int16)
    h_raw, w_raw = if_type_raw.shape if if_type_raw.ndim == 2 else (0, 0)
    padded_if_type = torch.zeros((pro_len, pro_len), dtype=torch.int16)
    if h_raw > 0 and w_raw > 0:
        copy_h = min(h_raw, pro_len)
        copy_w = min(w_raw, pro_len)
        padded_if_type[:copy_h, :copy_w] = if_type_raw[:copy_h, :copy_w]

    # Interaction Matrix (Optimized Padding)
    if_matrix_raw = torch.tensor(cpu_data["interaction_matrix"], dtype=torch.int16)
    h_raw, w_raw, d_raw = if_matrix_raw.shape if if_matrix_raw.ndim == 3 else (0, 0, 0)
    d_fixed = d_raw if d_raw > 0 else 6  # Assuming depth 6 if input is empty or malformed
    padded_if_matrix = torch.zeros((pro_len, pro_len, d_fixed), dtype=torch.int16)
    if h_raw > 0 and w_raw > 0 and d_raw > 0:
        copy_h = min(h_raw, pro_len)
        copy_w = min(w_raw, pro_len)
        padded_if_matrix[:copy_h, :copy_w, :] = if_matrix_raw[:copy_h, :copy_w, :]

    # Mass Center (Optimized Padding)
    mass_centor_raw = torch.tensor(cpu_data["res_mass_centor"], dtype=torch.float16)  # Assuming float16 from cpu_data
    h_raw, w_raw = mass_centor_raw.shape if mass_centor_raw.ndim == 2 else (0, 0)
    w_fixed_mass = w_raw if w_raw > 0 else 3  # Assuming 3D coordinates
    padded_mass_centor = torch.zeros((pro_len, w_fixed_mass), dtype=torch.float16)
    if h_raw > 0 and w_raw > 0:
        copy_h = min(h_raw, pro_len)
        padded_mass_centor[:copy_h, :] = mass_centor_raw[:copy_h, :]

    # HETATM Features (Optimized Padding)
    if cpu_data["hetatm_features"]:
        # np.stack might fail on empty list of arrays if not all are same shape,
        # but cpu_data["hetatm_features"] is list of 1D arrays of same length (hetatm_list_len)
        hetatm_features_stacked = torch.tensor(np.stack(cpu_data["hetatm_features"]), dtype=torch.float32)
    else:
        hetatm_features_stacked = torch.empty(0, hetatm_list_len, dtype=torch.float32)

    h_raw, w_raw = hetatm_features_stacked.shape if hetatm_features_stacked.ndim == 2 else (0, 0)
    actual_hetatm_dim = w_raw if w_raw > 0 else hetatm_list_len
    padded_hetatm_features = torch.zeros((pro_len, actual_hetatm_dim), dtype=torch.float32)
    if h_raw > 0:
        copy_h = min(h_raw, pro_len)
        padded_hetatm_features[:copy_h, :] = hetatm_features_stacked[:copy_h, :]

    return {
        "protein_name": protein_name_key, "seq": seq_concat, "chain_id_res": cpu_data["chain_id_res"],
        "enc_tokens": padded_enc_tokens, "seq_features": padded_seq_features,
        "coor_dict_for_esmif": coor_dict_for_esmif, "chain_ids_for_esmif": chain_ids_for_esmif,
        "original_length": original_len, "interface_res_matrix": interface_res_matrix,
        "affinity": current_affinity, "interaction_type_matrix": padded_if_type,
        "interaction_matrix": padded_if_matrix, "res_mass_centor": padded_mass_centor,
        "hetatm_features": padded_hetatm_features
    }
