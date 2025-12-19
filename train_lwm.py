# =============================================================================
# 1. IMPORTS AND WARNINGS SETUP
#    - Load necessary PyTorch modules, utilities, and suppress UserWarnings
# =============================================================================

# Fix matplotlib backend to avoid tkinter errors on Windows
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend

import os
import pickle
import random
import warnings
from collections import defaultdict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import pretrained_model  # Assuming this contains the LWM model definition
import mamba_model  # Mamba-based architecture
from utils import (
    count_parameters,
    create_train_dataloader,
    generate_channels_and_labels,
    tokenizer_train,
    train_lwm,
)

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 1.5. HDF5 HELPER FUNCTIONS
#     - Functions to save and load tokenized data in HDF5 format
# =============================================================================
def save_tokenized_to_hdf5(hdf5_path, tokenized_data_list, seq_length, chunk_size=32):
    """
    Save tokenized data to HDF5 file with optimal chunking for random access.

    Args:
        hdf5_path (str): Path to HDF5 file
        tokenized_data_list (list): List of [input_ids, masked_tokens, masked_pos] samples
        seq_length (int): Sequence length for this group
        chunk_size (int): Number of samples per chunk (default: 32)
    """
    num_samples = len(tokenized_data_list)
    if num_samples == 0:
        return

    # Convert first sample to numpy to get shapes
    sample = tokenized_data_list[0]
    input_ids_np = np.array(sample[0], dtype=np.float32)
    masked_tokens_np = np.array(sample[1], dtype=np.float32)
    masked_pos_np = np.array(sample[2], dtype=np.int32)

    input_ids_shape = input_ids_np.shape
    masked_tokens_shape = masked_tokens_np.shape
    masked_pos_shape = masked_pos_np.shape

    # Prepare arrays - convert all samples to numpy
    input_ids_array = np.array([np.array(s[0], dtype=np.float32) for s in tokenized_data_list], dtype=np.float32)
    masked_tokens_array = np.array([np.array(s[1], dtype=np.float32) for s in tokenized_data_list], dtype=np.float32)
    masked_pos_array = np.array([np.array(s[2], dtype=np.int32) for s in tokenized_data_list], dtype=np.int32)

    # Create or append to HDF5 file
    with h5py.File(hdf5_path, 'a') as f:
        # Create group for this sequence length
        group_name = f'length_{seq_length}'

        if group_name in f:
            # Group already exists, remove it to overwrite
            del f[group_name]

        group = f.create_group(group_name)

        # Create datasets with chunking and compression
        group.create_dataset(
            'input_ids',
            data=input_ids_array,
            chunks=(min(chunk_size, num_samples), *input_ids_shape),
            compression='gzip',
            compression_opts=4,
            dtype=np.float32
        )

        group.create_dataset(
            'masked_tokens',
            data=masked_tokens_array,
            chunks=(min(chunk_size, num_samples), *masked_tokens_shape),
            compression='gzip',
            compression_opts=4,
            dtype=np.float32
        )

        group.create_dataset(
            'masked_pos',
            data=masked_pos_array,
            chunks=(min(chunk_size, num_samples), *masked_pos_shape),
            compression='gzip',
            compression_opts=4,
            dtype=np.int32
        )

        # Store metadata
        group.attrs['num_samples'] = num_samples
        group.attrs['seq_length'] = seq_length
        group.attrs['input_ids_shape'] = input_ids_shape
        group.attrs['masked_tokens_shape'] = masked_tokens_shape
        group.attrs['masked_pos_shape'] = masked_pos_shape


# =============================================================================
# 2. SCENARIO LIST DEFINITION
#    - Define the list of scenario names to iterate over for data generation
# =============================================================================
def scenarios_list():
    scen_list = np.array(
        [
            "city_0_newyork_3p5_lwm",
            "city_1_losangeles_3p5_lwm",
            "city_2_chicago_3p5_lwm",
            "city_3_houston_3p5_lwm",
            "city_4_phoenix_3p5_lwm",
            "city_5_philadelphia_3p5_lwm",
            "city_6_miami_3p5_lwm",
            "city_7_sandiego_3p5_lwm",
            "city_8_dallas_3p5_lwm",
            "city_9_sanfrancisco_3p5_lwm",
            "city_10_austin_3p5_lwm",
            "city_11_santaclara_3p5_lwm",
            "city_12_fortworth_3p5_lwm",
            "city_13_columbus_3p5_lwm",
            "city_14_charlotte_3p5_lwm",
            "city_15_indianapolis_3p5_lwm",
            "city_16_sanfrancisco_3p5_lwm",
            "city_17_seattle_3p5_lwm",
            "city_18_denver_3p5_lwm",
            "city_19_oklahoma_3p5_lwm",
            "asu_campus_3p5",
            "o1_3p5",
            "boston5G_3p5",
        ]
    )
    return scen_list


# =============================================================================
# 3. SCENARIO PROPERTIES MAPPING
#    - Map each scenario name to its corresponding rows, antenna count, and subcarrier count
# =============================================================================


def scenario_prop():
    row_column_users = {
        "city_0_newyork_3p5_lwm": {
            "n_rows": 109,
            "n_per_row": 291,
            "grid_idx": 1,
            "n_ant_bs": 8,
            "n_subcarriers": 32,
        },
        "city_1_losangeles_3p5_lwm": {
            "n_rows": 142,
            "n_per_row": 201,
            "grid_idx": 1,
            "n_ant_bs": 8,
            "n_subcarriers": 64,
        },
        "city_2_chicago_3p5_lwm": {
            "n_rows": 139,
            "n_per_row": 200,
            "grid_idx": 1,
            "n_ant_bs": 8,
            "n_subcarriers": 128,
        },
        "city_3_houston_3p5_lwm": {
            "n_rows": 154,
            "n_per_row": 202,
            "grid_idx": 1,
            "n_ant_bs": 8,
            "n_subcarriers": 256,
        },
        "city_4_phoenix_3p5_lwm": {
            "n_rows": 198,
            "n_per_row": 214,
            "grid_idx": 1,
            "n_ant_bs": 8,
            "n_subcarriers": 512,
        },
        "city_5_philadelphia_3p5_lwm": {
            "n_rows": 239,
            "n_per_row": 164,
            "grid_idx": 1,
            "n_ant_bs": 8,
            "n_subcarriers": 1024,
        },
        "city_6_miami_3p5_lwm": {
            "n_rows": 199,
            "n_per_row": 216,
            "grid_idx": 1,
            "n_ant_bs": 16,
            "n_subcarriers": 32,
        },
        "city_7_sandiego_3p5_lwm": {
            "n_rows": 176,
            "n_per_row": 207,
            "grid_idx": 1,
            "n_ant_bs": 16,
            "n_subcarriers": 64,
        },
        "city_8_dallas_3p5_lwm": {
            "n_rows": 207,
            "n_per_row": 190,
            "grid_idx": 1,
            "n_ant_bs": 16,
            "n_subcarriers": 128,
        },
        "city_9_sanfrancisco_3p5_lwm": {
            "n_rows": 196,
            "n_per_row": 206,
            "grid_idx": 1,
            "n_ant_bs": 16,
            "n_subcarriers": 256,
        },
        "city_10_austin_3p5_lwm": {
            "n_rows": 255,
            "n_per_row": 137,
            "grid_idx": 1,
            "n_ant_bs": 16,
            "n_subcarriers": 512,
        },
        "city_11_santaclara_3p5_lwm": {
            "n_rows": 117,
            "n_per_row": 285,
            "grid_idx": 1,
            "n_ant_bs": 32,
            "n_subcarriers": 32,
        },
        "city_12_fortworth_3p5_lwm": {
            "n_rows": 214,
            "n_per_row": 179,
            "grid_idx": 1,
            "n_ant_bs": 32,
            "n_subcarriers": 64,
        },
        "city_13_columbus_3p5_lwm": {
            "n_rows": 178,
            "n_per_row": 240,
            "grid_idx": 1,
            "n_ant_bs": 32,
            "n_subcarriers": 128,
        },
        "city_14_charlotte_3p5_lwm": {
            "n_rows": 216,
            "n_per_row": 177,
            "grid_idx": 1,
            "n_ant_bs": 32,
            "n_subcarriers": 256,
        },
        "city_15_indianapolis_3p5_lwm": {
            "n_rows": 200,
            "n_per_row": 196,
            "grid_idx": 1,
            "n_ant_bs": 64,
            "n_subcarriers": 32,
        },
        "city_16_sanfrancisco_3p5_lwm": {
            "n_rows": 201,
            "n_per_row": 208,
            "grid_idx": 1,
            "n_ant_bs": 64,
            "n_subcarriers": 64,
        },
        "city_17_seattle_3p5_lwm": {
            "n_rows": 185,
            "n_per_row": 205,
            "grid_idx": 1,
            "n_ant_bs": 64,
            "n_subcarriers": 128,
        },
        "city_18_denver_3p5_lwm": {
            "n_rows": 212,
            "n_per_row": 204,
            "grid_idx": 1,
            "n_ant_bs": 128,
            "n_subcarriers": 32,
        },
        "city_19_oklahoma_3p5_lwm": {
            "n_rows": 204,
            "n_per_row": 188,
            "grid_idx": 1,
            "n_ant_bs": 128,
            "n_subcarriers": 64,
        },
        "asu_campus_3p5_v1": {
            "n_rows": [0, 1 * int(321 / 20)],
            "n_per_row": 411,
            "grid_idx": 1,
            "n_ant_bs": 8,
            "n_subcarriers": 32,
        },
        "asu_campus_3p5_v2": {
            "n_rows": [1 * int(321 / 20), 2 * int(321 / 20)],
            "n_per_row": 411,
            "grid_idx": 1,
            "n_ant_bs": 8,
            "n_subcarriers": 64,
        },
        "asu_campus_3p5_v3": {
            "n_rows": [2 * int(321 / 20), 3 * int(321 / 20)],
            "n_per_row": 411,
            "grid_idx": 1,
            "n_ant_bs": 8,
            "n_subcarriers": 128,
        },
        "asu_campus_3p5_v4": {
            "n_rows": [3 * int(321 / 20), 4 * int(321 / 20)],
            "n_per_row": 411,
            "grid_idx": 1,
            "n_ant_bs": 8,
            "n_subcarriers": 256,
        },
        "asu_campus_3p5_v5": {
            "n_rows": [4 * int(321 / 20), 5 * int(321 / 20)],
            "n_per_row": 411,
            "grid_idx": 1,
            "n_ant_bs": 8,
            "n_subcarriers": 512,
        },
        "asu_campus_3p5_v6": {
            "n_rows": [5 * int(321 / 20), 6 * int(321 / 20)],
            "n_per_row": 411,
            "grid_idx": 1,
            "n_ant_bs": 8,
            "n_subcarriers": 1024,
        },
        "asu_campus_3p5_v7": {
            "n_rows": [6 * int(321 / 20), 7 * int(321 / 20)],
            "n_per_row": 411,
            "grid_idx": 1,
            "n_ant_bs": 16,
            "n_subcarriers": 32,
        },
        "asu_campus_3p5_v8": {
            "n_rows": [7 * int(321 / 20), 8 * int(321 / 20)],
            "n_per_row": 411,
            "grid_idx": 1,
            "n_ant_bs": 16,
            "n_subcarriers": 64,
        },
        "asu_campus_3p5_v9": {
            "n_rows": [8 * int(321 / 20), 9 * int(321 / 20)],
            "n_per_row": 411,
            "grid_idx": 1,
            "n_ant_bs": 16,
            "n_subcarriers": 128,
        },
        "asu_campus_3p5_v10": {
            "n_rows": [9 * int(321 / 20), 10 * int(321 / 20)],
            "n_per_row": 411,
            "grid_idx": 1,
            "n_ant_bs": 16,
            "n_subcarriers": 256,
        },
        "asu_campus_3p5_v11": {
            "n_rows": [10 * int(321 / 20), 11 * int(321 / 20)],
            "n_per_row": 411,
            "grid_idx": 1,
            "n_ant_bs": 16,
            "n_subcarriers": 512,
        },
        "asu_campus_3p5_v12": {
            "n_rows": [11 * int(321 / 20), 12 * int(321 / 20)],
            "n_per_row": 411,
            "grid_idx": 1,
            "n_ant_bs": 32,
            "n_subcarriers": 32,
        },
        "asu_campus_3p5_v13": {
            "n_rows": [12 * int(321 / 20), 13 * int(321 / 20)],
            "n_per_row": 411,
            "grid_idx": 1,
            "n_ant_bs": 32,
            "n_subcarriers": 64,
        },
        "asu_campus_3p5_v14": {
            "n_rows": [13 * int(321 / 20), 14 * int(321 / 20)],
            "n_per_row": 411,
            "grid_idx": 1,
            "n_ant_bs": 32,
            "n_subcarriers": 128,
        },
        "asu_campus_3p5_v15": {
            "n_rows": [14 * int(321 / 20), 15 * int(321 / 20)],
            "n_per_row": 411,
            "grid_idx": 1,
            "n_ant_bs": 32,
            "n_subcarriers": 256,
        },
        "asu_campus_3p5_v16": {
            "n_rows": [15 * int(321 / 20), 16 * int(321 / 20)],
            "n_per_row": 411,
            "grid_idx": 1,
            "n_ant_bs": 64,
            "n_subcarriers": 32,
        },
        "asu_campus_3p5_v17": {
            "n_rows": [16 * int(321 / 20), 17 * int(321 / 20)],
            "n_per_row": 411,
            "grid_idx": 1,
            "n_ant_bs": 64,
            "n_subcarriers": 64,
        },
        "asu_campus_3p5_v18": {
            "n_rows": [17 * int(321 / 20), 18 * int(321 / 20)],
            "n_per_row": 411,
            "grid_idx": 1,
            "n_ant_bs": 64,
            "n_subcarriers": 128,
        },
        "asu_campus_3p5_v19": {
            "n_rows": [18 * int(321 / 20), 19 * int(321 / 20)],
            "n_per_row": 411,
            "grid_idx": 1,
            "n_ant_bs": 128,
            "n_subcarriers": 32,
        },
        "asu_campus_3p5_v20": {
            "n_rows": [19 * int(321 / 20), 20 * int(321 / 20)],
            "n_per_row": 411,
            "grid_idx": 1,
            "n_ant_bs": 128,
            "n_subcarriers": 64,
        },
        "boston5G_3p5_v1": {
            "n_rows": [812 + 0, 812 + 1 * int((1622 - 812) / 20)],
            "n_per_row": 595,
            "grid_idx": 2,
            "n_ant_bs": 8,
            "n_subcarriers": 32,
        },
        "boston5G_3p5_v2": {
            "n_rows": [
                812 + 1 * int((1622 - 812) / 20),
                812 + 2 * int((1622 - 812) / 20),
            ],
            "n_per_row": 595,
            "grid_idx": 2,
            "n_ant_bs": 8,
            "n_subcarriers": 64,
        },
        "boston5G_3p5_v3": {
            "n_rows": [
                812 + 2 * int((1622 - 812) / 20),
                812 + 3 * int((1622 - 812) / 20),
            ],
            "n_per_row": 595,
            "grid_idx": 2,
            "n_ant_bs": 8,
            "n_subcarriers": 128,
        },
        "boston5G_3p5_v4": {
            "n_rows": [
                812 + 3 * int((1622 - 812) / 20),
                812 + 4 * int((1622 - 812) / 20),
            ],
            "n_per_row": 595,
            "grid_idx": 2,
            "n_ant_bs": 8,
            "n_subcarriers": 256,
        },
        "boston5G_3p5_v5": {
            "n_rows": [
                812 + 4 * int((1622 - 812) / 20),
                812 + 5 * int((1622 - 812) / 20),
            ],
            "n_per_row": 595,
            "grid_idx": 2,
            "n_ant_bs": 8,
            "n_subcarriers": 512,
        },
        "boston5G_3p5_v6": {
            "n_rows": [
                812 + 5 * int((1622 - 812) / 20),
                812 + 6 * int((1622 - 812) / 20),
            ],
            "n_per_row": 595,
            "grid_idx": 2,
            "n_ant_bs": 8,
            "n_subcarriers": 1024,
        },
        "boston5G_3p5_v7": {
            "n_rows": [
                812 + 6 * int((1622 - 812) / 20),
                812 + 7 * int((1622 - 812) / 20),
            ],
            "n_per_row": 595,
            "grid_idx": 2,
            "n_ant_bs": 16,
            "n_subcarriers": 32,
        },
        "boston5G_3p5_v8": {
            "n_rows": [
                812 + 7 * int((1622 - 812) / 20),
                812 + 8 * int((1622 - 812) / 20),
            ],
            "n_per_row": 595,
            "grid_idx": 2,
            "n_ant_bs": 16,
            "n_subcarriers": 64,
        },
        "boston5G_3p5_v9": {
            "n_rows": [
                812 + 8 * int((1622 - 812) / 20),
                812 + 9 * int((1622 - 812) / 20),
            ],
            "n_per_row": 595,
            "grid_idx": 2,
            "n_ant_bs": 16,
            "n_subcarriers": 128,
        },
        "boston5G_3p5_v10": {
            "n_rows": [
                812 + 9 * int((1622 - 812) / 20),
                812 + 10 * int((1622 - 812) / 20),
            ],
            "n_per_row": 595,
            "grid_idx": 2,
            "n_ant_bs": 16,
            "n_subcarriers": 256,
        },
        "boston5G_3p5_v11": {
            "n_rows": [
                812 + 10 * int((1622 - 812) / 20),
                812 + 11 * int((1622 - 812) / 20),
            ],
            "n_per_row": 595,
            "grid_idx": 2,
            "n_ant_bs": 16,
            "n_subcarriers": 512,
        },
        "boston5G_3p5_v12": {
            "n_rows": [
                812 + 11 * int((1622 - 812) / 20),
                812 + 12 * int((1622 - 812) / 20),
            ],
            "n_per_row": 595,
            "grid_idx": 2,
            "n_ant_bs": 32,
            "n_subcarriers": 32,
        },
        "boston5G_3p5_v13": {
            "n_rows": [
                812 + 12 * int((1622 - 812) / 20),
                812 + 13 * int((1622 - 812) / 20),
            ],
            "n_per_row": 595,
            "grid_idx": 2,
            "n_ant_bs": 32,
            "n_subcarriers": 64,
        },
        "boston5G_3p5_v14": {
            "n_rows": [
                812 + 13 * int((1622 - 812) / 20),
                812 + 14 * int((1622 - 812) / 20),
            ],
            "n_per_row": 595,
            "grid_idx": 2,
            "n_ant_bs": 32,
            "n_subcarriers": 128,
        },
        "boston5G_3p5_v15": {
            "n_rows": [
                812 + 14 * int((1622 - 812) / 20),
                812 + 15 * int((1622 - 812) / 20),
            ],
            "n_per_row": 595,
            "grid_idx": 2,
            "n_ant_bs": 32,
            "n_subcarriers": 256,
        },
        "boston5G_3p5_v16": {
            "n_rows": [
                812 + 15 * int((1622 - 812) / 20),
                812 + 16 * int((1622 - 812) / 20),
            ],
            "n_per_row": 595,
            "grid_idx": 2,
            "n_ant_bs": 64,
            "n_subcarriers": 32,
        },
        "boston5G_3p5_v17": {
            "n_rows": [
                812 + 16 * int((1622 - 812) / 20),
                812 + 17 * int((1622 - 812) / 20),
            ],
            "n_per_row": 595,
            "grid_idx": 2,
            "n_ant_bs": 64,
            "n_subcarriers": 64,
        },
        "boston5G_3p5_v18": {
            "n_rows": [
                812 + 17 * int((1622 - 812) / 20),
                812 + 18 * int((1622 - 812) / 20),
            ],
            "n_per_row": 595,
            "grid_idx": 2,
            "n_ant_bs": 64,
            "n_subcarriers": 128,
        },
        "boston5G_3p5_v19": {
            "n_rows": [
                812 + 18 * int((1622 - 812) / 20),
                812 + 19 * int((1622 - 812) / 20),
            ],
            "n_per_row": 595,
            "grid_idx": 2,
            "n_ant_bs": 128,
            "n_subcarriers": 32,
        },
        "boston5G_3p5_v20": {
            "n_rows": [
                812 + 19 * int((1622 - 812) / 20),
                812 + 20 * int((1622 - 812) / 20),
            ],
            "n_per_row": 595,
            "grid_idx": 2,
            "n_ant_bs": 128,
            "n_subcarriers": 64,
        },
        "o1_3p5_v1": {
            "n_rows": [0 * int(3852 / 12), 1 * int(3852 / 12)],
            "n_per_row": 181,
            "grid_idx": 1,
            "n_ant_bs": 8,
            "n_subcarriers": 32,
        },
        "o1_3p5_v2": {
            "n_rows": [1 * int(3852 / 12), 2 * int(3852 / 12)],
            "n_per_row": 181,
            "grid_idx": 1,
            "n_ant_bs": 8,
            "n_subcarriers": 64,
        },
        "o1_3p5_v3": {
            "n_rows": [2 * int(3852 / 12), 3 * int(3852 / 12)],
            "n_per_row": 181,
            "grid_idx": 1,
            "n_ant_bs": 8,
            "n_subcarriers": 128,
        },
        "o1_3p5_v4": {
            "n_rows": [3 * int(3852 / 12), 4 * int(3852 / 12)],
            "n_per_row": 181,
            "grid_idx": 1,
            "n_ant_bs": 8,
            "n_subcarriers": 256,
        },
        "o1_3p5_v5": {
            "n_rows": [4 * int(3852 / 12), 5 * int(3852 / 12)],
            "n_per_row": 181,
            "grid_idx": 1,
            "n_ant_bs": 8,
            "n_subcarriers": 512,
        },
        "o1_3p5_v6": {
            "n_rows": [5 * int(3852 / 12), 6 * int(3852 / 12)],
            "n_per_row": 181,
            "grid_idx": 1,
            "n_ant_bs": 8,
            "n_subcarriers": 1024,
        },
        "o1_3p5_v7": {
            "n_rows": [6 * int(3852 / 12), 7 * int(3852 / 12)],
            "n_per_row": 181,
            "grid_idx": 1,
            "n_ant_bs": 16,
            "n_subcarriers": 32,
        },
        "o1_3p5_v8": {
            "n_rows": [7 * int(3852 / 12), 8 * int(3852 / 12)],
            "n_per_row": 181,
            "grid_idx": 1,
            "n_ant_bs": 16,
            "n_subcarriers": 64,
        },
        "o1_3p5_v9": {
            "n_rows": [8 * int(3852 / 12), 2750],
            "n_per_row": 181,
            "grid_idx": 1,
            "n_ant_bs": 16,
            "n_subcarriers": 128,
        },
        "o1_3p5_v10": {
            "n_rows": [2751, 10 * int(3852 / 12)],
            "n_per_row": 181,
            "grid_idx": 2,
            "n_ant_bs": 16,
            "n_subcarriers": 256,
        },
        "o1_3p5_v11": {
            "n_rows": [10 * int(3852 / 12), 11 * int(3852 / 12)],
            "n_per_row": 181,
            "grid_idx": 2,
            "n_ant_bs": 16,
            "n_subcarriers": 512,
        },
        "o1_3p5_v12": {
            "n_rows": [11 * int(3852 / 12), 3851],
            "n_per_row": 181,
            "grid_idx": 2,
            "n_ant_bs": 32,
            "n_subcarriers": 32,
        },
        "o1_3p5_v13": {
            "n_rows": [3852, 12 * int(3852 / 12) + 1 * int(1351 / 10)],
            "n_per_row": 361,
            "grid_idx": 3,
            "n_ant_bs": 32,
            "n_subcarriers": 64,
        },
        "o1_3p5_v14": {
            "n_rows": [
                12 * int(3852 / 12) + 1 * int(1351 / 10),
                12 * int(3852 / 12) + 2 * int(1351 / 10),
            ],
            "n_per_row": 181,
            "grid_idx": 3,
            "n_ant_bs": 32,
            "n_subcarriers": 128,
        },
        "o1_3p5_v15": {
            "n_rows": [
                12 * int(3852 / 12) + 2 * int(1351 / 10),
                12 * int(3852 / 12) + 3 * int(1351 / 10),
            ],
            "n_per_row": 181,
            "grid_idx": 3,
            "n_ant_bs": 32,
            "n_subcarriers": 256,
        },
        "o1_3p5_v16": {
            "n_rows": [
                12 * int(3852 / 12) + 3 * int(1351 / 10),
                12 * int(3852 / 12) + 4 * int(1351 / 10),
            ],
            "n_per_row": 181,
            "grid_idx": 3,
            "n_ant_bs": 64,
            "n_subcarriers": 32,
        },
        "o1_3p5_v17": {
            "n_rows": [
                12 * int(3852 / 12) + 4 * int(1351 / 10),
                12 * int(3852 / 12) + 5 * int(1351 / 10),
            ],
            "n_per_row": 181,
            "grid_idx": 3,
            "n_ant_bs": 64,
            "n_subcarriers": 64,
        },
        "o1_3p5_v18": {
            "n_rows": [
                12 * int(3852 / 12) + 5 * int(1351 / 10),
                12 * int(3852 / 12) + 6 * int(1351 / 10),
            ],
            "n_per_row": 181,
            "grid_idx": 3,
            "n_ant_bs": 64,
            "n_subcarriers": 128,
        },
        "o1_3p5_v19": {
            "n_rows": [
                12 * int(3852 / 12) + 6 * int(1351 / 10),
                12 * int(3852 / 12) + 7 * int(1351 / 10),
            ],
            "n_per_row": 181,
            "grid_idx": 3,
            "n_ant_bs": 128,
            "n_subcarriers": 32,
        },
        "o1_3p5_v20": {
            "n_rows": [
                12 * int(3852 / 12) + 7 * int(1351 / 10),
                12 * int(3852 / 12) + 8 * int(1351 / 10),
            ],
            "n_per_row": 181,
            "grid_idx": 3,
            "n_ant_bs": 128,
            "n_subcarriers": 64,
        },
    }
    return row_column_users


# =============================================================================
# 4. TRAINING PARAMETERS AND HYPERPARAMETERS
#    - Set training epochs, batch sizes, learning rates, model dimensions, etc.
# =============================================================================

EPOCHS = 100
BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
WARMUP_EPOCHS = 10
BASE_LR = 5e-4
MIN_LR = 1e-8
N_ROWS = 4
N_COLUMNS = 4
ELEMENT_LENGTH = N_ROWS * N_COLUMNS * 2
D_MODEL = 128
MAX_LEN = 513
N_LAYERS = 12
N_ANT_BS = 64
N_SUBCARRIERS = 64
device_idx = 0
WEIGHT_DECAY = 0.05
BETA1 = 0.9
BETA2 = 0.999
MASK_PERCENT = 0.4
N_HEADS = 8
DROPOUT = 0.1
task = [
    "LosNlosClassification",
    "BeamPrediction",
    "ChannelInterpolation",
    "ChannelEstimation",
    "ChannelCharting",
    None,
][-1]

# Model architecture selection: "transformer" or "mamba"
MODEL_ARCHITECTURE = "transformer"  # Change to "mamba" to use Mamba-based architecture
save_dir = f"pretrained_models_{MODEL_ARCHITECTURE}"

# Mamba-specific hyperparameters (only used if MODEL_ARCHITECTURE == "mamba")
D_STATE = 8  # SSM state dimension
D_CONV = 4    # Convolution kernel size
EXPAND = 1.2    # Expansion factor for Mamba blocks
BIDIRECTIONAL = True  # Use bidirectional Mamba for masked language modeling

# Filter for specific sequence lengths (set to None or [] to use all sequence lengths)
# Example: FILTER_SEQ_LENGTHS = [33, 65, 129] to train only on these lengths
FILTER_SEQ_LENGTHS = [17, 33, 65, 129, 257]  # Set to None or [] to use all, or specify list like [33, 65, 129]

# =============================================================================
# 5. DATA GENERATION LOOP
#    - Iterate over scenarios and base station indices to generate channel samples and labels
#    - Handle both full-scenario and zoned sub-scenarios for campus and Boston data
# =============================================================================
if __name__ == "__main__":
    scenarios = scenarios_list()

    channels = []
    labels = []
    scenario_properties = scenario_prop()
    preprocessed_data_dict = defaultdict(list)

    # Create cache directory for scenario data
    cache_dir = "/mnt/c/Users/tomer/scenarios_cache_lwm"
    os.makedirs(cache_dir, exist_ok=True)

    # Create directory for tokenized data organized by sequence length
    tokenized_base_dir = "/mnt/c/Users/tomer/tokenized_data_lwm"
    os.makedirs(tokenized_base_dir, exist_ok=True)

    for scenario in scenarios[:-3]:
        for bs_idx in range(1, 4):
            # Create a unique cache filename for this scenario + bs_idx combination
            cache_filename = f"{scenario}_bs{bs_idx}.pkl"
            cache_filepath = os.path.join(cache_dir, cache_filename)

            # Check if cached data exists
            if os.path.exists(cache_filepath):
                print(f"\nLoading cached data for scenario: {scenario}, BS #{bs_idx}")
                with open(cache_filepath, "rb") as f:
                    cached_data = pickle.load(f)
                    scenario_channels = cached_data["channels"]
                    scenario_labels = cached_data["labels"]
            else:
                # Generate data if cache doesn't exist
                scenario_channels, scenario_labels = generate_channels_and_labels(
                    n_ant_bs=scenario_properties[scenario]["n_ant_bs"],
                    n_subcarriers=scenario_properties[scenario]["n_subcarriers"],
                    bs_idx=bs_idx,
                    scenario_name=scenario,
                    task=task,
                    n_beams=64,
                )

                # Save to cache
                print(f"Saving data to cache: {cache_filepath}")
                with open(cache_filepath, "wb") as f:
                    pickle.dump(
                        {"channels": scenario_channels, "labels": scenario_labels}, f, protocol=5
                    )

            # Check if scenario has any valid channels before tokenizing
            if len(scenario_channels) == 0:
                print(f"\nSkipping scenario: {scenario}, BS #{bs_idx} (no valid channels)")
                continue

            # Check if tokenized HDF5 file already exists
            hdf5_filename = f"{scenario}_bs{bs_idx}.h5"
            hdf5_filepath = os.path.join(tokenized_base_dir, hdf5_filename)

            if os.path.exists(hdf5_filepath):
                print(f"\nTokenized data already exists for {scenario}, BS #{bs_idx}, skipping tokenization")
                continue

            # Tokenize the data
            print(f"\nTokenizing scenario: {scenario}, BS #{bs_idx}")
            scenario_preprocessed_dict = tokenizer_train(
                [scenario_channels],
                max_len=MAX_LEN,
                masking_percent=MASK_PERCENT,
                mask=True,
                seed=42,
            )

            # Save tokenized data to HDF5 file (one file per scenario+bs, with groups by sequence length)
            print(f"Saving tokenized data to: {hdf5_filepath}")
            for seq_length, tokenized_samples in scenario_preprocessed_dict.items():
                save_tokenized_to_hdf5(
                    hdf5_filepath,
                    tokenized_samples,
                    seq_length=int(seq_length),
                    chunk_size=32
                )
                print(f"  Saved {len(tokenized_samples)} samples for sequence length {seq_length}")

            labels.extend(scenario_labels)

    bs_idxs = [[1], [4, 15], [2]]
    for scenario_idx, scenario in enumerate(scenarios[-3:]):
        for bs_idx in bs_idxs[scenario_idx]:
            for zone in range(20):
                row_start = scenario_properties[scenario + f"_v{zone + 1}"]["n_rows"][0]
                row_end = scenario_properties[scenario + f"_v{zone + 1}"]["n_rows"][1]
                grid_idx = (
                    scenario_properties[scenario + f"_v{zone + 1}"]["grid_idx"] - 1
                )
                print(f"Processing scenario: {scenario}, BS #{bs_idx}, Zone {zone}")

                # Create a unique cache filename for this scenario + bs_idx combination
                cache_filename = f"{scenario}_bs{bs_idx}_zone{zone}.pkl"
                cache_filepath = os.path.join(cache_dir, cache_filename)

                # Check if cached data exists
                if os.path.exists(cache_filepath):
                    print(
                        f"\nLoading cached data for scenario: {scenario}, BS #{bs_idx}"
                    )
                    with open(cache_filepath, "rb") as f:
                        cached_data = pickle.load(f)
                        scenario_channels = cached_data["channels"]
                        scenario_labels = cached_data["labels"]
                else:
                    # Generate data if cache doesn't exist
                    scenario_channels, scenario_labels = generate_channels_and_labels(
                        n_ant_bs=scenario_properties[scenario + f"_v{zone + 1}"][
                            "n_ant_bs"
                        ],
                        n_subcarriers=scenario_properties[scenario + f"_v{zone + 1}"][
                            "n_subcarriers"
                        ],
                        grid_idx=grid_idx,
                        bs_idx=bs_idx,
                        scenario_name=scenario,
                        rows=np.arange(row_start, row_end),
                        task=task,
                        n_beams=64,
                    )

                    # Save to cache
                    print(f"Saving data to cache: {cache_filepath}")
                    with open(cache_filepath, "wb") as f:
                        pickle.dump(
                            {"channels": scenario_channels, "labels": scenario_labels},
                            f,
                            protocol=5
                        )

                # Check if scenario has any valid channels before tokenizing
                if len(scenario_channels) == 0:
                    print(f"\nSkipping scenario: {scenario}, BS #{bs_idx}, Zone {zone} (no valid channels)")
                    continue

                # Check if tokenized HDF5 file already exists
                hdf5_filename = f"{scenario}_bs{bs_idx}_zone{zone}.h5"
                hdf5_filepath = os.path.join(tokenized_base_dir, hdf5_filename)

                if os.path.exists(hdf5_filepath):
                    print(f"\nTokenized data already exists for {scenario}, BS #{bs_idx}, Zone {zone}, skipping tokenization")
                    continue

                # Tokenize the data
                print(f"\nTokenizing scenario: {scenario}, BS #{bs_idx}, Zone {zone}")
                scenario_preprocessed_dict = tokenizer_train(
                    [scenario_channels],
                    max_len=MAX_LEN,
                    masking_percent=MASK_PERCENT,
                    mask=True,
                    seed=42,
                )

                # Save tokenized data to HDF5 file (one file per scenario+bs+zone, with groups by sequence length)
                print(f"Saving tokenized data to: {hdf5_filepath}")
                for seq_length, tokenized_samples in scenario_preprocessed_dict.items():
                    save_tokenized_to_hdf5(
                        hdf5_filepath,
                        tokenized_samples,
                        seq_length=int(seq_length),
                        chunk_size=32
                    )
                    print(f"  Saved {len(tokenized_samples)} samples for sequence length {seq_length}")

                labels.extend(scenario_labels)

    # Collect tokenized data file metadata (lazy loading - don't load all data into memory)
    print("\n" + "=" * 80)
    print("Collecting tokenized data file metadata...")
    print("=" * 80)

    # Apply sequence length filter if specified
    if FILTER_SEQ_LENGTHS:
        print(f"\nFiltering to specific sequence lengths: {sorted(FILTER_SEQ_LENGTHS)}")
        filter_keys = set(str(length) for length in FILTER_SEQ_LENGTHS)
    else:
        print("\nUsing all sequence lengths")
        filter_keys = None

    file_metadata = defaultdict(list)  # {seq_len: [(filepath, group_name, num_samples), ...]}

    # Iterate through all HDF5 files in tokenized_base_dir
    for filename in os.listdir(tokenized_base_dir):
        if filename.endswith(".h5"):
            hdf5_filepath = os.path.join(tokenized_base_dir, filename)

            print(f"\nProcessing HDF5 file: {filename}")

            try:
                with h5py.File(hdf5_filepath, 'r') as f:
                    # Iterate through all groups (sequence lengths) in this file
                    for group_name in f.keys():
                        if group_name.startswith('length_'):
                            seq_len_key = group_name.replace('length_', '')

                            # Skip if filtering is enabled and this sequence length is not in the filter
                            if filter_keys and seq_len_key not in filter_keys:
                                continue

                            group = f[group_name]
                            num_samples = group.attrs['num_samples']

                            # Store: (filepath, group_name, num_samples)
                            file_metadata[seq_len_key].append((hdf5_filepath, group_name, num_samples))
                            print(f"  Found group '{group_name}' with {num_samples} samples")

            except (OSError, KeyError) as e:
                print(f"  WARNING: Error reading HDF5 file: {filename}")
                print(f"  Error: {e}")
                print(f"  Skipping this file.")

    # Print summary
    for seq_len_key in sorted(file_metadata.keys(), key=int):
        total_samples = sum(count for _, _, count in file_metadata[seq_len_key])
        print(f"\nSequence length {seq_len_key}: {total_samples} total samples from {len(file_metadata[seq_len_key])} file(s)")

    print("\n" + "=" * 80)
    print(f"Total sequence length keys: {len(file_metadata)}")
    print("=" * 80)

    # =============================================================================
    # 7. TRAIN/VALIDATION/TEST SPLIT
    #    - Split indices for each sequence length into train, validation, and test subsets
    # =============================================================================

    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_ratio = 0.8
    val_ratio = 0.2

    train_data = {}  # Will store: {seq_len: (file_metadata, train_indices)}
    val_data = {}
    test_data = {}

    for key in file_metadata.keys():
        print(f"\nSplitting data for sequence length: {key}")

        # Build global indices: list of (file_idx, sample_idx_within_file)
        indices = []
        for file_idx, (filepath, group_name, num_samples) in enumerate(file_metadata[key]):
            for sample_idx in range(num_samples):
                indices.append((file_idx, sample_idx))

        total_samples = len(indices)
        print(f"  Total samples: {total_samples}")

        # Shuffle with fixed seed for reproducibility
        random.shuffle(indices)

        # Split
        train_size = int(train_ratio * total_samples)
        val_size = int(val_ratio * total_samples)
        test_size = total_samples - train_size - val_size

        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]

        print(
            f"  Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}"
        )

        # Store file metadata and corresponding indices
        train_data[key] = (file_metadata[key], train_indices)
        val_data[key] = (file_metadata[key], val_indices)
        test_data[key] = (file_metadata[key], test_indices)

    # =============================================================================
    # 8. DATALOADER CREATION
    #    - Build PyTorch DataLoader objects for batched training and validation
    # =============================================================================

    train_loaders = create_train_dataloader(
        train_data, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loaders = create_train_dataloader(
        val_data, batch_size=VAL_BATCH_SIZE, shuffle=False
    )

    # =============================================================================
    # 9. MODEL INITIALIZATION
    #    - Instantiate the LWM model (transformer or mamba) and optionally load pre-trained weights
    #    - Wrap with DataParallel for multi-GPU support
    # =============================================================================

    gpu_ids = [0]  # device_idx
    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")

    # Select model architecture based on MODEL_ARCHITECTURE flag
    if MODEL_ARCHITECTURE.lower() == "mamba":
        print("=" * 80)
        print("Initializing Mamba-based LWM model")
        print(f"Bidirectional: {BIDIRECTIONAL}")
        print("=" * 80)
        model = mamba_model.lwm_mamba(
            element_length=ELEMENT_LENGTH,
            d_model=D_MODEL,
            n_layers=N_LAYERS,
            max_len=MAX_LEN,
            d_state=D_STATE,
            d_conv=D_CONV,
            expand=EXPAND,
            dropout=DROPOUT,
            bidirectional=BIDIRECTIONAL,
        ).to(device)
    elif MODEL_ARCHITECTURE.lower() == "transformer":
        print("=" * 80)
        print("Initializing Transformer-based LWM model")
        print("=" * 80)
        model = pretrained_model.lwm(
            element_length=ELEMENT_LENGTH,
            d_model=D_MODEL,
            n_layers=N_LAYERS,
            max_len=MAX_LEN,
            n_heads=N_HEADS,
            dropout=DROPOUT,
        ).to(device)
    else:
        raise ValueError(f"Unknown model architecture: {MODEL_ARCHITECTURE}. Choose 'transformer' or 'mamba'.")

    # Optional: Load pre-trained model
    load_model = False
    if load_model:
        model.load_state_dict(
            torch.load("models/model_checkpoint.pth", map_location=device)
        )
        print("Pre-trained model loaded successfully.")

    # Use DataParallel for multi-GPU support
    model = nn.DataParallel(model, device_ids=gpu_ids)
    print(f"Model loaded successfully on GPU {device.index}")
    n_parameters = count_parameters(model)
    print(f"Number of trainable parameters: {n_parameters:,}")
    print(f"Architecture: {MODEL_ARCHITECTURE}")
    if MODEL_ARCHITECTURE.lower() == "mamba":
        print(f"Bidirectional Mode: {BIDIRECTIONAL}")
    print("=" * 80)

    # =============================================================================
    # 10. OPTIMIZER AND LEARNING RATE SCHEDULER
    #     - Configure AdamW optimizer and a cosine-with-warmup LR schedule based on total steps
    # =============================================================================

    TOTAL_STEPS = sum(len(loader) for loader in train_loaders.values()) * EPOCHS
    WARMUP_STEPS = sum(len(loader) for loader in train_loaders.values()) * WARMUP_EPOCHS

    optimizer = AdamW(
        model.parameters(), lr=BASE_LR, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY
    )

    def lr_lambda(current_step):
        if current_step < WARMUP_STEPS:
            return current_step / WARMUP_STEPS
        else:
            scaled_progress = (current_step - WARMUP_STEPS) / (
                TOTAL_STEPS - WARMUP_STEPS
            )
            cosine_decay = 0.5 * (1 + np.cos(np.pi * scaled_progress))
            return cosine_decay * (BASE_LR - MIN_LR) / BASE_LR + MIN_LR / BASE_LR

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # =============================================================================
    # 11. PRE-TRAINING LOOP
    #     - Call the train_lwm utility to run the pre-training epochs, logging metrics and saving models
    # =============================================================================
    pretrained_model = train_lwm(
        model,
        train_loaders,
        val_loaders,
        optimizer,
        scheduler,
        EPOCHS,
        device=device,
        save_dir=save_dir,
        log_file="training_log.csv",
        max_batches_per_epoch=1000,  # Limit each dataloader to 1000 batches per epoch
    )