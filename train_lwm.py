# =============================================================================
# 1. IMPORTS AND WARNINGS SETUP
#    - Load necessary PyTorch modules, utilities, and suppress UserWarnings
# =============================================================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch.optim as optim
from utils import (generate_channels_and_labels, tokenizer_train, 
                   create_train_dataloader, count_parameters, train_lwm)
import numpy as np
import pretrained_model  # Assuming this contains the LWM model definition
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
import warnings
import os
import pickle
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# 2. SCENARIO LIST DEFINITION
#    - Define the list of scenario names to iterate over for data generation
# =============================================================================
def scenarios_list():
    scen_list = np.array([
        'city_0_newyork_3p5_lwm', 
        'city_1_losangeles_3p5_lwm', 
        'city_2_chicago_3p5_lwm', 
        'city_3_houston_3p5_lwm', 
        'city_4_phoenix_3p5_lwm', 
        'city_5_philadelphia_3p5_lwm', 
        'city_6_miami_3p5_lwm', 
        'city_7_sandiego_3p5_lwm',
        'city_8_dallas_3p5_lwm', 
        'city_9_sanfrancisco_3p5_lwm', 
        'city_10_austin_3p5_lwm', 
        'city_11_santaclara_3p5_lwm', 
        'city_12_fortworth_3p5_lwm', 
        'city_13_columbus_3p5_lwm', 
        'city_14_charlotte_3p5_lwm',
        'city_15_indianapolis_3p5_lwm',
        'city_16_sanfrancisco_3p5_lwm',  
        'city_17_seattle_3p5_lwm', 
        'city_18_denver_3p5_lwm', 
        'city_19_oklahoma_3p5_lwm', 
        'asu_campus_3p5',
        'o1_3p5',
        'boston5G_3p5'
    ])
    return scen_list  

# =============================================================================
# 3. SCENARIO PROPERTIES MAPPING
#    - Map each scenario name to its corresponding rows, antenna count, and subcarrier count
# =============================================================================
 
def scenario_prop():
    row_column_users = {
    'city_0_newyork_3p5_lwm': {
        'n_rows': 109,
        'n_per_row': 291,
        'grid_idx': 1,
        'n_ant_bs': 8,
        'n_subcarriers': 32
    },
    'city_1_losangeles_3p5_lwm': {
        'n_rows': 142,
        'n_per_row': 201,
        'grid_idx': 1,
        'n_ant_bs': 8,
        'n_subcarriers': 64
    },
    'city_2_chicago_3p5_lwm': {
        'n_rows': 139,
        'n_per_row': 200,
        'grid_idx': 1,
        'n_ant_bs': 8,
        'n_subcarriers': 128
    },
    'city_3_houston_3p5_lwm': {
        'n_rows': 154,
        'n_per_row': 202,
        'grid_idx': 1,
        'n_ant_bs': 8,
        'n_subcarriers': 256
    },
    'city_4_phoenix_3p5_lwm': {
        'n_rows': 198,
        'n_per_row': 214,
        'grid_idx': 1,
        'n_ant_bs': 8,
        'n_subcarriers': 512
    },
    'city_5_philadelphia_3p5_lwm': {
        'n_rows': 239,
        'n_per_row': 164,
        'grid_idx': 1,
        'n_ant_bs': 8,
        'n_subcarriers': 1024
    },
    'city_6_miami_3p5_lwm': {
        'n_rows': 199,
        'n_per_row': 216,
        'grid_idx': 1,
        'n_ant_bs': 16,
        'n_subcarriers': 32
    },
    'city_7_sandiego_3p5_lwm': {
        'n_rows': 176,
        'n_per_row': 207,
        'grid_idx': 1,
        'n_ant_bs': 16,
        'n_subcarriers': 64
    },
    'city_8_dallas_3p5_lwm': {
        'n_rows': 207,
        'n_per_row': 190,
        'grid_idx': 1,
        'n_ant_bs': 16,
        'n_subcarriers': 128
    },
    'city_9_sanfrancisco_3p5_lwm': {
        'n_rows': 196,
        'n_per_row': 206,
        'grid_idx': 1,
        'n_ant_bs': 16,
        'n_subcarriers': 256
    },
    'city_10_austin_3p5_lwm': {
        'n_rows': 255,
        'n_per_row': 137,
        'grid_idx': 1,
        'n_ant_bs': 16,
        'n_subcarriers': 512
    },
    'city_11_santaclara_3p5_lwm': {
        'n_rows': 117,
        'n_per_row': 285,
        'grid_idx': 1,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'city_12_fortworth_3p5_lwm': {
        'n_rows': 214,
        'n_per_row': 179,
        'grid_idx': 1,
        'n_ant_bs': 32,
        'n_subcarriers': 64
    },
    'city_13_columbus_3p5_lwm': {
        'n_rows': 178,
        'n_per_row': 240,
        'grid_idx': 1,
        'n_ant_bs': 32,
        'n_subcarriers': 128
    },
    'city_14_charlotte_3p5_lwm': {
        'n_rows': 216,
        'n_per_row': 177,
        'grid_idx': 1,
        'n_ant_bs': 32,
        'n_subcarriers': 256
    },
    'city_15_indianapolis_3p5_lwm': {
        'n_rows': 200,
        'n_per_row': 196,
        'grid_idx': 1,
        'n_ant_bs': 64,
        'n_subcarriers': 32
    },
    'city_16_sanfrancisco_3p5_lwm': {
        'n_rows': 201,
        'n_per_row': 208,
        'grid_idx': 1,
        'n_ant_bs': 64,
        'n_subcarriers': 64
    },
    'city_17_seattle_3p5_lwm': {
        'n_rows': 185,
        'n_per_row': 205,
        'grid_idx': 1,
        'n_ant_bs': 64,
        'n_subcarriers': 128
    },
    'city_18_denver_3p5_lwm': {
        'n_rows': 212,
        'n_per_row': 204,
        'grid_idx': 1,
        'n_ant_bs': 128,
        'n_subcarriers': 32
    },
    'city_19_oklahoma_3p5_lwm': {
        'n_rows': 204,
        'n_per_row': 188,
        'grid_idx': 1,
        'n_ant_bs': 128,
        'n_subcarriers': 64
    },
    'asu_campus_3p5_v1': {
        'n_rows': [0, 1*int(321/20)],
        'n_per_row': 411,
        'grid_idx': 1,
        'n_ant_bs': 8,
        'n_subcarriers': 32
    },
    'asu_campus_3p5_v2': {
        'n_rows': [1*int(321/20), 2*int(321/20)],
        'n_per_row': 411,
        'grid_idx': 1,
        'n_ant_bs': 8,
        'n_subcarriers': 64
    },
    'asu_campus_3p5_v3': {
        'n_rows': [2*int(321/20), 3*int(321/20)],
        'n_per_row': 411,
        'grid_idx': 1,
        'n_ant_bs': 8,
        'n_subcarriers': 128
    },
    'asu_campus_3p5_v4': {
        'n_rows': [3*int(321/20), 4*int(321/20)],
        'n_per_row': 411,
        'grid_idx': 1,
        'n_ant_bs': 8,
        'n_subcarriers': 256
    },
    'asu_campus_3p5_v5': {
        'n_rows': [4*int(321/20), 5*int(321/20)],
        'n_per_row': 411,
        'grid_idx': 1,
        'n_ant_bs': 8,
        'n_subcarriers': 512
    },
    'asu_campus_3p5_v6': {
        'n_rows': [5*int(321/20), 6*int(321/20)],
        'n_per_row': 411,
        'grid_idx': 1,
        'n_ant_bs': 8,
        'n_subcarriers': 1024
    },
    'asu_campus_3p5_v7': {
        'n_rows': [6*int(321/20), 7*int(321/20)],
        'n_per_row': 411,
        'grid_idx': 1,
        'n_ant_bs': 16,
        'n_subcarriers': 32
    },
    'asu_campus_3p5_v8': {
        'n_rows': [7*int(321/20), 8*int(321/20)],
        'n_per_row': 411,
        'grid_idx': 1,
        'n_ant_bs':16,
        'n_subcarriers': 64
    },
    'asu_campus_3p5_v9': {
        'n_rows': [8*int(321/20), 9*int(321/20)],
        'n_per_row': 411,
        'grid_idx': 1,
        'n_ant_bs': 16,
        'n_subcarriers': 128
    },
    'asu_campus_3p5_v10': {
        'n_rows': [9*int(321/20), 10*int(321/20)],
        'n_per_row': 411,
        'grid_idx': 1,
        'n_ant_bs': 16,
        'n_subcarriers': 256
    },
    'asu_campus_3p5_v11': {
        'n_rows': [10*int(321/20), 11*int(321/20)],
        'n_per_row': 411,
        'grid_idx': 1,
        'n_ant_bs': 16,
        'n_subcarriers': 512
    },
    'asu_campus_3p5_v12': {
        'n_rows': [11*int(321/20), 12*int(321/20)],
        'n_per_row': 411,
        'grid_idx': 1,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'asu_campus_3p5_v13': {
        'n_rows': [12*int(321/20), 13*int(321/20)],
        'n_per_row': 411,
        'grid_idx': 1,
        'n_ant_bs': 32,
        'n_subcarriers': 64
    },
    'asu_campus_3p5_v14': {
        'n_rows': [13*int(321/20), 14*int(321/20)],
        'n_per_row': 411,
        'grid_idx': 1,
        'n_ant_bs': 32,
        'n_subcarriers': 128
    },
    'asu_campus_3p5_v15': {
        'n_rows': [14*int(321/20), 15*int(321/20)],
        'n_per_row': 411,
        'grid_idx': 1,
        'n_ant_bs': 32,
        'n_subcarriers': 256
    },
    'asu_campus_3p5_v16': {
        'n_rows': [15*int(321/20), 16*int(321/20)],
        'n_per_row': 411,
        'grid_idx': 1,
        'n_ant_bs': 64,
        'n_subcarriers': 32
    },
    'asu_campus_3p5_v17': {
        'n_rows': [16*int(321/20), 17*int(321/20)],
        'n_per_row': 411,
        'grid_idx': 1,
        'n_ant_bs': 64,
        'n_subcarriers': 64 
    },
    'asu_campus_3p5_v18': {
        'n_rows': [17*int(321/20), 18*int(321/20)],
        'n_per_row': 411,
        'grid_idx': 1,
        'n_ant_bs': 64,
        'n_subcarriers': 128
    },
    'asu_campus_3p5_v19': {
        'n_rows': [18*int(321/20), 19*int(321/20)],
        'n_per_row': 411,
        'grid_idx': 1,
        'n_ant_bs': 128,
        'n_subcarriers': 32
    },
    'asu_campus_3p5_v20': {
        'n_rows': [19*int(321/20), 20*int(321/20)],
        'n_per_row': 411,
        'grid_idx': 1,
        'n_ant_bs': 128,
        'n_subcarriers': 64
    },
    'boston5G_3p5_v1': {
        'n_rows': [812 + 0, 812 + 1*int((1622-812)/20)],
        'n_per_row': 595,
        'grid_idx': 2,
        'n_ant_bs': 8,
        'n_subcarriers': 32
    },
    'boston5G_3p5_v2': {
        'n_rows': [812 + 1*int((1622-812)/20), 812 + 2*int((1622-812)/20)],
        'n_per_row': 595,
        'grid_idx': 2,
        'n_ant_bs': 8,
        'n_subcarriers': 64
    },
    'boston5G_3p5_v3': {
        'n_rows': [812 + 2*int((1622-812)/20), 812 + 3*int((1622-812)/20)],
        'n_per_row': 595,
        'grid_idx': 2,
        'n_ant_bs': 8,
        'n_subcarriers': 128
    },
    'boston5G_3p5_v4': {
        'n_rows': [812 + 3*int((1622-812)/20), 812 + 4*int((1622-812)/20)],
        'n_per_row': 595,
        'grid_idx': 2,
        'n_ant_bs': 8,
        'n_subcarriers': 256
    },
    'boston5G_3p5_v5': {
        'n_rows': [812 + 4*int((1622-812)/20), 812 + 5*int((1622-812)/20)],
        'n_per_row': 595,
        'grid_idx': 2,
        'n_ant_bs': 8,
        'n_subcarriers': 512
    },
    'boston5G_3p5_v6': {
        'n_rows': [812 + 5*int((1622-812)/20), 812 + 6*int((1622-812)/20)],
        'n_per_row': 595,
        'grid_idx': 2,
        'n_ant_bs': 8,
        'n_subcarriers': 1024
    },
    'boston5G_3p5_v7': {
        'n_rows': [812 + 6*int((1622-812)/20), 812 + 7*int((1622-812)/20)],
        'n_per_row': 595,
        'grid_idx': 2,
        'n_ant_bs': 16,
        'n_subcarriers': 32
    },
    'boston5G_3p5_v8': {
        'n_rows': [812 + 7*int((1622-812)/20), 812 + 8*int((1622-812)/20)],
        'n_per_row': 595,
        'grid_idx': 2,
        'n_ant_bs':16,
        'n_subcarriers': 64
    },
    'boston5G_3p5_v9': {
        'n_rows': [812 + 8*int((1622-812)/20), 812 + 9*int((1622-812)/20)],
        'n_per_row': 595,
        'grid_idx': 2,
        'n_ant_bs': 16,
        'n_subcarriers': 128
    },
    'boston5G_3p5_v10': {
        'n_rows': [812 + 9*int((1622-812)/20), 812 + 10*int((1622-812)/20)],
        'n_per_row': 595,
        'grid_idx': 2,
        'n_ant_bs': 16,
        'n_subcarriers': 256
    },
    'boston5G_3p5_v11': {
        'n_rows': [812 + 10*int((1622-812)/20), 812 + 11*int((1622-812)/20)],
        'n_per_row': 595,
        'grid_idx': 2,
        'n_ant_bs': 16,
        'n_subcarriers': 512
    },
    'boston5G_3p5_v12': {
        'n_rows': [812 + 11*int((1622-812)/20), 812 + 12*int((1622-812)/20)],
        'n_per_row': 595,
        'grid_idx': 2,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'boston5G_3p5_v13': {
        'n_rows': [812 + 12*int((1622-812)/20), 812 + 13*int((1622-812)/20)],
        'n_per_row': 595,
        'grid_idx': 2,
        'n_ant_bs': 32,
        'n_subcarriers': 64
    },
    'boston5G_3p5_v14': {
        'n_rows': [812 + 13*int((1622-812)/20), 812 + 14*int((1622-812)/20)],
        'n_per_row': 595,
        'grid_idx': 2,
        'n_ant_bs': 32,
        'n_subcarriers': 128
    },
    'boston5G_3p5_v15': {
        'n_rows': [812 + 14*int((1622-812)/20), 812 + 15*int((1622-812)/20)],
        'n_per_row': 595,
        'grid_idx': 2,
        'n_ant_bs': 32,
        'n_subcarriers': 256
    },
    'boston5G_3p5_v16': {
        'n_rows': [812 + 15*int((1622-812)/20), 812 + 16*int((1622-812)/20)],
        'n_per_row': 595,
        'grid_idx': 2,
        'n_ant_bs': 64,
        'n_subcarriers': 32
    },
    'boston5G_3p5_v17': {
        'n_rows': [812 + 16*int((1622-812)/20), 812 + 17*int((1622-812)/20)],
        'n_per_row': 595,
        'grid_idx': 2,
        'n_ant_bs': 64,
        'n_subcarriers': 64 
    },
    'boston5G_3p5_v18': {
        'n_rows': [812 + 17*int((1622-812)/20), 812 + 18*int((1622-812)/20)],
        'n_per_row': 595,
        'grid_idx': 2,
        'n_ant_bs': 64,
        'n_subcarriers': 128
    },
    'boston5G_3p5_v19': {
        'n_rows': [812 + 18*int((1622-812)/20), 812 + 19*int((1622-812)/20)],
        'n_per_row': 595,
        'grid_idx': 2,
        'n_ant_bs': 128,
        'n_subcarriers': 32
    },
    'boston5G_3p5_v20': {
        'n_rows': [812 + 19*int((1622-812)/20), 812 + 20*int((1622-812)/20)],
        'n_per_row': 595,
        'grid_idx': 2,
        'n_ant_bs': 128,
        'n_subcarriers': 64
    },
    'o1_3p5_v1': {
        'n_rows': [0*int(3852/12), 1*int(3852/12)],
        'n_per_row': 181,
        'grid_idx': 1,
        'n_ant_bs': 8,
        'n_subcarriers': 32
    },
    'o1_3p5_v2': {
        'n_rows': [1*int(3852/12), 2*int(3852/12)],
        'n_per_row': 181,
        'grid_idx': 1,
        'n_ant_bs': 8,
        'n_subcarriers': 64
    },
    'o1_3p5_v3': {
        'n_rows': [2*int(3852/12), 3*int(3852/12)],
        'n_per_row': 181,
        'grid_idx': 1,
        'n_ant_bs': 8,
        'n_subcarriers': 128
    },
    'o1_3p5_v4': {
        'n_rows': [3*int(3852/12), 4*int(3852/12)],
        'n_per_row': 181,
        'grid_idx': 1,
        'n_ant_bs': 8,
        'n_subcarriers': 256
    },
    'o1_3p5_v5': {
        'n_rows': [4*int(3852/12), 5*int(3852/12)],
        'n_per_row': 181,
        'grid_idx': 1,
        'n_ant_bs': 8,
        'n_subcarriers': 512
    },
    'o1_3p5_v6': {
        'n_rows': [5*int(3852/12), 6*int(3852/12)],
        'n_per_row': 181,
        'grid_idx': 1,
        'n_ant_bs': 8,
        'n_subcarriers': 1024
    },
    'o1_3p5_v7': {
        'n_rows': [6*int(3852/12), 7*int(3852/12)],
        'n_per_row': 181,
        'grid_idx': 1,
        'n_ant_bs': 16,
        'n_subcarriers': 32
    },
    'o1_3p5_v8': {
        'n_rows': [7*int(3852/12), 8*int(3852/12)],
        'n_per_row': 181,
        'grid_idx': 1,
        'n_ant_bs': 16,
        'n_subcarriers': 64
    },
    'o1_3p5_v9': {
        'n_rows': [8*int(3852/12), 2750],
        'n_per_row': 181,
        'grid_idx': 1,
        'n_ant_bs': 16,
        'n_subcarriers': 128
    },
    'o1_3p5_v10': {
        'n_rows': [2751, 10*int(3852/12)],
        'n_per_row': 181,
        'grid_idx': 2,
        'n_ant_bs': 16,
        'n_subcarriers': 256
    },
    'o1_3p5_v11': {
        'n_rows': [10*int(3852/12), 11*int(3852/12)],
        'n_per_row': 181,
        'grid_idx': 2,
        'n_ant_bs': 16,
        'n_subcarriers': 512
    },
    'o1_3p5_v12': {
        'n_rows': [11*int(3852/12), 3851],
        'n_per_row': 181,
        'grid_idx': 2,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'o1_3p5_v13': {
        'n_rows': [3852, 12*int(3852/12)+1*int(1351/10)],
        'n_per_row': 361,
        'grid_idx': 3,
        'n_ant_bs': 32,
        'n_subcarriers': 64
    },
    'o1_3p5_v14': {
        'n_rows': [12*int(3852/12)+1*int(1351/10), 12*int(3852/12)+2*int(1351/10)],
        'n_per_row': 181,
        'grid_idx': 3,
        'n_ant_bs': 32,
        'n_subcarriers': 128
    },
    'o1_3p5_v15': {
        'n_rows': [12*int(3852/12)+2*int(1351/10), 12*int(3852/12)+3*int(1351/10)],
        'n_per_row': 181,
        'grid_idx': 3,
        'n_ant_bs': 32,
        'n_subcarriers': 256
    },
    'o1_3p5_v16': {
        'n_rows': [12*int(3852/12)+3*int(1351/10), 12*int(3852/12)+4*int(1351/10)],
        'n_per_row': 181,
        'grid_idx': 3,
        'n_ant_bs': 64,
        'n_subcarriers': 32
    },
    'o1_3p5_v17': {
        'n_rows': [12*int(3852/12)+4*int(1351/10), 12*int(3852/12)+5*int(1351/10)],
        'n_per_row': 181,
        'grid_idx': 3,
        'n_ant_bs': 64,
        'n_subcarriers': 64
    },
    'o1_3p5_v18': {
        'n_rows': [12*int(3852/12)+5*int(1351/10), 12*int(3852/12)+6*int(1351/10)],
        'n_per_row': 181,
        'grid_idx': 3,
        'n_ant_bs': 64,
        'n_subcarriers': 128
    },
    'o1_3p5_v19': {
        'n_rows': [12*int(3852/12)+6*int(1351/10), 12*int(3852/12)+7*int(1351/10)],
        'n_per_row': 181,
        'grid_idx': 3,
        'n_ant_bs': 128,
        'n_subcarriers': 32
    },
    'o1_3p5_v20': {
        'n_rows': [12*int(3852/12)+7*int(1351/10), 12*int(3852/12)+8*int(1351/10)],
        'n_per_row': 181,
        'grid_idx': 3,
        'n_ant_bs': 128,
        'n_subcarriers': 64
    }}
    return row_column_users   

# =============================================================================
# 4. TRAINING PARAMETERS AND HYPERPARAMETERS
#    - Set training epochs, batch sizes, learning rates, model dimensions, etc.
# =============================================================================

EPOCHS = 50
BATCH_SIZE = 128
VAL_BATCH_SIZE = 64
WARMUP_EPOCHS = 5
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
task = ["LosNlosClassification", 
        "BeamPrediction", 
        "ChannelInterpolation", 
        "ChannelEstimation", 
        "ChannelCharting",
        None][-1]

# =============================================================================
# 5. DATA GENERATION LOOP
#    - Iterate over scenarios and base station indices to generate channel samples and labels
#    - Handle both full-scenario and zoned sub-scenarios for campus and Boston data
# =============================================================================

scenarios = scenarios_list()

channels = []
labels = []
scenario_properties = scenario_prop()
preprocessed_data = []

# Create cache directory for scenario data
cache_dir = r"C:\Users\tomer\scenarios_cache_lwm"
os.makedirs(cache_dir, exist_ok=True)

cities_scenarios = scenarios[:-3]
for scenario in cities_scenarios:
    for bs_idx in range (1,4):
        # Create a unique cache filename for this scenario + bs_idx combination
        cache_filename = f"{scenario}_bs{bs_idx}.pkl"
        cache_filepath = os.path.join(cache_dir, cache_filename)

        # Check if cached data exists
        if os.path.exists(cache_filepath):
            print(f"\nLoading cached data for scenario: {scenario}, BS #{bs_idx}")
            with open(cache_filepath, 'rb') as f:
                cached_data = pickle.load(f)
                scenario_channels = cached_data['channels']
                scenario_labels = cached_data['labels']
        else:
            # Generate data if cache doesn't exist
            scenario_channels, scenario_labels = generate_channels_and_labels(
                n_ant_bs=scenario_properties[scenario]["n_ant_bs"],
                n_subcarriers=scenario_properties[scenario]["n_subcarriers"],
                bs_idx=bs_idx,
                scenario_name=scenario,
                task=task,
                n_beams=64
            )

            # Save to cache
            print(f"Saving data to cache: {cache_filepath}")
            with open(cache_filepath, 'wb') as f:
                pickle.dump({
                    'channels': scenario_channels,
                    'labels': scenario_labels
                }, f)

        labels.extend(scenario_labels)
        channels.append(scenario_channels)


# bs_idxs = [[1], [4, 15], [2]]
# for scenario_idx, scenario in enumerate(scenarios[-3:]):
#     for bs_idx in bs_idxs[scenario_idx]:
#         for zone in range(20):
#             row_start = scenario_properties[scenario+f"_v{zone+1}"]["n_rows"][0]
#             row_end = scenario_properties[scenario+f"_v{zone+1}"]["n_rows"][1]
#             grid_idx = scenario_properties[scenario+f"_v{zone+1}"]["grid_idx"]-1

#             # Create a unique cache filename for this scenario + bs_idx + zone combination
#             cache_filename = f"{scenario}_bs{bs_idx}_zone{zone+1}.pkl"
#             cache_filepath = os.path.join(cache_dir, cache_filename)

#             # Check if cached data exists
#             if os.path.exists(cache_filepath):
#                 print(f"\nLoading cached data for scenario: {scenario}, BS #{bs_idx}, Zone {zone+1}")
#                 with open(cache_filepath, 'rb') as f:
#                     cached_data = pickle.load(f)
#                     scenario_channels = cached_data['channels']
#                     scenario_labels = cached_data['labels']
#             else:
#                 # Generate data if cache doesn't exist
#                 scenario_channels, scenario_labels = generate_channels_and_labels(
#                     n_ant_bs=scenario_properties[scenario+f"_v{zone+1}"]["n_ant_bs"],
#                     n_subcarriers=scenario_properties[scenario+f"_v{zone+1}"]["n_subcarriers"],
#                     grid_idx=grid_idx,
#                     bs_idx=bs_idx,
#                     scenario_name=scenario,
#                     rows=np.arange(row_start, row_end),
#                     task=task,
#                     n_beams=64
#                 )

#                 # Only save to cache if we got valid data
#                 if scenario_channels.numel() > 0:
#                     print(f"Saving data to cache: {cache_filepath}")
#                     with open(cache_filepath, 'wb') as f:
#                         pickle.dump({
#                             'channels': scenario_channels,
#                             'labels': scenario_labels
#                         }, f)

#             if scenario_channels.numel() == 0:
#                 print(f"No candidate user in zone {zone+1} for scenario {scenario} has a path to bs_idx {bs_idx} (All channels are zero)")
#                 continue

#             labels.extend(scenario_labels)
#             channels.append(scenario_channels)
# =============================================================================
# 6. DATA TOKENIZATION
#    - Tokenize channel matrices into input sequences with masking for pretraining
# =============================================================================

preprocessed_data = tokenizer_train(
    channels,
    max_len=MAX_LEN,
    masking_percent=MASK_PERCENT,
    mask=True,
    seed=42,
)

# =============================================================================
# 7. TRAIN/VALIDATION/TEST SPLIT
#    - Split each tokenized dataset into train, validation, and test subsets with a fixed random seed
# =============================================================================

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
train_ratio = 0.8
val_ratio = 0.2
train_data = {}
val_data = {}
test_data = {}

for key, samples in preprocessed_data.items():
    print(f"key: {key}")
    total_samples = len(samples)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size
    
    train_data[key], val_data[key], test_data[key] = random_split(
        samples, [train_size, val_size, test_size]
    )

# =============================================================================
# 8. DATALOADER CREATION
#    - Build PyTorch DataLoader objects for batched training and validation
# =============================================================================

train_loaders = create_train_dataloader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loaders = create_train_dataloader(val_data, batch_size=VAL_BATCH_SIZE, shuffle=False)

# =============================================================================
# 9. MODEL INITIALIZATION
#    - Instantiate the LWM transformer model and optionally load pre-trained weights
#    - Wrap with DataParallel for multi-GPU support
# =============================================================================

gpu_ids = [0,1] # device_idx
device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
model = pretrained_model.lwm(
    element_length=ELEMENT_LENGTH,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    max_len=MAX_LEN,
    n_heads=N_HEADS,
    dropout=DROPOUT
).to(device)

# Optional: Load pre-trained model
load_model = False
if load_model:
    model.load_state_dict(torch.load("models/model_checkpoint.pth", map_location=device))
    print("Pre-trained model loaded successfully.")

# Use DataParallel for multi-GPU support
model = nn.DataParallel(model, device_ids=gpu_ids)
print(f"Model loaded successfully on GPU {device.index}")
n_parameters = count_parameters(model)
print(f"Number of trainable parameters: {n_parameters:,}")

# =============================================================================
# 10. OPTIMIZER AND LEARNING RATE SCHEDULER
#     - Configure AdamW optimizer and a cosine-with-warmup LR schedule based on total steps
# =============================================================================

TOTAL_STEPS = sum(len(loader) for loader in train_loaders.values()) * EPOCHS
WARMUP_STEPS = sum(len(loader) for loader in train_loaders.values()) * WARMUP_EPOCHS

optimizer = AdamW(
    model.parameters(),
    lr=BASE_LR,
    betas=(BETA1, BETA2),
    weight_decay=WEIGHT_DECAY
)

def lr_lambda(current_step):
    if current_step < WARMUP_STEPS:
        return current_step / WARMUP_STEPS
    else:
        scaled_progress = (current_step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
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
    save_dir="pretrained_models",
    log_file="training_log.csv"
)