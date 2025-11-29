import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import pickle
import deepmimo as dm
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from math import pi
import zipfile
import shutil

def generate_channels_and_labels(
    n_ant_bs=16,
    n_subcarriers=64,
    bs_idx=3,
    grid_idx=0,
    scenario_name=0,
    scenario_idx=None,
    rows=None,
    task="LosNlosClassification",
    n_beams=64
    ):
    """
    Generate wireless channel samples and task-specific labels for pre-training or evaluation.

    Args:
        n_ant_bs (int): Number of antennas at the base station. Defaults to 16.
        n_subcarriers (int): Number of subcarriers per channel. Defaults to 64.
        bs_idx (int): Index of the base station to generate data for. Defaults to 3.
        scenario_idx (int): Index of the scenario to select from available scenarios. Defaults to 0.
        task (str): Task for which to generate labels. Defaults to "LosNlosClassification".
        n_beams (int): Number of beams for beam-related tasks. Defaults to 64.

    Returns:
        tuple: A tuple containing:
            - channels (list of torch.Tensor): List of generated channel tensors.
            - labels (list): Corresponding task-specific labels.
    """
    if scenario_idx:
        scenario_name = dm.search({})[scenario_idx]
        
    channels, labels = dataset_generator(
        n_ant_bs=n_ant_bs,
        n_subcarriers=n_subcarriers,
        grid_idx=grid_idx,
        scenario_name=scenario_name,
        rows=rows,
        bs_idx=bs_idx,
        task=task,
        n_beams=n_beams
    )

    return channels, labels

def dataset_generator(
        n_ant_bs=32, 
        n_subcarriers=32, 
        scenario_name="city_0_newyork_3p5", 
        grid_idx=0,
        rows=None,
        bs_idx=1, 
        save_dir="data", 
        task="LosNlosClassification", 
        n_beams=64, 
        snr=None, 
        seed=42
        ):
    """
    Generate wireless channel data and task-specific labels using DeepMIMO dataset.

    Args:
        n_ant_bs (int): Number of antennas at the base station. Defaults to 32.
        n_subcarriers (int): Number of subcarriers per channel. Defaults to 32.
        scenario_name (str): Name of the scenario for data generation. Defaults to "city_0_newyork_3p5".
        bs_idx (int): Index of the base station to generate data for. Defaults to 1.
        save_dir (str): Directory to save generated data. Defaults to "data".
        task (str): Task for which to generate labels. Defaults to "LosNlosClassification".
        n_beams (int): Number of beams for beam-related tasks. Defaults to 64.
        snr (float, optional): Signal-to-noise ratio for adding Gaussian noise in robust beamforming task.
            Defaults to None.
        seed (int): Random seed for reproducibility in noise generation. Defaults to 42.

    Returns:
        tuple: A tuple containing:
            - cleaned_deepmimo_data (torch.Tensor): Cleaned channel data tensor.
            - labels (torch.Tensor or list): Task-specific labels or NaN-filled tensor if task is None.
    """ 
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nGenerating data for scenario: {scenario_name}, BS #{bs_idx}")
    deepmimo_data = DeepMIMO_data_gen(scenario_name, 
                                      n_ant_bs, 
                                      1, n_subcarriers, 
                                      bs_idx=bs_idx, 
                                      row_indices=rows,
                                      grid_idx=grid_idx)
    if task is not None:
        labels = label_gen(deepmimo_data, task, scenario_name, bs_idx=bs_idx, n_beams=n_beams)
    else:
        n_channels = len(np.where(deepmimo_data.los != -1)[0])
        labels = np.nan * torch.ones(n_channels) 
    cleaned_deepmimo_data = deepmimo_data_cleaning(deepmimo_data)
    if snr is not None and task in["ChannelEstimation", "LosNlosClassification"]:
        cleaned_deepmimo_data = generate_gaussian_noise(cleaned_deepmimo_data, snr, seed=seed)
    return cleaned_deepmimo_data.squeeze(1), labels

def label_gen(data, task, scenario_name, bs_idx=1, n_beams=64):
    """
    Generate task-specific labels for wireless channel data.

    Args:
        data (object): DeepMIMO data object containing channel, LOS status, and receiver positions.
        task (str): Task for which to generate labels. Options: 'LosNlosClassification',
            'BeamPrediction', 'ChannelCharting', 'ChannelEstimation', 'ChannelInterpolation'.
        scenario_name (str): Name of the scenario for data generation.
        bs_idx (int): Index of the base station. Defaults to 1.
        n_beams (int): Number of beams for beam-related tasks. Defaults to 64.

    Returns:
        torch.Tensor: Task-specific labels for the valid data indices.
    """
    labels = 0
    idxs = np.where(data.los != -1)[0]

    if len(idxs) == 0: # Users with no path (do not need them for pre-training). You should set the task to None in train_lwm.py
        labels = np.full(data.n_ue, np.nan, dtype=float)

    else:        
        if task == 'LosNlosClassification':
            
            labels = data.los[idxs].astype(int)
            dm.plot_coverage(data.rx_pos[idxs], data.los[idxs], cbar_title='LoS status')
            
        elif task == 'BeamPrediction':
            
            parameters = get_parameters()
            n_users = len(data.channel)
            n_subbands = 1
            fov = 180

            # Setup Beamformers
            beam_angles = np.around(np.arange(-fov/2, fov/2+.1, fov/(n_beams-1)), 2)

            F1 = np.array([dm.steering_vec(data.ch_params.bs_antenna.shape, phi=azi).squeeze()
                for azi in beam_angles])

            full_dbm = np.zeros((n_beams, n_subbands, n_users), dtype=float)
            for ue_idx in tqdm(range(n_users), desc='Computing the channel for each user'):
                if data.los[ue_idx] == -1:
                    full_dbm[:,:,ue_idx] = np.nan
                else:
                    chs = F1 @ data.channel[ue_idx]
                    full_linear = np.abs(np.mean(chs.squeeze().reshape((n_beams, n_subbands, -1)), axis=-1))
                    full_dbm[:,:,ue_idx] = np.around(20*np.log10(full_linear) + 30, 1)

            best_beams = np.argmax(np.mean(full_dbm,axis=1), axis=0)
            best_beams = best_beams.astype(float)
            best_beams[np.isnan(full_dbm[0,0,:])] = np.nan
            
            dm.plot_coverage(data.rx_pos[idxs], best_beams[idxs], bs_pos=data.tx_pos, 
                            bs_ori=parameters.bs_antenna.rotation*np.pi/180, 
                            cbar_title='Best beam index')
            
            labels = best_beams[idxs].astype(int)
            
        elif task == 'ChannelCharting':
            
            labels = torch.tensor(data.rx_pos[:,:2][idxs]).to(dtype=torch.float32)
            
        elif task == 'ChannelEstimation':
            
            channels = torch.tensor(data.channel[idxs]*1e6, dtype=torch.complex64).squeeze(1)
            labels = torch.stack((channels.real, channels.imag), dim=1) 
            
        elif task == 'ChannelInterpolation':
            
            channels = torch.tensor(data.channel[idxs]*1e6, dtype=torch.complex64).squeeze(1)
            labels = torch.stack((channels.real, channels.imag), dim=1)
    
        labels = torch.tensor(labels)
    
    return labels        

def generate_gaussian_noise(data, snr_db, seed=42):
    """
    Add complex Gaussian noise to channel data based on a specified signal-to-noise ratio (SNR).

    Args:
        data (torch.Tensor): Input complex-valued channel data with shape (n_samples, 1, n_ant, n_sc).
        snr_db (float): Signal-to-noise ratio in decibels.
        seed (int): Random seed for reproducibility of noise generation. Defaults to 42.

    Returns:
        torch.Tensor: Noisy channel data with the same shape as the input, with added complex Gaussian noise.
    """
    torch.manual_seed(seed) 
    data = data.squeeze(1)  # Shape: (n_samples, n_ant, n_sc)
    flat_data = data.view(data.size(0), -1)  
    
    # Compute signal power
    signal_power = torch.mean(flat_data.abs() ** 2, dim=1, keepdim=True)  
    snr_linear = 10 ** (snr_db / 10)  
    noise_power = signal_power / snr_linear  

    # Generate noise
    noise_real = torch.randn_like(flat_data.real) * torch.sqrt(noise_power / 2)
    noise_imag = torch.randn_like(flat_data.imag) * torch.sqrt(noise_power / 2)
    noise = torch.complex(noise_real, noise_imag) 

    # Reshape noise and add to data
    noise = noise.view_as(data)
    noisy_data = data + noise
    noisy_data = noisy_data.unsqueeze(1) 

    return noisy_data

# REMOVE ZERO CHANNELS AND SCALE
def deepmimo_data_cleaning(deepmimo_data):
    """
    Clean DeepMIMO channel data by removing invalid channels and scaling the valid ones.

    Args:
        deepmimo_data (object): DeepMIMO data object containing channel data and LOS status.

    Returns:
        torch.Tensor: Cleaned and scaled channel data as a complex-valued tensor with dtype torch.complex64.
    """
    idxs = np.where(deepmimo_data.los != -1)[0]
    cleaned_deepmimo_data = deepmimo_data.channel[idxs]
    return torch.tensor(cleaned_deepmimo_data, dtype=torch.complex64) * 1e6

def manual_unzip_scenario(scenario_name):
    """Manually unzip a downloaded scenario to avoid DeepMIMO corruption issues."""
    scenarios_dir = os.path.join(os.getcwd(), "deepmimo_scenarios")
    zip_path = os.path.join(scenarios_dir, f"{scenario_name}_downloaded.zip")
    scenario_path = os.path.join(scenarios_dir, scenario_name)
    
    # Remove existing unzipped folder if it exists
    if os.path.exists(scenario_path):
        print(f"Removing existing scenario folder: {scenario_path}")
        shutil.rmtree(scenario_path)
    
    # Manually unzip the downloaded file
    if os.path.exists(zip_path):
        print(f"Manually unzipping: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(scenarios_dir)
        print(f"Successfully extracted to: {scenario_path}")
        return True
    else:
        print(f"Zip file not found: {zip_path}")
        return False   

# Data Generation 
def DeepMIMO_data_gen(scenario, num_ant_hor, num_ant_vert, n_subcarriers, bs_idx, row_indices, grid_idx):
    """
    Generates wireless channel data for a specified DeepMIMO scenario and base station configuration.

    This function downloads and loads the specified DeepMIMO scenario, filters the users (if row_indices are given),
    and computes the MIMO channels using the provided antenna and subcarrier configuration. It also plots the
    line-of-sight (LoS) coverage map for visualization.

    Args:
        scenario (str): Name of the DeepMIMO scenario (e.g., 'o1_3p5').
        num_ant_hor (int): Number of horizontal antennas at the base station (for the UPA configuration).
        num_ant_vert (int): Number of vertical antennas at the base station.
        n_subcarriers (int): Number of subcarriers to simulate per user.
        bs_idx (int): Index of the base station to extract channels from.
        row_indices (list or None): List of user row indices to subset the dataset. If None, all users are included.

    Returns:
        object: A DeepMIMO data object containing:
            - Computed channel matrices for selected users and base station.
            - User metadata including location, LoS status, and other parameters.
            - Scenario-specific attributes for analysis and visualization.
    """
    parameters = get_parameters(num_ant_hor, num_ant_vert, n_subcarriers)
    to_download_scenario = False
    if to_download_scenario:
        dm.download(scenario)
        # Manual unzip for LWM scenarios to avoid corruption
        if "lwm" in scenario:
            manual_unzip_scenario(scenario)

    data = dm.load(scenario, tx_sets=[bs_idx], rx_sets=[grid_idx])
    
    if row_indices is not None:
        if grid_idx == 0:
            row_idxs = data.get_row_idxs(row_indices) # example: row_indices = np.arange(40,60)
            data = data.subset(row_idxs)
        elif grid_idx == 1:
            if scenario == "o1_3p5":
                col_idxs = data.get_col_idxs(row_indices-2751)
                data = data.subset(col_idxs)
            elif scenario == "boston5G_3p5":
                row_idxs = data.get_row_idxs(row_indices-812)
                data = data.subset(row_idxs)
        elif grid_idx == 2:
            if scenario == "o1_3p5":
                col_idxs = data.get_col_idxs(row_indices-3852)
            data = data.subset(col_idxs)
    
    # data.plot_coverage(data.los)
    data.compute_channels(parameters)
    
    return data

def get_parameters(num_ant_hor=32, num_ant_vert=1, n_subcarriers=32):
    """
    Generate channel parameters for DeepMIMO dataset generation.

    Args:
        num_ant_hor (int): Number of horizontal antennas at the base station. Defaults to 32.
        num_ant_vert (int): Number of vertical antennas at the base station. Defaults to 1.
        n_subcarriers (int): Number of subcarriers per channel. Defaults to 32.
        bs_idx (int): Index of the base station. Defaults to 1.

    Returns:
        dm.ChannelGenParameters: Configured channel parameters object for DeepMIMO data generation.
    """
    # Create channel parameters with all options
    ch_params = dm.ChannelParameters()

    # Antenna parameters

    # Base station antenna parameters
    ch_params.bs_antenna.rotation = np.array([0, 0, -135])  # [az, el, pol] in degrees
    ch_params.bs_antenna.fov = np.array([360, 180])      # [az, el] in degrees
    ch_params.bs_antenna.shape = np.array([num_ant_hor, num_ant_vert])        # [horizontal, vertical] elements
    ch_params.bs_antenna.spacing = 0.5                   # Element spacing in wavelengths

    # User equipment antenna parameters
    ch_params.ue_antenna.rotation = np.array([0, 0, 0])  # [az, el, pol] in degrees
    ch_params.ue_antenna.fov = np.array([360, 180])      # [az, el] in degrees
    ch_params.ue_antenna.shape = np.array([1, 1])        # [horizontal, vertical] elements
    ch_params.ue_antenna.spacing = 0.5                   # Element spacing in wavelengths
    
    # Channel parameters
    ch_params.freq_domain = True  # Whether to compute frequency domain channels
    ch_params.num_paths = 20      # Number of paths

    # OFDM parameters
    subcarrier_spacing = 30e3                                 
    ch_params.ofdm.subcarriers = n_subcarriers                       # Number of subcarriers
    ch_params.ofdm.selected_subcarriers = np.arange(n_subcarriers)   # Which subcarriers to generate
    ch_params.ofdm.bandwidth = subcarrier_spacing * n_subcarriers    # Bandwidth in Hz
    ch_params.ofdm.rx_filter = 0     
    
    return ch_params

def tokenizer_train(channels,
                    max_len=513, 
                    masking_percent=0.40, 
                    mask=False, 
                    seed=42):
    """
    Tokenize wireless channel data into patches and optionally apply masking.

    Args:
        channels (torch.Tensor or list): Input channel data to be tokenized.
        max_len (int): Maximum sequence length for tokenized samples. Defaults to 513.
        masking_percent (float): Percentage of patches to mask if mask is True. Defaults to 0.40.
        mask (bool): Whether to apply masking to the tokenized samples. Defaults to False.
        seed (int): Random seed for reproducibility in masking. Defaults to 42.

    Returns:
        dict or torch.Tensor: If mask is True, returns a dictionary mapping sequence lengths to lists
            of tokenized samples. If mask is False, returns a tensor of stacked tokenized samples.
    """
    patches = [patch_maker(channel_set, patch_rows=4, patch_cols=4) for channel_set in channels]
    patches = [patch for patch_list in patches for patch in patch_list]
    print("\nTotal number of samples:", len(patches))
    
    grouped_data = defaultdict(list)  # Group samples by sequence length
    grouped_data_2 = []
    
    for user_idx in tqdm(range(len(patches)), desc="Processing items"):
        patch_size = patches[user_idx].shape[1]
        n_patches = patches[user_idx].shape[0]
        n_masks_half = int(masking_percent * n_patches)
        
        word2id = {
            '[CLS]': 0.2 * np.ones((patch_size)),
            '[MASK]': 0.1 * np.ones((patch_size))
        }
        
        sample = make_sample(
            user_idx, patches, word2id, n_patches, n_masks_half, patch_size, mask=mask, seed=seed
        )
        
        if mask:
            seq_length = len(sample[0]) 
            grouped_data[seq_length].append(sample)
        else:
            grouped_data_2.append(sample)
    
    if mask:
        normalized_grouped_data = {key: grouped_data[key] for key in sorted(grouped_data.keys())}
    else: 
        normalized_grouped_data = torch.stack(grouped_data_2, dim=0)
    
    return normalized_grouped_data 

def tokenizer(channels,
              max_len=513, 
              masking_percent=0.40, 
              mask=False, 
              seed=42):
    """
    Tokenize wireless channel data into patches and optionally apply masking.

    Args:
        channels (torch.Tensor or list): Input channel data to be tokenized.
        max_len (int): Maximum sequence length for tokenized samples. Defaults to 513.
        masking_percent (float): Percentage of patches to mask if mask is True. Defaults to 0.40.
        mask (bool): Whether to apply masking to the tokenized samples. Defaults to False.
        seed (int): Random seed for reproducibility in masking. Defaults to 42.

    Returns:
        dict or torch.Tensor: If mask is True, returns a dictionary mapping sequence lengths to lists
            of tokenized samples. If mask is False, returns a tensor of stacked tokenized samples.
    """
    patches = patch_maker(channels, patch_rows=4, patch_cols=4)
    print("\nTotal number of samples:", len(patches))
    
    grouped_data = defaultdict(list)  # Group samples by sequence length
    grouped_data_2 = []
    
    for user_idx in tqdm(range(len(patches)), desc="Processing items"):
        patch_size = patches[user_idx].shape[1]
        n_patches = patches[user_idx].shape[0]
        n_masks_half = int(masking_percent * n_patches)
        
        word2id = {
            '[CLS]': 0.2 * np.ones((patch_size)),
            '[MASK]': 0.1 * np.ones((patch_size))
        }
        
        sample = make_sample(
            user_idx, patches, word2id, n_patches, n_masks_half, patch_size, mask=mask, seed=seed
        )
        
        if mask:
            seq_length = len(sample[0]) 
            grouped_data[seq_length].append(sample)
        else:
            grouped_data_2.append(sample)
    
    if mask:
        normalized_grouped_data = {key: grouped_data[key] for key in sorted(grouped_data.keys())}
    else: 
        normalized_grouped_data = torch.stack(grouped_data_2, dim=0)
    
    return normalized_grouped_data 

def make_sample(user_idx, patch, word2id, n_patches, n_masks, patch_size, mask=True, seed=None):
    """
    Create a tokenized sample from patch data, optionally applying masking for a specific user.

    Args:
        user_idx (int): Index of the user whose patch data is to be processed.
        patch (numpy.ndarray or torch.Tensor): Patch data for all users.
        word2id (dict): Dictionary mapping special tokens ('[CLS]', '[MASK]') to their representations.
        n_patches (int): Number of patches in the input data.
        n_masks (int): Number of patches to mask if mask is True.
        patch_size (int): Size of each patch.
        mask (bool): Whether to apply masking to the sample. Defaults to True.
        seed (int, optional): Random seed for reproducibility in masking. Defaults to None.

    Returns:
        torch.Tensor or list: If mask is False, returns a tensor of input IDs with [CLS] prepended.
            If mask is True, returns a list containing input IDs, masked tokens, and masked positions.
    """
    if seed is not None:
        np.random.seed(seed)  
        
    # Step 1: Retrieve tokens and prepend [CLS]
    tokens = patch[user_idx]
    input_ids = np.vstack((word2id['[CLS]'], tokens))

    # Step 2: Mask real and imaginary patches
    tokens_size = int(n_patches)  # int(n_patches / 2)
    masked_pos = np.random.choice(range(1, tokens_size), size=n_masks, replace=False)

    masked_tokens = []
    for pos in masked_pos:
        original_masked_tokens = input_ids[pos].copy()
        masked_tokens.append(original_masked_tokens)
        if mask:
            rnd_num = np.random.rand()
            if rnd_num < 0.1:
                input_ids[pos] = np.random.rand(patch_size)  # Replace with random values
            elif rnd_num < 0.9:
                input_ids[pos] = word2id['[MASK]']  # Replace with [MASK]
    
    if not mask:
        return torch.tensor(input_ids)
    else:
        return [input_ids, masked_tokens, masked_pos]
    
# Patch GENERATION
def patch_maker(original_ch, patch_rows=4, patch_cols=4):
    """
    Converts complex-valued channel matrices into flattened, interleaved real-imaginary patch embeddings.

    This function takes a batch of complex-valued 2D channel matrices (one per sample), splits the real 
    and imaginary components, interleaves them along the last dimension, and divides the result into
    non-overlapping patches of specified size. The output is a set of flattened patches per sample,
    ready for use in models like Transformers.

    Args:
        original_ch (np.ndarray): Input array of shape (n_samples, n_rows, n_cols) with complex values.
        patch_rows (int): Number of rows per patch. Default is 4.
        patch_cols (int): Number of columns per patch. Default is 4.

    Returns:
        np.ndarray: Array of shape (n_samples, n_patches, patch_rows * patch_cols * 2), where each patch
                    is flattened and contains interleaved real and imaginary parts.
    """
    # Step 1: Remove the singleton channel dimension
    n_samples, n_rows, n_cols = original_ch.shape  # Unpack shape
    # original_ch = original_ch[:, 0]  # Remove the singleton dimension

    # Step 2: Split into real and imaginary parts and interleave them
    flat_real = original_ch.real
    flat_imag = original_ch.imag

    # Interleave real and imaginary parts along the last axis
    interleaved = np.empty((n_samples, n_rows, n_cols * 2), dtype=np.float32)
    interleaved[:, :, 0::2] = flat_real
    interleaved[:, :, 1::2] = flat_imag

    # Step 3: Compute the number of patches along rows and columns
    n_patches_rows = int(np.ceil(n_rows / patch_rows))
    n_patches_cols = int(np.ceil(n_cols / patch_cols))

    # Step 4: Pad the matrix if necessary to make it divisible by patch size
    padded_rows = n_patches_rows * patch_rows - n_rows
    padded_cols = n_patches_cols * patch_cols - n_cols
    if padded_rows > 0 or padded_cols > 0:
        interleaved = np.pad(
            interleaved,
            ((0, 0), (0, padded_rows), (0, padded_cols * 2)),  # Double padding for interleaved axis
            mode='constant',
            constant_values=0,
        )

    # Step 5: Create patches by dividing into blocks
    n_samples, padded_rows, padded_cols = interleaved.shape
    padded_cols //= 2  # Adjust for interleaving (real and imaginary parts count as one)
    patches = []

    for i in range(0, padded_rows, patch_rows):
        for j in range(0, padded_cols, patch_cols):
            patch = interleaved[:, i:i + patch_rows, j * 2:(j + patch_cols) * 2]
            patches.append(patch.reshape(n_samples, -1))  # Flatten each patch

    # Step 6: Stack patches to form the final array
    patches = np.stack(patches, axis=1)  # Shape: (num_samples, n_patches, patch_rows * patch_cols * 2)

    return patches

def patch_reconstructor(patches, original_rows, original_cols, patch_rows=4, patch_cols=4):
    """
    Reconstructs the original channel matrix with real and imaginary parts as separate channels using PyTorch.

    Args:
        patches (torch.Tensor): Patches of shape (n_samples, n_patches, patch_rows * patch_cols * 2)
        original_rows (int): Original number of rows (n_rows)
        original_cols (int): Original number of columns (n_cols)
        patch_rows (int): Number of rows per patch (default: 4)
        patch_cols (int): Number of columns per patch (default: 4)

    Returns:
        torch.Tensor: Reconstructed channel matrix of shape (n_samples, 2, original_rows, original_cols)
                     where channel 0 is real and channel 1 is imaginary
    """
    # Step 1: Extract dimensions
    n_samples, n_patches, patch_size = patches.shape
    assert patch_size == patch_rows * patch_cols * 2, "Patch size does not match patch_rows * patch_cols * 2"

    # Step 2: Compute the number of patches along rows and columns
    # Use integer division since no padding is needed
    n_patches_rows = original_rows // patch_rows
    n_patches_cols = original_cols // patch_cols
    assert n_patches == n_patches_rows * n_patches_cols, "Number of patches does not match expected grid"

    # Step 3: Reshape patches back into 2D blocks
    patches_2d = patches.reshape(n_samples, n_patches_rows, n_patches_cols, patch_rows, patch_cols * 2)

    # Step 4: Reconstruct the interleaved matrix
    # No padding, so use original dimensions directly
    interleaved = torch.zeros((n_samples, original_rows, original_cols * 2), dtype=torch.float32, device=patches.device)
    for i in range(n_patches_rows):
        for j in range(n_patches_cols):
            interleaved[:, i * patch_rows:(i + 1) * patch_rows, j * patch_cols * 2:(j + 1) * patch_cols * 2] = \
                patches_2d[:, i, j, :, :]

    # Step 5: De-interleave real and imaginary parts
    flat_real = interleaved[:, :, 0::2]
    flat_imag = interleaved[:, :, 1::2]

    # Step 6: Stack real and imaginary parts as separate channels along axis=1
    reconstructed = torch.stack((flat_real, flat_imag), dim=1)  # Shape: (n_samples, 2, original_rows, original_cols)

    return reconstructed

class LazyLoadDataset(torch.utils.data.Dataset):
    """
    Lazy-loading dataset that reads samples from pickle files on-demand.

    Args:
        pickle_file_paths (list): List of paths to pickle files
        indices (list): List of (file_idx, sample_idx) tuples indicating which samples to use
    """
    def __init__(self, pickle_file_paths, indices):
        self.pickle_file_paths = pickle_file_paths
        self.indices = indices  # [(file_idx, sample_idx), ...]
        self._cache = {}  # Cache loaded pickle files in memory

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.indices[idx]

        # Load pickle file if not cached
        if file_idx not in self._cache:
            with open(self.pickle_file_paths[file_idx], 'rb') as f:
                self._cache[file_idx] = pickle.load(f)

        # Get the sample: [input_ids, masked_tokens, masked_pos]
        sample = self._cache[file_idx][sample_idx]

        # Convert to tensors
        input_ids = torch.tensor(sample[0], dtype=torch.float32)
        masked_tokens = torch.tensor(sample[1], dtype=torch.float32)
        masked_pos = torch.tensor(sample[2], dtype=torch.long)

        return input_ids, masked_tokens, masked_pos

def create_train_dataloader(grouped_data, batch_size, shuffle):
    """
    Creates a dictionary of DataLoaders using lazy-loading datasets.

    Args:
        grouped_data (dict): Dictionary mapping seq_len to (file_metadata, indices)
            where file_metadata = [(filepath, num_samples), ...]
            and indices = [(file_idx, sample_idx), ...]
        batch_size (int): Batch size for DataLoaders
        shuffle (bool): Whether to shuffle data

    Returns:
        dict: Dictionary mapping seq_len to DataLoader
    """
    dataloaders = {}

    for seq_length, (file_metadata, indices) in grouped_data.items():
        print(f"\nCreating dataloader for sequence length: {seq_length}")
        print(f"  Number of samples: {len(indices)}")

        # Extract file paths from metadata
        file_paths = [filepath for filepath, _ in file_metadata]

        # Create lazy-loading dataset
        dataset = LazyLoadDataset(file_paths, indices)

        # Create DataLoader with multiple workers for parallel I/O
        # Note: On Windows, num_workers > 0 requires if __name__ == '__main__' guard
        # Setting to 0 for Windows compatibility (single-process loading)
        dataloaders[seq_length] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=0  # Set to 0 for Windows; use 2-4 on Linux/Mac
        )

        print(f"  DataLoader created with batch_size={batch_size}, num_workers=0")

    return dataloaders

def count_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The model to inspect.

    Returns:
        int: Total number of trainable parameters (i.e., those with requires_grad=True).
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def nmse_loss(y_true, y_pred):
    """Compute Normalized Mean Squared Error (NMSE) using PyTorch.
    
    Args:
        y_true (torch.Tensor): Ground truth tensor.
        y_pred (torch.Tensor): Predicted tensor.
        
    Returns:
        torch.Tensor: NMSE value.
    """
    # Ensure inputs are torch tensors
    y_true = torch.as_tensor(y_true)
    y_pred = torch.as_tensor(y_pred)
    
    # Compute NMSE: mean((y_true - y_pred)^2) / mean(y_true^2)
    squared_diff = torch.mean((y_true - y_pred) ** 2)
    squared_true = torch.mean(y_true ** 2)
    nmse = squared_diff / squared_true
    
    return nmse

def train_lwm(model, train_loaders, val_loaders, optimizer, scheduler, epochs, device, save_dir="models", log_file="training_log.csv"):
    """
    Trains the Large Wireless Model (LWM) using masked channel modeling on grouped datasets of various sequence lengths.

    The training alternates between training and evaluation every 2 epochs. For each sequence length, 
    a separate DataLoader is used. MSE is used for the training objective, and both MSE and NMSE are computed
    during validation for performance tracking.

    Args:
        model (torch.nn.Module): The LWM model to train.
        train_loaders (dict): Dictionary mapping sequence length to DataLoader for training data.
        val_loaders (dict): Dictionary mapping sequence length to DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer to use for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        epochs (int): Total number of epochs to train for.
        device (torch.device): Device to train on ('cuda' or 'cpu').
        save_dir (str, optional): Directory to save the best model checkpoints. Default is "models".
        log_file (str, optional): CSV path to log training/validation metrics. Default is "training_log.csv".

    Returns:
        model (torch.nn.Module): The trained model with the best checkpoint (based on validation MSE).
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Initialize loss criterion
    criterion = nn.MSELoss(reduction='sum')  # Sum reduction for manual averaging

    # Initialize lists to store losses
    train_mse_losses = []
    val_mse_losses = []
    val_nmse_losses = []
    best_val_mse = float('inf')

    for epoch in range(epochs):
        # Training loop
        model.train()
        train_mse = 0.0
        train_samples = 0

        print(f"\nEpoch {epoch + 1}/{epochs} [Training]")
        for length, train_loader in train_loaders.items():
            print(f"Processing sequences of length {length}")
            with tqdm(train_loader, desc=f"Length {length} [Training]", unit="batch") as t:
                for batch in t:
                    optimizer.zero_grad()

                    # Move data to device
                    input_ids, masked_tokens, masked_pos = [b.to(device) for b in batch]

                    # Forward pass
                    logits_lm = model(input_ids, masked_pos)[0]

                    # Compute MSE loss
                    loss = criterion(masked_tokens, logits_lm)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    train_mse += loss.item()
                    train_samples += input_ids.shape[0]

                    # Update progress bar with MSE
                    t.set_postfix({"mse": train_mse / train_samples, "lr": scheduler.get_last_lr()[0]})

        # Average MSE across training samples
        train_mse = train_mse / max(train_samples, 1)
        train_mse_losses.append(train_mse)

        # Validation loop every 2 epochs
        if epoch % 2 == 0:
            model.eval()
            val_mse = 0.0
            val_nmse = 0.0
            val_samples = 0

            with torch.no_grad():
                print(f"\nEpoch {epoch + 1}/{epochs} [Validation]")
                for length, val_loader in val_loaders.items():
                    print(f"Processing sequences of length {length}")
                    with tqdm(val_loader, desc=f"Length {length} [Validation]", unit="batch") as t:
                        for batch in t:
                            # Move data to device
                            input_ids, masked_tokens, masked_pos = [b.to(device) for b in batch]

                            # Forward pass
                            logits_lm = model(input_ids, masked_pos)[0]

                            # Compute MSE loss 
                            mse = criterion(masked_tokens, logits_lm)
                            val_mse += mse.item()

                            # Compute NMSE for reporting
                            masked_tokens_np = masked_tokens.cpu().numpy()
                            logits_lm_np = logits_lm.cpu().numpy()
                            nmse = nmse_loss(masked_tokens_np, logits_lm_np)
                            val_nmse += nmse * input_ids.shape[0]

                            val_samples += input_ids.shape[0]

                            # Update progress bar with both MSE and NMSE
                            t.set_postfix({"mse": val_mse / val_samples, "nmse": val_nmse / val_samples})

            # Average MSE and NMSE across validation samples
            val_mse = val_mse / max(val_samples, 1)
            val_nmse = val_nmse / max(val_samples, 1)
            val_mse_losses.append(val_mse)
            val_nmse_losses.append(val_nmse)

            # Save model if validation MSE improves
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                model_path = os.path.join(save_dir, f"lwm_epoch{epoch+1}_train{train_mse:.4f}_val{val_mse:.4f}.pth")
                torch.save(model.state_dict(), model_path)
                print(f"Model saved: {model_path}")

        # Log the results
        print(f"  Train MSE: {train_mse:.4f}")
        if epoch % 2 == 0:
            print(f"  Validation MSE: {val_mse:.4f}")
            print(f"  Validation NMSE: {val_nmse:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6e}")

        # Plot losses after each epoch
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_mse_losses) + 1), train_mse_losses, label="Train MSE")
        if val_mse_losses:  # Plot validation only if it exists
            plt.plot(range(1, len(val_mse_losses) + 1), val_mse_losses, label="Validation MSE")
            plt.plot(range(1, len(val_nmse_losses) + 1), val_nmse_losses, label="Validation NMSE")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Losses")
        plt.legend()
        plt.grid(True)
        plt.show()

    print("Training and validation complete.")
    return model

def inference(model, tokens, batch_size=128, device="cuda"):
    """
    Runs inference using a trained model to extract embeddings from input tokens.

    This function processes the input tokens in batches (without shuffling), moves them to the 
    specified device, passes them through the model, and aggregates the outputs.

    Args:
        model (torch.nn.Module): The trained model used for inference.
        tokens (torch.Tensor): Input tensor of shape (N, ...) representing the tokenized data.
        batch_size (int, optional): Batch size for inference. Default is 128.
        device (str or torch.device, optional): Device to run inference on. Default is "cuda".

    Returns:
        torch.Tensor: A tensor of shape (N, D), where D is the embedding dimension produced by the model.
    """
    dataset = TensorDataset(tokens)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    embeddings = []
    model.eval()
    with torch.no_grad():
        with tqdm(dataloader, desc="Inference", unit="batch") as t:
            for batch in t:
                
                input_ids = batch[0].to(device)
                output = model(input_ids)
                embeddings.append(output)
                    
    output_total = torch.cat(embeddings, dim=0).float()
    return output_total

def visualize_embeddings(embeddings, labels=None, method="tsne", label=None):
    """
    Visualizes high-dimensional embeddings in 2D using PCA, UMAP, or t-SNE.

    This function reduces the dimensionality of embeddings to two components and visualizes them
    with an optional color-coding based on provided labels. It supports three reduction methods:
    PCA (linear), UMAP (nonlinear, preserves local/global structure), and t-SNE (nonlinear, local structure).

    Args:
        embeddings (torch.Tensor or np.ndarray): Embedding matrix of shape (n_samples, n_features).
        labels (torch.Tensor or np.ndarray, optional): Class labels of shape (n_samples,). If provided,
            each class will be visualized with a distinct color.
        method (str): Dimensionality reduction method: one of {'pca', 'umap', 'tsne'}. Default is 'tsne'.
        label (str, optional): Title for the plot. Defaults to "Embedding Visualization" if not provided.

    Raises:
        ValueError: If an unsupported dimensionality reduction method is specified.

    Returns:
        None. Displays a 2D scatter plot of the embeddings.
    """
    # to numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # choose reducer
    m = method.lower()
    if m == "pca":
        reducer = PCA(n_components=2)
    elif m == "umap":
        reducer = umap.UMAP(n_components=2, n_neighbors=16, random_state=42)
    elif m == "tsne":
        reducer = TSNE(n_components=2, random_state=42, init="random")
    else:
        raise ValueError("Invalid method. Choose 'pca', 'umap', or 'tsne'.")

    Z = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    if labels is not None:
        num_classes = len(np.unique(labels))
        colors = plt.cm.get_cmap("tab10", num_classes)

        for class_idx in range(num_classes):
            class_points = Z[labels == class_idx]
            plt.scatter(
                class_points[:, 0], class_points[:, 1],
                label=f"Class {class_idx}",
                alpha=0.6,
                cmap=colors
            )
    else:
        plt.scatter(Z[:, 0], Z[:, 1], color="C0", alpha=0.6, label="Samples")

    title = label or "Embedding Visualization"
    plt.title(f"{title} ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()


def embedding_space_visual(model, data, input_type="cls_emb", device="cpu", batch_size=64, task=None, visualization=False, labels=None, visualization_method="tsne", selected_tokens=0):
    """
    Extracts embeddings from a model and optionally visualizes the embedding space.

    Supports different types of embeddings from the model (e.g., CLS token, mean-pooled embeddings),
    and provides 2D visualization using t-SNE, PCA, or UMAP. Also supports angular-based clustering 
    if the task is "ChannelCharting".

    Args:
        model (torch.nn.Module): The trained model for embedding extraction.
        data (torch.Tensor): Input tensor of shape (N, ...). For non-'raw' types, it's tokenized input.
        input_type (str): Type of embedding to extract. Options:
            - 'cls_emb': Extract CLS token (first token) embeddings.
            - 'channel_emb': Extract all non-CLS token embeddings.
            - 'combined': Use the full output from the model.
            - 'mean_pooled': Mean of all token embeddings.
            - 'arbitrary_concat': Select specific token index (given by `selected_tokens`).
            - 'arbitrary_meanPooled': Mean-pool over selected token indices.
            - 'raw': Use the input `data` as is.
        device (str or torch.device): The device for computation ('cuda' or 'cpu').
        batch_size (int): Batch size for inference.
        task (str, optional): If "ChannelCharting", performs position-based angular clustering.
        visualization (bool): Whether to visualize the embeddings.
        labels (torch.Tensor or np.ndarray, optional): Ground-truth labels or 2D positions used for coloring the plot.
        visualization_method (str): One of {'tsne', 'pca', 'umap'} for 2D visualization.
        selected_tokens (int, list, or tensor): Used for 'arbitrary_concat' or 'arbitrary_meanPooled' types.

    Returns:
        torch.Tensor: Output embedding tensor of shape (N, D) ready for downstream tasks or analysis.
    """
    print("\nPreparing for LWM inference and embedding space visualization ...")
    if input_type == "raw":
        output_total = data
    else:
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        embeddings = []
        model.eval()
        with torch.no_grad():
            with tqdm(dataloader, desc="Inference", unit="batch") as t:
                for batch in t:
                    
                    input_ids = batch[0].to(device)
                    output = model(input_ids)
                
                    if input_type == "cls_emb":
                        batch_embeddings = output[:, 0] 
                    elif input_type == "channel_emb":
                        batch_embeddings = output[:, 1:] 
                    elif input_type == "combined":
                        batch_embeddings = output
                    elif input_type == "mean_pooled":
                        batch_embeddings = torch.mean(output, dim=1).unsqueeze(1)
                    elif input_type == "arbitrary_concat":
                        batch_embeddings = output[:, selected_tokens]
                    elif input_type == "arbitrary_meanPooled":
                        batch_embeddings = torch.mean(output[:, selected_tokens], dim=1).unsqueeze(1)
                    
                    embeddings.append(batch_embeddings)
                    
                    
        output_total = torch.cat(embeddings, dim=0).float()
        
        if visualization:
            
            if task in ["ChannelCharting"]:
                
                positions = labels.cpu().numpy()
                x_coords = positions[:, 0] 
                y_coords = positions[:, 1]  
                center_x = np.mean(x_coords)
                center_y = np.mean(y_coords)
                x_shifted = x_coords - center_x
                y_shifted = y_coords - center_y
                angles = np.arctan2(y_shifted, x_shifted)
                angles = (angles + 2 * np.pi) % (2 * np.pi)
                n_clusters = 8
                sector_size = (2 * np.pi) / n_clusters  # Size of each sector in radians
                labels = np.floor(angles / sector_size).astype(int)
                labels = np.clip(labels, 0, n_clusters - 1)

                plt.figure(figsize=(10, 8))
                if labels is not None:
                    # Color-code by labels if provided
                    num_classes = len(np.unique(labels))
                    colors = plt.cm.get_cmap("tab10", num_classes)
    
                    for class_idx in range(num_classes):
                        class_points = positions[labels == class_idx]
                        plt.scatter(
                            class_points[:, 0], class_points[:, 1],
                            label=f"Class {class_idx}",
                            alpha=0.6,
                            cmap=colors
                        )
                else:
                    # Plot all points in a single color if no labels
                    plt.scatter(
                        positions[:, 0], positions[:, 1],
                        color="blue",  # Default color for unlabeled data
                        alpha=0.6,
                        label="Samples"
                    )
                plt.title("Original 2D Positions with 5 Clusters")
                plt.xlabel("X Coordinate")
                plt.ylabel("Y Coordinate")
                plt.legend()
                plt.grid(True, linestyle="--", alpha=0.3)
                plt.show()

            visualize_embeddings(output_total.view(output_total.size(0), -1), 
                                 labels=labels, 
                                 method=visualization_method, 
                                 label="Embedding Space")
        
    return output_total

def plot_radar_chart(task_names, optimized_scores, baseline_scores, title="Task Performance Comparison", figsize=(8, 8), save_path="submission/chart.png"):
    """
    Plot a dark-themed radar chart comparing optimized and baseline scores.

    Args:
        task_names (list): List of task names (e.g., ["LoS/NLoS Classification", ...]).
        optimized_scores (list): List of optimized performance scores.
        baseline_scores (list): List of baseline performance scores.
        title (str): Title of the chart (default: "Task Performance Comparison").
        figsize (tuple): Figure size (width, height) in inches (default: (8, 8)).
        save_path (str): Path to save the figure (default: "submission/chart.png").

    Raises:
        ValueError: If input lists have mismatched lengths or are empty.
    """
    # Input validation
    if not task_names or not optimized_scores or not baseline_scores:
        raise ValueError("All input lists (task_names, optimized_scores, baseline_scores) must not be empty")
    if not (len(task_names) == len(optimized_scores) == len(baseline_scores)):
        raise ValueError("All input lists must have the same length")

    # Number of variables (tasks)
    num_tasks = len(task_names)

    # Repeat the first score to close the radar chart
    angles = [n / float(num_tasks) * 2 * pi for n in range(num_tasks)]
    angles += angles[:1]
    optimized = optimized_scores + optimized_scores[:1]
    baseline = baseline_scores + baseline_scores[:1]

    # Figure & Axes Setup
    plt.style.use('default')  # Reset to default style to avoid global dark background
    fig, ax = plt.subplots(
        figsize=figsize, dpi=300,
        subplot_kw=dict(polar=True)
    )
    fig.patch.set_facecolor('#1a1a1a')  # Dark gray background for the figure (outside the circle)
    ax.set_facecolor('#1a1a1a')  # Dark background for the plot (inside the circle)
    ax.patch.set_alpha(1)

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Grid & Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(task_names, fontsize=14, fontweight='bold', color='white')
    ax.set_rlabel_position(0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=11, color='#bbbbbb')
    ax.set_ylim(0, 1.15)
    ax.grid(color='#bbbbbb', linestyle=':', linewidth=0.8)

    # Glow Effect Function
    def plot_with_glow(x, y, color, label):
        for lw, alpha in zip([15, 10, 6, 3], [0.02, 0.04, 0.06, 0.08]):
            ax.plot(x, y, linewidth=lw, color=color, alpha=alpha, zorder=1)
        ax.plot(x, y, linewidth=2.5, color=color, label=label, zorder=2)
        ax.scatter(x, y, s=100, color=color, edgecolors='white', linewidth=1.5, zorder=3)

    # Plot optimized & baseline
    plot_with_glow(angles, optimized, color='#00ffff', label='Optimized')
    plot_with_glow(angles, baseline, color='#ff5588', label='Baseline')

    # Layered Fill
    ax.fill_between(angles, optimized, color='#00ffff', alpha=0.12, zorder=1)
    ax.fill_between(angles, baseline, color='#ff5588', alpha=0.12, zorder=1)

    # Custom Legend
    legend_elements = [
        plt.Line2D([0], [0], color='#00ffff', lw=3, label='Optimized'),
        plt.Line2D([0], [0], color='#ff5588', lw=3, label='Baseline')
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper right', bbox_to_anchor=(1.15, 1.15),
        frameon=True, facecolor='#bbbbbb', edgecolor='#555555', fontsize=18
    )
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.show()
