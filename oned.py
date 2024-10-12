import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import seaborn as sns


################## 1D Histograms ##################
scatter_channels = ['FSC-A', 'FSC-H', 'FSC-W', 'SSC-A', 'SSC-H', 'SSC-W']
fluro_channels_panel = ['CD8', 'B220', 'CD44', 'CD3', 'IgD', 'CD25', 'Ly6C', 'NK1.1', 'IgM & CD4', 'LD', 'CD19', 'CD62L']
fluro_channels_enu = ['IgM', 'B2220', 'IgD', 'KLRG1', 'NK1.1', 'CD4', 'CD3', 'CD44', 'CD8']

# Histogram generation function
def generate_histogram(panel_np, index, min_val, max_val):
    range = (min_val, max_val)

    hist, bin_edges = np.histogram(panel_np[:, index], bins=200, range=range)
    hist = hist / np.sum(hist)  # Normalize the histogram
    return hist, bin_edges


def plot_fluoro_hist_compare(data_list, names=None, synth_batch=True):
    """
    Plots histograms comparing fluorescence data across multiple datasets.

    Parameters:
        data_list (list of numpy arrays): A list of datasets to be compared.
        name (str): Name used for labeling and saving the plot.
        synth_batch (bool): If True, use synthetic batch labels; otherwise, use default labels.
    """
    num_datasets = len(data_list)
    
    if num_datasets < 2:
        print("Error: At least two datasets are required for comparison.")
        return
    
    num_channels = data_list[0].shape[1]
    
    # Ensure all datasets have the same number of channels
    for data in data_list:
        if data.shape[1] != num_channels:
            print("Error: All datasets must have the same number of channels.")
            return

    # Plot the remaining channels, assuming we skip the first 6
    remaining_channels = num_channels - 6
    fig2, axs2 = plt.subplots(remaining_channels, 1, figsize=(8, 2 * remaining_channels))

    # Choose labels based on synth_batch flag
    if synth_batch:
        labels = fluro_channels_panel
    else:
        labels = fluro_channels_enu

    # Iterate over channels starting from the 7th (index 6)
    for i, ax in enumerate(axs2):
        channel_index = i + 6
        
        # Determine the min and max values across all datasets for the current channel
        min_val = np.min([np.min(data[:, channel_index]) for data in data_list])
        max_val = np.max([np.max(data[:, channel_index]) for data in data_list])
        
        # Plot histograms for each dataset
        for dataset_idx, data in enumerate(data_list):
            if (names is not None):
                data_name = names[dataset_idx]
            else:
                data_name = f'Dataset {dataset_idx+1}'
            hist, bin_edges = generate_histogram(data, channel_index, min_val, max_val)
            ax.bar(bin_edges[:-1], hist, width=(bin_edges[1] - bin_edges[0]), alpha=0.5, 
                   label=data_name)

        # Set axis labels using the provided labels
        ax.set_xlabel(labels[i] + " - Logicle Transformed Value")
        ax.set_ylabel('Frequency (Relative)')
        ax.legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f'fluro_histograms_compare.png')
    plt.close()


def plot_fluoro_hist_compare_channel(data_list, i, names=None, synth_batch=True):
    """
    Plots histograms comparing fluorescence data for a specific channel across multiple datasets.

    Parameters:
        data_list (list of numpy arrays): A list of datasets to be compared.
        i (int): The channel index to plot.
        names (list of str, optional): Names of datasets for labeling in the plot. Defaults to None.
        synth_batch (bool): If True, use synthetic batch labels; otherwise, use default labels.
    """
    num_datasets = len(data_list)

    if num_datasets < 2:
        print("Error: At least two datasets are required for comparison.")
        return
    
    num_channels = data_list[0].shape[1]

    # Ensure all datasets have the same number of channels
    for data in data_list:
        if data.shape[1] != num_channels:
            print("Error: All datasets must have the same number of channels.")
            return

    # Ensure the specified channel index `i` is valid
    if i < 0 or i >= num_channels:
        print(f"Error: Channel index {i} is out of bounds for the dataset.")
        return

    # Choose labels based on synth_batch flag
    if synth_batch:
        labels = fluro_channels_panel
    else:
        labels = fluro_channels_enu

    # Create a single plot for the specified channel `i`
    fig, ax = plt.subplots(figsize=(8, 6))

    # Determine the min and max values across all datasets for the specified channel
    min_val = np.min([np.min(data[:, i]) for data in data_list])
    max_val = np.max([np.max(data[:, i]) for data in data_list])

    # Plot histograms for each dataset
    for dataset_idx, data in enumerate(data_list):
        if names is not None:
            data_name = names[dataset_idx]
        else:
            data_name = f'Dataset {dataset_idx + 1}'
        
        # Generate histogram
        hist, bin_edges = generate_histogram(data, i, min_val, max_val)
        
        # Plot the histogram for this dataset
        ax.bar(bin_edges[:-1], hist, width=(bin_edges[1] - bin_edges[0]), alpha=0.5, label=data_name)

    # Set axis labels using the provided labels
    ax.set_xlabel(f"{labels[i]} - Logicle Transformed Value")
    ax.set_ylabel('Frequency (Relative)')
    ax.legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f'fluro_histograms_compare_channel_{i}.png')
    plt.close()