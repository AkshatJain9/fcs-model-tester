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


def plot_fluoro_hist_compare(data_list, names=None, idx=None, synth_batch=True, file_name=None):
    """
    Plots histograms comparing fluorescence data across multiple datasets.

    Parameters:
        data_list (list of numpy arrays): A list of datasets to be compared.
        names (list of str): Names used for labeling the datasets.
        idx (int or list of ints): Index or indices of columns to plot. If None, plot all columns starting from index 6.
        synth_batch (bool): If True, use synthetic batch labels; otherwise, use default labels.
        file_name (str): Name used for saving the plot.
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

    # Determine the indices of channels to plot
    if idx is not None:
        # If idx is a single integer, convert it to a list
        if isinstance(idx, int):
            channels_to_plot = [idx]
        else:
            channels_to_plot = idx  # Assume idx is iterable
    else:
        # Plot all channels starting from index 6
        channels_to_plot = list(range(6, num_channels))

    remaining_channels = len(channels_to_plot)
    if remaining_channels == 0:
        print("Error: No channels to plot.")
        return

    fig2, axs2 = plt.subplots(remaining_channels, 1, figsize=(8, 2 * remaining_channels))

    # If only one subplot, axs2 may not be a list, so make it into a list
    if remaining_channels == 1:
        axs2 = [axs2]

    # Choose labels based on synth_batch flag
    labels_full = [f'Channel {i}' for i in range(num_channels)]
    if synth_batch:
        labels_full[6:] = fluro_channels_panel
    else:
        labels_full[6:] = fluro_channels_enu

    colors = sns.color_palette("bright", len(data_list))

    for channel_index, ax in zip(channels_to_plot, axs2):
        # Determine the min and max values across all datasets for the current channel
        min_val = np.min([np.min(data[:, channel_index]) for data in data_list])
        max_val = np.max([np.max(data[:, channel_index]) for data in data_list])

        # Plot histograms for each dataset
        for dataset_idx, data in enumerate(data_list):
            if names is not None:
                data_name = names[dataset_idx]
            else:
                data_name = f'Dataset {dataset_idx+1}'
            hist, bin_edges = generate_histogram(data, channel_index, min_val, max_val)
            ax.plot(bin_edges[:-1], hist, label=data_name, color=colors[dataset_idx])

        # Set axis labels using the provided labels
        ax.set_xlabel(labels_full[channel_index] + " - Logicle Transformed Value")
        ax.set_ylabel('Frequency (Relative)')
        ax.legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    if file_name is not None:
        plt.savefig(f'fluro_histograms_compare_{file_name}.png')
    else:
        plt.savefig(f'fluro_histograms_compare.png')
    plt.close()



def plot_fluoro_hist_compare_channel(data_list, i, names=None, synth_batch=True, file_name=None):
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
    if file_name is not None:
        plt.savefig(f'fluro_histograms_compare_channel_{file_name}_{labels[i]}.png')
    else:
        plt.savefig(f'fluro_histograms_compare_channel_{labels[i]}.png')
    plt.close()

if __name__ == "__main__":
    dataset = "Synthetic"
    directory = "rawdata"

    batches = ["Panel1", "Panel3"]
    data_list = []
    names = []
    for batch in batches:
        filename = f"{dataset}/{directory}/{batch}.npy"
        data = np.load(filename)
        data_list.append(data)
        names.append(batch)

    # plot_fluoro_hist_compare(data_list, names, synth_batch=True, file_name=directory)
    plot_fluoro_hist_compare(data_list, idx=4, names=names, synth_batch=True, file_name=directory)