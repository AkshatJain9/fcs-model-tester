import os
import flowkit as fk
import numpy as np
import matplotlib.pyplot as plt
import platform
import glob


################# GLOBAL VARIABLES #################
scatter_channels = ['FSC-A', 'FSC-H', 'FSC-W', 'SSC-A', 'SSC-H', 'SSC-W']
fluro_channels = ['BUV 395-A', 'BUV737-A', 'Pacific Blue-A', 'FITC-A', 'PerCP-Cy5-5-A', 'PE-A', 'PE-Cy7-A', 'APC-A', 'Alexa Fluor 700-A', 'APC-Cy7-A','BV510-A','BV605-A']
new_channels = ['APC-Alexa 750 / APC-Cy7-A', 'Alexa 405 / Pac Blue-A', 'Qdot 605-A']
fluro_channels += new_channels

all_channels = scatter_channels + fluro_channels

transform = fk.transforms.LogicleTransform('logicle', param_t=262144, param_w=0.5, param_m=4.5, param_a=0)

dataset = "Synthetic"
processing_type = "rawdata"

if (platform.system() == "Windows"):
    somepath = ".\\" + dataset + "\\" + processing_type + "\\"
else:
    somepath = "./" + dataset + "/" + processing_type + "/"

####################################################


##################### UTILITY FUNCTIONS #####################
def print_array(arr):
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    print(np.array2string(arr, separator=', ', formatter={'float_kind':lambda x: f"{x:.2f}"}))

def load_data(panel: str) -> np.ndarray:
    panel_root = somepath
    if (os.path.exists(panel_root + panel + ".npy")):
        return np.load(panel_root + panel + ".npy")

    # Recursively search for all .fcs files in the directory and subdirectories
    fcs_files = glob.glob(os.path.join(panel_root, '**', '*.fcs'), recursive=True)
    fcs_files_np = []

    if (platform.system() == "Windows"):
        spillover = "C:\\Users\\aksha\\Documents\\ANU\\COMP4550_(Honours)\\Spillovers\\281122_Spillover_Matrix.csv"
    else:
        spillover = "/home/akshat/Documents/281122_Spillover_Matrix.csv"
    
    # Load each .fcs file into fk.Sample and print it
    for fcs_file in fcs_files:
        sample = fk.Sample(fcs_file)
        if "Panel" in panel:
            sample.apply_compensation(spillover)
        else:
            sample.apply_compensation(sample.metadata['spill'])
        sample.apply_transform(transform)
        fcs_files_np.append(get_np_array_from_sample(sample, subsample=True))

    stacked_np = np.vstack(fcs_files_np)
    np.save(panel_root + panel + ".npy", stacked_np)
    return stacked_np

def get_channel_source(channel: str) -> str:
    """ Get the source of the channel

    Args:
        channel: The channel to get the source of

    Returns:
        str: The source of the channel
    """

    if channel in scatter_channels:
        return 'raw'
    return 'xform'

def get_factor(channel: str) -> float:
    """ Get the factor to divide the channel by

    Args:
        channel: The channel to get the factor for

    Returns:
        float: The factor to divide the channel by
    """

    if channel in scatter_channels:
        return 262144.0
    return 1.0

def get_np_array_from_sample(sample: fk.Sample, subsample: bool) -> np.ndarray:
    """ Get a np.ndarray from a Sample object

    Args:
        sample: The Sample object to convert
        subsample: Whether to subsample the data

    Returns:
        np.ndarray: The np.ndarray representation of the Sample object
    """

    return np.array([
        sample.get_channel_events(sample.get_channel_index(ch), source=get_channel_source(ch), subsample=subsample) / get_factor(ch)
        for ch in all_channels if ch in sample.pnn_labels
    ]).T

###################################################



##################### BASIC SPREAD #####################
def print_mean_summary(panel_np):
    mean_vector = np.mean(panel_np, axis=0)
    return mean_vector


# Covariance Summary
def print_cov_summary(panel_np):
    cov_matrix = np.cov(panel_np, rowvar=False)
    return cov_matrix



###################### 1D EMD ######################
def generate_histogram(panel_np, index):
    range = (0, 1)

    hist, _ = np.histogram(panel_np[:, index], bins=100, range=range)
    hist = hist / np.sum(hist)
    return hist

def compute_emd(hist1, hist2):
    return np.sum(np.abs(hist1 - hist2))


def compute_all_emd(panel_np1, panel_np2):
    emds = []
    for i in range(6, panel_np1.shape[1]):
        hist1 = generate_histogram(panel_np1, i)
        hist2 = generate_histogram(panel_np2, i)
        emds.append(compute_emd(hist1, hist2))
    return emds


###################### 2D EMD ######################
def gen_2d_histogram(panel_np, index1, index2):
    assert((index1 < 6 and index2 < 6) or (index1 >= 6 and index2 >= 6))

    range = (0, 1)

    hist, xedges, yedges = np.histogram2d(panel_np[:, index1], panel_np[:, index2], bins=50, range=[range, range])
    hist = hist / np.sum(hist)
    # reverse
    hist = np.flip(hist, 0)
    return hist

def plot_2d_histogram(hist):
    plt.imshow(hist, interpolation='nearest')
    plt.show()


def compute_all_emd_2d(panel_np1, panel_np2):
    emds_scatter = []
    for i in range(len(scatter_channels)):
        emd_row = []
        for j in range(len(scatter_channels)):
            hist1 = gen_2d_histogram(panel_np1, i, j)
            hist2 = gen_2d_histogram(panel_np2, i, j)
            emd_row.append(compute_emd(hist1, hist2))
        emds_scatter.append(emd_row)

    emds_fluro = []
    for i in range(len(fluro_channels)):
        emd_row = []
        for j in range(len(fluro_channels)):
            hist1 = gen_2d_histogram(panel_np1, i + 6, j + 6)
            hist2 = gen_2d_histogram(panel_np2, i + 6, j + 6)
            emd_row.append(compute_emd(hist1, hist2))
        emds_fluro.append(emd_row)

    return emds_scatter, emds_fluro

###################### PLOTTING ######################
def plot_all_histograms(panel_np_1, panel_np_2):
    num_channels = len(all_channels)
    
    # Create a figure with subplots, one for each channel
    fig, axs = plt.subplots(num_channels, 1, figsize=(8, 2*num_channels))
    
    # If there's only one subplot, axs won't be an array, so wrap it in one
    if num_channels == 1:
        axs = [axs]
    
    for i, ax in enumerate(axs):
        hist1 = generate_histogram(panel_np_1, i)
        hist2 = generate_histogram(panel_np_2, i)
        
        # Plot on the current axis
        ax.plot(hist1, label='Panel 1')
        ax.plot(hist2, label='Panel 2')
        ax.legend()
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    b1 = load_data("Panel1")
    b2 = load_data("Panel2")

    emd = compute_all_emd(b1, b2)
    print(np.mean(emd))