import os
import flowkit as fk
import numpy as np
import matplotlib.pyplot as plt
import platform
import glob
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import mean_squared_error
from sklearn.mixture import GaussianMixture
import torch


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

def print_std_summary(panel_np):
    std_vector = np.std(panel_np, axis=0)
    return std_vector


###################### 1D TVD ######################
def generate_histogram(panel_np, index, min_val, max_val):
    range = (min_val, max_val)

    hist, _ = np.histogram(panel_np[:, index], bins=200, range=range)
    hist = hist / np.sum(hist)
    return hist

def compute_tvd(hist1, hist2):
    return np.sum(np.abs(hist1 - hist2))


def compute_all_tvd(panel_np1, panel_np2):
    tvds = []
    for i in range(panel_np1.shape[1]):
        min_val = np.min([np.min(panel_np1[:, i]), np.min(panel_np2[:, i])])
        max_val = np.max([np.max(panel_np1[:, i]), np.max(panel_np2[:, i])])

        hist1 = generate_histogram(panel_np1, i, min_val, max_val)
        hist2 = generate_histogram(panel_np2, i, min_val, max_val)
        tvds.append(compute_tvd(hist1, hist2))
    return tvds


###################### EMD ######################
def compute_all_emd(panel_np1, panel_np2):
    emds = []
    for i in range(panel_np1.shape[1]):
        min_val = np.min([np.min(panel_np1[:, i]), np.min(panel_np2[:, i])])
        max_val = np.max([np.max(panel_np1[:, i]), np.max(panel_np2[:, i])])

        hist1 = generate_histogram(panel_np1, i, min_val, max_val)
        hist2 = generate_histogram(panel_np2, i, min_val, max_val)
        emds.append(wasserstein_distance(hist1, hist2))
    return emds

###################### 2D TVD ######################
def gen_2d_histogram(panel_np, index1, index2, min_val, max_val):
    assert((index1 < 6 and index2 < 6) or (index1 >= 6 and index2 >= 6))

    range = (min_val, max_val)

    hist, _, _ = np.histogram2d(panel_np[:, index1], panel_np[:, index2], bins=50, range=[range, range])
    hist = hist / np.sum(hist)
    hist = np.flip(hist, 0)
    return hist

def plot_2d_histogram(hist):
    plt.imshow(hist, interpolation='nearest')
    plt.show()


def compute_all_tvd_2d(panel_np1, panel_np2):
    tvds_fluoro = []
    for i in range(panel_np1.shape[1] - 6):
        tvd_row = []
        for j in range(panel_np1.shape[1] - 6):
            min_val = np.min([np.min(panel_np1[:, i + 6, j + 6]), np.min(panel_np2[:, i + 6, j + 6])])
            max_val = np.max([np.max(panel_np1[:, i + 6, j + 6]), np.max(panel_np2[:, i + 6, j + 6])])
            hist1 = gen_2d_histogram(panel_np1, i + 6, j + 6, min_val, max_val)
            hist2 = gen_2d_histogram(panel_np2, i + 6, j + 6, min_val, max_val)
            tvd_row.append(compute_tvd(hist1, hist2))
        tvds_fluoro.append(tvd_row)

    return tvds_fluoro

###################### PLOTTING ######################
def plot_all_histograms(panel_np_1, panel_np_2):
    num_channels = panel_np_1.shape[1]
    
    # Create a figure with subplots, one for each channel
    fig, axs = plt.subplots(num_channels, 1, figsize=(8, 2*num_channels))
    
    # If there's only one subplot, axs won't be an array, so wrap it in one
    if num_channels == 1:
        axs = [axs]
    
    for i, ax in enumerate(axs):
        min_val = np.min([np.min(panel_np_1[:, i]), np.min(panel_np_2[:, i])])
        max_val = np.max([np.max(panel_np_1[:, i]), np.max(panel_np_2[:, i])])
        hist1 = generate_histogram(panel_np_1, i, min_val, max_val)
        hist2 = generate_histogram(panel_np_2, i, min_val, max_val)
        
        # Plot on the current axis
        ax.plot(hist1, label='Panel 1')
        ax.plot(hist2, label='Panel 2')
        ax.legend()
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.show()


############# CELL POPULATIONS ################
def get_main_cell_pops(data, k):
    gmm = GaussianMixture(n_components=k, random_state=0).fit(data)
    return gmm.means_, gmm.covariances_, gmm.predict(data)

# Function to compute the average distance between two sets of cluster centers
def average_cluster_distance(cluster_centers1, cluster_centers2):

    # Initialize an array to hold the MSE values between each pair of centers
    mse_matrix = np.zeros((cluster_centers1.shape[0], cluster_centers2.shape[0]))
    
    # Calculate the MSE between each pair of centers
    for i, centre in enumerate(cluster_centers1):
        for j, transformed_centre in enumerate(cluster_centers2):
            mse_matrix[i, j] = mean_squared_error(centre, transformed_centre)
    
    # Use the Hungarian algorithm to find the optimal matching
    row_ind, col_ind = linear_sum_assignment(mse_matrix)
    
    # Create a list to hold the aligned center pairs and their MSE values
    aligned_centers = []
    correspondence_array = []
    for i, j in zip(row_ind, col_ind):
        aligned_centers.append({
            'x_centre_index': i,
            'x_transformed_centre_index': j,
            'mse': mse_matrix[i, j]
        })
        correspondence_array.append([i, j])
    
    mean_mse = np.mean([pair['mse'] for pair in aligned_centers])

    return mean_mse, correspondence_array

def kl_divergence(cov1, cov2):
    n = cov1.shape[0]
    logdet_cov2 = np.linalg.slogdet(cov2)[1]
    logdet_cov1 = np.linalg.slogdet(cov1)[1]
    inv_cov2 = np.linalg.inv(cov2)
    trace_term = np.trace(np.matmul(inv_cov2, cov1))
    return 0.5 * (trace_term + logdet_cov2 - logdet_cov1 - n)

    # Calculate the difference between the two matrices
    # difference = cov1 - cov2
    
    # # Calculate the Frobenius norm
    # return np.sqrt(np.sum(np.abs(difference)**2))

###################### FINAL FUNCTION ######################
def compute_all_metrics(reference_batch, target_batches):
    file_name = somepath + "results.txt"

    # Open the file for writing the summary
    with open(file_name, "w") as file:
        # Print header with batch names
        file.write("Batches:\n")
        for batch_name in target_batches.keys():
            file.write(f"{batch_name}\n")
        file.write("\n-------------------------\n")

        # # Mean Summary for all batches
        # file.write("Mean Summaries:\n")
        # mean1 = print_mean_summary(reference_batch)
        # file.write("Mean Summary Reference Dataset:\n" + str(mean1) + "\n")

        # for batch_name, target_batch in target_batches.items():
        #     mean2 = print_mean_summary(target_batch)
        #     file.write(f"Mean Summary {batch_name}:\n" + str(mean2) + "\n")
        # file.write("\n-------------------------\n")

        # # MSE Difference in Means for all batches
        # file.write("MSE Difference in Means:\n")
        # for batch_name, target_batch in target_batches.items():
        #     mean2 = print_mean_summary(target_batch)
        #     mean_diff = np.mean((mean1 - mean2)**2)
        #     file.write(f"MSE Difference for {batch_name}: {mean_diff}\n")
        # file.write("\n-------------------------\n")

        # # Std Summary for all batches
        # file.write("Std Summaries:\n")
        # std1 = print_std_summary(reference_batch)
        # file.write("Std Summary Reference Dataset:\n" + str(std1) + "\n")

        # for batch_name, target_batch in target_batches.items():
        #     std2 = print_std_summary(target_batch)
        #     file.write(f"Std Summary {batch_name}:\n" + str(std2) + "\n")
        # file.write("\n-------------------------\n")

        # # MSE Difference in Std for all batches
        # file.write("MSE Difference in Standard Deviations:\n")
        # for batch_name, target_batch in target_batches.items():
        #     std2 = print_std_summary(target_batch)
        #     std_diff = np.mean((std1 - std2)**2)
        #     file.write(f"MSE Difference for {batch_name}: {std_diff}\n")
        # file.write("\n-------------------------\n")

        # # 1D TVD for all batches
        # file.write("1D TVD for each feature:\n")
        # for batch_name, target_batch in target_batches.items():
        #     tvds = compute_all_tvd(reference_batch, target_batch)
        #     file.write(f"1D TVD for {batch_name}:\n" + str(tvds) + "\n")
        #     file.write(f"Mean 1D TVD for {batch_name}:\n" + str(np.mean(tvds)) + "\n")
        # file.write("\n-------------------------\n")

        # # 1D EMD for all batches
        # file.write("1D EMD for each feature:\n")
        # for batch_name, target_batch in target_batches.items():
        #     emds = compute_all_emd(reference_batch, target_batch)
        #     file.write(f"1D EMD for {batch_name}:\n" + str(emds) + "\n")
        #     file.write(f"Mean 1D EMD for {batch_name}:\n" + str(np.mean(emds)) + "\n")
        # file.write("\n-------------------------\n")

        # Cluster Distance for all batches
        file.write("Average Cluster Distance:\n")
        cluster_centers1, cluster_cov1, _ = get_main_cell_pops(reference_batch[:, 6:], 7)

        for batch_name, target_batch in target_batches.items():
            cluster_centers2, cluster_cov2, _ = get_main_cell_pops(target_batch[:, 6:], 7)
            cluster_dist, correspondence_arr = average_cluster_distance(cluster_centers1, cluster_centers2)
            kl_divs = [kl_divergence(cluster_cov1[i], cluster_cov2[j]) for i, j in correspondence_arr]
            kl_divs_mean = np.mean(kl_divs)
            file.write(f"Average Cluster Distance for {batch_name}: {cluster_dist}\n")
            file.write(f"Average KL Divergence for {batch_name}: {kl_divs_mean}\n")

    print(f"Metrics summary saved to {file_name}")


if __name__ == "__main__":
    b1 = load_data("Panel1")
    b2 = load_data("Panel2")
    b3 = load_data("Panel1_frob")

    d = dict()
    d["Panel 2"] = b2
    d["Panel 1 Transformed"] = b3

    compute_all_metrics(b1, d)