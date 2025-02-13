import os
import flowkit as fk
import numpy as np
import matplotlib.pyplot as plt
import platform
import glob
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import mean_squared_error
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


################# GLOBAL VARIABLES #################
scatter_channels = ['FSC-A', 'FSC-H', 'FSC-W', 'SSC-A', 'SSC-H', 'SSC-W']
fluro_channels = ['BUV 395-A', 'BUV737-A', 'Pacific Blue-A', 'FITC-A', 'PerCP-Cy5-5-A', 'PE-A', 'PE-Cy7-A', 'APC-A', 'Alexa Fluor 700-A', 'APC-Cy7-A','BV510-A','BV605-A']
new_channels = ['APC-Alexa 750 / APC-Cy7-A', 'Alexa 405 / Pac Blue-A', 'Qdot 605-A']
fluro_channels += new_channels

all_channels = scatter_channels + fluro_channels

transform = fk.transforms.LogicleTransform('logicle', param_t=262144, param_w=0.5, param_m=4.5, param_a=0)

dataset = "Synthetic"
processing_type = "CytofBatchAdjust"

if (platform.system() == "Windows"):
    somepath = ".\\" + dataset + "\\" + processing_type + "\\"
else:
    somepath = "./" + dataset + "/" + processing_type + "/"

####################################################


##################### UTILITY FUNCTIONS #####################
def print_array(arr):
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    print(np.array2string(arr, separator=', ', formatter={'float_kind':lambda x: f"{x:.2f}"}))

def load_data(panel: str, load_orig: bool = False) -> np.ndarray:
    panel_root = somepath
    if (not load_orig and os.path.exists(panel_root + panel + ".npy")):
        return np.load(panel_root + panel + ".npy")

    # Recursively search for all .fcs files in the directory and subdirectories
    fcs_files = glob.glob(os.path.join(panel_root, '**', '*.fcs'), recursive=True)
    fcs_files = [f for f in fcs_files if panel in f]
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
        # else:
        #     sample.apply_compensation(sample.metadata['spill'])
        sample.apply_transform(transform)
        fcs_files_np.append(get_np_array_from_sample(sample, subsample=(not load_orig)))

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
            min_val = np.min([np.min(panel_np1[:, [i + 6, j + 6]]), np.min(panel_np2[:, [i + 6, j + 6]])])
            max_val = np.max([np.max(panel_np1[:, [i + 6, j + 6]]), np.max(panel_np2[:, [i + 6, j + 6]])])
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
    return gmm.means_

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

        # Mean Summary for all batches
        file.write("Mean Summaries:\n")
        mean1 = print_mean_summary(reference_batch)
        file.write("Mean Summary Reference Dataset:\n" + str(mean1) + "\n")

        for batch_name, target_batch in target_batches.items():
            mean2 = print_mean_summary(target_batch)
            file.write(f"Mean Summary {batch_name}:\n" + str(mean2) + "\n")
        file.write("\n-------------------------\n")

        # MSE Difference in Means for all batches
        file.write("MSE Difference in Means:\n")
        for batch_name, target_batch in target_batches.items():
            mean2 = print_mean_summary(target_batch)
            mean_diff = np.mean((mean1 - mean2)**2)
            file.write(f"MSE Difference for {batch_name}: {mean_diff}\n")
        file.write("\n-------------------------\n")

        # Std Summary for all batches
        file.write("Std Summaries:\n")
        std1 = print_std_summary(reference_batch)
        file.write("Std Summary Reference Dataset:\n" + str(std1) + "\n")

        for batch_name, target_batch in target_batches.items():
            std2 = print_std_summary(target_batch)
            file.write(f"Std Summary {batch_name}:\n" + str(std2) + "\n")
        file.write("\n-------------------------\n")

        # MSE Difference in Std for all batches
        file.write("MSE Difference in Standard Deviations:\n")
        for batch_name, target_batch in target_batches.items():
            std2 = print_std_summary(target_batch)
            std_diff = np.mean((std1 - std2)**2)
            file.write(f"MSE Difference for {batch_name}: {std_diff}\n")
        file.write("\n-------------------------\n")

        # 1D TVD for all batches
        file.write("1D TVD for each feature:\n")
        for batch_name, target_batch in target_batches.items():
            tvds = compute_all_tvd(reference_batch, target_batch)
            file.write(f"1D TVD for {batch_name}:\n" + str(tvds) + "\n")
            file.write(f"Mean 1D TVD for {batch_name}:\n" + str(np.mean(tvds)) + "\n")
        file.write("\n-------------------------\n")

        # 2D TVD for all batches
        file.write("2D TVD for each feature pair:\n")
        for batch_name, target_batch in target_batches.items():
            tvds_fluoro = compute_all_tvd_2d(reference_batch, target_batch)
            file.write(f"2D TVD for {batch_name}:\n" + str(tvds_fluoro) + "\n")
            file.write(f"Mean 2D TVD for {batch_name}:\n" + str(np.mean(tvds_fluoro)) + "\n")
        file.write("\n-------------------------\n")

        # Cluster Distance for all batches
        file.write("Average Cluster Distance:\n")
        cluster_centers1 = get_main_cell_pops(reference_batch[:, 6:], 13)

        for batch_name, target_batch in target_batches.items():
            cluster_centers2 = get_main_cell_pops(target_batch[:, 6:], 13)
            cluster_dist, correspondence_arr = average_cluster_distance(cluster_centers1, cluster_centers2)
            file.write(f"Average Cluster Distance for {batch_name}: {cluster_dist}\n")

    print(f"Metrics summary saved to {file_name}")

########## K MEANS CLUSTERING ############
def find_optimal_k(data, max_k):
    """
    Find the optimal k for k-means clustering using silhouette score and plot the results.
    
    Parameters:
    data (numpy.ndarray): 2D array of data points
    max_k (int): Maximum number of clusters to try
    
    Returns:
    int: Optimal number of clusters
    """
    silhouette_scores = []
    k_values = range(2, max_k + 1)
    
    # Try k from 2 to max_k
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # Find the index of the maximum silhouette score
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
    
    # Plot the silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
    plt.legend()
    plt.show()
    
    return optimal_k


def cytofBatchAdjust(ref_batch, target_batches):
    # For each clolumn, find the mean and variance of that column in the reference batch, and map the column in each target batch to the reference batch
    # For each column, find the mean and variance of that column in the reference batch
    ref_means = np.mean(ref_batch, axis=0)
    ref_vars = np.var(ref_batch, axis=0)

    # For each column in each target batch, find the mean and variance of that column
    for target_batch in target_batches:
        target_means = np.mean(target_batch, axis=0)
        target_vars = np.var(target_batch, axis=0)

        # For each column, calculate the scaling factor and offset to map the target batch to the reference batch
        for i in range(target_batch.shape[1]):
            target_batch[:, i] = (target_batch[:, i] - target_means[i]) * np.sqrt(ref_vars[i] / target_vars[i]) + ref_means[i]

    return target_batches





if __name__ == "__main__":
    # e2 = load_data("Plate 27902_N")
    # e4 = load_data("Plate 28528_N")
    # e7 = load_data("Plate 39630_N")

    # e3 = load_data("Plate 28332")
    # e5 = load_data("Plate 29178_N")


    # e1 = load_data("Plate 19635_CD8")

    # e6 = load_data("Plate 36841")

    
    # d = dict()
    # d["Plate 28528_N"] = e4
    # d["Plate 39630_N"] = e7
    # d["Plate 28332"] = e3
    # d["Plate 29178_N"] = e5
    # d["Plate 19635 _CD8"] = e1
    # # d["Plate 27902_N"] = e2
    # d["Plate 36841"] = e6

    # compute_all_metrics(e2, d)


    # b1 = load_data("Panel1")
    # b2 = load_data("Panel2")
    # b3 = load_data("Panel3")

    # d = dict()
    # d["Panel 2"] = b2
    # d["Panel 3"] = b3

    # compute_all_metrics(b1, d)

    # b1 = load_data("Panel1")
    # b2 = load_data("Panel1_sinkhorn_vr_clust13")

    # d = dict()
    # d["Panel 1 Sink"] = b2

    # compute_all_metrics(b1, d)
    b1 = load_data("Panel1")
    b2 = load_data("Panel2")
    b3 = load_data("Panel3")

    d = dict()
    d["Panel 2"] = b2
    d["Panel 3"] = b3

    compute_all_metrics(b1, d)


    # b2 = load_data("Panel1_sink_final")
    # find_optimal_k(b2, 15)