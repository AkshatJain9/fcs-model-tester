import os
import flowkit as fk
import numpy as np
import matplotlib.pyplot as plt


dataset = "Synthetic"

data = "autoencoder_cytonorm"
data = data + "/"

panel_1 = data + "Panel1"
panel_2 = data + "Panel2"
panel_3 = data + "Panel3"

print_mean = False
print_cov = False
print_emd = False
plot_all_histograms_bool = True
print_emd_2d = False

# Utils
scatter_channels = ['FSC-A', 'FSC-H', 'FSC-W', 'SSC-A', 'SSC-H', 'SSC-W']
fluro_channels = ['BUV 395-A', 'BUV737-A', 'Pacific Blue-A', 'FITC-A', 'PerCP-Cy5-5-A', 'PE-A', 'PE-Cy7-A', 'APC-A', 'Alexa Fluor 700-A', 'APC-Cy7-A','BV510-A','BV605-A']
all_channels = scatter_channels + fluro_channels
transform = fk.transforms.LogicleTransform('logicle', param_t=262144, param_w=0.5, param_m=4.5, param_a=0)

def print_array(arr):
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    print(np.array2string(arr, separator=', ', formatter={'float_kind':lambda x: f"{x:.2f}"}))


# Data retrieval
def get_np_array_from_sample(sample: fk.Sample, subsample: bool) -> np.ndarray:
    sample_np = []
    for ch in scatter_channels:
        sample_np.append(sample.get_channel_events(sample.get_channel_index(ch), source='raw', subsample=subsample) / 262144)
    
    for ch in fluro_channels:
        sample_np.append(sample.get_channel_events(sample.get_channel_index(ch), source='xform', subsample=subsample))
    
    sample_np = np.array(sample_np)
    sample_np = np.transpose(sample_np)
    return sample_np

def get_events(panel):
    if (os.path.exists(panel + ".npy")):
        return np.load(panel + ".npy")
    all_samples = []
    for root, _, files in os.walk(panel):
        event_count = 0
        for file in files:
            file_path = "./" + panel + "/" + os.path.relpath(os.path.join(root, file), panel)
            sample = fk.Sample(file_path)
            sample.apply_compensation("./281122_Spillover_Matrix.csv")
            sample.apply_transform(transform)
            all_samples.append(get_np_array_from_sample(sample, False))
            event_count += sample.event_count
        print(f"Total events in {root}: {event_count}")

    res = np.vstack(all_samples)
    np.save(panel + ".npy", res)
    return res


panel_1_np = get_events(panel_1)
panel_2_np = get_events(panel_2)
# panel_3_np = get_events(panel_3)

# Mean Summary
def print_mean_summary(panel_np):
    mean_vector = np.mean(panel_np, axis=0)
    return mean_vector

if print_mean:
    p1_mean = print_mean_summary(panel_1_np)
    p2_mean = print_mean_summary(panel_2_np)
    p3_mean = print_mean_summary(panel_3_np)


    print("Mean squared difference between panel 1 and panel 2: ", np.mean((p1_mean - p2_mean) ** 2))
    print("Mean squared difference between panel 1 and panel 3: ", np.mean((p1_mean - p3_mean) ** 2))


# Covariance Summary
def print_cov_summary(panel_np):
    cov_matrix = np.cov(panel_np, rowvar=False)
    return cov_matrix

if print_cov:
    p1_cov = print_cov_summary(panel_1_np)
    p2_cov = print_cov_summary(panel_2_np)
    p3_cov = print_cov_summary(panel_3_np)

    print("Mean squared difference (covariance) between panel 1 and panel 2: ", np.mean((p1_cov - p2_cov) ** 2))
    print("Mean squared difference (covariance) between panel 1 and panel 3: ", np.mean((p1_cov - p3_cov) ** 2))



# Earth Mover's Distance
def generate_histogram(panel_np, index):
    range = (0, 1)

    hist, _ = np.histogram(panel_np[:, index], bins=100, range=range)
    hist = hist / np.sum(hist)
    return hist

def compute_emd(hist1, hist2):
    return np.sum(np.abs(hist1 - hist2))


def compute_all_emd(panel_np1, panel_np2):
    emds = []
    for i in range(6, len(all_channels)):
        hist1 = generate_histogram(panel_np1, i)
        hist2 = generate_histogram(panel_np2, i)
        emds.append(compute_emd(hist1, hist2))
    return emds

if print_emd:
    emd_12 = compute_all_emd(panel_1_np, panel_2_np)
    # emd_13 = compute_all_emd(panel_1_np, panel_3_np)

    print("Earth Mover's Distance between panel 1 and panel 2: ", np.mean(emd_12))
    # print("Earth Mover's Distance between panel 1 and panel 3: ", np.mean(emd_13))


# Visualise the histograms
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

if plot_all_histograms_bool:
    plot_all_histograms(panel_1_np, panel_2_np)
    plot_all_histograms(panel_1_np, panel_3_np)


# 2D Histogram Stuff
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

if print_emd_2d:
    scatter_emd_12, fluro_emd_12 = compute_all_emd_2d(panel_1_np, panel_2_np)
    scatter_emd_13, fluro_emd_13 = compute_all_emd_2d(panel_1_np, panel_3_np)

    print("[2D] Earth Mover's Distance between panel 1 and panel 2 (scatter): ", np.mean(scatter_emd_12))
    print("[2D] Earth Mover's Distance between panel 1 and panel 3 (scatter): ", np.mean(scatter_emd_13))

    print("[2D] Earth Mover's Distance between panel 1 and panel 2 (fluro): ", np.mean(fluro_emd_12))
    print("[2D] Earth Mover's Distance between panel 1 and panel 3 (fluro): ", np.mean(fluro_emd_13))



