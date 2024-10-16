import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


scatter_channels = ['FSC-A', 'FSC-H', 'FSC-W', 'SSC-A', 'SSC-H', 'SSC-W']
fluro_channels_panel = ['CD8', 'B220', 'CD44', 'CD3', 'IgD', 'CD25', 'Ly6C', 'NK1.1', 'IgM / CD4', 'LD', 'CD19', 'CD62L']
fluro_channels_enu = ['IgM', 'B220', 'IgD', 'KLRG1', 'NK1.1', 'CD4', 'CD3', 'CD44', 'CD8']

def plot_twod(data_list, i, j, synth_batch=True, labels=None, filename=None):
    """
    Plots contour and scatter plots for the selected channel pairing (i, j) 
    for all datasets in data_list.

    Parameters:
        data_list (list): A list of datasets (numpy arrays) to be plotted.
        i (int): The index of the first channel to plot (x-axis).
        j (int): The index of the second channel to plot (y-axis).
    """
    if synth_batch:
        channel_list = scatter_channels + fluro_channels_panel
    else:
        channel_list = scatter_channels + fluro_channels_enu

    # Check if the datasets have at least i and j channels
    for idx, data in enumerate(data_list):
        if data.shape[1] <= max(i, j):
            print(f"Dataset {idx+1} does not have enough channels for indices {i} and {j}.")
            return

    # Use the first dataset for determining the density grid
    data1 = data_list[0]
    channel1_data1 = data1[:, i]
    channel2_data1 = data1[:, j]
    
    # Stack the channels to create a 2D array of points
    points_data1 = np.vstack([channel1_data1, channel2_data1]).T
    
    # If there are too many points, sample a subset
    if len(points_data1) > 20000:
        indices = np.random.choice(len(points_data1), 20000, replace=False)
        points_data1 = points_data1[indices]
    
    # Compute KDE on data1
    kde_data1 = gaussian_kde(points_data1.T)
    
    # Determine x and y ranges
    xmin, xmax = points_data1[:, 0].min(), points_data1[:, 0].max()
    ymin, ymax = points_data1[:, 1].min(), points_data1[:, 1].max()
    
    # Create a regular grid over the data1 range
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])
    
    # Evaluate KDE on grid
    density_grid = kde_data1(grid_coords).reshape(xx.shape)
    
    # Choose a single contour level to show the general shape
    density_values = density_grid.flatten()
    contour_level = np.percentile(density_values, 90)  # Adjust percentile as needed
    
    # Now, loop over datasets to create individual plots
    for idx, data in enumerate(data_list):
        # Create a figure for each dataset
        fig, ax = plt.subplots(figsize=(10, 10))

        # Extract the relevant channels
        channel1 = data[:, i]
        channel2 = data[:, j]
        
        # Stack the channels to create a 2D array of points
        points = np.vstack([channel1, channel2]).T
        
        # If there are too many points, sample a subset
        if len(points) > 20000:
            indices = np.random.choice(len(points), 20000, replace=False)
            points = points[indices]
        
        # Calculate the point density using Gaussian KDE
        kde = gaussian_kde(points.T)
        density = kde(points.T)
        
        # Sort the points by density so that the densest points are plotted last
        idx_density = density.argsort()
        x, y, density = points[:, 0][idx_density], points[:, 1][idx_density], density[idx_density]
        
        # Create the scatter plot
        scatter = ax.scatter(x, y, c=density, s=1, cmap='viridis')
        
        # Set xlim and ylim to match data1
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        
        # Overlay the contour plot from data1
        ax.contour(xx, yy, density_grid, levels=[contour_level], colors='red', linewidths=2)
        
        # Set plot titles and labels
        ax.set_xlabel(channel_list[i])
        ax.set_ylabel(channel_list[j])
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if (labels is not None):
            data_name = labels[idx]
        else:
            data_name = f'Dataset {idx+1}'
        # Set title for the dataset
        ax.set_title(data_name, fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure for this dataset with high resolution
        if filename is not None:
            plt.savefig(f'{filename}_{data_name}_{channel_list[i]}_{channel_list[j]}.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'{data_name}_{channel_list[i]}_{channel_list[j]}.png', dpi=300, bbox_inches='tight')
        plt.close()


                
if __name__ == '__main__':
    dataset = "ENU"
    directory = "AE_EmpBayes_27902"

    # batches = ["Panel1", "Panel3", "Panel3 uncorrected"]
    # batches = ["Plate 19635_CD8", "Plate 27902_N", "Plate 28332", "Plate 28528_N", "Plate 29178_N", "Plate 36841", "Plate 39630_N"]
    batches = ["Plate 27902_N", "Plate 28528_N", "Plate 28528_N uncorrected", "Plate 28528_N LL"]
    data_list = []
    names = []
    for batch in batches:
        if ("uncorrected" in batch):
            batch_name = batch.replace(" uncorrected", "")
            filename = f"{dataset}/rawdata/{batch_name}.npy"
        elif ("LL" in batch):
            batch_name = batch.replace(" LL", "")
            filename = f"{dataset}/AE_LL_CD8/{batch_name}.npy"
        else:
            filename = f"{dataset}/{directory}/{batch}.npy"
        data = np.load(filename)
        data_list.append(data)
        
        names.append(batch)

    plot_twod(data_list, 6, 7, synth_batch=False, labels=names, filename=directory)
