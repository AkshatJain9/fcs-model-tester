import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_all_enu(data):
    for i in range(6, data.shape[1]):
        for j in range(i + 1, data.shape[1]):
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
            idx = density.argsort()
            x, y, density = points[:, 0][idx], points[:, 1][idx], density[idx]
            
            # Create the scatter plot
            fig, ax = plt.subplots(figsize=(10, 10))
            scatter = ax.scatter(x, y, c=density, s=1, cmap='viridis')
            
            # Set plot titles and labels
            plt.title(f'Density Scatter Plot for Channels {i} vs {j}')
            plt.xlabel(f'Channel {i}')
            plt.ylabel(f'Channel {j}')
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Save the figure
            plt.savefig(f'channel_{i}_vs_{j}_density_scatter4.png', dpi=300, bbox_inches='tight')
            plt.close()


def plot_all_enu_together(data1, data2, data3, data4):
    data_list = [data1, data2, data3, data4]
    num_datasets = len(data_list)
    num_channels = data1.shape[1]  # Assuming all datasets have the same number of columns

    for i in range(6, num_channels):
        for j in range(i + 1, num_channels):
            # Create a figure with 2x2 subplots
            fig, axs = plt.subplots(2, 2, figsize=(20, 20))
            axs = axs.flatten()

            xmin, xmax, ymin, ymax = None, None, None, None

            for idx, data in enumerate(data_list):
                # Check if the data has enough columns
                if data.shape[1] <= max(i, j):
                    print(f"Data {idx+1} does not have enough columns for channels {i} and {j}.")
                    continue

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

                # For data1, store the axis limits
                if idx == 0:
                    xmin, xmax = x.min(), x.max()
                    ymin, ymax = y.min(), y.max()

                # Create the scatter plot in the appropriate subplot
                ax = axs[idx]
                scatter = ax.scatter(x, y, c=density, s=1, cmap='viridis')

                # Set xlim and ylim to match data1
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

                # Set plot titles and labels
                ax.set_title(f'Dataset {idx+1}: Channels {i} vs {j}')
                ax.set_xlabel(f'Channel {i}')
                ax.set_ylabel(f'Channel {j}')

                # Remove top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            # Adjust layout and add a main title
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.suptitle(f'Density Scatter Plots for Channels {i} vs {j}', fontsize=20)

            # Save the combined figure
            plt.savefig(f'zchannels_{i}_vs_{j}_density_scatter_combined2.png', dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == '__main__':
    data_ref = "Plate 27902_N"
    data_target = "Plate 19635_CD8"

    method_existing = "CytofBatchAdjust_27902_N"
    method_new = "AE_Spline_27902"

    raw_data_ref_path = f'./ENU/rawdata/{data_ref}.npy'
    raw_data_target_path = f'./ENU/rawdata/{data_target}.npy'
    target_processed_old = f'./ENU/{method_existing}/{data_target}.npy'
    target_processed_new = f'./ENU/{method_new}/{data_target}.npy'


    # data_path = './ENU/AE_Spline_27902/Plate 28332.npy'

    # data = np.load(data_path)
    # plot_all_enu(data)

    plot_all_enu_together(np.load(raw_data_ref_path), np.load(raw_data_target_path), np.load(target_processed_old), np.load(target_processed_new))