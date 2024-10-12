import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def plot_cluster_centers_pca(data_list, K, dataset_names=None):
    """
    Runs KMeans clustering on each dataset in data_list with K clusters.
    Performs PCA on the cluster centers of the first dataset.
    Projects all cluster centers onto the PCA components.
    Plots and saves the projected cluster centers for all datasets.

    Parameters:
        data_list (list of numpy arrays): List of datasets to cluster and project.
        K (int): Number of clusters for KMeans.
        dataset_names (list of str, optional): Names of the datasets for labeling. Defaults to Dataset 1, Dataset 2, etc.
    """
    num_datasets = len(data_list)
    
    if num_datasets == 0:
        print("Error: data_list is empty.")
        return

    # Ensure all datasets have the same number of features
    num_features = data_list[0].shape[1]
    for idx, data in enumerate(data_list):
        if data.shape[1] != num_features:
            print(f"Error: Dataset {idx+1} does not have the same number of features as the first dataset.")
            return

    # If dataset_names is not provided, create default names
    if dataset_names is None:
        dataset_names = [f'Dataset {i+1}' for i in range(num_datasets)]

    # Step 1: Run KMeans clustering on each dataset to get cluster centers
    cluster_centers_list = []
    for idx, data in enumerate(data_list):
        print(f"Running KMeans for {dataset_names[idx]}...")
        kmeans = KMeans(n_clusters=K, random_state=0)
        kmeans.fit(data)
        cluster_centers = kmeans.cluster_centers_
        cluster_centers_list.append(cluster_centers)
        print(f"KMeans for {dataset_names[idx]} completed.")

    # Step 2: Perform PCA on the cluster centers of the first dataset
    print(f"Performing PCA on cluster centers of {dataset_names[0]}...")
    pca = PCA(n_components=2)
    cluster_centers_first = cluster_centers_list[0]
    pca.fit(cluster_centers_first)
    print("PCA fitting completed.")

    # Step 3: Project all cluster centers onto the PCA components
    projected_centers_list = []
    for idx, cluster_centers in enumerate(cluster_centers_list):
        projected_centers = pca.transform(cluster_centers)
        projected_centers_list.append(projected_centers)

    # Step 4: Plot the projected cluster centers
    plt.figure(figsize=(10,8))
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h']  # markers for up to 10 datasets
    colors = plt.cm.get_cmap('tab10', num_datasets)
    for idx, projected_centers in enumerate(projected_centers_list):
        plt.scatter(projected_centers[:,0], projected_centers[:,1], 
                    label=dataset_names[idx],
                    marker=markers[idx % len(markers)],
                    color=colors(idx))
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Projected Cluster Centers of Datasets')
    plt.legend()
    plt.tight_layout()
    plt.savefig('cluster_centers_pca.png')
    plt.close()
    print("Plot saved as 'cluster_centers_pca.png'.")
