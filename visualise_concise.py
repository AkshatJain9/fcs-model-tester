import re

# File parsing function
def parse_results(file_content):
    # Initialize dictionary to store results
    parsed_data = {
        'batches': [],
        'mse_difference_means': {},
        'mse_difference_stds': {},
        'mean_1d_tvd': {},
        'mean_2d_tvd': {},
        'average_cluster_distance': {}
    }

    # Extract Batches
    batches_section = re.search(r'Batches:\n(.*?)\n\n', file_content, re.DOTALL)
    if batches_section:
        parsed_data['batches'] = batches_section.group(1).splitlines()

    # Extract MSE Difference in Means
    mse_diff_means_section = re.search(r'MSE Difference in Means:(.*?)\n\n', file_content, re.DOTALL)
    if mse_diff_means_section:
        mse_diff_means = re.findall(r'MSE Difference for ([^:]+): ([0-9.e+-]+)', mse_diff_means_section.group(1))
        parsed_data['mse_difference_means'] = {panel.strip(): float(mse) for panel, mse in mse_diff_means}

    # Extract MSE Difference in Standard Deviations
    mse_diff_stds_section = re.search(r'MSE Difference in Standard Deviations:(.*?)\n\n', file_content, re.DOTALL)
    if mse_diff_stds_section:
        mse_diff_stds = re.findall(r'MSE Difference for ([^:]+): ([0-9.e+-]+)', mse_diff_stds_section.group(1))
        parsed_data['mse_difference_stds'] = {panel.strip(): float(mse) for panel, mse in mse_diff_stds}

    # Extract Mean 1D TVD
    tvd_1d_section = re.search(r'1D TVD for each feature:(.*?)\n\n', file_content, re.DOTALL)
    if tvd_1d_section:
        mean_1d_tvd = re.findall(r'Mean 1D TVD for ([^:]+):\s*([0-9.e+-]+)', tvd_1d_section.group(1))
        parsed_data['mean_1d_tvd'] = {panel.strip(): float(mean) for panel, mean in mean_1d_tvd}

    # Extract Mean 2D TVD
    tvd_2d_section = re.search(r'2D TVD for each feature pair:(.*?)\n\n', file_content, re.DOTALL)
    if tvd_2d_section:
        mean_2d_tvd = re.findall(r'Mean 2D TVD for ([^:]+):\s*([0-9.e+-]+)', tvd_2d_section.group(1))
        parsed_data['mean_2d_tvd'] = {panel.strip(): float(mean) for panel, mean in mean_2d_tvd}

    # Extract Average Cluster Distance
    avg_cluster_dist_section = re.search(r'Average Cluster Distance:(.*?)(?:\n\n|$)', file_content, re.DOTALL)
    if avg_cluster_dist_section:
        avg_cluster_dist = re.findall(r'Average Cluster Distance for ([^:]+): ([0-9.e+-]+)', avg_cluster_dist_section.group(1))
        parsed_data['average_cluster_distance'] = {panel.strip(): float(dist) for panel, dist in avg_cluster_dist}

    return parsed_data

# Function to write concise tables side by side
def write_latex_summary_tables_side_by_side(results, filename):
    batches = results['batches']
    mse_difference_means = results['mse_difference_means']
    mse_difference_stds = results['mse_difference_stds']
    mean_1d_tvd = results['mean_1d_tvd']
    mean_2d_tvd = results['mean_2d_tvd']
    average_cluster_distance = results['average_cluster_distance']

    def format_value(value):
        """Formats value in scientific notation if it's below 10^-5, otherwise keeps it in float format."""
        if abs(value) < 1e-3:
            return f"{value:.2e}"
        else:
            return f"{value:.8f}"

    with open(filename, 'w') as f:
        # Begin the table environment
        f.write(r"\begin{table*}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        
        # First minipage for MSE differences
        f.write(r"\begin{minipage}[t]{0.48\linewidth}" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\begin{tabular}{|l|" + "c|" * len(batches) + "}\n")
        f.write(r"\hline" + "\n")
        
        # Write the table header
        header = " Metric & " + " & ".join(batches) + r" \\\hline" + "\n"
        f.write(header)
        
        # MSE Difference in Means
        row = "MSE Difference in Means"
        for batch in batches:
            value = mse_difference_means.get(batch, 0)
            row += f" & {format_value(value)}"
        row += r" \\" + "\n"
        f.write(row)
        
        # MSE Difference in Stds
        row = "MSE Difference in Stds"
        for batch in batches:
            value = mse_difference_stds.get(batch, 0)
            row += f" & {format_value(value)}"
        row += r" \\" + "\n"
        f.write(row)
        
        # End the table
        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\caption{MSE Differences across Batches}" + "\n")
        f.write(r"\label{tab:mse_differences}" + "\n")
        f.write(r"\end{minipage}" + "\n")
        
        f.write(r"\hfill" + "\n")  # Horizontal fill to separate the two tables
        
        # Second minipage for TVD and Cluster Distance
        f.write(r"\begin{minipage}[t]{0.48\linewidth}" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\begin{tabular}{|l|" + "c|" * len(batches) + "}\n")
        f.write(r"\hline" + "\n")
        
        # Write the table header
        f.write(header)
        
        # Mean 1D TVD
        row = "Mean 1D TVD"
        for batch in batches:
            value = mean_1d_tvd.get(batch, 0)
            row += f" & {format_value(value)}"
        row += r" \\" + "\n"
        f.write(row)
        
        # Mean 2D TVD
        row = "Mean 2D TVD"
        for batch in batches:
            value = mean_2d_tvd.get(batch, 0)
            row += f" & {format_value(value)}"
        row += r" \\" + "\n"
        f.write(row)
        
        # Average Cluster Distance
        row = "Average Cluster Distance"
        for batch in batches:
            value = average_cluster_distance.get(batch, 0)
            row += f" & {format_value(value)}"
        row += r" \\" + "\n"
        f.write(row)
        
        # End the table
        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\caption{TVD and Cluster Distance across Batches}" + "\n")
        f.write(r"\label{tab:tvd_cluster}" + "\n")
        f.write(r"\end{minipage}" + "\n")
        
        # End the table environment
        f.write(r"\end{table*}" + "\n")

# Example usage
if __name__ == "__main__":
    batch = "Synthetic"
    method = "rawdata"

    dir_path = f"{batch}/{method}"
    results_path = f"{dir_path}/results.txt"
    latex_path = f"{dir_path}/latex.txt"

    # Read the contents of the file (replace 'your_file.txt' with the actual file path)
    with open(results_path, 'r') as f:
        file_content = f.read()

    # Parse the content
    results = parse_results(file_content)

    # Write the concise tables side by side
    write_latex_summary_tables_side_by_side(results, latex_path)
