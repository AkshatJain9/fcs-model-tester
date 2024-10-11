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

# Function to write a single summary table
def write_latex_summary_table(results, filename):
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

        # Begin the tabular environment
        num_columns = len(batches) + 1  # +1 for the Metric column
        f.write(r"\begin{tabular}{|" + "l|" + "c|" * len(batches) + "}\n")
        f.write(r"\hline" + "\n")

        # Write the table header
        header = " Metric & " + " & ".join(batches) + r" \\\hline" + "\n"
        f.write(header)

        # Metrics to include
        metrics = [
            ("MSE Difference in Means", mse_difference_means),
            ("MSE Difference in Stds", mse_difference_stds),
            ("Mean 1D TVD", mean_1d_tvd),
            ("Mean 2D TVD", mean_2d_tvd),
            ("Average Cluster Distance", average_cluster_distance)
        ]

        # Write each metric row
        for metric_name, metric_data in metrics:
            row = metric_name
            for batch in batches:
                value = metric_data.get(batch, 0)
                row += f" & {format_value(value)}"
            row += r" \\" + "\n"
            f.write(row)

        # End the table
        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\caption{Summary of Metrics across Batches}" + "\n")
        f.write(r"\label{tab:summary_metrics}" + "\n")
        f.write(r"\end{table*}" + "\n")

# Example usage
if __name__ == "__main__":
    batch = "ENU"
    method = "CytoRUV"

    dir_path = f"{batch}/{method}"
    results_path = f"{dir_path}/results_e2.txt"
    latex_path = f"{dir_path}/latex_concise2.txt"

    # Read the contents of the file (replace 'results.txt' with the actual file path)
    with open(results_path, 'r') as f:
        file_content = f.read()

    # Parse the content
    results = parse_results(file_content)

    # Write the concise single summary table
    write_latex_summary_table(results, latex_path)
