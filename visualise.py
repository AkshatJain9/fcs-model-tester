import re
import numpy as np
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import ast


scatter_channels = ['FSC-A', 'FSC-H', 'FSC-W', 'SSC-A', 'SSC-H', 'SSC-W']
fluro_channels_panel = ['CD8', 'B220', 'CD44', 'CD3', 'IgD', 'CD25', 'Ly6C', 'NK1.1', 'IgM / CD4', 'LD', 'CD19', 'CD62L']
fluro_channels_enu = ['IgM', 'B2220', 'IgD', 'KLRG1', 'NK1.1', 'CD4', 'CD3', 'CD44', 'CD8']

# File parsing function
def parse_results(file_content):
    # Initialize dictionary to store results
    parsed_data = {
        'batches': [],
        'mean_summaries': {},
        'mse_difference_means': {},
        'std_summaries': {},
        'mse_difference_stds': {},
        '1d_tvd': {},
        'mean_1d_tvd': {},
        '2d_tvd': {},
        'mean_2d_tvd': {},
        'average_cluster_distance': {}
    }

    # Helper function to fix arrays without commas
    def fix_array_syntax(array_str):
        # Remove newlines and ensure proper spacing and commas
        array_str = array_str.replace('\n', '')
        array_str = re.sub(r'(\d)\s+(\d)', r'\1, \2', array_str)  # Add missing commas between numbers
        # Ensure correct closure of brackets (remove any extra spaces around the brackets)
        array_str = re.sub(r'\s*(\])\s*', r'\1', array_str)
        return array_str

    # Extract Batches
    batches_section = re.search(r'Batches:\n(.*?)\n\n', file_content, re.DOTALL)
    if batches_section:
        parsed_data['batches'] = batches_section.group(1).splitlines()

    # Extract Mean Summaries
    mean_summary_section = re.search(r'Mean Summaries:(.*?)\n\n', file_content, re.DOTALL)
    if mean_summary_section:
        mean_summary_section = mean_summary_section.group(1).split("\nMean Summary")
        for line in mean_summary_section:
            match = re.search(r"(\w+\s?\w+):\s*(\[[^]]+\])", line)
            if match:
                key = match.group(1).strip()
                value_str = match.group(2)  # No need to fix array syntax
                # Extract all numbers, including those in scientific notation
                value_list = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', value_str)
                # Convert the list of strings to floats
                value = np.array([float(x) for x in value_list])
                parsed_data['mean_summaries'][key] = value

    # Extract MSE Difference in Means
    mse_diff_means_section = re.search(r'MSE Difference in Means:(.*?)\n\n', file_content, re.DOTALL)
    if mse_diff_means_section:
        mse_diff_means = re.findall(r'MSE Difference for ([^:]+): ([0-9.e+-]+)', mse_diff_means_section.group(1))
        parsed_data['mse_difference_means'] = {panel.strip(): float(mse) for panel, mse in mse_diff_means}

    # Extract Std Summaries
    std_summary_section = re.search(r'Std Summaries:(.*?)\n\n', file_content, re.DOTALL)
    if std_summary_section:
        std_summary_section = std_summary_section.group(1).split("\nStd Summary")
        for line in std_summary_section:
            match = re.search(r"(\w+\s?\w+):\s*(\[[^]]+\])", line)
            if match:
                key = match.group(1).strip()
                value_str = match.group(2)  # No need to fix array syntax
                # Extract all numbers, including those in scientific notation
                value_list = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', value_str)
                # Convert the list of strings to floats
                value = np.array([float(x) for x in value_list])
                parsed_data['std_summaries'][key] = value

    # Extract MSE Difference in Standard Deviations
    mse_diff_stds_section = re.search(r'MSE Difference in Standard Deviations:(.*?)\n\n', file_content, re.DOTALL)
    if mse_diff_stds_section:
        mse_diff_stds = re.findall(r'MSE Difference for ([^:]+): ([0-9.e+-]+)', mse_diff_stds_section.group(1))
        parsed_data['mse_difference_stds'] = {panel.strip(): float(mse) for panel, mse in mse_diff_stds}


    # Extract 1D TVD for each feature
    tvd_1d_section = re.search(r'1D TVD for each feature:(.*?)\n\n', file_content, re.DOTALL)
    if tvd_1d_section:
        tvd_1d_panels = re.findall(r'1D TVD for ([^:]+):\n(\[[^\]]+\])', tvd_1d_section.group(1))
        mean_1d_tvd = re.findall(r'Mean 1D TVD for ([^:]+):\s*([0-9.e+-]+)', tvd_1d_section.group(1))
        parsed_data['1d_tvd'] = {panel.strip(): np.array(eval(fix_array_syntax(values))) for panel, values in tvd_1d_panels}
        parsed_data['mean_1d_tvd'] = {panel.strip(): float(mean) for panel, mean in mean_1d_tvd}

    # Extract 2D TVD for each feature pair
    tvd_2d_section = re.search(r'2D TVD for each feature pair:(.*?)\n\n', file_content, re.DOTALL)
    if tvd_2d_section:
        # Adjust the regex to capture the entire array, including multiline arrays
        tvd_2d_panels = re.findall(
            r'^2D TVD for ([^:]+):\n(.*?)(?=^Mean 2D TVD|^2D TVD for|^-------------------------)',
            tvd_2d_section.group(1),
            re.DOTALL | re.MULTILINE
        )

        mean_2d_tvd = re.findall(r'Mean 2D TVD for ([^:]+):\s*([0-9.e+-]+)', tvd_2d_section.group(1))

        # Handle multiline 2D arrays
        for panel, values in tvd_2d_panels:
            panel = panel.strip()
            if panel in parsed_data['2d_tvd']:
                continue
            # Remove newlines within arrays and ensure correct syntax
            values_fixed = fix_array_syntax(values)
            parsed_data['2d_tvd'][panel] = np.array(eval(values_fixed))

        parsed_data['mean_2d_tvd'] = {panel.strip(): float(mean) for panel, mean in mean_2d_tvd}

    # Extract Average Cluster Distance
    avg_cluster_dist_section = re.search(r'Average Cluster Distance:(.*?)(?:\n\n|$)', file_content, re.DOTALL)
    if avg_cluster_dist_section:
        avg_cluster_dist = re.findall(r'Average Cluster Distance for ([^:]+): ([0-9.e+-]+)', avg_cluster_dist_section.group(1))
        parsed_data['average_cluster_distance'] = {panel.strip(): float(dist) for panel, dist in avg_cluster_dist}

    return parsed_data



# New function to write mean and std tables side by side
def write_latex_mean_std_tables_side_by_side(results, filename):
    batches = results['batches']
    mean_summaries = results['mean_summaries']
    std_summaries = results['std_summaries']
    mse_difference_means = results['mse_difference_means']
    mse_difference_stds = results['mse_difference_stds']

    mse_differences_means = [(mean_summaries[batch] - mean_summaries['Reference Dataset']) ** 2 for batch in batches]
    mse_differences_stds = [(std_summaries[batch] - std_summaries['Reference Dataset']) ** 2 for batch in batches]

    def format_value(value):
        """Formats value in scientific notation if it's below 10^-5, otherwise keeps it in float format."""
        if abs(value) < 1e-3:
            return f"{value:.2e}"
        else:
            return f"{value:.8f}"

    with open(filename, 'w') as f:
        # Begin the table environment
        f.write(r"\begin{table*}[htbp]")  # Use table* for a full-width table
        f.write(r"\centering")

        # Begin first minipage for Mean MSE table
        f.write(r"\begin{minipage}[t]{0.48\linewidth}")
        f.write(r"\centering")
        f.write(r"\label{tab:mse_means}")
        f.write(r"\begin{tabular}{|l|" + "c|" * len(batches) + "}\n")
        f.write(r"\hline")

        # Write the table header for Mean MSE
        header = " Channel & " + " & ".join(batches) + r" \\\hline" + "\n"
        f.write(header)

        # Combine channels
        all_channels = scatter_channels + fluro_channels_panel

        # Write Mean MSE table rows
        for i, channel in enumerate(all_channels):
            row = channel
            for j, batch in enumerate(batches):
                mse_value = mse_differences_means[j][i]
                row += f" & {format_value(mse_value)}"
            row += r" \\" + "\n"
            f.write(row)

        # End Mean MSE table
        f.write(r"\hline ")
        f.write(r"Mean & " + " & ".join([format_value(mse) for mse in mse_difference_means.values()]) + r" \\" + "\n")
        f.write(r"\hline")
        f.write(r"\end{tabular}")
        f.write(r"\end{minipage}")
        f.write(r"\hfill")  # Horizontal fill to separate the two tables

        # Begin second minipage for Std MSE table
        f.write(r"\begin{minipage}[t]{0.48\linewidth}")
        f.write(r"\centering")
        f.write(r"\caption{MSE Differences for Means (Left) and Standard Deviations (Right) across Batches}")
        f.write(r"\label{tab:mse_stds}")
        f.write(r"\begin{tabular}{|l|" + "c|" * len(batches) + "}\n")
        f.write(r"\hline")

        # Write the table header for Std MSE
        f.write(header)  # Header is the same as before

        # Write Std MSE table rows
        for i, channel in enumerate(all_channels):
            row = channel
            for j, batch in enumerate(batches):
                mse_value = mse_differences_stds[j][i]
                row += f" & {format_value(mse_value)}"
            row += r" \\" + "\n"
            f.write(row)

        # End Std MSE table
        f.write(r"\hline ")
        f.write(r"Mean & " + " & ".join([format_value(mse) for mse in mse_difference_stds.values()]) + r" \\" + "\n")
        f.write(r"\hline")
        f.write(r"\end{tabular}")
        f.write(r"\end{minipage}")

        # End the table environment
        f.write(r"\end{table*}")

def write_latex_1d_tvd_table(results, filename):
    batches = results['batches']
    tvd_1d = results['1d_tvd']
    mean_1d_tvd = results['mean_1d_tvd']

    def format_tvd_value(tvd_value):
        """Formats TVD value in scientific notation if it's below 10^-5, otherwise keeps it in float format."""
        if abs(tvd_value) < 1e-4:
            return f"{tvd_value:.2e}"
        else:
            return f"{tvd_value:.8f}"

    with open(filename, 'a') as f:  # 'a' mode to append to the existing file
        # Begin the new table for 1D TVD
        f.write("\n")  # Add a newline for separation between tables
        f.write(r"\begin{table}[H]")
        f.write(r"\centering")
        f.write(r"\begin{tabular}{|l|" + "c|" * len(batches) + "}\n")
        f.write(r"\hline")

        # Write the table header
        header = " Channel & " + " & ".join(batches) + r" \\\hline" + "\n"
        f.write(header)

        # Combine scatter channels and fluoro channels into a single list
        all_channels = scatter_channels + fluro_channels_panel

        # Write each row: channel name + 1D TVD values for each batch
        for i, channel in enumerate(all_channels):
            row = channel
            for j, batch in enumerate(batches):
                tvd_value = tvd_1d[batch][i]
                # Use the format_tvd_value function to format the number
                row += f" & {format_tvd_value(tvd_value)}"
            row += r" \\" + "\n"
            f.write(row)

        # End the table
        f.write(r"\hline ")
        # Write the mean 1D TVD values across scatter and fluoro channels
        f.write(r"Mean 1D TVD & " + " & ".join([format_tvd_value(tvd) for tvd in mean_1d_tvd.values()]) + r" \\\hline")
        f.write(r"\end{tabular}")
        f.write(r"\caption{1D TVD for Scatter and Fluoro Channels across Batches}")
        f.write(r"\label{tab:1d_tvd}")
        f.write(r"\end{table}")

def write_latex_cluster_mse_table(results, filename):
    batches = results['batches']
    cluster_mse = results['average_cluster_distance']

    def format_mse_value(mse_value):
        """Formats MSE value in scientific notation if it's below 10^-5, otherwise keeps it in float format."""
        if abs(mse_value) < 1e-4:
            return f"{mse_value:.2e}"
        else:
            return f"{mse_value:.8f}"

    with open(filename, 'a') as f:  # 'a' mode to append to the existing file
        # Begin the new table for Cluster MSE Differences
        f.write("\n")  # Add a newline for separation between tables
        f.write(r"\begin{table}[htbp]")
        f.write(r"\centering")
        f.write(r"\begin{tabular}{|l|" + "c|" * len(batches) + "}\n")
        f.write(r"\hline")

        # Write the table header
        header = " Batch & " + " & ".join(batches) + r" \\\hline" + "\n"
        f.write(header)

        # Write the single row with the MSE values for each batch
        row = "Cluster MSE"
        for batch in batches:
            mse_value = cluster_mse.get(batch, 0)  # cluster_mse[batch] should be a float
            row += f" & {format_mse_value(mse_value)}"
        row += r" \\" + "\n"
        f.write(row)

        # End the table
        f.write(r"\hline")
        f.write(r"\end{tabular}")
        f.write(r"\caption{Average Cluster MSE Differences across Batches}")
        f.write(r"\label{tab:cluster_mse}")
        f.write(r"\end{table}")

def create_2d_tvd_heatmaps(results, fluro_channels_panel, dir_path, latex_filename, batch, tech):
    
    batches = results['batches']
    all_channels = fluro_channels_panel

    for batch in batches:
        if batch == 'Reference Dataset':
            continue  # Skip reference dataset if necessary

        tvd_matrix = results['2d_tvd'][batch]  # This should be a 2D numpy array

        # Get indices of fluro_channels_panel in all_channels
        indices = [all_channels.index(ch) for ch in fluro_channels_panel]

        # Extract the submatrix corresponding to fluro_channels_panel x fluro_channels_panel
        tvd_submatrix = tvd_matrix[np.ix_(indices, indices)]

        num_channels = len(fluro_channels_panel)

        # Calculate figure size based on number of channels
        cell_size = 1.0  # Increase this value to make cells larger
        fig_width = cell_size * num_channels
        fig_height = cell_size * num_channels
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Remove colors by setting a monochrome colormap and uniform data
        im = ax.imshow(np.zeros_like(tvd_submatrix), cmap='gray', vmin=0, vmax=1)

        # Place x-axis labels on top
        ax.xaxis.tick_top()
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, labelsize=14)

        # Set ticks and labels with increased font size
        ax.set_xticks(np.arange(num_channels))
        ax.set_yticks(np.arange(num_channels))
        ax.set_xticklabels(fluro_channels_panel, rotation=90, fontsize=14)
        ax.set_yticklabels(fluro_channels_panel, fontsize=14)

        # Remove grid lines and frame
        ax.grid(False)
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        # Annotate each cell with its corresponding value, with increased font size
        for i in range(num_channels):
            for j in range(num_channels):
                value = tvd_submatrix[i, j]
                ax.text(j, i, f"{value:.2e}", ha='center', va='center', color='white', fontsize=12)

        # Draw gridlines to separate cells
        ax.set_xticks(np.arange(-0.5, num_channels, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_channels, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        plt.tight_layout()
        image_filename = f'{batch}_{tech}_2d_tvd_{batch}.png'
        plt.savefig(f'{dir_path}/{image_filename}', bbox_inches='tight', dpi=300)  # Higher DPI for better resolution
        plt.close()

        # Append LaTeX code to include this image
        with open(latex_filename, 'a') as f:
            f.write('\n')  # Add a newline
            f.write(r'\begin{figure}[H]')
            f.write(r'\centering')
            f.write(f'\\includegraphics[width=0.7\\linewidth]{{{f"figures/results/{image_filename}"}}}')
            f.write(f'\\caption{{2D TVD Matrix for {batch}}}')
            f.write(f'\\label{{fig:{batch}_{tech}_2d_tvd_{batch}}}')
            f.write(r'\end{figure}')


# Example usage
if __name__ == "__main__":
    batch = "Synthetic"
    method = "Spline_AE"

    dir_path = f"{batch}/{method}"
    results_path = f"{dir_path}/results.txt"
    latex_path = f"{dir_path}/latex.txt"

    # Read the contents of the file (replace 'your_file.txt' with the actual file path)
    with open(results_path, 'r') as f:
        file_content = f.read()

    # Parse the content
    results = parse_results(file_content)

    write_latex_mean_std_tables_side_by_side(results, latex_path)
    write_latex_cluster_mse_table(results, latex_path)
    write_latex_1d_tvd_table(results, latex_path)
    create_2d_tvd_heatmaps(results, fluro_channels_panel, dir_path, latex_path, batch, method)