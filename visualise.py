import re
import numpy as np

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
                value_str = fix_array_syntax(match.group(2))
                value = np.array(eval(value_str))
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
                value_str = fix_array_syntax(match.group(2))
                value = np.array(eval(value_str))
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


# Example usage
if __name__ == "__main__":
    batch = "Synthetic"
    method = "rawdata"

    path = f"{batch}/{method}/results.txt"

    # Read the contents of the file (replace 'your_file.txt' with the actual file path)
    with open(path, 'r') as f:
        file_content = f.read()

    # Parse the content
    results = parse_results(file_content)

    # Now the results variable contains the parsed data
    print(results)
