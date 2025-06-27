import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, ttest_ind, mannwhitneyu
import matplotlib.image as mpimg
import sys # Import sys for stdout redirection

# --- Configuration ---
# Base directory where your hunter_prey_v5.py script saves its results
BASE_RESULTS_DIR = "simulation_results_dynamic"

# Names of the scenarios as they appear in your simulation script's output directories
SCENARIOS = ["no_dynamic", "with_dynamic"]

# Display names for the scenarios, used in plots and output
SCENARIO_DISPLAY_NAMES = {
    SCENARIOS[0]: "no_dynamic",
    SCENARIOS[1]: "with_dynamic",
}

# Variables from the _metrics.json files that we want to analyze statistically
# The key is a descriptive name for the variable, the value is the key in the JSON file.
VARIABLES_TO_ANALYZE = {
    "rabbit_extinction_time": "extinction_tick_rabbits",
    "fox_extinction_time": "extinction_tick_foxes",
    "peak_rabbits": "peak_rabbits",
    "peak_foxes": "peak_foxes",
    # New variables derived from population time series - these will be calculated per run
    "average_rabbit_population_over_time": None, # Set to None as it's not a direct JSON key
    "average_fox_population_over_time": None,       # Set to None as it's not a direct JSON key
}

# The maximum simulation duration defined in LVConfig (hunter_prey_v5.py),
# used to treat "N/A" extinction times (meaning the species did not go extinct)
# as reaching the maximum possible simulation duration.
MAX_SIMULATION_DURATION = 2000

# --- Helper Functions ---

def sanitize_filename(name: str) -> str:
    """Converts a display name into a suitable filename component."""
    return name.lower().replace(' ', '_').replace('-', '_')

def load_scenario_data(scenario_name: str) -> dict:
    """
    Loads all relevant metrics from _metrics.json files and calculates
    average populations from _population_data.csv for a given scenario.
    Treats "N/A" extinction times as MAX_SIMULATION_DURATION.

    Args:
        scenario_name (str): The name of the scenario directory (e.g., "energy_free_grid_search").

    Returns:
        dict: A dictionary where keys are descriptive variable names (from VARIABLES_TO_ANALYZE)
              and values are lists of collected data points for that variable.
    """
    scenario_dir = os.path.join(BASE_RESULTS_DIR, scenario_name)
    
    # Initialize data dictionary with lists for each analysis variable
    data = {var_name: [] for var_name in VARIABLES_TO_ANALYZE.keys()}

    if not os.path.exists(scenario_dir):
        print(f"Warning: Scenario directory not found: '{scenario_dir}'. "
              f"Please ensure the simulation script has been run to generate results.")
        return data

    run_dirs = [d for d in os.listdir(scenario_dir) if os.path.isdir(os.path.join(scenario_dir, d)) and d.startswith('combo_')]
    
    if not run_dirs:
        print(f"Warning: No simulation run directories found in '{scenario_dir}'. "
              f"Statistical analysis requires multiple runs per scenario ( ideally 30-50+).")

    for run_dir in run_dirs:
        run_path = os.path.join(scenario_dir, run_dir)
        metrics_file = os.path.join(run_path, f"{run_dir}_metrics.json")
        population_data_file = os.path.join(run_path, f"{run_dir}_population_data.csv")
        
        # Load scalar metrics from JSON
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                # Iterate through VARIABLES_TO_ANALYZE to populate data
                for var_key, json_key in VARIABLES_TO_ANALYZE.items():
                    # Only process if it's a direct metric from JSON (json_key is not None)
                    if json_key is not None and json_key in metrics: 
                        value = metrics.get(json_key)
                        if value == "N/A":
                            data[var_key].append(MAX_SIMULATION_DURATION) 
                        elif value is not None:
                            data[var_key].append(value)
            except json.JSONDecodeError:
                print(f"Error: Could not parse JSON from {metrics_file}. Skipping scalar metrics for this run.")
        else:
            print(f"Warning: Metrics file not found for run: '{metrics_file}'. Skipping scalar metrics for this run.")

        # Calculate and load average populations from CSV
        if os.path.exists(population_data_file):
            try:
                df_pop = pd.read_csv(population_data_file)
                
                if not df_pop.empty:
                    if 'Rabbits' in df_pop.columns and "average_rabbit_population_over_time" in VARIABLES_TO_ANALYZE:
                        data["average_rabbit_population_over_time"].append(df_pop['Rabbits'].mean())
                    if 'Foxes' in df_pop.columns and "average_fox_population_over_time" in VARIABLES_TO_ANALYZE:
                        data["average_fox_population_over_time"].append(df_pop['Foxes'].mean())
                else:
                    print(f"Warning: Population data CSV '{population_data_file}' is empty. Appending zeros for averages.")
                    if "average_rabbit_population_over_time" in VARIABLES_TO_ANALYZE:
                        data["average_rabbit_population_over_time"].append(0)
                    if "average_fox_population_over_time" in VARIABLES_TO_ANALYZE:
                        data["average_fox_population_over_time"].append(0)
            except Exception as e:
                print(f"Error processing population data from {population_data_file}: {e}. Skipping population data for this run.")
        else:
            print(f"Warning: Population data file not found for run: '{population_data_file}'. Skipping population data for this run.")
            # Ensure consistency even if population data is missing
            if "average_rabbit_population_over_time" in VARIABLES_TO_ANALYZE:
                data["average_rabbit_population_over_time"].append(0)
            if "average_fox_population_over_time" in VARIABLES_TO_ANALYZE:
                data["average_fox_population_over_time"].append(0)
    
    # After processing all runs, ensure all lists in 'data' have the same length
    # This handles cases where some runs might be missing certain files.
    # The minimum length of populated lists determines how many valid data points we have for each variable.
    # We will pad or truncate lists to this minimum length to ensure consistent array creation.
    min_len = float('inf')
    for var_list in data.values():
        if var_list: # Only consider non-empty lists for min_len calculation
            min_len = min(min_len, len(var_list))
    
    if min_len == float('inf'): # No data collected at all
        return {var_name: [] for var_name in VARIABLES_TO_ANALYZE.keys()}

    # Truncate all lists to the minimum length
    for var_key in data:
        data[var_key] = data[var_key][:min_len]

    return data

def plot_histogram(data: np.ndarray, title: str, xlabel: str, output_dir: str, filename: str, plot_paths: list):
    """
    Generates and saves a histogram for the given data.
    Also appends the path of the saved plot to the plot_paths list.

    Args:
        data (np.ndarray): The NumPy array of numerical data points to plot.
        title (str): The title of the histogram.
        xlabel (str): The label for the x-axis.
        output_dir (str): The directory where the plot will be saved.
        filename (str): The name of the file to save the plot as (e.g., "rabbit_ext_time_hist.png").
        plot_paths (list): A list to append the path of the generated plot.
    """
    if data.size == 0:
        print(f"Cannot create histogram for '{title}': No data provided.")
        return

    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=10, edgecolor='black', alpha=0.7)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    full_path = os.path.join(output_dir, filename)
    plt.savefig(full_path)
    plt.close() # Close the plot to free memory
    print(f"Histogram saved: {full_path}")
    plot_paths.append(full_path)


def check_normality(data: np.ndarray, variable_name: str) -> bool:
    """
    Performs the Shapiro-Wilk test for normality on the given data.

    Args:
        data (np.ndarray): The NumPy array of numerical data points.
        variable_name (str): A descriptive name for the variable being tested.

    Returns:
        bool: True if the data appears normally distributed (p > 0.05), False otherwise.
    """
    # Shapiro-Wilk test requires at least 3 data points
    if len(data) < 3: 
        print(f"\nSkipping Shapiro-Wilk test for '{variable_name}': Not enough data points ({len(data)}).")
        print("  (Statistical analysis requires at least 3 data points for normality test and ideally 30-50+ for robust results).")
        return False 

    stat, p = shapiro(data)
    print(f"\n--- Normality Test for {variable_name} ---")
    print(f"Shapiro-Wilk Test Statistic: {stat:.4f}, p-value: {p:.4f}")
    if p > 0.05:
        print("Conclusion: Data appears to be normally distributed (p > 0.05). ✅")
        return True
    else:
        print("Conclusion: Data does NOT appear to be normally distributed (p <= 0.05). ❌")
        return False

def perform_statistical_test(data1: np.ndarray, data2: np.ndarray, var_name: str, overall_is_normal: bool, corrected_alpha: float):
    """
    Applies the correct statistical test (t-test or Mann-Whitney U) and interprets the results.
    Applies Bonferroni correction to the alpha level.

    Args:
        data1 (np.ndarray): Data from the first group (e.g., energy-free).
        data2 (np.ndarray): Data from the second group (e.g., energy-enabled).
        var_name (str): Descriptive name of the variable being compared.
        overall_is_normal (bool): True if both datasets are considered normally distributed.
        corrected_alpha (float): The alpha level after Bonferroni correction.
    
    Returns:
        dict: A dictionary containing the variable name, p-value, corrected alpha, 
              and the accepted hypothesis, or None if skipped due to insufficient data.
    """
    print(f"\n--- Statistical Test for {var_name} ---")
    
    if data1.size == 0 or data2.size == 0:
        print(f"Skipping statistical test for '{var_name}': Insufficient data from one or both scenarios.")
        return None
    
    # Formulate Hypotheses based on the example in the prompt
    print(f"Hypotheses for '{var_name}':")
    print(f"  H₀ (Null Hypothesis): There is no difference in {var_name} between {SCENARIO_DISPLAY_NAMES[SCENARIOS[1]].lower()} and {SCENARIO_DISPLAY_NAMES[SCENARIOS[0]].lower()} scenarios.")
    print(f"  H₁ (Alternative Hypothesis): {var_name} differs between the two scenarios.")
    print(f"  (Using Bonferroni corrected alpha = {corrected_alpha:.4f})")

    # Determine which test to run based on normality
    if overall_is_normal and len(data1) >= 2 and len(data2) >= 2: # t-test needs at least 2 data points for variance
        # Use Welch's t-test (equal_var=False) which does not assume equal variances, making it more robust.
        stat, p = ttest_ind(data1, data2, equal_var=False)
        test_type = "Independent Two-Sample T-test (Welch's)"
    else:
        # Use Mann-Whitney U test (non-parametric) for non-normal data or small sample sizes
        stat, p = mannwhitneyu(data1, data2)
        test_type = "Mann-Whitney U Test (Non-Parametric)"
    
    print(f"Test Type: {test_type}")
    print(f"Test Statistic: {stat:.4f}, p-value: {p:.4f}")

    result = {
        "variable_name": var_name,
        "p_value": p,
        "corrected_alpha": corrected_alpha,
        "accepted_hypothesis": ""
    }

    # Interpret p-value using the corrected alpha
    if p < corrected_alpha:
        print(f"Interpretation (p < {corrected_alpha:.4f}): Reject Null Hypothesis.")
        print(f"Conclusion: There is a statistically significant difference in {var_name} between the {SCENARIO_DISPLAY_NAMES[SCENARIOS[1]].lower()} and {SCENARIO_DISPLAY_NAMES[SCENARIOS[0]].lower()} scenarios. ✅")
        result["accepted_hypothesis"] = "H1 Accepted ✅"
    else:
        print(f"Interpretation (p >= {corrected_alpha:.4f}): Fail to Reject Null Hypothesis.")
        print(f"Conclusion: There is no statistically significant difference in {var_name} between the {SCENARIO_DISPLAY_NAMES[SCENARIOS[1]].lower()} and {SCENARIO_DISPLAY_NAMES[SCENARIOS[0]].lower()} scenarios. ❌")
        result["accepted_hypothesis"] = "H0 Accepted ❌"
    
    return result

def plot_scenario_comparison_bar_chart(data_free: np.ndarray, data_enabled: np.ndarray, 
                                       title: str, ylabel: str, output_dir: str, filename: str, plot_paths: list):
    """
    Generates and saves a bar chart comparing the means of two scenarios with standard deviation error bars.
    Also appends the path of the saved plot to the plot_paths list.

    Args:
        data_free (np.ndarray): Data from the energy-free scenario.
        data_enabled (np.ndarray): Data from the energy-enabled scenario.
        title (str): The title of the chart.
        ylabel (str): The label for the y-axis.
        output_dir (str): The directory where the plot will be saved.
        filename (str): The name of the file to save the plot as.
        plot_paths (list): A list to append the path of the generated plot.
    """
    if data_free.size == 0 and data_enabled.size == 0:
        print(f"Cannot create comparison bar chart for '{title}': No data provided for either scenario.")
        return

    means = []
    stds = []
    labels = [SCENARIO_DISPLAY_NAMES[SCENARIOS[0]], SCENARIO_DISPLAY_NAMES[SCENARIOS[1]]]
    colors = ['lightcoral', 'skyblue'] 

    # Only add data if the array is not empty
    if data_free.size > 0:
        means.append(np.mean(data_free))
        stds.append(np.std(data_free))
    else:
        means.append(0)
        stds.append(0) # or np.nan, depending on desired behavior for empty data

    if data_enabled.size > 0:
        means.append(np.mean(data_enabled))
        stds.append(np.std(data_enabled))
    else:
        means.append(0)
        stds.append(0) # or np.nan

    plt.figure(figsize=(8, 6))
    # Plot bars with error bars representing standard deviation
    bars = plt.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Scenario', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    full_path = os.path.join(output_dir, filename)
    plt.savefig(full_path)
    plt.close()
    print(f"Scenario comparison bar chart saved: {full_path}")
    plot_paths.append(full_path)

def combine_plots(plot_paths: list, output_dir: str, combined_filename: str, is_horizontal: bool = False):
    """
    Combines multiple individual plot images into a single, high-quality image,
    with fixed layouts and minimal margins.

    Args:
        plot_paths (list): A list of file paths to the individual PNG plot images.
        output_dir (str): The directory where the combined plot will be saved.
        combined_filename (str): The name of the file to save the combined plot as.
        is_horizontal (bool): If True, arranges plots in a horizontal layout (6x3);
                              otherwise, vertical layout (3x6).
    """
    if not plot_paths:
        print(f"No plots to combine for {combined_filename}. Skipping combined image generation.")
        return

    num_plots = len(plot_paths) # Should be 18 for current VARIABLES_TO_ANALYZE * 3 plot types

    # Define fixed grid dimensions
    if is_horizontal:
        ncols = 6
        nrows = 3
    else: # Vertical
        ncols = 3
        nrows = 6

    # Calculate overall figure size based on desired compactness for each subplot
    # These are empirical values to achieve compactness while maintaining readability.
    subplot_width_inches = 4.0 # Adjust as needed for clarity vs. compactness
    subplot_height_inches = 3.0 # Adjust as needed

    fig_width = subplot_width_inches * ncols
    fig_height = subplot_height_inches * nrows

    # Create the figure and subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), dpi=150)
    
    # Flatten the axes array for easy iteration, handling single plot case
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Iterate through plot paths and place images in subplots
    for i, plot_path in enumerate(plot_paths):
        if i < len(axes): # Ensure we don't exceed available subplots
            ax = axes[i]
            try:
                img = mpimg.imread(plot_path)
                ax.imshow(img)
                ax.axis('off') # Hide axes ticks and labels for image subplots
                
                # Extract and format title from filename
                base_name = os.path.basename(plot_path).replace('.png', '')
                
                # Dynamic replacement based on scenario display names
                title = base_name.replace(f'_histogram_{sanitize_filename(SCENARIO_DISPLAY_NAMES[SCENARIOS[0]])}', f'\n(Hist {SCENARIO_DISPLAY_NAMES[SCENARIOS[0]]})') \
                                  .replace(f'_histogram_{sanitize_filename(SCENARIO_DISPLAY_NAMES[SCENARIOS[1]])}', f'\n(Hist {SCENARIO_DISPLAY_NAMES[SCENARIOS[1]]})') \
                                  .replace('_comparison_bar_chart', '\n(Comparison)') \
                                  .replace('_', ' ').title()
                ax.set_title(title, fontsize=10, pad=5) # Smaller font, add padding
            except Exception as e:
                print(f"Error loading or displaying image {plot_path}: {e}")
                ax.text(0.5, 0.5, 'Image Load Error', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.axis('off')
    
    # Hide any unused subplots if num_plots is less than nrows * ncols
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout to minimize padding between subplots
    plt.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.2) # Very minimal padding
    
    # Save the combined figure
    combined_path = os.path.join(output_dir, combined_filename)
    plt.savefig(combined_path, dpi=300, bbox_inches='tight') # Save with high DPI and tight bounding box
    plt.close()
    print(f"\nCombined analysis plots saved to: {combined_path}")

def generate_results_table(results: list):
    """
    Generates and prints a formatted table of statistical test results.

    Args:
        results (list): A list of dictionaries, where each dictionary contains 
                        results for a single variable (variable_name, p_value, 
                        corrected_alpha, accepted_hypothesis).
    """
    if not results:
        print("\nNo statistical test results to display in the table.")
        return

    print("\n\n=========================================================")
    print("📊 Statistical Test Results Summary 📊")
    print("=========================================================")
    
    headers = ["Metric", "P-value", f"Corrected Alpha ({SCENARIO_DISPLAY_NAMES[SCENARIOS[0]]} vs {SCENARIO_DISPLAY_NAMES[SCENARIOS[1]]})", "Accepted Hypothesis"]
    
    # Determine column widths for formatting
    col_widths = {header: len(header) for header in headers}
    for res in results:
        col_widths["Metric"] = max(col_widths["Metric"], len(res["variable_name"]))
        col_widths["P-value"] = max(col_widths["P-value"], len(f"{res['p_value']:.4f}"))
        col_widths[f"Corrected Alpha ({SCENARIO_DISPLAY_NAMES[SCENARIOS[0]]} vs {SCENARIO_DISPLAY_NAMES[SCENARIOS[1]]})"] = \
            max(col_widths[f"Corrected Alpha ({SCENARIO_DISPLAY_NAMES[SCENARIOS[0]]} vs {SCENARIO_DISPLAY_NAMES[SCENARIOS[1]]})"], len(f"{res['corrected_alpha']:.4f}"))
        col_widths["Accepted Hypothesis"] = max(col_widths["Accepted Hypothesis"], len(res["accepted_hypothesis"]))

    # Print header
    header_line = (
        f"{headers[0]:<{col_widths[headers[0]]}} | "
        f"{headers[1]:<{col_widths[headers[1]]}} | "
        f"{headers[2]:<{col_widths[headers[2]]}} | "
        f"{headers[3]:<{col_widths[headers[3]]}}"
    )
    print(header_line)
    print("-" * len(header_line))

    # Print results rows
    for res in results:
        row_str = (
            f"{res['variable_name']:<{col_widths[headers[0]]}} | "
            f"{res['p_value']:<{col_widths[headers[1]]}.4f} | "
            f"{res['corrected_alpha']:<{col_widths[headers[2]]}.4f} | "
            f"{res['accepted_hypothesis']:<{col_widths[headers[3]]}}"
        )
        print(row_str)
    print("=========================================================\n")


# --- Main Analysis Logic ---

def main():
    # Capture all console output to a file
    stats_output_dir = os.path.join(BASE_RESULTS_DIR, "statistical_analysis")
    os.makedirs(stats_output_dir, exist_ok=True) # Ensure directory exists for the log file
    log_file_path = os.path.join(stats_output_dir, "statistical_analysis_output.txt")
    
    original_stdout = sys.stdout
    sys.stdout = open(log_file_path, 'w')

    try:
        print("🚀 Starting Statistical Analysis of Lotka-Volterra Simulation Results 🚀")
        print(f"Looking for simulation results in: {BASE_RESULTS_DIR}")

        print(f"Statistical outputs will be saved to: {stats_output_dir}\n")

        # List to store paths of all generated individual plots for later combination
        generated_plot_paths = []
        # List to store results for the final summary table
        all_test_results = []

        # 1. Load Data for each scenario
        print(f"Loading data for '{SCENARIO_DISPLAY_NAMES[SCENARIOS[0]]}' scenario...")
        energy_free_data = load_scenario_data(SCENARIOS[0])
        print(f"Loading data for '{SCENARIO_DISPLAY_NAMES[SCENARIOS[1]]}' scenario...")
        energy_enabled_data = load_scenario_data(SCENARIOS[1])

        # Calculate Bonferroni corrected alpha
        # Use len(VARIABLES_TO_ANALYZE) to get the count of all variables we intend to test
        num_tests = len(VARIABLES_TO_ANALYZE) 
        original_alpha = 0.05
        corrected_alpha = original_alpha / num_tests
        print(f"\nApplying Bonferroni correction: Original alpha={original_alpha}, Number of tests={num_tests}, Corrected alpha={corrected_alpha:.4f}\n")


        # Check if essential data exists for comparison across any variable
        # This checks if at least one variable has data for both scenarios
        any_data_for_comparison = False
        for var_key in VARIABLES_TO_ANALYZE.keys():
            if var_key in energy_free_data and var_key in energy_enabled_data:
                if len(energy_free_data[var_key]) > 0 and len(energy_enabled_data[var_key]) > 0:
                    any_data_for_comparison = True
                    break

        if not any_data_for_comparison:
            print("\n🛑 Error: Insufficient data loaded for analysis from one or both scenarios for ANY variable. "
                  "Please ensure your 'hunter_prey_v5.py' script has generated results "
                  "in the specified 'simulation_results' directory, and that there are "
                  "multiple runs (e.g., via a proper grid search or repetitions) for each scenario. 🛑")
            return

        # 2. Analyze each variable of interest and collect plot paths
        for var_key, json_key in VARIABLES_TO_ANALYZE.items():
            # Clean variable name for titles and labels
            display_var_name = var_key.replace('_', ' ').title()

            print(f"\n=========================================================")
            print(f"📈 Analyzing Variable: {display_var_name} 📊")
            print(f"=========================================================")

            # Get data for the current variable, ensuring it's a list for np.array conversion
            data_free = energy_free_data.get(var_key, [])
            data_enabled = energy_enabled_data.get(var_key, [])

            # Convert to numpy arrays for statistical functions
            data_free_np = np.array(data_free)
            data_enabled_np = np.array(data_enabled)

            # Skip analysis if no data for either scenario for THIS specific variable
            if data_free_np.size == 0 and data_enabled_np.size == 0:
                print(f"Skipping analysis for {display_var_name}: No data found for this variable in either scenario.")
                continue
            elif data_free_np.size == 0:
                print(f"Skipping comparison for {display_var_name}: No data found for '{SCENARIO_DISPLAY_NAMES[SCENARIOS[0]]}' scenario. Histograms and summary for {SCENARIO_DISPLAY_NAMES[SCENARIOS[1]]} will still be generated if data exists.")
                # Still proceed to plot histogram for enabled if data exists, and then skip statistical test
                if data_enabled_np.size > 0:
                     plot_histogram(data_enabled_np, 
                                   title=f'Histogram of {display_var_name} ({SCENARIO_DISPLAY_NAMES[SCENARIOS[1]]})', 
                                   xlabel=display_var_name, 
                                   output_dir=stats_output_dir, 
                                   filename=f'{var_key}_histogram_{sanitize_filename(SCENARIO_DISPLAY_NAMES[SCENARIOS[1]])}.png',
                                   plot_paths=generated_plot_paths) # Pass plot_paths
                continue
            elif data_enabled_np.size == 0:
                print(f"Skipping comparison for {display_var_name}: No data found for '{SCENARIO_DISPLAY_NAMES[SCENARIOS[1]]}' scenario. Histograms and summary for {SCENARIO_DISPLAY_NAMES[SCENARIOS[0]]} will still be generated if data exists.")
                # Still proceed to plot histogram for free if data exists, and then skip statistical test
                if data_free_np.size > 0:
                    plot_histogram(data_free_np, 
                                   title=f'Histogram of {display_var_name} ({SCENARIO_DISPLAY_NAMES[SCENARIOS[0]]})', 
                                   xlabel=display_var_name, 
                                   output_dir=stats_output_dir, 
                                   filename=f'{var_key}_histogram_{sanitize_filename(SCENARIO_DISPLAY_NAMES[SCENARIOS[0]])}.png',
                                   plot_paths=generated_plot_paths) # Pass plot_paths
                continue


            print(f"\n--- Data Summary for {display_var_name} ---")
            print(f"{SCENARIO_DISPLAY_NAMES[SCENARIOS[0]]} (N={len(data_free_np)}): Mean={np.mean(data_free_np):.2f}, Median={np.median(data_free_np):.2f}, Std Dev={np.std(data_free_np):.2f}")
            print(f"{SCENARIO_DISPLAY_NAMES[SCENARIOS[1]]} (N={len(data_enabled_np)}): Mean={np.mean(data_enabled_np):.2f}, Median={np.median(data_enabled_np):.2f}, Std Dev={np.std(data_enabled_np):.2f}")


            # 1. Generate histograms for each scenario
            plot_histogram(data_free_np, 
                           title=f'Histogram of {display_var_name} ({SCENARIO_DISPLAY_NAMES[SCENARIOS[0]]})', 
                           xlabel=display_var_name, 
                           output_dir=stats_output_dir, 
                           filename=f'{var_key}_histogram_{sanitize_filename(SCENARIO_DISPLAY_NAMES[SCENARIOS[0]])}.png',
                           plot_paths=generated_plot_paths) # Pass plot_paths
            
            plot_histogram(data_enabled_np, 
                           title=f'Histogram of {display_var_name} ({SCENARIO_DISPLAY_NAMES[SCENARIOS[1]]})', 
                           xlabel=display_var_name, 
                           output_dir=stats_output_dir, 
                           filename=f'{var_key}_histogram_{sanitize_filename(SCENARIO_DISPLAY_NAMES[SCENARIOS[1]])}.png',
                           plot_paths=generated_plot_paths) # Pass plot_paths
            
            # 2. Check normality for each scenario
            is_normal_free = check_normality(data_free_np, f"{display_var_name} ({SCENARIO_DISPLAY_NAMES[SCENARIOS[0]]})")
            is_normal_enabled = check_normality(data_enabled_np, f"{display_var_name} ({SCENARIO_DISPLAY_NAMES[SCENARIOS[1]]})")

            # For comparing two groups, if *either* group's data is non-normal,
            # or if sample sizes are too small for normality testing,
            # it is generally safer to use a non-parametric test.
            overall_is_normal = is_normal_free and is_normal_enabled
            
            # 3. Perform the appropriate statistical test and interpret results
            test_result = perform_statistical_test(data_free_np, data_enabled_np, display_var_name, overall_is_normal, corrected_alpha)
            if test_result:
                all_test_results.append(test_result)

            # 4. Generate scenario comparison bar chart with standard deviation
            plot_scenario_comparison_bar_chart(data_free_np, data_enabled_np,
                                               title=f'Comparison of {display_var_name} Across Scenarios',
                                               ylabel=display_var_name,
                                               output_dir=stats_output_dir,
                                               filename=f'{var_key}_comparison_bar_chart.png',
                                               plot_paths=generated_plot_paths) # Pass plot_paths
        
        # Generate the final results table
        generate_results_table(all_test_results)

        # Combine all generated plots into one image (Vertical A4 format)
        combine_plots(generated_plot_paths, stats_output_dir, combined_filename="combined_analysis_plots_vertical.png", is_horizontal=False)

        # Combine all generated plots into another image (Horizontal A4 format)
        combine_plots(generated_plot_paths, stats_output_dir, combined_filename="combined_analysis_plots_horizontal.png", is_horizontal=True)

        print("\nNote on Repeated-Measures ANOVA: A full Repeated-Measures ANOVA on every time step of the population data would require significant data restructuring and a dedicated statistical model (e.g., using statsmodels). For this script, we are addressing the 'population time series data' comparison by calculating the average population over time for each run and comparing these averages between scenarios using independent sample tests, while also providing visual comparison through histograms and bar charts.")
        print("\n✅ Statistical analysis complete. All plots and detailed results are saved "
              f"in the '{stats_output_dir}' directory. ✅")
    finally:
        # Restore original stdout
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"\nConsole output saved to: {log_file_path}") # Print this to the actual console

if __name__ == "__main__":
    main()