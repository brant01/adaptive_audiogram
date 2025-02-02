from datetime import datetime
import json
from matplotlib import pyplot as plt
from pathlib import Path
import polars as pl
import re
from typing import Union

def load_json_to_tidy_polars(json_file: str) -> pl.DataFrame:
    """
    Load simulation results from a JSON file into a tidy Polars DataFrame.

    Args:
        json_file (Path): Path to the JSON file.

    Returns:
        pl.DataFrame: A tidy DataFrame with columns: patient_id, audiogram_date, steps, frequency, ear, final_threshold, threshold_error.
    """
    # Load the JSON file
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # Initialize a list to store tidy rows
    tidy_rows = []
    
    # Iterate over each entry in the JSON data
    for entry in data:
        patient_id = entry["patient_id"]
        audiogram_date = entry["audiogram_date"]
        steps = entry["steps"]
        
        # Iterate over final_thresholds and threshold_error
        for key, final_threshold in entry["final_thresholds"].items():
            frequency, ear = key.split("_")  # Split key into frequency and ear
            threshold_error = entry["threshold_error"].get(key, None)  # Get corresponding error
            
            # Append a tidy row
            tidy_rows.append({
                "patient_id": patient_id,
                "audiogram_date": audiogram_date,
                "steps": steps,
                "frequency": int(frequency),
                "ear": ear,
                "final_threshold": final_threshold,
                "threshold_error": threshold_error
            })
    
    # Convert the list of rows into a Polars DataFrame
    tidy_df = pl.DataFrame(tidy_rows)
    
    # Get the threshold from the end of the file name between thr and .json
    stop_threshold, alpha = extract_threshold_alpha_from_filename(json_file)
    
    # Add a column for the threshold
    tidy_df = tidy_df.with_columns(
        pl.lit(int(stop_threshold)).alias("stop_threshold"),
        pl.lit(float(alpha)).alias("alpha"),
        )
    return tidy_df



def load_folder_jsons_to_polars(
    folder: Union[str, Path]
) -> pl.DataFrame:
    """
    Loads all JSON files in the given folder using `json_loader_function` 
    and concatenates them vertically into one Polars DataFrame.

    Args:
        folder (str | Path): Path to the folder containing JSON files.
        json_loader_function (Callable[[str], pl.DataFrame]):
            A function that takes a JSON file path (string) and returns a Polars DataFrame.
            e.g., your custom `load_json_to_tidy_polars(json_file: str) -> pl.DataFrame`.

    Returns:
        pl.DataFrame: The concatenated DataFrame from all JSON files.
                      If no files, returns an empty DataFrame.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"Provided folder path does not exist or is not a directory: {folder}")

    # Gather all .json files (not descending into subfolders by default)
    json_files = sorted(folder_path.glob("*.json"))

    # If no JSON files, return empty DataFrame
    if not json_files:
        print(f"No .json files found in {folder_path}")
        return pl.DataFrame()

    # Process each JSON file using your loader function
    dfs: list[pl.DataFrame] = []
    for json_file in json_files:
        # Convert Path to string if needed for your loader
        df = load_json_to_tidy_polars(str(json_file))
        dfs.append(df)

    # Concatenate them vertically
    combined_df = pl.concat(dfs, how="vertical")
    
    combined_df = combined_df.sort("stop_threshold")

    return combined_df


def extract_threshold_alpha_from_filename(file_str: str) -> tuple[int, float]:
    """
    Extracts stop_threshold and alpha values from a given filename.

    Args:
        file_str (str): The filename string.

    Returns:
        dict: A dictionary containing 'stop_threshold' (int) and 'alpha' (float).
    
    Example:
        file_str = "20250128_BayesianAdaptiveModel_alph0.2_thr2.json"
        Returns: {'stop_threshold': 2, 'alpha': 0.2}
    """
    # Extract the filename from the path
    file_name = file_str.split("/")[-1]  # or use os.path.basename(file_str)
    
    # Updated pattern to match both alpha and threshold values
    pattern = r"_alph([\d\.]+)_thr(\d+)\.json$"
    match = re.search(pattern, file_name)
    
    if not match:
        raise ValueError(f"File name does not match the expected pattern: {file_name}")
    
    # Extract values and convert to correct types
    alpha = float(match.group(1))   # Extracts '0.2' as float
    stop_threshold = int(match.group(2))  # Extracts '2' as int

    return stop_threshold, alpha

def plot_violin_threshold_error(df):
    """
    Create a violin plot of threshold_error (y-axis) vs. threshold (x-axis).
    Args:
        df (pl.DataFrame or pd.DataFrame): Must have 'threshold' and 'threshold_error' columns.
    """

    # 1) Extract unique thresholds in ascending order
    unique_thresholds = sorted(df['stop_threshold'].unique())

    # 2) Group threshold_error by threshold
    grouped_errors = []
    for thr in unique_thresholds:
        # Filter rows matching this threshold
        # Adjust syntax depending on Polars vs. pandas
        if hasattr(df, "filter"):  # Polars
            errors_for_thr = df.filter(df['stop_threshold'] == thr)['threshold_error'].to_list()
        else:  # pandas
            errors_for_thr = df.loc[df['stop_threshold'] == thr, 'threshold_error'].tolist()
        
        grouped_errors.append(errors_for_thr)

    # 3) Create the violin plot
    fig, ax = plt.subplots(figsize=(8, 6))
    # violinplot returns a dictionary of collections
    parts = ax.violinplot(grouped_errors, showmeans=True, showextrema=True)

    # 4) Label the x-axis with the actual threshold values
    ax.set_xticks(range(1, len(unique_thresholds) + 1))
    ax.set_xticklabels(unique_thresholds)

    ax.set_xlabel("Stopping Threshold")
    ax.set_ylabel("Threshold Error")
    ax.set_title("Violin Plot of Threshold Error by Stopping Threshold")

    plt.show()
    
    
def plot_violin_setps_threshold(df):
    """
    Create a violin plot of threshold_error (y-axis) vs. threshold (x-axis).
    Args:
        df (pl.DataFrame or pd.DataFrame): Must have 'threshold' and 'threshold_error' columns.
    """

    # 1) Extract unique stopping thresholds in ascending order
    unique_thresholds = sorted(df['stop_threshold'].unique())

    # 2) Group threshold_error by stopping threshold
    grouped_errors = []
    for thr in unique_thresholds:
        # Filter rows matching this threshold
        # Adjust syntax depending on Polars vs. pandas
        if hasattr(df, "filter"):  # Polars
            errors_for_thr = df.filter(df['stop_threshold'] == thr)['steps'].to_list()
        else:  # pandas
            errors_for_thr = df.loc[df['stop_threshold'] == thr, 'steps'].tolist()
        
        grouped_errors.append(errors_for_thr)

    # 3) Create the violin plot
    fig, ax = plt.subplots(figsize=(8, 6))
    # violinplot returns a dictionary of collections
    parts = ax.violinplot(grouped_errors, showmeans=True, showextrema=True)

    # 4) Label the x-axis with the actual threshold values
    ax.set_xticks(range(1, len(unique_thresholds) + 1))
    ax.set_xticklabels(unique_thresholds)

    ax.set_xlabel("Stopping Threshold")
    ax.set_ylabel("Number of Steps")
    ax.set_title("Violin Plot of Number of Steps by Stopping Threshold")

    plt.show()
    
def plot_steps_vs_threshold_error(df):
    """
    Plots a scatter of 'threshold_error' vs 'steps'.
    
    Args:
        df (pl.DataFrame or pd.DataFrame):
            Must have 'steps' and 'threshold_error' columns.
    """
    
    # Extract the data (adjust if you're using Polars vs. pandas)
    if hasattr(df, "filter"):  # Polars
        x_data = df["steps"].to_list()
        y_data = df["threshold_error"].to_list()
    else:  # pandas
        x_data = df["steps"].values
        y_data = df["threshold_error"].values
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_data, y_data, alpha=0.7)
    
    ax.set_xlabel("Steps")
    ax.set_ylabel("Threshold Error")
    ax.set_title("Scatter Plot: Steps vs. Threshold Error")

    plt.show()
    
    
def plot_audiogram_polars(df: pl.DataFrame, patient_id: str, date_str: str):
    """
    Plots an audiogram from a Polars DataFrame containing columns:
      ['patient_id', 'date', 'frequency', 'ear', 'value'].

    - x-axis: frequency (linear)
    - y-axis: threshold in dB HL (inverted)
    - Left ear in blue, right ear in red.

    Args:
        df (pl.DataFrame): Must have 'patient_id', 'date', 'frequency', 'ear', 'value'.
        patient_id (str): The patient's ID to filter on.
        date (str): The date to filter on (exact string match).
    """
    # 1. Filter the Polars DataFrame
    
    date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    print(date_obj)
    subset = df.filter(
        (pl.col("pt_ID") == patient_id) &
        (pl.col("Audiogram_Date") == date_obj)
    )
    
    print(subset.shape)

    # 2. Separate left-ear and right-ear, then sort by frequency
    left_ear = subset.filter(pl.col("Ear") == "L").sort("Frequency")
    right_ear = subset.filter(pl.col("Ear") == "R").sort("Frequency")

    # 3. Extract frequency and value as Python lists
    x_left = left_ear["Frequency"].to_list()
    print(x_left)
    y_left = left_ear["Value"].to_list()
    print(y_left)
    x_right = right_ear["Frequency"].to_list()
    print(x_right)
    y_right = right_ear["Value"].to_list()
    print(y_right)  

    # 4. Set up Matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # 5. Plot left ear (blue) and right ear (red)
    ax.plot(x_left, y_left, marker="o", color="blue", label="Left Ear")
    ax.plot(x_right, y_right, marker="o", color="red",  label="Right Ear")

    # 6. Invert the y-axis so higher dB appear lower
    ax.invert_yaxis()

    # 7. Label axes and set title
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Volume (dB HL)")
    ax.set_title(f"Audiogram for Patient {patient_id} on {date_str}")

    # 8. Show legend and plot
    ax.legend()
    plt.show()