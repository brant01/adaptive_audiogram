import numpy as np
from pathlib import Path
import polars as pl

from src.utils.constants import FREQUENCIES

# use 0.05 at -10 dB, 0.25 at -5 db, 0.5 at threshold, 0.75 at +5 db, 0.95 at +10 db


def main() -> None:
    pass

audio_data = Path("data/cleaned_data.parquet")

def compute_frequency_means(df: pl.DataFrame, frequencies: list= FREQUENCIES) -> np.ndarray:
    """
    Compute the mean of 'Value' for each unique 'Frequency' in the DataFrame.

    Args:
        df (pl.DataFrame): Input Polars DataFrame with 'Frequency' and 'Value' columns.

    Returns:
        DataFrame: Polars DataFrame with each column as a frequency, one row representing the mean.
    """
    
    # filter Frequencies are in frequencies
    df = (
        df
        .filter(pl.col('Frequency').is_in(frequencies))
    )
    
    # Group by Frequency and compute the mean of Value
    df = (
        df.group_by('Frequency')
        .agg(pl.col('Value')
        .mean()
        .alias('Mean_Value'))
        ).sort('Frequency')

    df = (
    df
        .with_columns(pl.lit(1).alias("group"))  # Add a dummy group column
        .pivot(
            index="group",                      # Use the dummy group as the index
            columns="Frequency",                # Pivot on the 'Frequency' column
            values="Mean_Value"                 # Values to populate the pivoted columns
        )
        .drop("group")                          # Remove the dummy group column
        #.select(sorted(df_means["Frequency"].unique()))  # Sort columns by frequency
    )
    
    # convert to numpy array
    array = df.to_numpy().flatten()
    
    return array


def compute_frequency_ear_means(df: pl.DataFrame, frequencies: list = FREQUENCIES) -> pl.DataFrame:
    """
    Fills nulls with the mean per Frequency and Ear, computes the mean Value per Frequency and Ear,
    and pivots to a single-row wide DataFrame with one column per Frequency_Ear combination.
    
    Args:
        df (pl.DataFrame): Input long-format Polars DataFrame with 'Frequency', 'Ear', and 'Value' columns.
        frequencies (list): List of frequency values to include.
    
    Returns:
        pl.DataFrame: Single-row DataFrame with mean values for each Frequency_Ear combination.
        the order of returs is: 250_L┆ 250_R┆ 500_L┆ 500_R ┆ … ┆ 4000_L ┆ 4000_R┆ 8000_L ┆ 8000_R
    """
    # Fill nulls with mean per Frequency and Ear
    df_filled = df.with_columns(
        pl.col("Value").fill_null(
            pl.col("Value").mean().over(["Frequency", "Ear"])
        )
    )
    
    # Filter to include only specified frequencies
    df_filtered = df_filled.filter(pl.col("Frequency").is_in(frequencies))
    
    # Group by Frequency and Ear, compute mean Value
    df_grouped = df_filtered.group_by(["Frequency", "Ear"]).agg(
        pl.col("Value").mean().alias("Mean_Value")
    )
    
    # Create combined Frequency_Ear column (e.g., '250_L', '500_R')
    df_grouped = df_grouped.with_columns(
        (pl.col("Frequency").cast(pl.Utf8) + "_" + pl.col("Ear")).alias("Freq_Ear")
    )
    
    # Add a dummy group column for pivoting
    df_grouped = df_grouped.with_columns(pl.lit(1).alias("group"))
    
    # Pivot to wide format
    df_pivot = df_grouped.pivot(
        index="group",
        columns="Freq_Ear",
        values="Mean_Value"
    ).drop("group")
    
    # Sort the columns in ascending order based on Frequency and Ear
    sorted_freqs = sorted(
        df_grouped["Freq_Ear"].unique().to_list(),
        key=lambda x: (int(x.split("_")[0]), x.split("_")[1])
    )
    df_sorted = df_pivot.select(sorted_freqs)
    
    array = df_sorted.to_numpy().flatten()
    
    return array

def get_covariance_matrix(df: pl.DataFrame, frequencies: list = FREQUENCIES) -> np.ndarray:
    """
    Filters the DataFrame to include specified frequencies, fills nulls with the mean per frequency,
    groups by pt_ID, Audiogram_Date, and Ear to ensure unique rows, pivots to wide format,
    and computes the covariance matrix as a NumPy array.
    
    Args:
        df (pl.DataFrame): Input long-format Polars DataFrame with 'Frequency', 'Value', 'pt_ID', 'Audiogram_Date', and 'Ear' columns.
        frequencies (list): List of frequency values to include.
    
    Returns:
        np.ndarray: Covariance matrix as a NumPy array.
    """
    # 1. Filter to specified frequencies
    df_filtered = df.filter(pl.col("Frequency").is_in(frequencies))
    
    # 2. Fill nulls with mean per Frequency
    df_filled = df_filtered.with_columns(
        pl.col("Value").fill_null(pl.col("Value").mean().over("Frequency")).alias("Value")
    )
    
    # 3. Group by pt_ID, Audiogram_Date, and Ear to ensure unique rows
    #    and aggregate values by taking the mean if duplicates exist
    df_grouped = df_filled.group_by(["pt_ID", "Audiogram_Date", "Ear", "Frequency"]).agg(
        pl.col("Value").mean().alias("Value")
    )
    
    # 4. Pivot to wide format: one row per (pt_ID, Audiogram_Date, Ear), columns as frequencies
    df_pivot = df_grouped.pivot(
        index=["pt_ID", "Audiogram_Date", "Ear"],  # Unique identifier for each row
        columns="Frequency",                        # Frequencies as columns
        values="Value"                              # Values to populate the pivoted columns
    )
    
    # 5. Drop rows with any remaining nulls to ensure complete data for covariance
    df_pivot_clean = df_pivot.drop_nulls()
    
    # 6. Check if there are enough observations to compute covariance
    if df_pivot_clean.height < 2:
        raise ValueError("Not enough observations to compute covariance matrix.")
    
    # 7. Drop non-frequency columns: pt_ID	Audiogram_Date	Ear
    df_pivot_clean = df_pivot_clean.drop(["pt_ID", "Audiogram_Date", "Ear"])
    
    # 8. Convert to Pandas DataFrame for covariance computation
    df_pandas = df_pivot_clean.to_pandas()
    
    # 9. Compute covariance matrix
    cov_matrix = df_pandas.cov().to_numpy()
    
    return cov_matrix


def get_covariance_ear_matrix(df: pl.DataFrame, frequencies: list = FREQUENCIES) -> np.ndarray:
    """
    Filters the DataFrame to include specified frequencies, fills nulls with the mean per frequency,
    groups by pt_ID, Audiogram_Date, and Ear to ensure unique rows, pivots to wide format for each ear,
    combines left and right ear data into a single matrix, and computes the covariance matrix.
    
    Columns in the covariance matrix are ordered as follows:
    [250_L, 250_R, 500_L, 500_R, ..., 8000_L, 8000_R].
    
    Args:
        df (pl.DataFrame): Input long-format Polars DataFrame with 'Frequency', 'Value', 'pt_ID', 'Audiogram_Date', and 'Ear' columns.
        frequencies (list): List of frequency values to include.
    
    Returns:
        np.ndarray: Covariance matrix as a NumPy array.
    """
    # 1. Filter to specified frequencies
    df_filtered = df.filter(pl.col("Frequency").is_in(frequencies))
    
    # 2. Fill nulls with mean per Frequency
    df_filled = df_filtered.with_columns(
        pl.col("Value").fill_null(pl.col("Value").mean().over("Frequency")).alias("Value")
    )
    
    # 3. Group by pt_ID, Audiogram_Date, Ear, and Frequency
    df_grouped = df_filled.group_by(["pt_ID", "Audiogram_Date", "Ear", "Frequency"]).agg(
        pl.col("Value").mean().alias("Value")
    )
    
    # 4. Pivot to wide format for each ear
    df_pivot = df_grouped.pivot(
        index=["pt_ID", "Audiogram_Date"],  # Unique identifier for each row
        columns=["Frequency", "Ear"],      # Multi-level columns: Frequency and Ear
        values="Value"                     # Values to populate the pivoted columns
    )
    
    # 5. Extract and reformat column names
    def parse_column_name(col_name):
        if isinstance(col_name, str) and col_name.startswith("{") and col_name.endswith("}"):
            # Extract the frequency and ear from the string
            freq, ear = col_name.strip("{}").split(",")
            return f"{freq}_{ear.strip().strip('\"')}"  # Format as freq_ear (e.g., 250_L)
        return col_name
    
    formatted_columns = [parse_column_name(col) for col in df_pivot.columns]
    df_pivot.columns = formatted_columns

    # 6. Enforce the desired column order
    ordered_columns = [f"{freq}_{ear}" for freq in frequencies for ear in ["L", "R"]]
    
    # Ensure all ordered columns exist in the DataFrame
    missing_columns = [col for col in ordered_columns if col not in formatted_columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")
    
    df_pivot = df_pivot.select(ordered_columns)
    
    # 7. Drop rows with any remaining nulls to ensure complete data for covariance
    df_pivot_clean = df_pivot.drop_nulls()
    
    # 8. Check if there are enough observations to compute covariance
    if df_pivot_clean.height < 2:
        raise ValueError("Not enough observations to compute covariance matrix.")
    
    # 9. Convert to Pandas DataFrame for covariance computation
    df_pandas = df_pivot_clean.to_pandas()
    
    # 10. Compute covariance matrix
    cov_matrix = df_pandas.cov().to_numpy()
    
    return cov_matrix


if __name__ == '__main__':
    main()