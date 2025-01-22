import numpy as np
from pathlib import Path
import polars as pl
from typing import Optional

from src.utils.constants import FREQUENCIES, AUDIOGRAM_FILE_PATH


def main() -> None:
    pass


def load_audiogram_data(file_path: Path = AUDIOGRAM_FILE_PATH) -> pl.DataFrame:
    """
    Loads the audiogram data from the specified file path.

    Args:
        file_path (Path): Path to the audiogram data file.

    Returns:
        pl.DataFrame: Loaded Polars DataFrame.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Audiogram data file not found at: {file_path}")
    return pl.read_parquet(file_path)


def compute_frequency_ear_means(
    df: Optional[pl.DataFrame] = None, frequencies: list = FREQUENCIES
) -> np.ndarray:
    """
    Computes mean values per Frequency and Ear, returning as a 1D Numpy array.

    Args:
        df (Optional[pl.DataFrame]): Input Polars DataFrame with 'Frequency', 'Ear', and 'Value' columns.
                                     If None, the default dataset will be loaded.
        frequencies (list): List of frequency values to include.
        default_value (float): Default value to fill for missing frequency-ear combinations.

    Returns:
        np.ndarray: 1D array of mean values in the order:
                    [250_L, 250_R, 500_L, 500_R, ..., 8000_L, 8000_R].
    """
    if df is None:
        df = load_audiogram_data()

    # Fill nulls with mean per Frequency and Ear
    df_filled = df.with_columns(
        pl.col("Value").fill_null(pl.col("Value").mean().over(["Frequency", "Ear"]))
    )

    # Filter to specified frequencies
    df_filtered = df_filled.filter(pl.col("Frequency").is_in(frequencies))

    # Group by Frequency and Ear, compute means
    df_grouped = (
        df_filtered.group_by(["Frequency", "Ear"])
        .agg(pl.col("Value").mean().alias("Mean_Value"))
        .with_columns(
            (pl.col("Frequency").cast(pl.Utf8) + "_" + pl.col("Ear")).alias("Freq_Ear")
        )
    ).sort(["Frequency", "Ear"])
    
    return df_grouped["Mean_Value"].to_numpy()



def get_covariance_ear_matrix(
                              df: Optional[pl.DataFrame] = None, 
                              frequencies: list = FREQUENCIES
                              ) -> np.ndarray:
    """
    Computes the covariance matrix with combined left and right ear data.

    Args:
        df (pl.DataFrame): Input long-format Polars DataFrame with 'Frequency', 'Value',
                           'pt_ID', 'Audiogram_Date', and 'Ear' columns.
        frequencies (list): List of frequency values to include.

    Returns:
        np.ndarray: Covariance matrix ordered as [250_L, 250_R, ..., 8000_L, 8000_R].
    """
    
    if df is None:
        df = load_audiogram_data()
    
    # Filter to specified frequencies
    df_filtered = df.filter(pl.col("Frequency").is_in(frequencies))

    # Fill nulls with mean per Frequency
    df_filled = df_filtered.with_columns(
        pl.col("Value").fill_null(pl.col("Value").mean().over("Frequency")).alias("Value")
    )

    # Group by pt_ID, Audiogram_Date, Ear, and Frequency
    df_grouped = df_filled.group_by(["pt_ID", "Audiogram_Date", "Ear", "Frequency"]).agg(
        pl.col("Value").mean().alias("Value")
    )

    # Pivot to wide format
    df_pivot = df_grouped.pivot(
        index=["pt_ID", "Audiogram_Date"], 
        columns=["Frequency", "Ear"], 
        values="Value"
    )

    # Parse and format column names to match '250_L', '250_R', etc.
    def format_column_name(col_name):
        if isinstance(col_name, str) and col_name.startswith("{") and col_name.endswith("}"):
            freq, ear = col_name.strip("{}").split(",")
            return f"{freq.strip()}_{ear.strip().strip('\"')}"  # Format as 'freq_ear'
        return col_name

    # Apply formatting to all column names
    formatted_columns = [format_column_name(col) for col in df_pivot.columns]
    df_pivot.columns = formatted_columns

    # Ensure column order matches [250_L, 250_R, ..., 8000_L, 8000_R]
    ordered_columns = [f"{freq}_{ear}" for freq in frequencies for ear in ["L", "R"]]

    # Add missing columns if necessary
    for col in ordered_columns:
        if col not in df_pivot.columns:
            df_pivot = df_pivot.with_columns(pl.lit(None).alias(col))

    # Select and reorder columns
    df_pivot = df_pivot.select(ordered_columns)

    # Drop rows with any remaining nulls
    df_pivot_clean = df_pivot.drop_nulls()

    # Ensure enough observations for covariance
    if df_pivot_clean.height < 2:
        raise ValueError("Not enough observations to compute covariance matrix.")

    # Compute covariance matrix
    cov_matrix = df_pivot_clean.to_pandas().cov().to_numpy()
    return cov_matrix

if __name__ == "__main__":
    main()