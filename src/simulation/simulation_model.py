from datetime import datetime, date
import json
from pathlib import Path
import polars as pl
import random
from scipy.stats import norm
from typing import Optional

from src.models.model_base import AdaptiveModel
from src.sound_utils.sound import Sound
from src.utils.constants import AUDIOGRAM_FILE_PATH, SIMULATION_FILE_PATH, FREQUENCIES
from src.utils.enums import Response

# if not using psychometric response use 0.05 at -10 dB, 0.25 at -5 db, 0.5 at threshold, 0.75 at +5 db, 0.95 at +10 db

class SimulationBase:
    def __init__(self, model_class: AdaptiveModel, 
                 threshold: int = 5,
                 alpha: float = 0.3):
        """
        Base class for simulations.

        Args:
            model_class (Type[AdaptiveModel]): The class of the model to use (e.g., BayesianAdaptiveModel).
            threshold (int): Stopping criterion for uncertainty (default: 5 dB).
        """
        self.model_class = model_class
        self.threshold = threshold
        self.alpha = alpha
        self.model = None  # Placeholder for the model instance

    def initialize_model(self):
        """Initializes the model instance."""
        if not self.model_class:
            raise ValueError("Model class not provided.")
        self.model = self.model_class(threshold=self.threshold, alpha=self.alpha)

    @staticmethod
    def psychometric_response(loudness: float, threshold: int, slope: float = 4.0) -> Response:
        """
        Determines whether a patient hears the sound based on the psychometric curve.

        Args:
            loudness (float): Presented loudness in dB.
            threshold (float): Hearing threshold in dB.
            slope (float): Standard deviation of the psychometric curve (default: 4 dB).

        Returns:
            Response: Response.Yes if the patient hears the sound, Response.No otherwise.
        """
        probability = norm.cdf(loudness, loc=threshold, scale=slope)
        return Response.Yes if random.random() < probability else Response.No


    def log_results(self, results: list[dict], output_file: Optional[Path] = None):
        """
        Logs or stores results for further analysis.

        Args:
            results (List[dict]): Results of the simulation.
            output_file (Optional[Path]): Path to save the results as a JSON file.
        """
        def default_serializer(obj):
            """Custom serializer for non-JSON-serializable objects."""
            if isinstance(obj, date):
                return obj.isoformat()  # Convert date to ISO format string
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        if output_file:
            # Ensure the output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the results to the specified file
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4, default=default_serializer)
            print(f"Results saved to {output_file}")
        else:
            print("Simulation Results:")
            print(results)
            #for result in results:
            #    for key, value in result.items():
            #        print(f"{key}: {value} dB")


class ManualSimulation(SimulationBase):
    def run_model(self):
        """
        Runs the simulation manually with user input for each step.
        """
        self.initialize_model()
        print("Manual Simulation Started. Enter responses for each sound.")

        while not self.model.stop_flag:
            sound: Sound = self.model.current_sound
            print(f"Presenting: {sound.frequency} Hz ({sound.ear.name}) at {sound.volume} dB")
            response_in = input("Enter response (1 for Yes, 0 for No): ")
            response = Response.Yes if response_in == "1" else Response.No

            self.model.pass_response(response)
            print(f"Updated Means: {self.model.means}")
            print(f"History: {self.model.history[-1]}")

        print("\nFinal Results:")
        self.log_results(self.model.get_results())


class AutomatedSimulation(SimulationBase):
    def run_model(self, 
                  patient_data_file: Path = AUDIOGRAM_FILE_PATH, 
                  output_folder: Path = SIMULATION_FILE_PATH,
                  test_pct: float = 0.3, 
                  is_traning: bool = True):
        """
        Runs the simulation for multiple subjects using predefined data.

        Args:
            patient_data_file (Path): Path to the file containing patient thresholds.
            output_folder (Path): Folder to save the results as a JSON file.
            test_pct (float): Percentage of data to hold out for testing (default: 0.3).
        """
        # Load patient data
        patient_data = pl.read_parquet(patient_data_file)
        
        # Reformat the data into wide format
        wide_data = self._reformat_to_wide(patient_data)
        
        # Split the data into training and testing sets
        train_data, test_data = self._split_data(wide_data, test_pct)
        
        data = train_data if is_traning else test_data
        print(f"Data shape: {data.shape}")
        
        all_results = []
        pt_num = 0
        # Run the simulation on the training set
        for row in data.iter_rows(named=True):
            patient_id = row["pt_ID"]
            audiogram_date = row["Audiogram_Date"]
            
            self.initialize_model()
            step_count = 0  # Track the number of steps

            # Create a dictionary to store thresholds for each frequency and ear
            thresholds = {key: row[key] for key in row if key not in ["pt_ID", "Audiogram_Date"]}

            while not self.model.stop_flag:
                sound = self.model.current_sound
                key = f"{sound.frequency}_{sound.ear.name[0]}"  # Match the key format
                threshold = thresholds.get(key, None)
                
                if threshold is None:
                    raise ValueError(f"No threshold found for frequency {sound.frequency} and ear {sound.ear.name}.")
                
                response = self.psychometric_response(sound.volume, threshold)
                self.model.pass_response(response)
                step_count += 1

            # Collect results
            results: dict = self.model.get_results()
            
            # Create sorted final_thresholds and threshold_error dictionaries
            final_thresholds = {}
            threshold_error = {}
            
            for freq in FREQUENCIES:
                for ear in ["L", "R"]:
                    key = f"{freq}_{ear}"
                    final_mean = results.get(key, None)
                    true_threshold = thresholds.get(key, None)
                    
                    if final_mean is not None and true_threshold is not None:
                        final_thresholds[key] = final_mean
                        threshold_error[key] = final_mean - true_threshold
                    else:
                        raise ValueError(f"Missing data for frequency {freq} and ear {ear}.")

            if pt_num % 100 == 0:
                print(f"Running simulation for patient ID: {patient_id}, Date: {audiogram_date}, Number: {pt_num}/{data.shape[0]}")
                print(f"Threshold Error: {threshold_error}")
            pt_num += 1

            all_results.append({
                "patient_id": patient_id,
                "audiogram_date": audiogram_date,
                "steps": step_count,
                "final_thresholds": final_thresholds,
                "threshold_error": threshold_error,
                "results": results
            })

        print("\nAll Simulations Completed.")
        
        # Generate the output file name
        current_date = datetime.now().strftime("%Y%m%d")  # Format: YYYYMMDD
        model_name = self.model.__class__.__name__  # Get the model's class name
        threshold_value = int(self.threshold)  # Get the threshold value
        alpha_value = self.alpha
        output_file_name = f"{current_date}_{model_name}_alph{alpha_value}_thr{threshold_value}.json"
        output_file_path = output_folder / output_file_name
        
        # Save the results
        self.log_results(all_results, output_file_path)

    def _reformat_to_wide(self, patient_data: pl.DataFrame) -> pl.DataFrame:
        """
        Reformat the patient data into wide format with columns for each frequency and ear.
        Ensure only frequencies of interest are included.
        Using a single 'FreqEar' column so pivoted columns become '250_L', '500_R', etc.
        """
        
        # Filter to specified frequencies
        df = patient_data.filter(pl.col("Frequency").is_in(FREQUENCIES))

        
        # Create a single combined column
        df = df.with_columns(
            (pl.col("Frequency").cast(pl.Utf8) + "_" + pl.col("Ear")).alias("FreqEar")
        )

        # Pivot on that one column
        wide_data = df.pivot(
            values="Value",
            index=["pt_ID", "Audiogram_Date"],
            columns="FreqEar",          # <--- pivot on the combined col
            aggregate_function="first"
        )

        # remove rows with any null values
        wide_data = wide_data.drop_nulls()
        
        print(f"Filtered data shape: {wide_data.shape}")
        
        # Now wide_data has columns like:
        # ["pt_ID", "Audiogram_Date", "250_L", "250_R", "500_L", "500_R", ..., "8000_R"]

        return wide_data

    def _split_data(self, wide_data: pl.DataFrame, test_pct: float) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Split the data into training and testing sets.

        Args:
            wide_data (pl.DataFrame): The patient data in wide format.
            test_pct (float): Percentage of data to hold out for testing.

        Returns:
            tuple[pl.DataFrame, pl.DataFrame]: The training and testing datasets.
        """
        # Shuffle the data
        wide_data = wide_data.sample(fraction=1.0, shuffle=False)
        
        # Split the data
        split_idx = int(len(wide_data) * (1 - test_pct))
        train_data = wide_data[:split_idx]
        test_data = wide_data[split_idx:]
        
        return train_data, test_data