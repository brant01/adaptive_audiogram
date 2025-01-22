import json
import polars as pl
import random
from scipy.stats import norm
from typing import Optional

from src.models.model_base import AdaptiveModel
from src.sound_utils.sound import Sound
from src.utils.enums import Response

# if not using psychometric response use 0.05 at -10 dB, 0.25 at -5 db, 0.5 at threshold, 0.75 at +5 db, 0.95 at +10 db

class SimulationBase:
    def __init__(self, model_class: AdaptiveModel, threshold: int = 5):
        """
        Base class for simulations.

        Args:
            model_class (Type[AdaptiveModel]): The class of the model to use (e.g., BayesianAdaptiveModel).
            threshold (int): Stopping criterion for uncertainty (default: 5 dB).
        """
        self.model_class = model_class
        self.threshold = threshold
        self.model = None  # Placeholder for the model instance

    def initialize_model(self):
        """Initializes the model instance."""
        if not self.model_class:
            raise ValueError("Model class not provided.")
        self.model = self.model_class(threshold=self.threshold)

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

    def log_results(self, results: list[dict], output_file: Optional[str] = None):
        """
        Logs or stores results for further analysis.

        Args:
            results (List[dict]): Results of the simulation.
            output_file (Optional[str]): Path to save the results as a JSON file.
        """
        if output_file:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to {output_file}")
        else:
            print("Simulation Results:")
            for result in results:
                print(result)


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
            response = input("Enter response (1 for Yes, 0 for No): ")
            response = Response.Yes if response == "1" else Response.No

            self.model.pass_response(response)
            print(f"Updated Means: {self.model.means}")
            print(f"History: {self.model.history[-1]}")

        print("\nFinal Results:")
        self.log_results(self.model.get_results())


class AutomatedSimulation(SimulationBase):
    def run_model(self, patient_data_file: str, threshold_column_format: str = "{freq}_{ear}"):
        """
        Runs the simulation for multiple subjects using predefined data.

        Args:
            patient_data_file (str): Path to the file containing patient thresholds.
            threshold_column_format (str): Format string for threshold columns (default: "{freq}_{ear}").
        """
        # Load patient data
        patient_data = pl.read_parquet(patient_data_file)
        all_results = []

        for patient in patient_data.iter_rows(named=True):
            print(f"Running simulation for patient ID: {patient['patient_id']}")
            self.initialize_model()

            while not self.model.stop_flag:
                sound = self.model.current_sound
                threshold_col = threshold_column_format.format(freq=sound.frequency, ear=sound.ear.name)
                threshold = patient[threshold_col]
                response = self.psychometric_response(sound.volume, threshold)
                self.model.pass_response(response)

            results = self.model.get_results()
            all_results.append({"patient_id": patient["patient_id"], "results": results})

        print("\nAll Simulations Completed.")
        self.log_results(all_results)