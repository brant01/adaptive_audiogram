
import polars as pl
import random
from scipy.stats import norm

from src.utils.enums import ModelType, Response, Ear
from src.sound_utils.sound import Sound

class SimulationBase:
    def __init__(self, model_class: ModelType, threshold: int = 5):
        """
        Base class for simulations.

        Args:
            model_class (Type): The class of the model to use (e.g., BayesianAdaptiveModel).
            threshold (int): Stopping criterion for uncertainty (default: 5 dB).
        """
        self.model_class = model_class
        self.threshold = threshold
        self.model = None  # Placeholder for the model instance

    def initialize_model(self):
        """Initializes the model with the given class and threshold."""
        self.model = self.model_class(threshold=self.threshold)
        
    # Define a psychometric response function from the psychometric curve
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

    def log_results(self, results: list[dict]):
        """
        Logs or stores results for further analysis.
        
        Args:
            results (List[dict]): Results of the simulation.
        """
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
        
        print("\nFinal Results:")
        self.log_results(self.model.get_results())
        

class AutomatedSimulation(SimulationBase):
    def run_model(self, patient_data_file: str):
        """
        Runs the simulation for multiple subjects using predefined data.

        Args:
            patient_data_file (str): Path to the file containing patient thresholds.
        """
        # Load patient data
        patient_data = pl.read_parquet(patient_data_file)
        all_results = []

        for patient in patient_data.iter_rows(named=True):
            print(f"Running simulation for patient ID: {patient['patient_id']}")
            self.initialize_model()
            
            # Simulate responses based on psychometric curve
            while not self.model.stop_flag:
                sound = self.model.current_sound
                threshold = patient[f"{sound.frequency}_{sound.ear.name}"]
                response = self.psychometric_response(sound.loudness, self.threshold)
                self.model.pass_response(response)
            
            # Log results for the patient
            results = self.model.get_results()
            all_results.append({"patient_id": patient["patient_id"], "results": results})
        
        print("\nAll Simulations Completed.")
        self.log_results(all_results)