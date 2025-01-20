from abc import ABC, abstractmethod

from src.utils.enums import Response
from src.sound_utils.sound import Sound

class AdaptiveModel(ABC):
    def __init__(self, threshold: int = 5):
        """
        Base class for adaptive models.

        Args:
            threshold (int): Stopping criterion for uncertainty (default = 5 dB).
        """
        self.threshold = threshold
        self.sound = None
        self.stop_flag = False
        self.history = []

    @property
    def current_sound(self) -> Sound:
        """
        Provides read-only access to the current sound to be presented.

        Returns:
            Sound: The current sound.
        """
        return self.sound

    @property
    def results(self):
        """
        Returns the final thresholds for each frequency and ear.

        Returns:
            list: List of tuples (label, threshold).
        """
        return self.get_results()

    @abstractmethod
    def get_results(self):
        """
        Abstract method to retrieve the final thresholds.

        Returns:
            list: List of tuples (label, threshold).
        """
        pass

    @abstractmethod
    def get_detailed_info(self):
        """
        Abstract method to retrieve model-specific detailed information.

        Returns:
            dict: Detailed information specific to the model.
        """
        pass
    
    def pass_response(self, response: Response):
        """
        Updates the model state based on the subject's response.

        Args:
            response (Response): Response.Yes if the patient hears the sound, Response.No otherwise.
        """
        if not self.sound:
            raise ValueError("No sound selected. Call `select_next_sound` first.")

        # Perform any shared updates (e.g., logging response)
        self.history.append({
            "sound": self.sound,
            "response": response.name,
        })

        # Check for convergence
        self.check_convergence()

        # Update the next sound if the process hasn't stopped
        if not self.stop_flag:
            self.sound = self.select_next_sound()
            
    def check_convergence(self):
        """
        Abstract method to check if the model has converged.
        Derived classes can extend or override this as needed.
        """
        # Placeholder: Derived classes should implement their own logic.
        self.stop_flag = True
