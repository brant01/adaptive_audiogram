import numpy as np

from src.models.model_base import AdaptiveModel
from src.utils.enums import Response, Ear
from src.sound_utils.sound import Sound
from src.models.bayes_model.bayesian_data import compute_frequency_ear_means, get_covariance_ear_matrix
from src.utils.constants import FREQUENCIES


class BayesianAdaptiveModel(AdaptiveModel):
    """
    Bayesian adaptive model for audiology testing.
    """
    def __init__(self, threshold: int = 5):
        """
        Initializes the BayesianAdaptiveModel with default means and covariance matrix.

        Args:
            threshold (int): Stopping criterion for uncertainty (in dB).
        """
        super().__init__(threshold)
        try:
            self.means = compute_frequency_ear_means(FREQUENCIES)
            self.cov_matrix = get_covariance_ear_matrix(FREQUENCIES)
        except Exception as e:
            raise ValueError(f"Error initializing means or covariance: {e}")
        
        self.sound = self._select_next_sound() # initial sound selection from baseline mean/cov matrix
        self.stop_flag = False # when all uncertainties are below threshold
    

    def _select_next_sound(self) -> Sound:
        """
        Selects the next sound to present based on maximum uncertainty.

        Returns:
            Sound: The next sound to be presented.
        """
        freq_idx = np.argmax(np.diag(self.cov_matrix))  # Most uncertain frequency
        frequency = FREQUENCIES[freq_idx // 2]
        ear = Ear.LEFT if freq_idx % 2 == 0 else Ear.RIGHT
        loudness = self.means[freq_idx]
        loudness = max(min(round(loudness / 5) * 5, 120), -20)  # Clamp to valid range
        return Sound(frequency, loudness, ear)

    def pass_response(self, response: Response):
        """
        Updates the model state based on the subject's response.

        Args:
            response (Response): Response.Yes if the patient hears the sound, Response.No otherwise.
        """
        # Identify the current frequency index
        freq_idx = np.argmax(np.diag(self.cov_matrix))

        # Perform Bayesian update
        likelihood_variance = 5.0
        kalman_gain = self.cov_matrix[freq_idx, freq_idx] / (
            self.cov_matrix[freq_idx, freq_idx] + likelihood_variance
        )

        # Update the posterior mean
        adjustment = kalman_gain * (self.sound.loudness - self.means[freq_idx])
        if response == Response.Yes:
            self.means[freq_idx] += adjustment
        elif response == Response.No:
            self.means[freq_idx] -= adjustment

        # Update the posterior covariance
        self.cov_matrix[freq_idx, freq_idx] *= (1 - kalman_gain)

        # Log the update
        self.history.append({
            "freq_idx": freq_idx,
            "frequency": self.sound.frequency,
            "ear": self.sound.ear.name,
            "loudness": self.sound.loudness,
            "response": response.name,
            "updated_means": self.means.copy(),
            "updated_covariance": self.cov_matrix.copy(),
        })

        # Check for convergence
        uncertainties = np.sqrt(np.diag(self.cov_matrix))
        self.stop_flag = np.all(uncertainties <= self.threshold)

        # Select the next sound if the process hasn't stopped
        if not self.stop_flag:
            self.sound = self._select_next_sound()

    def get_results(self):
        """
        Returns the final thresholds for each frequency and ear.

        Returns:
            list: List of tuples (label, threshold).
        """
        return list(zip(
            [f"{freq}_{ear}" for freq in FREQUENCIES for ear in ["L", "R"]],
            self.means
        ))

    def get_detailed_info(self):
        """
        Returns model-specific details such as uncertainties and history.

        Returns:
            dict: Detailed information specific to the Bayesian model.
        """
        uncertainties = np.sqrt(np.diag(self.cov_matrix))
        return {
            "thresholds": self.get_results(),
            "uncertainties": list(zip(
                [f"{freq}_{ear}" for freq in FREQUENCIES for ear in ["L", "R"]],
                uncertainties
            )),
            "history": self.history
        }