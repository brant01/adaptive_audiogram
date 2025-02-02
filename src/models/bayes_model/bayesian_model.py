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
    def __init__(self, 
                 stop_threshold: int = 3, 
                 alpha: float = 0.3,
                 update_factor: float = 1.0,
                 **model_params):
        """
        Initializes the BayesianAdaptiveModel with default means and covariance matrix.

        Args:
            stop_threshold (int): Stopping criterion for uncertainty (in dB).
            alpha (float): Psychometric function slope.
            **kwargs: Additional parameters for flexibility.
        """
        
        # Ensure that known parameters are extracted and not passed to super()
        self.stop_threshold = model_params.pop("stop_threshold", stop_threshold)
        self.alpha = model_params.pop("alpha", alpha)
        self.update_factor = model_params.pop("update_factor", update_factor)
        
        super().__init__(**model_params)
        
        try:
            self.means = compute_frequency_ear_means().copy()
            self.cov_matrix = get_covariance_ear_matrix().copy()
        except Exception as e:
            raise ValueError(f"Error initializing means or covariance: {e}")
        
        self.sound = self._select_next_sound() # initial sound selection from baseline mean/cov matrix
        self.stop_flag = False # when all uncertainties are below stop_threshold
        
    def _select_next_sound(self) -> Sound:
        """
        Selects the next sound to present based on maximum uncertainty.
        """
        # 1. Identify the dimension with largest variance
        freq_idx = np.argmax(np.diag(self.cov_matrix))
        
        # 2. Save it so we can retrieve this in pass_response
        self.last_tested_index = freq_idx

        # 3. Build the Sound object
        frequencies_per_ear = [freq for freq in FREQUENCIES for _ in range(2)]
        frequency = frequencies_per_ear[freq_idx]
        ear = Ear.LEFT if freq_idx % 2 == 0 else Ear.RIGHT
        loudness = self.means[freq_idx]
        loudness = max(min(round(loudness / 5) * 5, 120), -20)  # clamp to valid range
        
        return Sound(frequency, loudness, ear)

    def pass_response(self, response: Response):
        """
        Updates the model state based on the subject's response.
        """
        # Retrieve the dimension we just tested
        freq_idx = self.last_tested_index
        
        # Do the EKF/Bayesian update (using freq_idx, self.sound.volume, etc.)
        self.means, self.cov_matrix = self.ekf_update_yes_no(
            self.means,
            self.cov_matrix,
            freq_idx,
            self.sound.volume,
            response,
        )

        # Log it
        self.history.append({
            "freq_idx": freq_idx,
            "frequency": self.sound.frequency,
            "ear": self.sound.ear.name,
            "loudness": self.sound.volume,
            "response": response.name,
            "updated_means": self.means.copy(),
            "updated_covariance": self.cov_matrix.copy(),
        })

        # Check for stopping
        uncertainties = np.sqrt(np.diag(self.cov_matrix))
        self.stop_flag = np.all(uncertainties <= self.stop_threshold)

        # If not stopping, select next sound
        if not self.stop_flag:
            self.sound = self._select_next_sound()

    def get_results(self) -> dict:
        """
        Returns the final thresholds for each frequency and ear.
        Rounds/clamps each threshold to the nearest 5 dB within [-20, 120].
        """
        labels = [f"{freq}_{ear}" for freq in FREQUENCIES for ear in ["L", "R"]]
        
        # Round to nearest 5 dB and clamp between -20 and 120
        rounded_means = [
            max(min(round(m / 5) * 5, 120), -20)
            for m in self.means
        ]
        
        # Combine into a dictionary
        return {label: mean for label, mean in zip(labels, rounded_means)}

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
        
    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def _psychometric_probability(self, volume, threshold):
        s = self._sigmoid(self.alpha * (volume - threshold))
        return s
    
    def ekf_update_yes_no(self, current_thresholds, current_covariance, tested_dimension, test_volume, observed_response):
        """
        Performs a single Bayesian (EKF-like) update for the audiogram model.
        
        Args:
            current_thresholds (np.array): Current threshold estimates (one per frequency/ear).
            current_covariance (np.array): Current covariance matrix of the threshold estimates.
            tested_dimension (int): Index corresponding to the frequency/ear that was just tested.
            test_volume (float): The volume (in dB) that was presented.
            observed_response (Response): The subject's response (Yes/No).
            
        Returns:
            updated_thresholds (np.array): Updated threshold estimates.
            updated_covariance (np.array): Updated covariance matrix.
        """
        num_dimensions = len(current_thresholds)

        # Compute the predicted probability of a "Yes" response based on the current threshold estimate.
        predicted_probability = self._psychometric_probability(test_volume, current_thresholds[tested_dimension])
        
        # Calculate the derivative of the psychometric function with respect to the threshold.
        # h(θ) = predicted_probability, so the derivative is:
        derivative_wrt_threshold = -self.alpha * predicted_probability * (1 - predicted_probability)
        
        # Build the measurement Jacobian (1 x D) that only has a nonzero entry at the tested dimension.
        measurement_jacobian = np.zeros((1, num_dimensions))
        measurement_jacobian[0, tested_dimension] = derivative_wrt_threshold

        # Convert the response into a binary measurement: 1 for "Yes", 0 for "No".
        binary_measurement = 1 if observed_response == Response.Yes else 0

        # Compute the innovation (measurement residual): difference between the actual response and the predicted probability.
        measurement_residual = binary_measurement - predicted_probability

        # The variance for a Bernoulli process is p*(1-p); we clamp it to avoid near-zero values.
        measurement_noise_variance = max(predicted_probability * (1 - predicted_probability), 1e-4)

        # Compute the innovation covariance.
        innovation_covariance = measurement_jacobian @ current_covariance @ measurement_jacobian.T + measurement_noise_variance
        
        # If the innovation covariance is extremely small, clamp it to avoid division by zero.
        innovation_covariance_value = max(innovation_covariance[0, 0], 1e-4)
        
        # Calculate the Kalman gain.
        kalman_gain = current_covariance @ measurement_jacobian.T / innovation_covariance_value
        
        # Apply an update factor to make the update more aggressive if desired
        effective_gain = self.update_factor * kalman_gain

        # Update the threshold estimates.
        updated_thresholds = current_thresholds + effective_gain[:, 0] * measurement_residual

        # Update the covariance matrix.
        identity_matrix = np.eye(num_dimensions)
        updated_covariance = (identity_matrix - kalman_gain @ measurement_jacobian) @ current_covariance

        return updated_thresholds, updated_covariance