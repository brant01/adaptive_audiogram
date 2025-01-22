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
    def __init__(self, threshold: int = 1):
        """
        Initializes the BayesianAdaptiveModel with default means and covariance matrix.

        Args:
            threshold (int): Stopping criterion for uncertainty (in dB).
        """
        super().__init__(threshold)
        try:
            self.means = compute_frequency_ear_means().copy()
            self.cov_matrix = get_covariance_ear_matrix().copy()
        except Exception as e:
            raise ValueError(f"Error initializing means or covariance: {e}")
        
        self.sound = self._select_next_sound() # initial sound selection from baseline mean/cov matrix
        self.stop_flag = False # when all uncertainties are below threshold
    

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
        self.means, self.cov_matrix = BayesianAdaptiveModel.ekf_update_yes_no(
            self.means,
            self.cov_matrix,
            freq_idx,
            self.sound.volume,
            response,
            alpha=0.2
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
        self.stop_flag = np.all(uncertainties <= self.threshold)

        # If not stopping, select next sound
        if not self.stop_flag:
            self.sound = self._select_next_sound()

    def get_results(self):
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
        
        return list(zip(labels, rounded_means))

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

    @staticmethod
    def _psychometric_probability(volume, threshold, alpha=0.2):
        s = BayesianAdaptiveModel._sigmoid(alpha * (volume - threshold))
        return s
        
    def ekf_update_yes_no(mu, Sigma, i, volume, response, alpha=0.2):
        """
        mu:    (D,) current mean
        Sigma: (D,D) current covariance
        i:     dimension index for the tested frequency/ear
        volume: the dB level tested
        response: 1 if yes, 0 if no

        Returns:
        (mu_new, Sigma_new)
        """
        D = len(mu)

        # Predicted probability of yes at the current mean
        p_yes = BayesianAdaptiveModel._psychometric_probability(volume, mu[i], alpha=alpha)

        # Jacobian: derivative of h wrt dimension i
        # h(\theta_i) = p_yes => derivative = -alpha * p_yes * (1 - p_yes)
        dh_dtheta_i = -alpha * p_yes * (1.0 - p_yes)

        # H is 1 x D, all zeros except H[0, i] = dh_dtheta_i
        H = np.zeros((1, D))
        H[0, i] = dh_dtheta_i

        # Observed measurement: y in {0,1}
        y = 1 if response == Response.Yes else 0
        # Innovation
        innovation = y - p_yes  # scalar

        # The local variance of a Bernoulli with mean p_yes is p_yes*(1 - p_yes)
        R = p_yes*(1 - p_yes)
        # We can clamp R to avoid numerical blow-ups if p_yes is near 0 or 1
        R = max(R, 1e-4)

        # S = H * Sigma * H.T + R
        S = H @ Sigma @ H.T + R  # shape (1,1)
        # Kalman gain K = Sigma * H.T * (S^-1)
        K = Sigma @ H.T / S[0,0]  # shape (D,1)

        # Updated mean
        mu_new = mu + (K[:,0] * innovation)
        # Updated covariance
        I = np.eye(D)
        Sigma_new = (I - K @ H) @ Sigma

        return mu_new, Sigma_new