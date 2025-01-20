
import numpy as np
import sounddevice as sd

from src.utils.enums import Ear
from src.utils.constants import CALIBRATION_DICT


class Sound():
    """
    Sound class
    """
    
    def __init__ (self,
                  frequency: int,
                  volume: int,
                  ear: Ear,
                  sample_rate: int = 44100,
                  duration: int = 1,):
        
        self.frequency: int = frequency
        self.volume: int = volume
        self.ear: Ear = ear
        self.calibration_dict = CALIBRATION_DICT #TODO: Fix this
        self.sample_rate: int = sample_rate
        self.duration: int = duration
        self.signal: np.ndarray = self.generate_signal()
        
    def generate_signal(self) -> np.ndarray:
        """
        Generate a sine wave signal
        """
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        
        signal = self.volume * np.sin(2 * np.pi * self.frequency * t)
        
        # return an array of two signals, one for each ear
        # if the ear is left, the left signal is the sound and the right signal is zero
        
        if self.ear == Ear.LEFT:
            return np.array([signal, np.zeros_like(signal)])
        elif self.ear == Ear.RIGHT:
            return np.array([np.zeros_like(signal), signal])
        else: # Ear.BOTH
            return np.array([signal, signal])
        
    def play_sound(self) -> None:
        """
        Play the sound
        """
        sd.play(self.signal * self.calibration_dict[str(self.frequency)], samplerate=self.sample_rate)