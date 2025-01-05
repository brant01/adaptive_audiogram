from dataclasses import dataclass
import numpy as np

from enums import Ear

@dataclass
class Sound():
    """
    Sound class
    """
    duration: int # in seconds
    frequency: int
    volume: float
    ear: Ear
    sample_rate: int = 44100
    signal: np.ndarray = None