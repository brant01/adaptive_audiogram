from dataclasses import dataclass
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