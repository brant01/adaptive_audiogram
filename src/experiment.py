

import numpy as np
import sounddevice as sd

from enums import Ear, TrialType
from d_classes import Sound
from testing import Decision
    
class Experiment():
    """
    Experiment class
    """
    def __init__(self, type: TrialType, patient_id: int) -> None:
        self.type = type
        self.stop_test = False # Flag for ending test
        self.patient_id = patient_id
    
    def play_sound(self) -> None:
        
        sound = self.select_sound()
        
        sd.play(sound.signal, samplerate=sound.sample_rate)
        #sd.wait()
    
    def select_sound(self) -> Sound:
        
        sound = Sound(
            duration=1,
            frequency=1000,
            volume=0.5,
            ear=Ear.LEFT,
            sample_rate=44100,
            signal=None
        )
        
        t = np.linspace(0, sound.duration, int(sound.sample_rate * sound.duration), endpoint=False)  # Time array
        sound.signal = sound.volume * np.sin(2 * np.pi * sound.frequency * t)  # Sine wave
        
        return sound
        
    def record(self, decision: Decision) -> None:
        print(f"Recording decision: {decision.name}")
        return None

