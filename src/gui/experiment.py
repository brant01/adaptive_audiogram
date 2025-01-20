

from pathlib import Path

from src.models.bayes_model import bayesian_model as bayes
from src.gui.calibrate import load_calibration
from src.utils.enums import TrialType, ModelType
from src.sound_utils.sound import Sound
from src.utils.enums import Response
    
class Experiment():
    """
    Experiment class
    """
    def __init__(self, type: TrialType, 
                 patient_id: int, 
                 calibration_file: Path,
                 model: ModelType) -> None:
        
        self.type = type
        self.stop_test = False # Flag for ending test
        self.patient_id = patient_id
        self.calibration_dict = load_calibration(calibration_file)
        self.sound = self.get_first_sound()
        self.model = model
    
    def get_first_sound(self) -> Sound:
        
        match self.model:
            case ModelType.Bayesian:
                return bayes.get_inital_sound()
    
    def play_sound(self) -> None:
        
        self.sound.play_sound()
    
    def select_next_sound(self) -> Sound:
        
        match self.model:
            case ModelType.Bayesian:
                self.sound = bayes.get_next_sound()
        
    def record(self, decision: Response) -> None:
        print(f"Recording decision: {decision.name}")
        return None

