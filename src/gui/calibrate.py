from pathlib import Path
import json
import tkinter as tk

from src.utils.constants import FREQUENCIES, CALIBRATION_DIR, DEFAULT_CALIBRATION_FILE

def create_default_calibration() -> dict:
    """
    Create a default calibration dictionary.
    Frequencies map to a default calibration value of 1.0.
    """
    return {freq: 1.0 for freq in FREQUENCIES}


def save_calibration(calibration: dict, filename: Path = DEFAULT_CALIBRATION_FILE) -> None:
    """
    Save the calibration dictionary to a JSON file.
    """
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists
    with filename.open("w") as f:
        json.dump(calibration, f, indent=4)
    print(f"Calibration saved to {filename}")


def load_calibration(filename: Path = DEFAULT_CALIBRATION_FILE) -> dict:
    """
    Load the calibration dictionary from a JSON file.
    If the file does not exist, create a default calibration.
    """
    if not filename.exists():
        print("Calibration file not found. Creating default calibration.")
        calibration = create_default_calibration()
        save_calibration(calibration, filename)
        return calibration

    with filename.open("r") as f:
        return json.load(f)

class CalibrationMode(tk.Frame):
    """
    Calibration mode class
    """
    def __init__(self, parent: tk.Widget, controller: tk.Tk) -> None:
        super().__init__(parent)
        self.controller = controller
        
if __name__ == '__main__':
    print("Creating default calibration...")
    default_dict = create_default_calibration()
    save_calibration(default_dict)