from pathlib import Path

# Frequencies in Hz per ear, left then right
FREQUENCIES = [250, 250, 500, 500, 1000, 1000, 2000, 2000, 4000, 4000, 8000, 8000]  # Hz


LOUDNESSES = [x for x in range(-20, 121, 5)]  # -10 to 120 dB in 5 dB intervals

CALIBRATION_DICT = {} #TODO: Fix this
CALIBRATION_DIR = Path(__file__).resolve().parent.parent / "calibration_files"
DEFAULT_CALIBRATION_FILE = CALIBRATION_DIR / "default_calibration.json"