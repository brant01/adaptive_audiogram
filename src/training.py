import tkinter as tk

from testing import TestMode
from enums import  TrialType

class TrainMode(TestMode):
    """
    Train mode class
    """
    def __init__(self, parent: tk.Widget, controller: tk.Tk) -> None:
        super().__init__(parent, controller)
        self.type = TrialType.Train