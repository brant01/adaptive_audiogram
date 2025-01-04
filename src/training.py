import tkinter as tk

class TrainMode(tk.Frame):
    """
    Train mode class
    """
    def __init__(self, parent: tk.Widget, controller: tk.Tk) -> None:
        super().__init__(parent)
        self.controller = controller