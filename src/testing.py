import tkinter as tk

class TestMode(tk.Frame):
    """
    Test mode class
    """
    def __init__(self, parent: tk.Widget, controller: tk.Tk) -> None:
        super().__init__(parent)
        self.controller = controller