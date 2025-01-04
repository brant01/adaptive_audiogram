import tkinter as tk

from main_menu import MainMenu
from testing import TestMode
from training import TrainMode
from calibrate import CalibrationMode

def main() -> None:
    app = Application()
    app.mainloop()
            
class Application(tk.Tk):
    """
    Main application class
    """
    def __init__(self) -> None:
        super().__init__()
        self.title("Adaptive Audiogram")
    
        # Container to stack frames
        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        
        # Dictionary to store frames
        self.frames = {}
        
        # Initialize all frames
        for F in (MainMenu, TestMode, TrainMode, CalibrationMode):
            frame = F(self.container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
            
        # Start with Main Menu    
        self.show_main_menu()
                  
    def show_main_menu(self) -> None:
        """Show the main menu."""
        self.frames[MainMenu].tkraise()

    def show_test_mode(self) -> None:
        """Show the test mode."""
        self.frames[TestMode].tkraise()

    def show_train_mode(self) -> None:
        """Show the training mode."""
        self.frames[TrainMode].tkraise()

    def show_calibration_mode(self) -> None:
        """Show the calibration mode."""
        self.frames[CalibrationMode].tkraise()
            
if __name__ == '__main__':
    main()