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
        self.geometry("800x600")
    
        # Container to stack frames
        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        
        # Dictionary to store frames
        self.frames = {}
        
        # Initialize all frames
        for F in (MainMenu, TestMode, TrainMode, CalibrationMode):
            frame = F(self.container, self)
            print(f"Initializing frame: {F.__name__}, widget path: {str(frame)}")
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
            
        # Start with Main Menu    
        self.show_frame(MainMenu)
                  
    def show_frame(self, screen_class) -> None:
        frame = self.frames[screen_class]
        frame.tkraise() # Bring chosen frame to front
            
if __name__ == '__main__':
    main()