import tkinter as tk

class MainMenu(tk.Frame):
    """
    Main menu class
    """
    def __init__(self, parent: tk.Widget, controller: tk.Tk) -> None:
        super().__init__(parent)
        self.controller = controller
        self.display_main_menu()
        
    def display_main_menu(self) -> None:

        # Add menu components
        tk.Label(self.controller, text="Main Menu", font=("Arial", 20)).pack(pady=20)
 
        tk.Button(self.controller, text="Test", command=self.temp, width=15).pack(pady=10)
        tk.Button(self.controller, text="Train Model", command=self.temp, width=15).pack(pady=10)
        tk.Button(self.controller, text="Calibrate Headphones", command=self.temp, width=15).pack(pady=10)
        tk.Button(self.controller, text="Exit", command=self.controller.quit, width=15).pack(pady=20)
        
    def temp(self) -> None:
        print("Test")