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
        tk.Label(self, text="Main Menu", font=("Arial", 20)).pack(pady=20)

        # Buttons to navigate between modes
        tk.Button(
            self,
            text="Test",
            command=self.controller.show_test_mode,  # Call the controller's method
            width=15
        ).pack(pady=10)

        tk.Button(
            self,
            text="Train",
            command=self.controller.show_train_mode,  # Call the controller's method
            width=15
        ).pack(pady=10)

        tk.Button(
            self,
            text="Calibrate",
            command=self.controller.show_calibration_mode,  # Call the controller's method
            width=15
        ).pack(pady=10)

        tk.Button(
            self,
            text="Exit",
            command=self.controller.quit,
            width=15
        ).pack(pady=20)