from enums import TrialType, Decision
import tkinter as tk


from experiment import Experiment 
#from main_menu import MainMenu


class TestMode(tk.Frame):
    """
    Test mode class
    """
    def __init__(self, parent: tk.Widget, controller: tk.Tk) -> None:
        super().__init__(parent)
        self.controller = controller
        self.type = TrialType.Test
        self.exp = Experiment(self.type)
        self.running = False
        
        # Start test button
        self.start_button = tk.Button(
            self,
            text="Start Test",
            font=("Arial", 18),
            command=self.start_test
        )
        self.start_button.pack(pady=50)  # Center it initially
        
    def start_test(self) -> None:
        """Begin the test and set up the UI."""
        self.running = True
        self.start_button.destroy()  # Remove the Start button

        # Add Play, Yes, No, and Stop buttons
        self.play_button = tk.Button(
            self,
            text="Play Sound",
            bg="red",
            font=("Arial", 15),
            command=self.play_sound
        )
        self.play_button.grid(row=0, column=0, columnspan=2, pady=20)

        self.yes_button = tk.Button(
            self,
            text="Yes",
            font=("Arial", 14),
            command=lambda: self.record_response(Decision.Yes)
        )
        self.yes_button.grid(row=1, column=0, padx=20, pady=10)

        self.no_button = tk.Button(
            self,
            text="No",
            font=("Arial", 14),
            command=lambda: self.record_response(Decision.No)
        )
        self.no_button.grid(row=1, column=1, padx=20, pady=10)

        self.stop_button = tk.Button(
            self,
            text="Stop Test",
            font=("Arial", 14),
            command=self.stop_test
        )
        self.stop_button.grid(row=2, column=0, columnspan=2, pady=20)

        # Start the test loop
        self.test_loop()

    def play_sound(self) -> None:
        """Play a sound and update the button state."""
        self.play_button.config(bg="green")
        self.exp.play_sound()  # Assume Experiment has a play_sound method
        self.play_button.config(bg="red")

    def record_response(self, decision: Decision) -> None:
        """Record the user's decision."""
        self.exp.record(decision)

    def test_loop(self) -> None:
        """Simulate the test loop."""
        if self.running:
            # Perform periodic updates or checks here
            print("Test is running...")
            self.after(500, self.test_loop)  # Schedule next check in 500ms

    def stop_test(self) -> None:
        """Stop the test and reset the UI."""
        self.running = False
        print("Test stopped")
        self.controller.show_frame(MainMenu)  # Return to main menu or reset the UI