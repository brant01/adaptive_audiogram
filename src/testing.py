from enums import TrialType, Decision
import tkinter as tk
from experiment import Experiment


class TestMode(tk.Frame):
    """
    Test mode class
    """
    def __init__(self, parent: tk.Widget, controller: tk.Tk) -> None:
        super().__init__(parent)
        self.controller = controller
        self.type = TrialType.Test

        # Patient ID Label
        self.patient_label = tk.Label(self, text="Patient ID:", font=("Arial", 14))
        self.patient_label.pack(pady=(20, 5))

        # Patient ID Entry
        self.patient_id_var = tk.StringVar()
        self.patient_id_entry = tk.Entry(self, textvariable=self.patient_id_var, font=("Arial", 14))
        self.patient_id_entry.pack(pady=5)
        self.patient_id_var.trace_add("write", self.validate_patient_id)

        # Validation Message
        self.validation_label = tk.Label(self, text="", font=("Arial", 12), fg="red")
        self.validation_label.pack(pady=5)

        # Start Test Button (Initially Disabled)
        self.start_button = tk.Button(
            self,
            text="Start Test",
            font=("Arial", 18),
            command=self.start_test,
            state=tk.DISABLED  # Disabled by default
        )
        self.start_button.pack(pady=20)

    def validate_patient_id(self, *args) -> None:
        """Validate the Patient ID input."""
        patient_id = self.patient_id_var.get()
        if patient_id.isdigit():  # Check if the input is a valid integer
            self.start_button.config(state=tk.NORMAL)
            self.validation_label.config(text="")  # Clear any validation message
        else:
            self.start_button.config(state=tk.DISABLED)
            self.validation_label.config(text="Patient ID must be a valid integer.")

    def start_test(self) -> None:
        """Initialize the test UI with Patient ID."""
        patient_id = int(self.patient_id_var.get())  # Convert to integer
        print(f"Starting test for Patient ID: {patient_id}")
        self.exp = Experiment(self.type, patient_id=patient_id)  # Pass Patient ID to Experiment

        # Remove Start Test UI
        self.start_button.destroy()
        self.patient_label.destroy()
        self.patient_id_entry.destroy()
        self.validation_label.destroy()

        # Play Sound button
        self.play_button = tk.Button(
            self,
            text="Play Sound",
            bg="red",
            font=("Arial", 15),
            command=self.play_sound
        )
        self.play_button.grid(row=0, column=0, columnspan=2, pady=20)

        # Yes button
        self.yes_button = tk.Button(
            self,
            text="Yes",
            font=("Arial", 14),
            command=lambda: self.record_response(Decision.Yes)
        )
        self.yes_button.grid(row=1, column=0, padx=20, pady=10)

        # No button
        self.no_button = tk.Button(
            self,
            text="No",
            font=("Arial", 14),
            command=lambda: self.record_response(Decision.No)
        )
        self.no_button.grid(row=1, column=1, padx=20, pady=10)

        # Stop Test button
        self.stop_button = tk.Button(
            self,
            text="Stop Test",
            font=("Arial", 14),
            command=self.stop_test
        )
        self.stop_button.grid(row=2, column=0, columnspan=2, pady=20)

    def play_sound(self) -> None:
        """Play a sound."""
        self.play_button.config(bg="green")  # Indicate sound is playing
        self.exp.play_sound()  # Play sound using the Experiment class
        self.play_button.config(bg="red")  # Reset button color after playback

    def record_response(self, decision: Decision) -> None:
        """Record the user's decision and configure the next sound."""
        self.exp.record(decision)  # Record the response in Experiment
        self.exp.select_sound()  # Configure the next sound
        print(f"Decision recorded: {decision}")

    def stop_test(self) -> None:
        """Stop the test and return to the main menu."""
        print("Test stopped")
        self.controller.show_main_menu()  # Navigate back to the main menu