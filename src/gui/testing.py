from pathlib import Path
from tkinter import ttk
from src.utils.enums import Response, TrialType
import tkinter as tk
from src.gui.experiment import Experiment

class TestMode(tk.Frame):
    """
    Test mode class
    """
    def __init__(self, parent: tk.Widget, controller: tk.Tk) -> None:
        super().__init__(parent)
        self.controller = controller
        self.type = TrialType.Test
        self.selected_file = tk.StringVar(value="")  # For the selected calibration file

        # Patient ID Label
        self.patient_label = tk.Label(self, text="Patient ID:", font=("Arial", 14))
        self.patient_label.pack(pady=(20, 5))

        # Patient ID Entry
        self.patient_id_var = tk.StringVar()
        self.patient_id_entry = tk.Entry(self, textvariable=self.patient_id_var, font=("Arial", 14))
        self.patient_id_entry.pack(pady=5)
        self.patient_id_var.trace_add("write", self.validate_inputs)

        # Calibration File Dropdown Label
        self.file_label = tk.Label(self, text="Select Calibration File:", font=("Arial", 14))
        self.file_label.pack(pady=(20, 5))

        # Calibration File Dropdown
        self.file_dropdown = ttk.Combobox(
            self,
            textvariable=self.selected_file,
            font=("Arial", 12),
            state="readonly"
        )
        self.file_dropdown.pack(pady=5)
        self.populate_calibration_files()
        self.file_dropdown.bind("<<ComboboxSelected>>", self.validate_inputs)

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

    def populate_calibration_files(self) -> None:
        """Populate the dropdown with calibration file names."""
        calibration_dir = Path(__file__).resolve().parent.parent / "calibration_files"
        files = calibration_dir.glob("*.json")  # List all .json files
        file_names = [file.stem for file in files]  # Extract file names without extensions
        self.file_dropdown["values"] = file_names  # Populate the dropdown

    def validate_inputs(self, *args) -> None:
        """Validate Patient ID and calibration file selection."""
        patient_id = self.patient_id_var.get()
        selected_file = self.selected_file.get()

        if not patient_id.isdigit():
            self.start_button.config(state=tk.DISABLED)
            self.validation_label.config(text="Patient ID must be a valid integer.")
            return

        if not selected_file:
            self.start_button.config(state=tk.DISABLED)
            self.validation_label.config(text="A calibration file must be selected.")
            return

        # If both inputs are valid, enable the Start Test button
        self.start_button.config(state=tk.NORMAL)
        self.validation_label.config(text="")

    def start_test(self) -> None:
        """Initialize the test UI with Patient ID and calibration file."""
        patient_id = int(self.patient_id_var.get())  # Convert to integer
        selected_file = self.selected_file.get()  # Get selected calibration file
        print(f"Starting test for Patient ID: {patient_id} using {selected_file} calibration.")

        # Load calibration file data (future functionality)
        calibration_dir = Path(__file__).resolve().parent.parent.parent / "calibration_files"
        calibration_file = calibration_dir / f"{selected_file}.json"

        # Initialize Experiment with patient ID and calibration file
        self.exp = Experiment(self.type, patient_id=patient_id, calibration_file=Path(calibration_file))

        # Remove Start Test UI
        self.start_button.destroy()
        self.patient_label.destroy()
        self.patient_id_entry.destroy()
        self.file_label.destroy()
        self.file_dropdown.destroy()
        self.validation_label.destroy()

        # Add test buttons (Play, Yes, No, Stop)
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
            command=lambda: self.record_response(Response.Yes)
        )
        self.yes_button.grid(row=1, column=0, padx=20, pady=10)

        self.no_button = tk.Button(
            self,
            text="No",
            font=("Arial", 14),
            command=lambda: self.record_response(Response.No)
        )
        self.no_button.grid(row=1, column=1, padx=20, pady=10)

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

    def record_response(self, decision: Response) -> None:
        """Record the user's decision and configure the next sound."""
        self.exp.record(decision)
        self.exp.select_sound()
        print(f"Decision recorded: {decision}")

    def stop_test(self) -> None:
        """Stop the test and return to the main menu."""
        print("Test stopped")
        self.controller.show_main_menu()