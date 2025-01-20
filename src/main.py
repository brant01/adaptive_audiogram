import sys
from pathlib import Path

# Dynamically add the project root directory to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import using absolute imports
from src.gui.app import start_app

def main():
    print(f"Running from project root: {project_root}")
    start_app()

if __name__ == "__main__":
    main()
    