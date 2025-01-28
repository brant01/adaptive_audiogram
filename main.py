
from src.gui.app import start_app
from src.models.bayes_model.bayesian_model import BayesianAdaptiveModel
from src.simulation.simulation_model import ManualSimulation, AutomatedSimulation

def main():
    start_app()

def manual_sim_bayes():
    simulation = ManualSimulation(BayesianAdaptiveModel, threshold=4)
    
    simulation.run_model()
    
def auto_sim_bayes():
    # Initialize the automated simulation
    automated_sim = AutomatedSimulation(model_class=BayesianAdaptiveModel, threshold=5)

    # Run the simulation on the dataset
    automated_sim.run_model()

if __name__ == "__main__":
    #auto_sim_bayes()
    manual_sim_bayes()
    