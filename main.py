
from src.gui.app import start_app
from src.models.bayes_model.bayesian_model import BayesianAdaptiveModel
from src.simulation.simulation_model import ManualSimulation

def main():
    start_app()

def manual_sim_bayes():
    simulation = ManualSimulation(BayesianAdaptiveModel)
    
    simulation.run_model()
    
    
if __name__ == "__main__":
    manual_sim_bayes()
    