from nqs import Ising1DHamiltonian
from nqs import NeuralQuantumState
from nqs import StochasticReconfigOptimizer

def runOptimizer():
    # Define the Hamiltonian
    hamiltonian = Ising1DHamiltonian(40, 0.5, True)

    # Instantiate the Neural Network
    nqs = NeuralQuantumState()

    # Define the parameters for the Monte-Carlo Sampling
    sampler_params = {'numSweeps': 10000, 'thermFactor': 0., 'sweepFactor': 1, 'numFlips': 1}

    # Instantiate the optimizer
    optimizer = StochasticReconfigOptimizer(hamiltonian, nqs, samplerParams=sampler_params, learningRate=0.01)

    # Run the optimizer
    optimizer.run(100)

if __name__ == '__main__':
    runOptimizer()