import tensorflow as tf
from hamiltonians.ising import Ising1D
from sampler.sampler import MetropolisHastingsSampler

class OptimizerWrapper:
    """ This class implements a wrapper for a TensorFlow optimizer object.

    Call Arguments
    --------------
    hamiltonian: Ising1D
        The Hamiltonian object for the system.
        (Presently only the Ising1D Hamiltonian is supported)
    nqs
        The neural network quantum state, Ψ. Any TensorFlow Model object is supported.
    optimizer
        A TensorFlow Optimizer object.
    sampler_params
        A dictionary containing sampler parameters.
    """
    def __init__(
        self,
        hamiltonian: Ising1D,
        nqs: tf.keras.models.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        sampler_params: dict = None
    ):
        self.hamiltonian = hamiltonian
        self.nqs = nqs
        self.optimizer = optimizer
        self.sampler = None

        if sampler_params is None:
            sampler_params = {
                'num_sweeps': 1000,
                'therm_factor': 0.,
                'sweep_factor': 1,
                'num_flips': None
            }

        # Sampler parameters
        self.num_sweeps = sampler_params['num_sweeps']
        self.therm_factor = sampler_params['therm_factor']
        self.sweep_factor = sampler_params['sweep_factor']
        self.num_flips = sampler_params['num_flips']

    def calc_grads(self, num_sweeps):
        """ Calculates the gradients of the network which are passed to the
        optimizer.

        For a detailed description of the gradient calculation, see 
        https://doi.org/10.1002/adts.202000269
        """
        dpsis = self.sampler.dpsi_list
        nqs_energy = self.sampler.nqs_energy
        elocs_times_dpsi = self.sampler.eloc_times_dpsi_list

        gradient = (2/num_sweeps) * (
            [sum(col) for col in zip(*elocs_times_dpsi)]
        - nqs_energy.numpy() * [sum(col) for col in zip(*dpsis)]
        )
        return gradient

    def run(self, num_epochs):
        """ Runs the optimizer.
        """
        init_state = None
        for epoch in range(num_epochs):
            print(f"\n\nEpoch {epoch+1}/{num_epochs}")
            self.sampler = MetropolisHastingsSampler(
                hamiltonian=self.hamiltonian,
                nqs=self.nqs,
                init_state=init_state
                )
            self.sampler.run(num_sweeps=self.num_sweeps)
            init_state = self.sampler.current_state
            weights = self.nqs.trainable_variables
            grads = self.calc_grads(self.num_sweeps)
            self.optimizer.apply_gradients(zip(grads, weights))

        