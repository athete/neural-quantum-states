from hamiltonians.ising import Ising1D
import numpy as np
import tensorflow as tf
import os

class MetropolisHastingsSampler:
    """ Implements Markov Chain Monte Carlo Sampling of the wavefunction.

    This class implements a Varitional Monte Carlo Sampler that uses the Metropolis-Hastings
    algorithm. When instantiated, the current state of the sampler is randomly initialized. 

    The sampler runs for `num_sweeps` iterations, that is `num_sweeps` samples are drawn for Monte-Carlo
    statistics. In each run, the sampler burns in or discards the first `sweep_factor * num_spins` samples 
    drawn. With the last drawn state, the sampler computes the local energy and simultaneously also computes the
    quantity d(ln Ψ) which is used by the optimizer. 

    Call Arguments
    --------------
    hamiltonian: Ising1D
        The Hamiltonian object for the system.
        (Presently only the Ising1D Hamiltonian is supported)
    nqs
        The neural network quantum state, Ψ. Any TensorFlow Model object is supported.
    init_state
        A state to initialize the sampler.
    write_energies
        If True, writes the computed ground state energy to ./data/energies.txt
    """
    def __init__(
        self,
        hamiltonian: Ising1D,
        nqs: tf.keras.models.Model,
        init_state=None,
        write_energies: bool=True
    ):
        self.hamiltonian = hamiltonian
        self.num_spins = self.hamiltonian.get_spins()
        self.nqs = nqs

        # Stores the states admitted by the sampler
        self.state_history = []
        # Stores the computed local energies
        self.local_energies = []
        # Stores the computed d(log Ψ) values
        self.dpsi_list = []
        # Stores local energy * d(log Ψ) values
        self.eloc_times_dpsi_list = []

        # The computed ground state energy 
        self.nqs_energy = 0
        # The computed error in the ground state energy
        self.nqs_energy_error = 0
        # The last computed local energy
        self.current_eloc = 0
        
        # The random number generator object for the sampler
        self.rng = np.random.default_rng()
        self.write_energies = write_energies
        
        if init_state is None:
            self.init_random_state()
        else:
            self.current_state = init_state

        if self.write_energies is True:
            if not os.path.exists('.data/'):
                os.makedirs('./data', exist_ok=True)

    
    def init_random_state(self):
        """ Initializes the current state of the sampler to a randomly generated state. """
        self.current_state = self.rng.uniform(size=[1, self.num_spins])
        self.current_state = np.where(self.current_state < 0.5, -1.0, 1.0)

    def choose_random_site(self):
        """ Chooses a random site in the configuration/state to flip. """
        return [self.rng.integers(low=0, high=self.num_spins - 1)]
    
    def reset_state_history(self):
        """ Resets the state history """
        self.state_history = []
    
    def flip_spin(self):
        """ Given a randomly chosen site, flips the spin at that site. """
        site = self.choose_random_site()
        candidate = np.copy(self.current_state)
        candidate[0][site] *= -1.0

        return candidate

    def amplitude_ratio(self, state, candidate):
        """ Given a two states 'state' and 'candidate', computes Ψ(candidate)/Ψ(state)
        """
        a = self.nqs(candidate)
        b = self.nqs(state)
        return tf.math.divide(a, b)

    def move(self):
        """ Defines one move of the sampler.

        A move consists of flipping a random site in the current state to generate
        a candidate state. If the ratio |Ψ(candidate)/Ψ(state)|^2 > uniform(0, 1), the 
        candidate state is accepted and set to the current state in the Markov chain.
        """
        candidate = self.flip_spin()
        psi_ratio = self.amplitude_ratio(self.current_state, candidate)
        accept_prob = tf.math.square(tf.math.abs(psi_ratio))
        if accept_prob.numpy() > self.rng.random():
            self.current_state = candidate

    def burn_in(self, sweep_factor):
        """ Burns in the sampler.

        Burn-in is performed by running the sampler for `sweep_factor * num_spins` iterations
        but discarding the drawn samples by not counted them towards any Monte-Carlo statistics. 
        """
        for _ in range(sweep_factor * self.num_spins):
                self.move()
    
    def run(self, num_sweeps, sweep_factor=1):
        """ Defines one run of the sampler. 

        One run of the sampler consists of performing the following steps `num_sweeps` times:
        1. Performing a burn-in
        2. Computing the local energy using the last drawn state
        3. Computing the quantities d(log Ψ) and local energy * d(log Ψ).
        """

        print("Starting Monte-Carlo Sampling...")
        print(f"{num_sweeps} sweeps will be perfomed.")

        self.reset_state_history()

        for _ in range(num_sweeps):
            self.burn_in(sweep_factor)
            self.current_eloc = self.find_local_energy()
            self.local_energies.append(self.current_eloc)
            self.state_history.append(self.current_state)
            self.dpsi_over_psi()
        
        print('Completed Monte-Carlo Sampling.')

        self.estimate_ground_energy()

    def dpsi_over_psi(self):
        """ Computes the quantity d(log Ψ) ≡ dΨ/Ψ and local energy * d(log Ψ)

        For a detailed description of the procedure involved, see 
        https://doi.org/10.1002/adts.202000269
        """
        with tf.GradientTape() as tape:
            psi = self.nqs(self.current_state)
        weights = self.nqs.trainable_variables
        dpsi = tape.gradient(psi, weights)
        self.dpsi_list.append([(1/psi.numpy())*grad for grad in dpsi])
        self.eloc_times_dpsi_list.append([(self.current_eloc.numpy()/psi.numpy())*grad for grad in dpsi])


    def find_local_energy(self):
        """ Computes the local energy for a drawn state. """
        state = self.current_state
        (mat_elements, spin_flip_sites) = self.hamiltonian.find_nonzero_elements(state)

        flipped_states = [np.copy(state) for _ in spin_flip_sites]
        for i, site in enumerate(spin_flip_sites):
            flipped_states[i][0][site] *= -1

        energies = [self.amplitude_ratio(state, flipped_states[i])* element for (i, element) in enumerate(mat_elements)]
        return sum(energies)

    
    def estimate_ground_energy(self):
        """ Computes a stochastic estimate of the ground state energy. 

        This computations uses blocking to account for autocorrelation between the drawn samples.
        """
        nblocks = 50
        blocksize = len(self.local_energies) // nblocks
        enmean = 0
        enmeansq = 0
        enmean_unblocked = 0
        enmean_sq_unblocked = 0

        for block in range(nblocks):
            eblock = 0
            for j in range(block*blocksize, (block + 1) * blocksize):
                eblock += self.local_energies[j]
                delta = self.local_energies[j] - enmean_unblocked
                enmean_unblocked += delta / (j + 1)
                delta2 = self.local_energies[j] - enmean_unblocked
                enmean_sq_unblocked += delta * delta2
            eblock /= blocksize
            delta = eblock - enmean
            enmean += delta / (block + 1)
            delta2 = eblock - enmean
            enmeansq += delta * delta2

        enmeansq /= (nblocks - 1)
        enmean_sq_unblocked /= (nblocks * blocksize - 1)
        est_avg = enmean / self.num_spins
        est_error = np.sqrt(enmeansq / nblocks) / self.num_spins

        self.nqs_energy = enmean
        self.nqs_energy_error = np.sqrt(enmeansq / nblocks)
        print(f"Estimated ground state energy: {self.nqs_energy} +/- {self.nqs_energy_error}")
        energy_report = f"Estimated average energy per spin: {est_avg} +/- {est_error}"
        print(energy_report)

        if self.write_energies:
            with open('./data/energies.txt', 'ab') as f:
                np.savetxt(f, np.array([self.nqs_energy]))

        bin_report = f'Error estimated with binning analysis consisting of {nblocks} bins of {blocksize} samples each.'
        print(bin_report)

        self.correlation_time = 0.5 * blocksize * enmeansq / enmean_sq_unblocked
        autocorrelation_report = f'Estimated autocorrelation time is {self.correlation_time}'

        print(autocorrelation_report)