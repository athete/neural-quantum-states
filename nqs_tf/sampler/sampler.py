import numpy as np
import tensorflow as tf

class MetropolisHastingsSampler:
    def __init__(
        self,
        hamiltonian,
        nqs,
        init_state=None
    ):
        self.hamiltonian = hamiltonian
        self.num_spins = self.hamiltonian.num_spins
        self.nqs = nqs
        self.num_moves = 0
        self.num_acceptances = 0
        self.state_history = []
        self.local_energies = []
        self.nqs_energy = []
        self.nqs_energy_error = []
        self.current_eloc = 0
        self.current_energy = 0
        

        if init_state is None:
            self.init_random_state()
        else:
            self.current_state = init_state

    
    def init_random_state(self):
        self.current_state = np.random.uniform(size=[1, self.num_spins])
        self.current_state = np.where(self.current_state < 0.5, -1.0, 1.0)

    def choose_random_site(self):
        return [np.random.randint(low=0, high=self.num_spins - 1)]
    
    def acceptance_rate(self):
        return self.num_acceptances / self.num_moves
    
    def reset_sampler_stats(self):
        self.num_moves = 0
        self.num_acceptances = 0
        self.state_history = []
    
    def flip_spin(self):
        site = self.choose_random_site()
        candidate_ = np.copy(self.current_state)
        candidate_[0][site] *= -1.0

        return candidate_

    def amplitude_ratio(self, state, candidate):
        a = self.nqs(candidate)
        b = self.nqs(state)

        a_sum = tf.reduce_sum(tf.math.square(a))
        b_sum = tf.reduce_sum(tf.math.square(b))
        ratio = tf.math.divide(a_sum, b_sum)
        return ratio

    def move(self):
    
        candidate_ = self.flip_spin()
        psi_ratio = self.amplitude_ratio(self.current_state, candidate_)
        accept_prob = tf.math.square(tf.math.abs(psi_ratio))
        if accept_prob > np.random.random():
            self.current_state = candidate_
            self.num_acceptances += 1
        
        self.num_moves += 1

    def thermalize(self, sweep_factor):
        for _ in range(sweep_factor * self.num_spins):
                self.move()
    
    def run(self, num_sweeps, tape, sweep_factor=1):

        print("Starting Monte-Carlo Sampling...")
        print(f"{num_sweeps} sweeps will be perfomed.")

        self.reset_sampler_stats()

        for _ in range(num_sweeps):
            with tape.stop_recording():
                self.thermalize(sweep_factor)
            self.current_eloc = self.find_local_energy()
            self.local_energies.append(self.current_eloc)
            self.state_history.append(self.current_state)
        print('Completed Monte-Carlo Sampling.')

        self.estimate_ground_energy()

        return self.current_energy

    def find_local_energy(self):
        state = self.current_state
        (mat_elements, spin_flip_sites) = self.hamiltonian.find_nonzero_elements(state)

        flipped_states = [np.copy(state) for _ in spin_flip_sites]
        for i, site in enumerate(spin_flip_sites):
            flipped_states[i][0][site] *= -1

        energies = [self.amplitude_ratio(state, flipped_states[i]) * element for (i, element) in enumerate(mat_elements)]

        return sum(energies)

    def estimate_ground_energy(self) -> None:
        '''
        Computes a stochastic estimate of the energy of the NQS and if writeOutput is set to True, 
        it writes the energy to a file 'computed_energies.txt'
        '''

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
        self.nqs_energy = np.squeeze(est_avg)
        self.nqs_energy_error = np.squeeze(est_error)

        self.current_energy = est_avg
        energy_report = f"Estimated average energy per spin: {est_avg} +/- {est_error}"
        print(energy_report)

        bin_report = f'Error estimated with binning analysis consisting of {nblocks} bins of {blocksize} samples each.'
        print(bin_report)

        self.correlation_time = 0.5 * blocksize * enmeansq / enmean_sq_unblocked
        autocorrelation_report = f'Estimated autocorrelation time is {self.correlation_time}'

        print(autocorrelation_report)
