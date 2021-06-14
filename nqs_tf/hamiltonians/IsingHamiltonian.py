import tensorflow as tf
import numpy as np

class Ising1D:
    """ This class implements the transverse-field Ising (TFI) model Hamiltonian in one dimension. 
    The TFI Hamiltonian has the form 
    H_{TFI} = -h \sum_{i}\sigma^x_i - \sum_{ij}\sigma^z_i \sigma^z_j
    where \sigma^i are the Pauli matrices in the i-th direction and h is the value of the transverse field. 
    """

    def __init__(self, num_spins, h_field, is_PBC):
        self.num_spins = num_spins
        self.h_field = h_field
        self.is_PBC = is_PBC

        self.min_spin_flips = 1
        self.mat_elements = np.full(shape=self.num_spins + 1, fill_value=-self.h_field)
        self.spin_flip_sites = [[]] + [[i] for i in range(num_spins)]
    
    def find_nonzero_elements(self, state):
        """ Finds all the non-zero matrix elements of the Hamiltonian on the given state, state. In other words,
        all state' such that <state'|H|state> is not zero
        """

        state = np.squeeze(state)
        mat_elements = self.mat_elements
    
        # Find the Sz*Sz interaction
        mat_elements[0] = 0.0

        for i in range(self.num_spins - 1):
            mat_elements[0] -= state[i] * state[i + 1]
        
        if self.is_PBC:
            mat_elements[0] -= state[self.num_spins - 1] * state[0]

        return mat_elements, self.spin_flip_sites