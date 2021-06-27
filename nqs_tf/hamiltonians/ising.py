import numpy as np

class Ising1D:
    """This class implements the transverse-field Ising (TFI) Hamiltonian in one dimension. 
    
    The TFI Hamiltonian has the form 

    H_{TFI} = -h \sum_{i}\sigma^x_i - \sum_{ij}\sigma^z_i \sigma^z_j

    where \sigma^i are the Pauli matrices in the i-th direction and h is the value of the transverse field. 

    Call Arguments
    ---------
    num_spins: int
        The number of spins in the system.
    h_field: double
        The magnitude of the transverse magnetic field.
    is_PBC: boolean
        Set to True if the Hamiltonian obeys periodic boundary conditions, else False.
    """

    def __init__(self, num_spins, h_field, is_PBC):
        self.num_spins = num_spins
        self.h_field = h_field
        self.is_PBC = is_PBC

        # Minimum number of spin flips permitted during Monte-Carlo Sampling
        self.min_spin_flips = 1
        # Non-zero matrix elements of the Hamiltonian
        self.mat_elements = np.full(shape=self.num_spins + 1, fill_value=-self.h_field)
        # Sites that have to be flipped for local energy computation
        self.spin_flip_sites = [[]] + [[i] for i in range(num_spins)]

    def get_spins(self):
        """ Returns the number of spins in the system. """
        return self.num_spins
    
    def find_nonzero_elements(self, state):
        """ Returns all the non-zero matrix elements of the Hamiltonian on the given state. In other words, returns
        all state' such that <state'|H|state> is not zero. 

        For a detailed description of this implementation, 
        see https://kits.ucas.ac.cn/images/articles/2017/Machine_Learning/GiuseppeCarleo.pdf

        Parameters
        ----------
        state
            The configuration/state whose non-zero overlaps have to be determined. 

        Returns
        --------
        mat_elements: numpy.ndarray
            Non-zero matrix elements of the Hamiltonian
        spins_flip_sites: List[list]
            Sites that have to be flipped for local energy computation
        """

        state = np.squeeze(state)
        mat_elements = self.mat_elements
    
        # The Sz*Sz interaction
        mat_elements[0] = 0.0

        # First term of the hamiltonian
        for i in range(self.num_spins - 1):
            mat_elements[0] -= state[i] * state[i + 1]
        
        # PBC condition
        if self.is_PBC:
            mat_elements[0] -= state[self.num_spins - 1] * state[0]

        return mat_elements, self.spin_flip_sites