from hamiltonians.IsingHamiltonian import Ising1D
from models.rbm import RBM
from optimizers.optimizer import Optimizer

if __name__ == '__main__':
    import tensorflow as tf

    # Select Model
    rbm = RBM(num_hidden=40)
    ham = Ising1D(num_spins=40, h_field=0.5, is_PBC=True)
    

    # Define sampler parameters
    sampler_params = {'num_sweeps': 100, 'therm_factor': 0., 'sweep_factor': 1, 'num_flips': 1}

    # Instantiate an optimizer and run
    sgd = tf.keras.optimizers.SGD(learning_rate=0.01)
    optimizer = Optimizer(hamiltonian=ham, nqs=rbm, optimizer=sgd, sampler_params=sampler_params)
    num_epochs = 25
    optimizer.run(num_epochs)
