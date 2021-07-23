from hamiltonians.ising import Ising1D
from models.ffnn import FFNN
from optimizers.optimizer import OptimizerWrapper

if __name__ == '__main__':
    import tensorflow as tf
    import numpy as np
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    # Define the neural network state
    ffnn = FFNN()
   
    # Select the Hamiltonian
    ham = Ising1D(num_spins=4, h_field=0.5, is_PBC=True)

    # Define sampler parameters
    sampler_params = {'num_sweeps': 1000, 'therm_factor': 0., 'sweep_factor': 1, 'num_flips': 1}

    # Instantiate an optimizer and run
    sgd = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.2, nesterov=True)
    #adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.9, epsilon=1e-8)
    optimizer = OptimizerWrapper(hamiltonian=ham, nqs=ffnn, optimizer=sgd, sampler_params=sampler_params)
    num_epochs = 250
    optimizer.run(num_epochs)  

