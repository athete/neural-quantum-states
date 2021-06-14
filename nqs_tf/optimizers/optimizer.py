import tensorflow as tf
import numpy as np
from tensorflow.python.eager.monitoring import Sampler
from sampler.sampler import MetropolisHastingsSampler

class Optimizer:
    def __init__(self, hamiltonian, nqs, optimizer, sampler_params = None):
        self.hamiltonian = hamiltonian
        self.nqs = nqs
        self.optimizer = optimizer
        self.sampler = None

        if sampler_params is None:
            sampler_params = {
                'num_sweeps': 100,
                'therm_factor': 0.,
                'sweep_factor': 1,
                'num_flips': None
            }

        self.num_sweeps = sampler_params['num_sweeps']
        self.therm_factor = sampler_params['therm_factor']
        self.sweep_factor = sampler_params['sweep_factor']
        self.num_flips = sampler_params['num_flips']

        self.loss = []
        self.error = []

    def run(self, num_epochs):
        init_state = None
        for epoch in range(num_epochs):
            print(f"\n\nEpoch {epoch+1}/{num_epochs}")
            self.sampler = MetropolisHastingsSampler(
                hamiltonian=self.hamiltonian,
                nqs=self.nqs,
                init_state=init_state
                )
            init_state = self.sampler.current_state
            with tf.GradientTape() as tape:
                energy = self.sampler.run(self.num_sweeps, tape)
            weights = self.nqs.trainable_weights
            grads = tape.gradient(energy, weights) 
            self.optimizer.apply_gradients(zip(grads, weights))

        