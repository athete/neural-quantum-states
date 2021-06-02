'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% COPYRIGHT NOTICE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Permission is granted for anyone to copy, use, modify, or distribute the
accompanying programs and documents for any purpose, provided this copyright
notice is retained and prominently displayed, along with a complete citation of
the published version of the paper:
 ______________________________________________________________________________
| G. Carleo, and M. Troyer                                                     |
| Solving the quantum many-body problem with artificial neural-networks        |
|______________________________________________________________________________|

The programs and documents are distributed without any warranty, express or
implied.

These programs were written for research purposes only, and are meant to
demonstrate and reproduce the main results obtained in the paper.

All use of these programs is entirely at the user's own risk.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

import numpy as np
from .sampler import Sampler
from typing import Any, List, Tuple

class StochasticReconfigOptimizer:
    '''
    Implements the Stochastic Reconfiguration algorithm. Refer to Sorella et al (https://doi.org/10.1063/1.2746035) for a detailed 
    description of the algorithm. 
    '''
    def __init__(self, hamiltonian, neuralNetState, learningRate=0.5e-2, samplerParams=None) -> None:
        self.hamiltonian = hamiltonian
        self.neuralNetState = neuralNetState
        
        if samplerParams is None:
            samplerParams = {
                'numSweeps': 10000, 
                'thermFactor': 0.,
                'sweepFactor': 1,
                'numFlips': None
            }

        self.numSweeps = samplerParams['numSweeps']
        self.thermFactor = samplerParams['thermFactor']
        self.sweepFactor = samplerParams['sweepFactor']
        self.numFlips = samplerParams['numFlips']
        self.learningRate = learningRate
        self.loss = []
        self.error = []
        

    def SRGRadients(self, sampler, p) -> Tuple[Any, List]:
        '''
        Determines the gradients of the cost function with respect
        to the network parameter. 
        '''
        states = np.array(sampler.stateHistory, dtype=complex)
        Elocs = np.array(sampler.localEnergies).reshape((len(sampler.localEnergies), 1))
        
        print('Computing stochastic reconfiguration updates...')
        self.loss.append(sampler.neuralNetEnergy)
        self.error.append(sampler.neuralNetEnergyError)
        B = self.neuralNetState.b
        weights = self.neuralNetState.W

        update, derivs = self.findDerivatives(p, Elocs, B, weights, states, 
                                                self.neuralNetState.numVisible, self.neuralNetState.numHidden, len(states))

        return update, derivs

    def findDerivatives(self, p, Eloc, B, weights, sigmas, numSpins, numHidden, numSamples) -> Tuple[Any, List]:
        '''
        Finds the variational derivatives. Refer to the supplementary materials of G. Carleo, and M. Troyer for a 
        description of the formulae used. 
        '''
        theta = B + np.dot(weights.transpose(), sigmas.transpose())
        dA = sigmas.transpose()
        dB = np.tanh(theta)
        dW = sigmas.transpose().reshape((numSpins, 1, numSamples)) * \
             np.tanh(theta.reshape(1, numHidden, numSamples))
        derivs = np.concatenate([dA, dB, dW.reshape(numSpins * numHidden, numSamples)])

        avgDerivs = np.sum(derivs, axis=1, keepdims=True) / numSamples
        avgDerivsMat = np.conjugate(avgDerivs.reshape(derivs.shape[0], 1))
        avgDerivsMat = avgDerivsMat * avgDerivs.reshape(1, derivs.shape[0])

        derivDerivConj = np.einsum('ik,jk->ij', np.conjugate(derivs), derivs) / numSamples
        S_kk = np.subtract(derivDerivConj, avgDerivsMat)

        F_p = np.sum(Eloc.transpose() * np.conjugate(derivs), axis=1) / numSamples
        F_p -= np.sum(Eloc.transpose(), axis=1) * np.sum(np.conjugate(derivs), axis=1) / (numSamples**2)

        S_kk2 = np.zeros(S_kk.shape, dtype=complex)
        row, col = np.diag_indices(S_kk.shape[0])
        S_kk2[row, col] = self.lambd(p) * np.diagonal(S_kk)
        S_reg = S_kk + S_kk2
        update = np.dot(np.linalg.inv(S_reg), F_p).reshape(derivs.shape[0], 1)

        return update, derivs

    def run(self, numEpochs) -> None:
        '''
        Runs the Stochastic Reconfiguration algorithm. 
        '''

        initialState = None

        for p in range(numEpochs):
            print(f"\n\nIteration {p}")
            sampler = Sampler(self.hamiltonian, self.neuralNetState, initialState=initialState)
            initialState = sampler.currentState
            sampler.run(self.numSweeps, self.thermFactor, self.sweepFactor, self.numFlips)
            updateVals, _ = self.SRGRadients(sampler, p+1)
            self.neuralNetState.updateNetworkParams(self.learningRate * updateVals)

    @staticmethod
    def lambd(p, lambd0=100, b=0.9, lambdMin=1e-4) -> Any:
        '''
        Method to enable explicit regularizattion of the S matrix. Refer to the supplementary materials of G. Carleo, and M. Troyer for a 
        description of the formulae used. 
        '''
        return max(lambd0 * (b ** p), lambdMin)
