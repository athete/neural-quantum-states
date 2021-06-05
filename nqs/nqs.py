'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% COPYRIGHT NOTICE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Permission is granted for anyone to copy, use, modify, or distribute the
accompanying programs and documents for any purpose, provided this copyright
notice is retained and prominently displayed, along with a complete citation of
the published version of the paper:
 ____________________________________________________________________________
| G. Carleo, and M. Troyer                                                   |
| Solving the quantum many-body problem with artificial neural-networks      |
|____________________________________________________________________________|

The programs and documents are distributed without any warranty, express or
implied.

These programs were written for research purposes only, and are meant to
demonstrate and reproduce the main results obtained in the paper.

All use of these programs is entirely at the user's own risk.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

from typing import Any
import numpy as np


class NeuralQuantumState:
    '''
    A class to implement a Neural Quantum State (NQS).
    A NQS is essentially a restricted Boltzmann machine (RBM) consisting of a
    visible layer (corresponing to the spins) and a hidden layer.
    '''

    def __init__(self, numHidden=40, numVisible=40, initWeight=0.001) -> None:
        self.numHidden = numHidden
        self.numVisible = numVisible
        # Hidden Layer Biases, randomly initialised
        self.a = (np.random.rand(self.numHidden, 1) - 0.5) * initWeight
        self.a = self.a.astype(dtype=np.cdouble)
        # Visible Layer Biases, randomly initialised
        self.b = (np.random.rand(self.numVisible, 1) - 0.5) * initWeight
        self.b = self.b.astype(dtype=np.cdouble)
        # Network weights
        self.W = (np.random.rand(self.numVisible,
                  self.numHidden) - 0.5) * initWeight
        self.W = self.W.astype(dtype=np.cdouble)

        self.theta = self.b

    def logAmplitudeRatio(self, state, flips) -> Any:
        '''
        Computes the logarithm of a ratio of two wavefunctions.
        For the formulas used here, refer to 
        https://kits.ucas.ac.cn/images/articles/2017/Machine_Learning/GiuseppeCarleo.pdf
        '''
        if len(flips) == 0:
            return 0.

        x = np.zeros(shape=(self.numVisible, 1))
        for flip in flips:
            x[flip] = 2 * state[flip]

        logPop = - np.dot(self.a.T, x) + np.sum(
            np.log(np.cosh(self.theta - np.dot(self.W.T, x))) -
            np.log(np.cosh(self.theta))
        )

        return logPop

    def amplitudeRatio(self, state, flips) -> Any:
        '''
        Computes the ratio of two wavefunctions.
        '''
        return np.exp(self.logAmplitudeRatio(state, flips))

    def initLookupTables(self, state) -> None:
        '''
        Initializes the lookup tables. Given a state, we update the list of 
        theta values used to compute the prefactor of a state.
        '''

        state = np.array(state).astype(float).reshape((self.numVisible, 1))
        self.theta = self.b + np.dot(self.W.T, state)

    def updateLookupTables(self, state, flipSites) -> None:
        '''
        Updates the lookup tables after a spin flip in the Metropolis-Hastings
        algorithm
        '''

        if len(flipSites) == 0:
            return

        x = np.empty(shape=(self.numVisible, 1))
        for flip in flipSites:
            x[flip] = 2 * state[flip]

        self.theta -= np.dot(self.W.T, x)

    def updateNetworkParams(self, updatedParams) -> None:
        '''
        Update the weights and biases of the network.
        '''
        dA = updatedParams[:self.numVisible]
        dB = updatedParams[self.numVisible: self.numVisible + self.numHidden]
        dW = updatedParams[self.numVisible + self.numHidden:]
        dW = dW.reshape((self.numVisible, self.numHidden))

        self.a -= dA
        self.b -= dB
        self.W -= dW
        self.theta = self.b