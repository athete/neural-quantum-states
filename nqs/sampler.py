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


from typing import List, Any, Tuple
import numpy as np
import math
import os


class Sampler:
    '''
    This class runs the Metropolis-Hastings algorithm to generate spin configuaration 
    samples from the network.
    '''

    def __init__(self, hamiltonian, neuralNetState, zeroMagnetization=True, filename=None,
                 initialState=None, writeOutput=True) -> None:
        self.hamiltonian = hamiltonian
        self.neuralNetState = neuralNetState
        self.neuralNetEnergy = None
        self.neuralNetEnergyError = None
        self.localEnergies = []
        self.zeroMagnetization = zeroMagnetization
        self.samplesFile = filename
        self.numSpins = self.hamiltonian.numSpins
        self.stateHistory = []
        self.currentLocalEnergy = None
        self.correlation_time = None
        self.writeOutput = writeOutput

        # Sampling Statistics
        self.acceptances = None
        self.numMoves = None

        # path to store energies
        self.dataDir = './data/'

        if not os.path.exists(self.dataDir):
            os.makedirs(self.dataDir)

        if initialState is None:
            self.initRandomState()
        else:
            self.currentState = initialState

    def initRandomState(self) -> None:
        '''
        Generates a random state by sampling a uniform distribution. 
        '''

        # Generate a random state by sampling a uniform distribution [0, numSpins)
        self.currentState = np.random.uniform(size=self.numSpins)
        # If a[i] < 0.5, set a[i] to -1, else set it to 1
        self.currentState = np.where(self.currentState < 0.5, -1, 1)

        # Ensure that the total magnetization of the state is zero, that is, sum(state) = 0
        if self.zeroMagnetization:
            if self.numSpins % 2:
                raise ValueError("Cannot initialize a random state with zero magnetization for an odd number of spins\n")

            totalMagnetization = np.sum(self.currentState)
        # If not zero, set the defaulting location to -1 or +1 to reduce/increase
        # the total magnetization
            if totalMagnetization > 0:
                while totalMagnetization != 0:
                    randomSite = self.chooseRandomSite()
                    while self.currentState[randomSite] < 0:
                        randomSite = self.chooseRandomSite()
                    self.currentState[randomSite] = -1
                    totalMagnetization = totalMagnetization - 1

            elif totalMagnetization < 0:
                while totalMagnetization != 0:
                    randomSite = self.chooseRandomSite()
                    while self.currentState[randomSite] > 0:
                        randomSite = self.chooseRandomSite()
                    self.currentState[randomSite] = 1
                    totalMagnetization = totalMagnetization + 1

    def randomSpinFlips(self, numFlips) -> List:
        '''
        For the Metropolis-Hastings algorithm, randomly flips at most two sites in a 
        state. 
        '''
        # Choose a site randomly to flip
        firstSite = self.chooseRandomSite()

        if numFlips == 2:
            secondSite = self.chooseRandomSite()

            if self.zeroMagnetization:
                if self.currentState[firstSite] == self.currentState[secondSite]:
                    return []
                else:
                    return [firstSite, secondSite]
            else:
                if firstSite == secondSite:
                    return []
                else:
                    return [firstSite, secondSite]
        else:
            return [firstSite]

    def resetSamplerStats(self) -> None:
        '''
        Resets the sampler statistics, i.e, the number of acceptances, the total number of
        moves, and the state history. 
        '''
        self.acceptances = 0
        self.numMoves = 0
        self.stateHistory = []

    def acceptanceRate(self) -> float:
        '''
        Calculates the acceptance rate of the sampling
        '''
        return self.acceptances / self.numMoves

    def move(self, numFlips) -> None:
        flipSites = self.randomSpinFlips(numFlips)
        if len(flipSites) > 0:
            # Find the acceptance probability
            psiRatio = self.neuralNetState.amplitudeRatio(
                self.currentState, flipSites)
            acceptanceProbability = np.square(np.abs(psiRatio))

            # Metropolis-Hastings Test
            if acceptanceProbability > np.random.random():
                self.neuralNetState.updateLookupTables(
                    self.currentState, flipSites)

                # Test passed, set the current state to the flipped version
                for flip in flipSites:
                    self.currentState[flip] *= -1
                self.acceptances += 1

        self.numMoves += 1

    def computeLocalEnergy(self) -> float:
        '''
        Finds the value of the local energy on the current state
        '''

        # Find the non-zero matrix elements of the Hamiltonian
        state = self.currentState
        (matElements, spinFlips) = self.hamiltonian.findNonZeroElements(state)

        energies = [self.neuralNetState.amplitudeRatio(state, spinFlips[i])*element
                    for (i, element) in enumerate(matElements)]
        return sum(energies)

    def run(self, numSweeps, thermFactor=0.1, sweepFactor=1, numFlips=None) -> None:
        '''
        Runs the Monte-Carlo Sampling for the NQS. 
        A sweep consists of (numSpins * sweepFactor) steps; sweep to consists of
        flipping each spin an expected number of numFlips times.
        '''

        if numFlips is None:
            numFlips = self.hamiltonian.minSpinFlips

        if numFlips < 1 and numFlips > 2:
            raise ValueError("Number of spin flips must be equal to 1 or 2.\n")

        if not (0 <= thermFactor <= 1):
            raise ValueError(
                "The thermalization factor should be a real number between 0 and 1.\n")

        if numSweeps < 50:
            raise ValueError(
                "Please enter a number of sweeps sufficiently large (>50).\n")

        print("Starting Monte-Carlo Sampling...")
        print(f"{numSweeps} sweeps will be perfomed.")

        self.resetSamplerStats()
        self.neuralNetState.initLookupTables(self.currentState)

        if thermFactor != 0:
            print('Starting Thermalization...')

            numMoves = (int)((thermFactor * numSweeps)
                             * (sweepFactor * self.numSpins))
            for _ in range(numMoves):
                self.move(numFlips)

            print('Done.')

        self.resetSamplerStats()

        for _ in range(numSweeps):
            for _ in range(sweepFactor * self.numSpins):
                self.move(numFlips)
            self.currentLocalEnergy = self.computeLocalEnergy()
            self.localEnergies.append(self.currentLocalEnergy)
            self.stateHistory.append(np.array(self.currentState))

        print('Completed Monte-Carlo Sampling.')

        return self.estimateOutputEnergy()

    def estimateOutputEnergy(self) -> None:
        '''
        Computes a stochastic estimate of the energy of the NQS and if writeOutput is set to True, 
        it writes the energy to a file 'computed_energies.txt'
        '''

        nblocks = 50
        blocksize = len(self.localEnergies) // nblocks
        enmean = 0
        enmeansq = 0
        enmeanUnblocked = 0
        enmeanSqUnblocked = 0

        for block in range(nblocks):
            eblock = 0
            for j in range(block*blocksize, (block + 1) * blocksize):
                eblock += self.localEnergies[j].real
                delta = self.localEnergies[j].real - enmeanUnblocked
                enmeanUnblocked += delta / (j + 1)
                delta2 = self.localEnergies[j].real - enmeanUnblocked
                enmeanSqUnblocked += delta * delta2
            eblock /= blocksize
            delta = eblock - enmean
            enmean += delta / (block + 1)
            delta2 = eblock - enmean
            enmeansq += delta * delta2

        enmeansq /= (nblocks - 1)
        enmeanSqUnblocked /= (nblocks * blocksize - 1)
        estAvg = enmean / self.numSpins
        estError = math.sqrt(enmeansq / nblocks) / self.numSpins
        self.neuralNetEnergy = np.squeeze(estAvg)
        self.neuralNetEnergyError = np.squeeze(estError)

        energyReport = f"Estimated average energy per spin: {estAvg} +/- {estError}"
        print(energyReport)

        binReport = f'Error estimated with binning analysis consisting of {nblocks} bins of {blocksize} samples each.'
        print(binReport)

        self.correlation_time = 0.5 * blocksize * enmeansq / enmeanSqUnblocked
        autocorrelation = f'Estimated autocorrelation time is {self.correlation_time}'

        print(autocorrelation)

        if self.writeOutput:
            # Save the computed energies to a file
            with open(self.dataDir + 'computed_energies.txt', 'a+') as f:
                np.savetxt(f, estAvg)

    def chooseRandomSite(self) -> int:
        '''
        Generates a random integer in the range [0, numSpins) that
        serves as a random index for the currentState attribute.
        '''
        return np.random.randint(low=0, high=self.numSpins-1)
