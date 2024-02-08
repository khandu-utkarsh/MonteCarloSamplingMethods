import subprocess
import sys
import time
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('numpy')
install('matplotlib')
install('joblib')



import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import logging

class Jarzynski:
    def __init__(self, beta, L, M, N):
        self.M = M
        self.L = L
        self.N = N
        self.initSpins = np.zeros((M, L, L), dtype=int)
        self.beta = beta
        self.hamiltonian = np.zeros((M, 1), dtype=int)
        self.Magnetization = np.zeros((M, N))  # Because of samples going from 0, 1, 2, 3, ... N - 1
        self.Weights = np.ones((M, N))
        self.update_seq = np.zeros((N, 1), dtype=[('x', 'int'), ('y', 'int')])  # It would be same for all three lattice

    def SetInitialHamiltonian(self):
        rows = self.L
        cols = self.L
        for index in range(self.M):
            spins = self.initSpins[index]
            hamiltonian = 0
            for row in range(rows):
                for col in range(cols):
                    sumNeighbors = 0

                    sumNeighbors += spins[(row), (col - 1) % cols] #Left
                    sumNeighbors += spins[(row - 1) % rows, (col)]  #Top
                    sumNeighbors += spins[(row), (col + 1) % cols] #Right
                    sumNeighbors += spins[(row + 1) % rows, (col)] #Bottom
                    sumNeighbors = int(sumNeighbors/2)
                    prod = sumNeighbors * spins[row, col]
                    hamiltonian += prod

            self.hamiltonian[index] = hamiltonian
        # print('Hamiltonian initialized: ', self.hamiltonian)

    def GetNeighborsSum(self, x_pos, y_pos):
        sumVal = self.initSpins[:, (x_pos) % self.L, (y_pos - 1) % self.L] + self.initSpins[:, (x_pos - 1) % self.L, (y_pos) % self.L] + self.initSpins[:,(x_pos) % self.L, (y_pos + 1) % self.L] + self.initSpins[:,(x_pos + 1) % self.L,(y_pos) % self.L]
        return sumVal

    def GetUpdateSequence(self):
        latticeCount = self.L * self.L
        count = 0
        while (count < self.N):
            for iNum in range(latticeCount):
                if (count < self.N):
                    x_pos = int(iNum // self.L)
                    y_pos = int(iNum - x_pos * self.L)
                    self.update_seq[count] = (x_pos, y_pos)
                    count += 1
                    # print('Updated Seq:',self.update_seq)

    def InitializeLattice(self, inputInitValues):
        initializedValues = inputInitValues
        #initializedValues = np.random.randint(0, 2, size=(self.L, self.L)) * 2 - 1  # Randomly create int b/w 0 and 1 and then multiply by 2 and sub by 1
        self.initSpins[:] = initializedValues
        self.SetInitialHamiltonian()
        partialSum = np.sum(self.initSpins, 2)
        partialSum = np.sum(partialSum, 1)
        self.Magnetization[:, 0] = partialSum
        # print('Magnetization: ',self.Magnetization)
        self.Weights[:, 0] = self.Weights[:, 0] / np.sum(self.Weights[:, 0])
        # print('Weights: ',self.Weights)
        self.GetUpdateSequence()  # Storing the update sequence

    def CustomBernaulli(self, p):
        randNum = np.random.uniform(0.0, 1.0, p.shape)
        boolArray = randNum < p
        return boolArray

    def GinalFinalSpinToUpdate(self, x_pos, y_pos, index):
        def flipSpinFxn(oldSpin):
            if (oldSpin == 1):
                return -1
            else:
                return 1

        oldSpin = self.initSpins[:, x_pos, y_pos]
        vfunc = np.vectorize(flipSpinFxn)
        newSpin = vfunc(oldSpin)
        deltaSpinChange = newSpin - oldSpin
        sumVal = self.GetNeighborsSum(x_pos, y_pos)
        delta_Hamiltonian = deltaSpinChange * sumVal

        p_accept = np.minimum(1.0, np.exp((self.beta * index / self.N) * delta_Hamiltonian))  #
        boolOutArray = self.CustomBernaulli(p_accept)
        finalSpin = np.where(boolOutArray, newSpin, oldSpin)  # Directly update this spin
        return sumVal, finalSpin

    # Weights should be normalized
    def DoMultinomialResampling(self, index):
        out = np.random.choice(self.M, self.M, replace=True, p=self.Weights[:, index])

        newWeights = np.ones((self.M, ))

        newSpins = self.initSpins[out, :, :]
        newHamiltonin = self.hamiltonian[out, :]
        newMagnetization = self.Magnetization[out, index]
        newWeights = newWeights / self.M

        self.initSpins = newSpins
        self.hamiltonian = newHamiltonin
        self.Magnetization[:, index] = newMagnetization
        self.Weights[:, index] = newWeights
        return

    def UpdateSpins(self, x_pos, y_pos, index, resample=False):
        oldSpin = self.initSpins[:, x_pos, y_pos]
        nSum, finalSpin = self.GinalFinalSpinToUpdate(x_pos, y_pos, index)
        deltaSpinChange = finalSpin - oldSpin
        self.initSpins[:, x_pos, y_pos] = finalSpin  # Spins updated in object #This is working fine
        delta_Hamiltonian = deltaSpinChange * nSum
        self.hamiltonian[:, 0] = self.hamiltonian[:, 0] + delta_Hamiltonian  # Hamiltonian updated
        omega_num = np.exp((self.beta / self.N) * self.hamiltonian)  # Using the updated Hamiltonian, after updating flips
        omega_num = np.reshape(omega_num, (omega_num.shape[0],))
        self.Weights[:, index] = self.Weights[:, index - 1] * omega_num  # Have to normalize it
        self.Weights[:, index] = self.Weights[:, index] / np.sum(self.Weights[:, index])  # Normalized weights
        self.Magnetization[:, index] = self.Magnetization[:, index - 1] + deltaSpinChange
        if (resample == True):
            self.DoMultinomialResampling(index)

    def GenerateMagnetizationSamples(self, resample = False):
        counts, _ = self.update_seq.shape
        for index in range(1,counts):
            x_pos = self.update_seq[index][0][0]
            y_pos = self.update_seq[index][0][1]
            self.UpdateSpins(x_pos, y_pos, index, resample)

        nthSampleMag = self.Magnetization[:,-1]
        nthSampleWeights = self.Weights[:,-1]
        return nthSampleMag, nthSampleWeights

def PlotHistogram(magnetization,beta, L,M,N, resampleString):
    fig, ax = plt.subplots() # fig : figure object, ax : Axes object
    ax.hist(magnetization);
    ax.set_xlabel('Magnetization')
    titleString = 'J D ' + ' ' + resampleString + " |N: " + str(N) + " |beta: " + str(beta) + " |L: " + str(L) + " |M: " + str(M)
    saveFigName = 'J_D ' + '_' + resampleString + '_N_' + str(N) + '_beta_' + str(beta) + '_L_' + str(L) + '_M_' + str(M)
    saveFigName = saveFigName.replace('.', 'd')
    ax.set_title(titleString)
    plt.savefig(saveFigName)
    plt.close(fig)

def GetVarianceOfSamples(L,M,N, initialSpins,beta = 0.05, resampling=False):
    start = time.perf_counter()
    model = Jarzynski(beta,L,M,N)
    model.InitializeLattice(initialSpins)
    magnetization, weights = model.GenerateMagnetizationSamples(resampling)
    average = magnetization * weights
    average = np.sum(average)
    #print('Shape mag: ',magnetization.shape)
    #print('Shape weights: ',weights.shape)
    #print('Average shape: ',average.shape)

    #average = np.mean(magnetization)
    #variance = np.var(weights)    #Have to plot variance of the weights
    variance = np.var(magnetization)    #Have to plot variance of the magnetization
    reStr = 'noR'
    if(resampling == True):
        reStr = 'R'
    PlotHistogram(magnetization,beta, L,M,N, reStr)
    end = time.perf_counter()
    timeTaken = end - start
    return average, variance, timeTaken


def Run(L,M,beta,resampling):
    initializedValues = np.random.randint(0, 2, size=(L, L)) * 2 - 1  # Randomly create int b/w 0 and 1 and then multiply by 2 and sub by 1
    Ns = [1000000,2000000,3000000,4000000,5000000]
    #Ns = [10000,15000,20000,25000,30000,35000,40000,45000,50000]
    #Ns = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    output = Parallel(n_jobs=len(Ns),verbose=10)(
        delayed(
            GetVarianceOfSamples
        )(L,M,N,initializedValues,beta,resampling) for N in Ns)

    unzipped = list(zip(*output))

    averages = unzipped[0]
    variances = unzipped[1]
    consumedTime = unzipped[2]
    fig, ax = plt.subplots()  # fig : figure object, ax : Axes object
    ax.plot(Ns,averages,'-o')
    ax.set_xlabel('N')
    ax.set_ylabel('Magnetization Estimator mean')
    ax.set_title('Plot of mean vs N')
    ax.set_xticks(Ns)
    if(resampling == True):
        plt.savefig('Jarzynski_MeanPlot_resampling_from_' + str(Ns[0]) + '_to_' + str(Ns[-1]))
    else:
        plt.savefig('Jarzynski_MeanPlot_no_resampling_from_' + str(Ns[0]) + '_to_' + str(Ns[-1]))
    plt.close(fig)

    fig, ax = plt.subplots()  # fig : figure object, ax : Axes object
    ax.plot(Ns,variances,'-o')
    ax.set_xlabel('N')
    ax.set_ylabel('Magnetization Estimator Variance')
    ax.set_title('Plot of variance vs N')
    ax.set_xticks(Ns)
    if(resampling == True):
        plt.savefig('Jarzynski_VariancePlot_resampling_from_' + str(Ns[0]) + '_to_' + str(Ns[-1]))
    else:
        plt.savefig('Jarzynski_VariancePlot_no_resampling_from_' + str(Ns[0]) + '_to_' + str(Ns[-1]))
    plt.close(fig)

    fig, ax = plt.subplots()  # fig : figure object, ax : Axes object
    ax.plot(Ns,consumedTime,'-o')
    ax.set_xlabel('N')
    ax.set_ylabel('Estimator time in seconds')
    ax.set_title('Plot of Time Taken by Estimator vs N')
    ax.set_xticks(Ns)
    if(resampling == True):
        plt.savefig('Jarzynski_TimePlot_resampling_from_' + str(Ns[0]) + '_to_' + str(Ns[-1]))
    else:
        plt.savefig('Jarzynski_TimePlot_no_resampling_from_' + str(Ns[0]) + '_to_' + str(Ns[-1]))
    plt.close(fig)



#def Run(L,M,beta,resampling):
#    initializedValues = np.random.randint(0, 2, size=(L, L)) * 2 - 1  # Randomly create int b/w 0 and 1 and then multiply by 2 and sub by 1
#    Ns = 10000
#    GetVarianceOfSamples(L,M,Ns,initializedValues,beta,resampling)

#Run(10, 1000, 0.2, True)
