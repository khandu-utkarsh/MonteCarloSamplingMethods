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

                    if (col == 0):  # Consider left
                        sumNeighbors += spins[(row), (col - 1) % cols]

                    if (row == 0):  # Consider top
                        sumNeighbors += spins[(row - 1) % rows, (col)]

                    # Always consider right
                    sumNeighbors += spins[(row), (col + 1) % cols]

                    # Always consider bottom
                    sumNeighbors += spins[(row + 1) % rows, (col)]

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
        copyCounts = np.random.multinomial(self.M, self.Weights[:, index])
        newSpins = np.zeros((self.M, self.L, self.L), dtype=int)
        newHamiltonin = np.zeros((self.M, 1), dtype=int)
        newMagnetization = np.zeros((self.M, 1))
        newWeights = np.ones((self.M, 1))

        nextIndex = 0
        for i in range(self.M):  # Looping over all trajectories
            cc = copyCounts[i]
            newSpins[nextIndex:nextIndex + cc, :, :] = self.initSpins[i, :, :]
            newHamiltonin[nextIndex:nextIndex + cc, :] = self.hamiltonian[i, :]
            newMagnetization[nextIndex:nextIndex + cc, :] = self.Magnetization[i, index]
            newWeights[nextIndex:nextIndex + cc, :] = 1 / self.M
            nextIndex = nextIndex + cc

        # Copy back to original object data structure
        self.initSpins = newSpins
        self.hamiltonian = newHamiltonin
        newMagnetization = newMagnetization.reshape((self.M,))
        self.Magnetization[:, index] = newMagnetization
        newWeights = newWeights.reshape((self.M,))
        self.Weights[:, index] = newWeights
        return

    def UpdateSpins(self, x_pos, y_pos, index, resample=False):

        # print('--------------------------------------------')
        # print('Index: ', index)

        oldSpin = self.initSpins[:, x_pos, y_pos]
        nSum, finalSpin = self.GinalFinalSpinToUpdate(x_pos, y_pos, index)
        deltaSpinChange = finalSpin - oldSpin

        # print('Old Spin: ', oldSpin)
        # print('Final Spin: ', finalSpin)
        # print('Spin Change: ',deltaSpinChange)

        # Update the spins
        # print('Before updating: ', self.initSpins)

        self.initSpins[:, x_pos, y_pos] = finalSpin  # Spins updated in object #This is working fine

        # print('After updating: ', self.initSpins)

        # Updating the hamiltonina:

        delta_Hamiltonian = deltaSpinChange * nSum

        # print('Delta Hamil: ', delta_Hamiltonian)

        # print('Hamiltonian before update: ', self.hamiltonian)
        self.hamiltonian[:, 0] = self.hamiltonian[:, 0] + delta_Hamiltonian  # Hamiltonian updated
        # print('Hamiltonian after update: ',self.hamiltonian)

        # Omega Weight Ratios
        omega_num = np.exp(
            (self.beta / self.N) * self.hamiltonian)  # Using the updated Hamiltonian, after updating flips
        omega_num = np.reshape(omega_num, (omega_num.shape[0],))
        # print('Omega Weights Before Update: ',self.Weights)

        self.Weights[:, index] = self.Weights[:, index - 1] * omega_num  # Have to normalize it
        self.Weights[:, index] = self.Weights[:, index] / np.sum(self.Weights[:, index])  # Normalized weights
        # print('Weights: ', self.Weights)
        # print('Omega Weights After Update: ',self.Weights)

        # Update magnetization
        #Have to multiply it by weights

        self.Magnetization[:, index] = self.Magnetization[:, index - 1] + deltaSpinChange

        # print('Magnetization: ', self.Magnetization)
        # print('--------------------------------------------')

        if (resample == True):
            self.DoMultinomialResampling(index)

    def GenerateMagnetizationSamples(self, resample = False):
      counts, _ = self.update_seq.shape
      for index in range(1,counts):
        x_pos = self.update_seq[index][0][0]
        y_pos = self.update_seq[index][0][1]
        #print("Positions: ",x_pos, y_pos)
        self.UpdateSpins(x_pos, y_pos, index, resample)

      nthSampleMag = self.Magnetization[:,-1]
      nthSampleWeights = self.Weights[:,-1]

      #Have to get nth sample weights
      #New Magnetization will be nth sample magnetization multiplied by weights

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
    #def runProcessJarzynski(L,M,N, beta = 0.05, resampling=False):
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
    variance = np.var(weights)    #Have to plot variance of the weights
    reStr = 'noR'
    if(resampling == True):
        reStr = 'R'
    PlotHistogram(magnetization,beta, L,M,N, reStr)
    #print('Mean magnetization: ',np.mean(magnetization), '|resample: ',reStr,'|beta: ', beta, '|L: ', L, '|M: ', M, '|N: ', N)
    end = time.perf_counter()
    timeTaken = end - start
    return average, variance, timeTaken


def Run(L,M,beta,resampling):
    initializedValues = np.random.randint(0, 2, size=(L, L)) * 2 - 1  # Randomly create int b/w 0 and 1 and then multiply by 2 and sub by 1
    Ns = [100000,200000,300000,400000,500000,600000,700000,800000,900000,100000]
    #Ns = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    output = Parallel(n_jobs=16)(
        delayed(
            GetVarianceOfSamples
        )(L,M,N,initializedValues,beta,resampling) for N in Ns)

    #print('Output: ', output)

    unzipped = list(zip(*output))
    #print('Unzipped: ', unzipped)

    averages = unzipped[0]
    #print('Averages: ', averages)
    variances = unzipped[1]
    #print('Variance: ', variances)
    consumedTime = unzipped[2]
    #print('Time: ', consumedTime)
    fig, ax = plt.subplots()  # fig : figure object, ax : Axes object
    ax.plot(Ns,averages,'-o')
    ax.set_xlabel('N')
    ax.set_ylabel('Magnetization Estimator mean')
    ax.set_title('Plot of mean vs N')
    ax.set_xticks(Ns)
    plt.savefig('Jarzynski_MeanPlot_no_reampling_from_' + str(Ns[0]) + '_to_' + str(Ns[-1]))
    plt.close(fig)

    fig, ax = plt.subplots()  # fig : figure object, ax : Axes object
    ax.plot(Ns,variances,'-o')
    ax.set_xlabel('N')
    ax.set_ylabel('Estimator weights Variance')
    ax.set_title('Plot of variance vs N')
    ax.set_xticks(Ns)
    plt.savefig('Jarzynski_VariancePlot_no_reampling_from_' + str(Ns[0]) + '_to_' + str(Ns[-1]))
    plt.close(fig)

    fig, ax = plt.subplots()  # fig : figure object, ax : Axes object
    ax.plot(Ns,consumedTime,'-o')
    ax.set_xlabel('N')
    ax.set_ylabel('Estimator time in seconds')
    ax.set_title('Plot of Estimator Variance vs N')
    ax.set_xticks(Ns)
    plt.savefig('Jarzynski_TimePlot_no_reampling_from_' + str(Ns[0]) + '_to_' + str(Ns[-1]))
    plt.close(fig)
