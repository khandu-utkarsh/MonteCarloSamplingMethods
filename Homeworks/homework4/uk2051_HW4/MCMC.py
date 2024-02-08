import subprocess
import sys
import time

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('emcee')
install('joblib')
install('numpy')
install('matplotlib')
import numpy as np
import multiprocessing
from emcee.autocorr import AutocorrError, integrated_time
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(message)s %(threadName)s %(processName)s',
                    )

class MCMC:
#Intialize the lattice
#M are the trajectory count
    def __init__(self, beta, latticeSize, M, N_max, method):
        self.L = latticeSize
        self.initSpins = np.zeros((M, latticeSize, latticeSize), dtype=int) #Every lattice has the trajectory count
        self.beta = beta
        self.M = M
        self.N_max = N_max
        self.update_seq = np.zeros((N_max, 1), dtype=[('x', 'int'), ('y', 'int')])  # It would be same for all three lattice
        self.Magnetization = np.zeros((M, N_max))  # Because of samples going from 0, 1, 2, 3, ... N - 1
        self.IATs = np.zeros((M,20)) #20, considering 5 percent frequency of calculation
        self.type = 'Deterministic'
        self.method = method
        self.Ns = np.zeros((20,),dtype=int)
    def GetPositionsDeterministically(self):
        #pos_list = np.zeros((K,1), dtype=[('x', 'int'), ('y', 'int')])
        nums = self.L * self.L
        count = 0
        while(count < self.N_max):
            for iNum in range(nums):
                if(count < self.N_max):
                    x_pos = int(iNum//self.L)
                    y_pos = int(iNum - x_pos * self.L)
                    self.update_seq[count] = (x_pos, y_pos)
                    #pos_list[count] = (x_pos, y_pos)
                    count += 1

    def GetPositionsRandomly(self):
        #pos_list = np.zeros((K,1), dtype=[('x', 'int'), ('y', 'int')])
        nums = self.L * self.L
        count = 0
        while(count < self.N_max):
            iNum = np.random.randint(0,nums)
            x_pos = int(iNum//self.L)
            y_pos = int(iNum - x_pos * self.L)
            #pos_list[count] = (x_pos, y_pos)
            self.update_seq[count] = (x_pos, y_pos)
            count += 1

    def InitializeLattice(self, inputInitValues, methodType):
        #self.initSpins = np.random.randint(0, 2, size = (self.L,self.L)) * 2 - 1 #Randomly create int b/w 0 and 1 and then multiply by 2 and sub by 1
        self.initSpins[:] = inputInitValues
        partialSum = np.sum(self.initSpins, 2)
        partialSum = np.sum(partialSum, 1)
        self.Magnetization[:, 0] = partialSum
        if(methodType == 'Random'):
            self.GetPositionsRandomly()
            self.type = 'Random'
        else:
            self.GetPositionsDeterministically()
            self.type = 'Deterministic'
    def GetNeighborsSum(self, x_pos, y_pos):
        sumVal = self.initSpins[:, (x_pos) % self.L, (y_pos - 1) % self.L] + self.initSpins[:, (x_pos - 1) % self.L, (y_pos) % self.L] + self.initSpins[:,(x_pos) % self.L, (y_pos + 1) % self.L] + self.initSpins[:,(x_pos + 1) % self.L,(y_pos) % self.L]
        return sumVal

    def CustomBernaulli(self, p):
        randNum = np.random.uniform(0.0, 1.0, p.shape)
        boolArray = randNum < p
        return boolArray

    def GenerateHistograms(self, progressIndex):
        titleString = self.method + ' ' + self.type + " | Sample Size: " + str(self.Ns[progressIndex]) + " | beta: " + str(self.beta) + " | L: " + str(self.L)
        saveFigName = self.method + '_' + self.type + "_Sample Size_" + str(self.Ns[progressIndex]) + "_beta_" + str(self.beta) + "_L_" + str(self.L)
        saveFigName = saveFigName.replace('.', 'd')
        fig, ax = plt.subplots()  # fig : figure object, ax : Axes object
        #ax.hist(self.Magnetization[0, 0:self.Ns[progressIndex]]);
        ax.hist(self.Magnetization[0:self.Ns[progressIndex]]);
        ax.set_xlabel('Magnetization')
        ax.set_title(titleString)
        fig.savefig('Histogram_' + saveFigName)
        plt.close(fig)

    def GenerateIATGraphs(self, progressIndex):
        titleString = self.method + ' ' + self.type + " | Sample Size: " + str(self.Ns[progressIndex]) + " | beta: " + str(self.beta) + " | L: " + str(self.L)
        saveFigName = self.method + '_' + self.type + "_Sample Size_" + str(self.Ns[progressIndex]) + "_beta_" + str(self.beta) + "_L_" + str(self.L)
        saveFigName = saveFigName.replace('.', 'd')

        x = np.arange(self.M)
        fig, ax = plt.subplots()  # fig : figure object, ax : Axes object
        ax.plot(x, self.IATs[:, progressIndex], '-o');
        ax.set_xlabel('Trajectories Index')
        ax.set_ylabel('Integrated Autocorrelation Time')

        ax.set_title(titleString)
        ax.set_xticks(x)
        fig.savefig(saveFigName)
        plt.close(fig)

    def GenerateGraphs(self):
        for i in range(20):
            self.GenerateHistograms(i)
            self.GenerateIATGraphs(i)

    def ComputeIAT_Of_AllTrajectories(self,index,progressIndex):
        taus = np.zeros((self.M,))
        for t in range(self.M):
            taus[t] = integrated_time(self.Magnetization[t, 0:index + 1], c=5, tol=50, quiet=True)
        self.IATs[:, progressIndex - 1] = taus

    def GenerateMagnetizationData(self):
        counts, _ = self.update_seq.shape
        fivePercent = int(counts/20)
        progressIndex = 1
        for index in range(1, counts):
            x_pos = self.update_seq[index][0][0]
            y_pos = self.update_seq[index][0][1]
            self.UpdateSpins(x_pos, y_pos, index)
            if(index + 1 == progressIndex * fivePercent):
                self.ComputeIAT_Of_AllTrajectories(index, progressIndex)
                self.Ns[progressIndex - 1] = index + 1
                progressIndex = progressIndex + 1
        return



class MCMC_Gibbs(MCMC):#class MCMC_Gibbs:
    def __init__(self, beta, latticeSize, M, N_max):
      super().__init__(beta, latticeSize, M, N_max, 'G')
      #print('temp')

    def UpdateSpins(self, x_pos, y_pos, index):
        old_spin = self.initSpins[:,x_pos,y_pos]
        sum_neigh = self.GetNeighborsSum(x_pos, y_pos)
        cond_1_succ = 1/(1 + np.exp(-2 * self.beta * sum_neigh))
        randNumBoolArray = self.CustomBernaulli(cond_1_succ)
        finalSpin = np.where(randNumBoolArray, 1, -1)  # Directly update this spin
        deltaSpinChange = finalSpin - old_spin
        self.initSpins[:, x_pos, y_pos] = finalSpin
        self.Magnetization[:, index] = self.Magnetization[:, index - 1] + deltaSpinChange

class MCMC_MH(MCMC):#class MCMC_Gibbs:
    def __init__(self, beta, latticeSize, M, N_max):
        super().__init__(beta, latticeSize, M, N_max, 'MH')

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

        p_accept = np.minimum(1.0, np.exp((self.beta) * delta_Hamiltonian))  #
        boolOutArray = self.CustomBernaulli(p_accept)
        finalSpin = np.where(boolOutArray, newSpin, oldSpin)  # Directly update this spin
        return sumVal, finalSpin

    def UpdateSpins(self, x_pos, y_pos, index, resample=False):
        oldSpin = self.initSpins[:, x_pos, y_pos]
        nSum, finalSpin = self.GinalFinalSpinToUpdate(x_pos, y_pos, index)
        deltaSpinChange = finalSpin - oldSpin
        self.initSpins[:, x_pos, y_pos] = finalSpin  # Spins updated in object #This is working fine
        self.Magnetization[:, index] = self.Magnetization[:, index - 1] + deltaSpinChange

def runBasicGibbs(N_max, beta, L, M, initSpins, type):
    model = MCMC_Gibbs(beta, L, M, N_max)  # Initializing Spins
    model.InitializeLattice(initSpins, type)
    model.GenerateMagnetizationData()
    return model

def runBasicMH(N_max, beta, L, M, initSpins, type):
    model = MCMC_MH(beta, L, M, N_max)  # Initializing Spins
    model.InitializeLattice(initSpins, type)
    model.GenerateMagnetizationData()
    return model

def RunMC(N_max, beta, L, initSpins,methodName,type, M = 10):
    logging.debug('debug logging: ')
    if(methodName == 'G'):
        model = runBasicGibbs(N_max, beta, L, M, initSpins, type)
    else:
        model = runBasicMH(N_max, beta, L, M, initSpins, type)

    #Make sure to delete the non needed data
    model.initSpins = 0
    model.update_seq = 0
    model.Magnetization = model.Magnetization[0,:]
    return  model