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


class MCMC:
#Intialize the lattice
  def __init__(self, beta, latticeSize):
      self.L = latticeSize
      self.initSpins = np.zeros((latticeSize, latticeSize), dtype=int)
      self.beta = beta

  def InitializeLattice(self):
    self.initSpins = np.random.randint(0, 2, size = (self.L,self.L)) * 2 - 1 #Randomly create int b/w 0 and 1 and then multiply by 2 and sub by 1

  def CalculateSumNeighborSpins(self, x_pos, y_pos, spins):
    rows, cols = spins.shape
    sum = spins[(x_pos) % rows, (y_pos - 1) % cols] + \
          spins[(x_pos - 1) % rows,(y_pos) % cols] + \
          spins[(x_pos) % rows, (y_pos + 1) % cols] + \
          spins[(x_pos + 1) % rows,(y_pos) % cols]
    return sum

  def GetPositionsDeterministically(self, K):
    pos_list = np.zeros((K,1), dtype=[('x', 'int'), ('y', 'int')])
    nums = self.L * self.L
    count = 0
    while(count < K):
      for iNum in range(nums):
        if(count < K):
          x_pos = int(iNum//self.L)
          y_pos = int(iNum - x_pos * self.L)
          pos_list[count] = (x_pos, y_pos)
          count += 1
    return pos_list

  def GetPositionsRandomly(self, K):
    pos_list = np.zeros((K,1), dtype=[('x', 'int'), ('y', 'int')])
    nums = self.L * self.L
    count = 0
    while(count < K):
      iNum = np.random.randint(0,nums)
      x_pos = int(iNum//self.L)
      y_pos = int(iNum - x_pos * self.L)
      pos_list[count] = (x_pos, y_pos)
      count += 1
    return pos_list

  def CustomBernaulli(self,p):
    x = np.random.uniform()
    if x < p:
      return 1
    else:
      return 0
  	
  def SetParamBeta(self, beta):
    self.beta = beta

  def ChangeLatticeSize(self,L):
    self.L = L
    self.InitializeLattice()

class MCMC_Gibbs(MCMC):#class MCMC_Gibbs:
    def __init__(self, beta, latticeSize):
      super().__init__(beta, latticeSize)
      #print('temp')

    def UpdateSpins(self, x_pos, y_pos, spins):
      sum_neigh = self.CalculateSumNeighborSpins(x_pos, y_pos, spins)
      cond_1_succ = 1/(1 + np.exp(-2 * self.beta * sum_neigh))
      randNum = self.CustomBernaulli(cond_1_succ)
      oldSpin = spins[x_pos][y_pos]
      newSpin = oldSpin
      updatedSpins = spins
      if(randNum == 1): #Success
        updatedSpins[x_pos][y_pos] = 1
        newSpin = 1
      else:
        updatedSpins[x_pos][y_pos] = -1
        newSpin = -1

      return updatedSpins, oldSpin, newSpin

class MCMC_MH(MCMC):#class MCMC_Gibbs:
    def __init__(self, beta, latticeSize):
      super().__init__(beta, latticeSize)

    def FlipSpin(self, oldSpin):
      if oldSpin == 1:
        return -1
      else:
        return 1

    def UpdateSpins(self, x_pos, y_pos, spins):
      oldSpin = spins[x_pos][y_pos]
      newSpin = self.FlipSpin(oldSpin)
      newSum = self.CalculateSumNeighborSpins(x_pos, y_pos, spins)
      prod = np.exp(self.beta * (newSpin * newSum - oldSpin * newSum))
      succProbab = min(1.0, prod)
      #print(type(succProbab))
      randNum = self.CustomBernaulli(succProbab)
      updatedSpins = spins
      finalSpin = oldSpin
      if randNum == 1:  # Success
        updatedSpins[x_pos][y_pos] = newSpin
        finalSpin = newSpin

      return updatedSpins, oldSpin, finalSpin


def GetMagnetization(model, N):
  #Deterministic Positions
  deterministicPos = model.GetPositionsDeterministically(N)
  spins = model.initSpins
  d_count = deterministicPos.shape[0]
  deter_mag = np.zeros((d_count,))
  initialMagD = np.sum(spins)
  for i in range(d_count):
    x_pos = deterministicPos[i][0][0]
    y_pos = deterministicPos[i][0][1]
    updatedSpins, oldSpin, newSpin = model.UpdateSpins(x_pos, y_pos, spins)

    spins = updatedSpins
    initialMagD = initialMagD - oldSpin + newSpin
    deter_mag[i] = initialMagD

  #Random Positions
  randomPos = model.GetPositionsRandomly(N)
  spins = model.initSpins
  r_count = randomPos.shape[0]
  rand_mag = np.zeros((d_count,))
  initialMagR = np.sum(spins)
  for i in range(d_count):
    x_pos = randomPos[i][0][0]
    y_pos = randomPos[i][0][1]
    updatedSpins, oldSpin, newSpin = model.UpdateSpins(x_pos, y_pos, spins)
    spins = updatedSpins
    #magVal = np.sum(spins)
    initialMagR = initialMagR - oldSpin + newSpin
    rand_mag[i] = initialMagR
    #rand_mag[i] = magVal

  return deter_mag, rand_mag


def ComputeIATMultiDim(x):
  return integrated_time(x, c=5, tol=50, quiet=True)

def ComputeIAT(N, model):
    trajectoriesCount = 10
    tauD = np.zeros((trajectoriesCount, 1))
    tauR = np.zeros((trajectoriesCount, 1))
    for i in range(trajectoriesCount):
      d, r = GetMagnetization(model, N)
      #d, r = g.GetMagnetization(model, N)
      tau_deter = integrated_time(d, c=5, tol=50, quiet=True)
      tau_rand = integrated_time(r, c=5, tol=50, quiet=True)
      tauD[i] = tau_deter
      tauR[i] = tau_rand

    return tauD, tauR, d, r

#Sequence type
def GenerateAndSaveGraph(mag, tau, type, beta, L, N, method):
    titleString = method + ' ' + type + " | Sample Size: " + str(N) + " | beta: " + str(beta) + " | L: " + str(L)
    saveFigName = method + '_' + type + "_Sample Size_" + str(N) + "_beta_" + str(beta) + "_L_" + str(L)
    saveFigName = saveFigName.replace('.', 'd')
    trajectoriesCount, _ = tau.shape
    x = np.arange(trajectoriesCount)
    fig, ax = plt.subplots() # fig : figure object, ax : Axes object
    ax.plot(x, tau, '-o');
    ax.set_xlabel('Trajectories Index')
    ax.set_ylabel('Integrated Autocorrelation Time')

    ax.set_title(titleString)
    ax.set_xticks(x)
    #saveName = str(N) + type + method
    plt.savefig(saveFigName)
    plt.close(fig)

    fig, ax = plt.subplots() # fig : figure object, ax : Axes object
    ax.hist(mag);
    ax.set_xlabel('Magnetization')
    ax.set_title(titleString)
    plt.savefig('Histogram_' + saveFigName)
    plt.close(fig)

def runProcessGibbs(N, beta, L):
    start = time.perf_counter()
    #logging.basicConfig(level=logging.DEBUG,format='%(message)s %(threadName)s %(processName)s',)
    model = MCMC_Gibbs(beta, L) #Initializing Spins
    #model = g.MCMC_Gibbs(beta, L) #Initializing Spins
    model.InitializeLattice()
    tauD, tauR, magD, magR = ComputeIAT(N, model)
    GenerateAndSaveGraph(magD, tauD, "Deterministic", model.beta, model.L, N, "G")
    GenerateAndSaveGraph(magR, tauR, "Random", model.beta, model.L, N, "G")
#    GenerateAndSaveGraph(magD, tauD, "Deterministic", model.beta, model.L, N, "Gibbs")
#    GenerateAndSaveGraph(magR, tauR, "Random", model.beta, model.L, N, "Gibbs")
    end = time.perf_counter()
    return end - start
    #print('Time taken by Gibbs process: ',(end - start), '| N: ', N)

def runProcessMH(N, beta, L):
    start = time.perf_counter()
    #logging.basicConfig(level=logging.DEBUG,format='%(message)s %(threadName)s %(processName)s',)
    model = MCMC_MH(beta, L) #Initializing Spins
    #model = g.MCMC_MH(beta, L) #Initializing Spins
    model.InitializeLattice()
    tauD, tauR, magD, magR = ComputeIAT(N, model)
    GenerateAndSaveGraph(magD, tauD, "Deterministic", model.beta, model.L, N, "MH")
    GenerateAndSaveGraph(magR, tauR, "Random", model.beta, model.L, N, "MH")
    end = time.perf_counter()
    return end - start

def RunMCMC_Both(N, beta = 0.0005, L = 50):
  gibsTimeTaken = runProcessGibbs(N, beta, L)
  mhTimeTaken = runProcessMH(N, beta, L)