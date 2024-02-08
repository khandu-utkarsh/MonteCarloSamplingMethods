import subprocess
import sys
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('emcee')
install('joblib')
install('numpy')
install('matplotlib')
#install('logging')
import numpy as np
import MCMC as g
from joblib import Parallel, delayed
import multiprocessing
from emcee.autocorr import AutocorrError, integrated_time
import matplotlib.pyplot as plt
import logging
def ComputeIATMultiDim(x):
  return integrated_time(x, c=5, tol=50, quiet=True)

def ComputeIAT(N, model):
    trajectoriesCount = 10
    tauD = np.zeros((trajectoriesCount, 1))
    tauR = np.zeros((trajectoriesCount, 1))
    for i in range(trajectoriesCount):
      d, r = g.GetMagnetization(model, N)
      tau_deter = integrated_time(d, c=5, tol=50, quiet=True)
      tau_rand = integrated_time(r, c=5, tol=50, quiet=True)
      tauD[i] = tau_deter
      tauR[i] = tau_rand

    return tauD, tauR

def GenerateAndSaveGraph(tau, type, N, method):
    trajectoriesCount, _ = tau.shape
    x = np.arange(trajectoriesCount)
    fig, ax = plt.subplots() # fig : figure object, ax : Axes object
    ax.plot(x, tau, '-o');
    ax.set_xlabel('Trajectories Index')
    ax.set_ylabel('Integrated Autocorrelation Time')
    titleString = type + " | Sample Size: " + str(N)
    ax.set_title(titleString)
    ax.set_xticks(x)
    saveName = str(N) + type + method
    plt.savefig(saveName)


def runProcessGibbs(N):
    logging.basicConfig(level=logging.DEBUG,format='%(message)s %(threadName)s %(processName)s',)
    model = g.MCMC_Gibbs(0.0005, 50) #Initializing Spins
    model.InitializeLattice()
    tauD, tauR = ComputeIAT(N, model)
    GenerateAndSaveGraph(tauD, "Deterministic", N, "_Gibbs")
    GenerateAndSaveGraph(tauR, "Random", N, "_Gibbs")

def runProcessMH(N):
    model = g.MCMC_MH(0.0005, 50) #Initializing Spins
    model.InitializeLattice()
    tauD, tauR = ComputeIAT(N, model)
    GenerateAndSaveGraph(tauD, "Deterministic", N, "_MetropolisHastings")
    GenerateAndSaveGraph(tauR, "Random", N, "_MetropolisHastings")

sampleSizes = []
Ns = 10
constSize = 1000#1000000
for i in range(Ns):
    N = constSize + constSize * i
    sampleSizes.append(N)

#runProcessGibbs(12000)
#runProcessMH(10000)
resultsD = Parallel(n_jobs=len(sampleSizes), backend="threading")(delayed(runProcessGibbs)(i) for i in sampleSizes)
#resultsR = Parallel(n_jobs=len(sampleSizes), backend="threading")(delayed(runProcessMH)(i) for i in sampleSizes)

#sampleSizes = []
#Ns = 15
#constSize = 1000#1000000
#for i in range(10,Ns):
#    N = constSize + constSize * i
#    sampleSizes.append(N)

#runProcessGibbs(12000)
#runProcessMH(10000)
#resultsD = Parallel(n_jobs=len(sampleSizes), backend="threading")(delayed(runProcessGibbs)(i) for i in sampleSizes)
#resultsR = Parallel(n_jobs=len(sampleSizes), backend="threading")(delayed(runProcessMH)(i) for i in sampleSizes)
