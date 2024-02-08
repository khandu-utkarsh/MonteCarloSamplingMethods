import subprocess
import sys
import time

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

def runProcessGibbs(N, beta = 0.0005, L = 50):
    start = time.perf_counter()
    #logging.basicConfig(level=logging.DEBUG,format='%(message)s %(threadName)s %(processName)s',)
    model = g.MCMC_Gibbs(beta, L) #Initializing Spins
    #model = g.MCMC_Gibbs(0.0005, 50) #Initializing Spins
    model.InitializeLattice()
    tauD, tauR, magD, magR = ComputeIAT(N, model)
    GenerateAndSaveGraph(magD, tauD, "Deterministic", model.beta, model.L, N, "G")
    GenerateAndSaveGraph(magR, tauR, "Random", model.beta, model.L, N, "G")
#    GenerateAndSaveGraph(magD, tauD, "Deterministic", model.beta, model.L, N, "Gibbs")
#    GenerateAndSaveGraph(magR, tauR, "Random", model.beta, model.L, N, "Gibbs")
    end = time.perf_counter()
    #print('Time taken by Gibbs process: ',(end - start), '| N: ', N)

def runProcessMH(N, beta = 0.0005, L = 50):
    start = time.perf_counter()
    #logging.basicConfig(level=logging.DEBUG,format='%(message)s %(threadName)s %(processName)s',)
    model = g.MCMC_MH(beta, L) #Initializing Spins
    model.InitializeLattice()
    tauD, tauR, magD, magR = ComputeIAT(N, model)
    GenerateAndSaveGraph(magD, tauD, "Deterministic", model.beta, model.L, N, "MH")
    GenerateAndSaveGraph(magR, tauR, "Random", model.beta, model.L, N, "MH")
#    GenerateAndSaveGraph(magD, tauD, "Deterministic", model.beta, model.L, N, "MetropolisHastings")
#    GenerateAndSaveGraph(magR, tauR, "Random", model.beta, model.L, N, "MetropolisHastings")
    end = time.perf_counter()
    #print('Time taken by MH process: ',(end - start), '| N: ', N)


#This is part a)

print('Part a started')
sampleSizes = []
Ns = 20
constSize = 1000000
for i in range(Ns):
    N = constSize + constSize * i
    sampleSizes.append(N)

resultsG = Parallel(n_jobs=len(sampleSizes))(delayed(runProcessGibbs)(i) for i in sampleSizes)
resultsMH = Parallel(n_jobs=len(sampleSizes))(delayed(runProcessMH)(i) for i in sampleSizes)
print('Part a ended')

#Vary b)
print('Part b started')
N = 10000000 #10 million samples
betas = [0.000025, 0.00005, 0.000075, 0.00025, 0.0005,0.00075 , 0.0025, 0.005, 0.0075, 0.025, 0.05, 0.075, 0.25, 0.5, 0.75]
resultsG_b = Parallel(n_jobs=len(betas))(delayed(runProcessGibbs)(N,beta, 50) for beta in betas)
resultsMH_b = Parallel(n_jobs=len(betas))(delayed(runProcessMH)(N,beta, 50) for beta in betas)
print('Part b ended')


#Vary L)
print('Part c started')
N = 10000000 #10 million samples
beta = 0.005
Ls = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
resultsG_b = Parallel(n_jobs=len(Ls))(delayed(runProcessGibbs)(N,beta, L) for L in Ls)
resultsMH_b = Parallel(n_jobs=len(Ls))(delayed(runProcessMH)(N,beta, L) for L in Ls)
print('Part c ended')