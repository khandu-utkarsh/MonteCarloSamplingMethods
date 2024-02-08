import subprocess
import sys
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('joblib')
from joblib import Parallel, delayed
import MCMC as mcmc
import numpy as np
import time
import logging
import matplotlib.pyplot as plt

logging.debug('debug logging: ')



N_max = 10000000 #10 million samples
#N_max = 1000 #10 million samples
betas = [0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
L = 30

initializedSpinValues = np.random.randint(0, 2, size=(L, L)) * 2 - 1  # Randomly create int b/w 0 and 1 and then multiply by 2 and sub by 1

def TypesParallelized(initSpins, L, beta, N_max, method):
    twoTypes = ['Random', 'Deterministic']
    out = Parallel(n_jobs=2)(delayed(mcmc.RunMC)(N_max, beta, L, initSpins, method, type) for type in twoTypes)
    return out
def MethodsParallelized(initSpins, L, beta, N_max):
    methods = ['G', 'MH']
    trueOut = Parallel(n_jobs=2)(delayed(TypesParallelized)(initSpins, L, beta, N_max, method) for method in methods)
    return trueOut

def BetaParallelized(initSpins, L, betas, N_max):
    betaOut = Parallel(n_jobs=len(betas))(delayed(MethodsParallelized)(initSpins, L, beta, N_max) for beta in betas)
    return  betaOut

out = BetaParallelized(initializedSpinValues, L, betas, N_max)
out = np.array(out)
out = np.reshape(out, (-1,4))

rows, cols = out.shape

for col in range(cols):
    N = np.zeros((rows, 20), dtype=int)
    Means = np.zeros((rows, 20))
    StdDevs = np.zeros((rows, 20))
    for row in range(rows):
        out[row,col].GenerateGraphs()
        N[row,:] = out[row,col].Ns
        Means[row, :] = np.mean(out[row,col].IATs, axis = 0)
        StdDevs[row,:] = np.std(out[row,col].IATs, axis = 0)

    #Mean
    fig, ax = plt.subplots()  # fig : figure object, ax : Axes object
    rc, cc = N.shape
    for r in range(rc):
        labelString = 'beta: ' + str(betas[r])
        ax.plot(N[r], Means[r], '-o', label = labelString);

    ax.set_xlabel('Samples Count')
    ax.set_ylabel('Mean of IAT')
    ax.legend()
    titleString = 'Mean_IAT ' + " | " + str(out[0,col].method)  + " | " + str(out[0,col].type) + " | " + " | L: " + str(L)
    saveFigName = 'Variation_beta_Mean_IAT_' + str(out[0,col].method)  + '_' + str(out[0,col].type) + '_' + "L_" + str(L)
    saveFigName = saveFigName.replace('.', 'd')
    ax.set_title(titleString)
    plt.savefig(saveFigName)
    plt.close(fig)

    #Standard Deviation
    fig, ax = plt.subplots()  # fig : figure object, ax : Axes object
    rc, cc = N.shape
    for r in range(rc):
        labelString = 'beta: ' + str(betas[r])
        ax.plot(N[r], StdDevs[r], '-o', label = labelString);

    ax.set_xlabel('Samples Count')
    ax.set_ylabel('Standard Deviation of IAT')
    ax.legend()
    titleString = 'StdDeviation_IAT ' + " | " + str(out[0,col].method)  + " | " + str(out[0,col].type) + " | " + " | L: " + str(L)
    saveFigName = 'Variation_beta_StdDeviation_IAT_' + str(out[0,col].method)  + '_' + str(out[0,col].type) + '_' + "L_" + str(L)
    saveFigName = saveFigName.replace('.', 'd')
    ax.set_title(titleString)
    plt.savefig(saveFigName)
    plt.close(fig)