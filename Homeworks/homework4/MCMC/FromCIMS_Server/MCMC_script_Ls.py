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
beta = 0.1
Ls = [10,20,30,40,50,60,70,80,90,100,110,120]

def TypesParallelized(initSpins, L, beta, N_max, method):
    twoTypes = ['Random', 'Deterministic']
    out = Parallel(n_jobs=2)(delayed(mcmc.RunMC)(N_max, beta, L, initSpins, method, type) for type in twoTypes)
    return out
def MethodsParallelized(initSpins, L, beta, N_max):
    methods = ['G', 'MH']
    trueOut = Parallel(n_jobs=2)(delayed(TypesParallelized)(initSpins, L, beta, N_max, method) for method in methods)
    return trueOut

def LParallelized(Ls, beta, N_max):
    lOut = Parallel(n_jobs=len(Ls))(delayed(MethodsParallelized)(np.random.randint(0, 2, size=(L, L)) * 2 - 1, L, beta, N_max) for L in Ls)
    return  lOut

out = LParallelized(Ls, beta, N_max)
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
        labelString = 'L: ' + str(Ls[r])
        ax.plot(N[r], Means[r], '-o', label = labelString);

    ax.set_xlabel('Samples Count')
    ax.set_ylabel('Mean of IAT')
    ax.legend()
    titleString = 'Mean_IAT ' + " | " + str(out[0,col].method)  + " | " + str(out[0,col].type) + " | " + " | beta: " + str(beta)
    saveFigName = 'Variation_beta_Mean_IAT_' + str(out[0,col].method)  + '_' + str(out[0,col].type) + '_' + "beta_" + str(beta)
    saveFigName = saveFigName.replace('.', 'd')
    ax.set_title(titleString)
    plt.savefig(saveFigName)
    plt.close(fig)

    #Standard Deviation
    fig, ax = plt.subplots()  # fig : figure object, ax : Axes object
    rc, cc = N.shape
    for r in range(rc):
        labelString = 'L: ' + str(Ls[r])
        ax.plot(N[r], StdDevs[r], '-o', label = labelString);

    ax.set_xlabel('Samples Count')
    ax.set_ylabel('Standard Deviation of IAT')
    ax.legend()
    titleString = 'StdDeviation_IAT ' + " | " + str(out[0,col].method)  + " | " + str(out[0,col].type) + " | " + " | beta: " + str(beta)
    saveFigName = 'Variation_beta_StdDeviation_IAT_' + str(out[0,col].method)  + '_' + str(out[0,col].type) + '_' + "beta_" + str(beta)
    saveFigName = saveFigName.replace('.', 'd')
    ax.set_title(titleString)
    plt.savefig(saveFigName)
    plt.close(fig)