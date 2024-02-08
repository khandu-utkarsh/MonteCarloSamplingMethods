import subprocess
import sys
import time
import logging

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('joblib')
from joblib import Parallel, delayed
import MCMC as mcmc
import numpy as np
import matplotlib.pyplot as plt


logging.debug('debug logging: ')

L = 50
beta = 0.0005
N_max = 20000000
#N_max = 2000

initializedSpinValues = np.random.randint(0, 2, size=(L, L)) * 2 - 1  # Randomly create int b/w 0 and 1 and then multiply by 2 and sub by 1

def BothMethodRun(initSpins, L, beta, N_max, method):
    twoTypes = ['Random', 'Deterministic']
    out = Parallel(n_jobs=2,verbose=10)(delayed(mcmc.RunMC)(N_max, beta, L, initSpins, method, type) for type in twoTypes)
    return out
def MainCall(initSpins, L, beta, N_max):
    methods = ['G', 'MH']
    trueOut = Parallel(n_jobs=2,verbose=10)(delayed(BothMethodRun)(initSpins, L, beta, N_max, method) for method in methods)
    return trueOut

result = MainCall(initializedSpinValues, L, beta, N_max)

print('Graph Plotting Started')
#This is list of list
npresult = np.array(result)
npresult = npresult.flatten()

N = []
Means = []
StDeviation = []
Methods = []
Types = []

for i in range(npresult.size):
        npresult[i].GenerateGraphs()
        N.append(npresult[i].Ns)
        Means.append(np.mean(npresult[i].IATs, axis =0))
        StDeviation.append(np.std(npresult[i].IATs, axis = 0))
        Methods.append(npresult[i].method)
        Types.append(npresult[i].type)

#Mean
fig, ax = plt.subplots()  # fig : figure object, ax : Axes object
for i in range(len(N)):
    labelString = Methods[i] + '_' + Types[i]
    ax.plot(N[i], Means[i], '-o', label = labelString);

ax.set_xlabel('Samples Count')
ax.set_ylabel('Mean of IAT')
ax.legend()
titleString = 'Mean_IAT ' + " | beta: " + str(beta) + " | L: " + str(L)
saveFigName = 'Mean_IAT_' + '_' + "Nmax_" + str(N_max) + "_beta_" + str(beta) + "_L_" + str(L)
saveFigName = saveFigName.replace('.', 'd')
ax.set_title(titleString)
plt.savefig(saveFigName)
plt.close(fig)

#Standard Deviation
fig, ax = plt.subplots()  # fig : figure object, ax : Axes object
for i in range(len(N)):
    labelString = Methods[i] + '_' + Types[i]
    ax.plot(N[i], StDeviation[i], '-o', label = labelString);

ax.set_xlabel('Samples Count')
ax.set_ylabel('Mean of IAT')
ax.legend()
titleString = 'StdDeviation_IAT ' + " | beta: " + str(beta) + " | L: " + str(L)
saveFigName = 'StdDeviation_IAT_' + '_' + "Nmax_" + str(N_max) + "_beta_" + str(beta) + "_L_" + str(L)
saveFigName = saveFigName.replace('.', 'd')
ax.set_title(titleString)
plt.savefig(saveFigName)
plt.close(fig)
