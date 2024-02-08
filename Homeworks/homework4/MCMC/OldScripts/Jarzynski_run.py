import subprocess
import sys
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

#install('emcee')
install('joblib')
install('numpy')
install('matplotlib')

import numpy as np
import MCMC as g
import JarzynskiMCMC as jmc

from joblib import Parallel, delayed
import matplotlib.pyplot as plt


def PlotHistogram(magnetization,L,M,N):
    fig, ax = plt.subplots() # fig : figure object, ax : Axes object
    ax.hist(magnetization);
    ax.set_xlabel('Magnetization')
    titleString = 'Jarzynski ' + '|' + 'L:' +str(L) +  ' M:' + str(M) + ' N:' + str(N)
    saveString = 'Jarzynski' + '_' + '_L:' +str(L) +  '_M:' + str(M) + '_N:' + str(N)
    ax.set_title(titleString)
    plt.savefig(saveString)

def runProcessJarzynski(L,M,N):
    model = jmc.Jarzynski(0.005,L,M,N)
    model.InitializeLattice()
    magnetization = model.GenerateMagnetizationSamples()
    PlotHistogram(magnetization,L,M,N)


runProcessJarzynski(10,1000,1000)