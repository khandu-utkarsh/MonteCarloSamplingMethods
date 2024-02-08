import subprocess
import sys
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('joblib')
from joblib import Parallel, delayed
import MCMC as mcmc
import numpy as np
L = 50
beta = 0.005
N_max = 2000
initializedSpinValues = np.random.randint(0, 2, size=(L, L)) * 2 - 1  # Randomly create int b/w 0 and 1 and then multiply by 2 and sub by 1

fourCases = [('G', 'Random'),('G','Deterministic'),('MH','Random'),('MH','Deterministic')]

def RunMC(N_max, beta, L, initSpins,type, methodName):
    results = Parallel(n_jobs=4)(delayed(mcmc.RunMC)(N_max, beta, L, initializedSpinValues, case[0], case[1])
                                 for case in fourCases)
