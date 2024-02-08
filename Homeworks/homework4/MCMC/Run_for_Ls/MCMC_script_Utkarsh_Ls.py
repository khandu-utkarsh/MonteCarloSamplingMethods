import subprocess
import sys
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('joblib')
from joblib import Parallel, delayed
import MCMC as mcmc

#Vary L)
print('Part c started')
N = 5000000 #5 million samples
beta = 0.005
Ls = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,110,120]
results = Parallel(n_jobs=16)(delayed(mcmc.RunMCMC_Both)(N,beta, L) for L in Ls)
print('Part c ended')