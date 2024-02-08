import subprocess
import sys
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('joblib')
from joblib import Parallel, delayed
import MCMC as mcmc

#Vary b)
print('Part b started')
N = 5000000 #5 million samples
betas = [0.000025, 0.00005, 0.000075, 0.00025, 0.0005,0.00075 , 0.0025, 0.005, 0.0075, 0.025, 0.05, 0.075, 0.25, 0.5, 0.75]
results = Parallel(n_jobs=16)(delayed(mcmc.RunMCMC_Both)(N,beta, 50) for beta in betas)
print('Part b ended')
