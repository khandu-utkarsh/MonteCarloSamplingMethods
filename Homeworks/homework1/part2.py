import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
import pandas as pd

def GenerateOneSamplePointFromGammaSum(N):
  return (np.random.gamma(N,1)/N)

def GetManySamplePoints(M, N):
  currSamples = []
  for i in range(M):
    currSamples.append(GenerateOneSamplePointFromGammaSum(N))
  df = pd.DataFrame(currSamples)
  return df

def ComputeGivenProbability(M,N):
  pointsFrame = GetManySamplePoints(M, N)
  col = pointsFrame[0]
  greaterCount = col[col > 1.1].count()
  probab = greaterCount/M
  return probab

def ComputeProabDecay(M,N):
  return np.log(ComputeGivenProbability(M,N))/N

def GetProbabilities():
  M = 100000
  N = []
  for i in range(1,31):
    N.append(i * 50)
  probabs = []
  for n in N:
    probabs.append(ComputeProabDecay(M,n))
  return N, probabs

def GenerateDecayGraph(counts, probabs):
  fig, axs = plt.subplots(nrows=1, ncols=1,  figsize=(16, 8))
  axs.plot(counts, probabs)
  axs.invert_yaxis()
  axs.set_title('Decay of tail probability')
  axs.set_xlabel('N')
  axs.set_ylabel('log(P(x - 1 > 0.1))/N')
  name = '/Users/utkarsh/NYU/Monte Carlo Methods/Homeworks/homework1/' + 'decayGraph' + '.png'
  plt.savefig(name)


counts, probabs = GetProbabilities()
GenerateDecayGraph(counts, probabs)