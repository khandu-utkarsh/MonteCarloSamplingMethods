#Importing Dependencies
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
import pandas as pd

#Functions needed for Part 1 of the Problem 13
def GenerateOneSamplePoint(N):
  generatedRands = np.random.exponential(size = N)
  sampleMean = np.mean(generatedRands)
  return sampleMean

def GetMSamplePoints(M, N):
  currSamples = []
  for i in range(M):
    currSamples.append(GenerateOneSamplePoint(N))
  return np.array(currSamples)

def GetScaledError(M, N):
  points = GetMSamplePoints(M, N)
  return np.sqrt(N) * (points - 1)

def GenerateGraphs(M,N):
  points = GetScaledError(M, N)
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.5))
  ax1.set_title('Histogram')
  ax2.set_title('QQ Plot')
  fig.suptitle('N = ' + str(N))
  ax1.hist(points)
  sm.qqplot(points,fit = True, line = "45", ax = ax2)
  plt.show()
  #name = '/Users/utkarsh/NYU/Monte Carlo Methods/Homeworks/homework1/' + 'histAndQQ' + str(N) + '.png'
  #plt.savefig(name)

#Functions needed for Part 2 of Problem 13
def GenerateOneSamplePointFromGammaSum(N):
  return (np.random.gamma(N,1)/N)

def GetManySamplePoints(M, N):
  currSamples = []
  for i in range(M):
    currSamples.append(GenerateOneSamplePointFromGammaSum(N))
  df = pd.DataFrame(currSamples)
  return df

def ComputeQ_N(M,N):
  pointsFrame = GetManySamplePoints(M, N)
  col = pointsFrame[0]
  greaterCount = col[col > 1.1].count()
  probab = greaterCount/M
  return probab

def ComputeProabDecay(M,N):
  return np.log(ComputeQ_N(M,N))/N

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
  #name = '/Users/utkarsh/NYU/Monte Carlo Methods/Homeworks/homework1/' + 'decayGraph' + '.png'
  #plt.savefig(name)
  plt.show()

def GenerateGraphsForPart1():
  #Computing graphs for Part 1
  M = 100
  N = [1, 10, 100, 1000, 10000]
  for n in N:
    GenerateGraphs(M, n)

def GenerateGraphsForPart2():
  # Computing graphs for Part 2
  counts, probabs = GetProbabilities()
  #print(counts)
  #print(probabs)
  GenerateDecayGraph(counts, probabs)

def Run():
  GenerateGraphsForPart1()
  GenerateGraphsForPart2()

Run()