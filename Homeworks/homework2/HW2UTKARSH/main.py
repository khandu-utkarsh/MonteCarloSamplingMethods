import numpy as np
from matplotlib import pyplot as plt
import time
from scipy.stats import norm
import math
from tabulate import tabulate


def GenerateNSamplesFromDistribution(N):
  points =  np.random.uniform(0,1,N)
  return (points ** 2)

def GivenDensity(x):
  return 1/(2 * np.sqrt(x))

def GetnerateQQPlot(samples):
  percent = (np.ones(samples.shape))/(len(samples))
  cumPercent = np.cumsum(percent)
  sortedSamples = np.sort(samples)
  theo = np.sqrt(sortedSamples)

  x_axis = np.linspace(0,1,100)
  y_axis = np.linspace(0,1,100);
  plt.scatter(theo, cumPercent, c='green')
  plt.plot(theo, cumPercent, x_axis, y_axis)
  plt.title('QQ Plot')
  plt.xlabel('Theoretical')
  plt.ylabel('Sampled')
  name = '/Users/utkarsh/NYU/Monte Carlo Methods/Homeworks/homework2/' + 'exercise16QQPlot' + '.png'
  plt.show()
  #plt.savefig(name)

def ExecuteExercise16():
    xa = np.linspace(0, 1, 80)
    xa = xa[1:-1]
    curve = GivenDensity(xa)

    name = '/Users/utkarsh/NYU/Monte Carlo Methods/Homeworks/homework2/' + 'exercise16Histogram' + '.png'
    plt.hist(GenerateNSamplesFromDistribution(100000), density = True)
    plt.plot(xa, curve, 'k', linewidth=2)
    title = "Histogram and the given density"
    plt.title(title)
    plt.show()
    #plt.savefig(name)

    plt.clf()
    GetnerateQQPlot(GenerateNSamplesFromDistribution(500))

# Exercise 18 code starts
def GenerateUniformPointsOnADisc(N):
  u1 = np.random.uniform(0,1,N)
  u2 = np.random.uniform(0,1,N)
  r = np.sqrt(u1)
  theta = 2 * np.pi * u2
  iters = 2 * N

  x = np.multiply(r, np.cos(theta))
  y = np.multiply(r, np.sin(theta))
  return (x, y, iters)

def Plot2DHistrogramAndScatterPlot(x, y, grpahName):
  fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,  figsize=(16, 7))
  ax1.set_title('2D Histogram')
  ax2.set_title('Scatter Plot')
  fig.suptitle('Uniform Distribution over a unit disc')
  ax1.hist2d(x,y, density= True)
  ax2.scatter(x,y)

  #plt.savefig(grpahName)
  plt.show()

def ExecuteExercise18():
    N_disc = 10000
    t_disc_initial = time.process_time()
    x, y, iterCounts = GenerateUniformPointsOnADisc(N_disc)
    t_disc_final = time.process_time()
    time_per_sample = (t_disc_final - t_disc_initial) / N_disc
    print('Time taken to generate one sample (in seconds):  ', time_per_sample)
    print('Mean iterations required to for exercise 18: ', iterCounts / N_disc)
    x, y, counts = GenerateUniformPointsOnADisc(1000)
    name = '/Users/utkarsh/NYU/Monte Carlo Methods/Homeworks/homework2/' + 'exercise18Diagrams' + '.png'
    Plot2DHistrogramAndScatterPlot(x, y, name)


#Functions needed for exercise 19
def GenerateOneSamplePointUsingRejectionSampling(k):
    iters = 0
    while(True):
      xC =  np.random.uniform(-1,1,1)
      yC =  np.random.uniform(-1,1,1)
      u =  np.random.uniform(0,1,1)
      iters += 1
      if (xC ** 2 + yC ** 2 < 1 and u < 4/(math.pi * k)):
        break
    return (xC, yC, iters)

def GenerateNSamples(N,k):
  X = []
  Y = []
  counts = []
  timeElapsed = 0.
  for i in range(N):
    t_init = time.process_time()
    x, y, iters = GenerateOneSamplePointUsingRejectionSampling(k)
    t_fin = time.process_time()
    timeElapsed = timeElapsed + (t_fin - t_init)
    X.append(x.item())
    Y.append(y.item())
    counts.append(iters)
  timeElapsed = timeElapsed/N
  return X, Y, counts, timeElapsed

def ExecuteExercise19():
    X, Y, counts, timeElapsed = GenerateNSamples(2000, 4 / math.pi)
    name = '/Users/utkarsh/NYU/Monte Carlo Methods/Homeworks/homework2/' + 'exercise19Diagrams' + '.png'
    Plot2DHistrogramAndScatterPlot(X, Y, name)
    print("Time taken for generating one sample (in seconds): ", timeElapsed)
    print("Mean iterations needed to compute on sample: ", np.mean(counts))


#Functions for exercsie 20
def SamplePoint(m, sigma):
  return np.random.normal(m, sigma)

def GetWeight(x, m, sigma):
  return 1/( sigma) * np.exp(1/2 * (((x - m) ** 2)/(sigma * sigma) - x**2/1))

def CalculateProbabilityExercise20(m, sigma, N):
  samples = np.random.normal(m, sigma, N)
  weights = GetWeight(samples, m, sigma)
  indicators = np.where(samples > 2, 1, 0)

  fw = np.multiply(indicators, weights)
  weightsSum = np.sum(weights)

  expectation = np.mean(fw)

  trueMean = 1 - norm.cdf(2,loc = 0,scale = 1)

#  fMinusMu = fw - expectation
  fMinusMu = fw - trueMean
  squareFMinusMu = fMinusMu ** 2

  variance = np.mean(squareFMinusMu)
  rmse = np.sqrt(variance/N)

  return (np.around(expectation, 6), np.around(variance , 6), np.around(rmse , 6))

def ExecuteExercise20():
    means = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    sigmas = [1.00, 1.50, 2.00, 2.50, 3.00]
    data = []
    for i in range(len(means)):
        for j in range(len(sigmas)):
            m_20, v_20, r_20 = CalculateProbabilityExercise20(means[i], sigmas[j], 10000)
            data.append((means[i], sigmas[j], m_20, v_20, r_20))

    print(tabulate(data, headers=['m', 'sigma', 'Mean', 'Variance', 'RMSE']))
def ComputeNormalizationConstant(N):
  sum = 0
  for i in range(N):
    x = np.random.normal(0,1)
    sum += (math.exp(-(abs(x)** 3)))/(math.exp((-x * x/2)))
  Z_p = math.sqrt(2 * math.pi)/N * sum
  return Z_p

def ExecuteExercise21():
    Z = []
    Ns = []
    for i in range(1, 21):
        N = 10000 * i
        Ns.append(N)
        Z.append(ComputeNormalizationConstant(N))
    plt.ylim(0, 2)
    plt.plot(Ns, Z)
    groupName = '/Users/utkarsh/NYU/Monte Carlo Methods/Homeworks/homework2/' + 'exercise20Diagrams' + '.png'
    plt.title('Sample Size vs Normalization Constant')
    plt.xlabel('N')
    plt.ylabel('Normalization Constant')
    #plt.savefig(groupName)

def CalculateProbabilityExercise22(m, sigma, N):
  samples = np.random.normal(m, sigma, N)
  weights = GetWeight(samples, m, sigma)
  indicators = np.where(samples > 2, 1, 0)
  weightsSum = np.sum(weights)

  normalizedWeights = weights/weightsSum

  fnw = np.multiply(indicators, normalizedWeights)
  expectation = np.sum(fnw)

  trueMean = 1 - norm.cdf(2,loc = 0,scale = 1)


#  fMinusMu = fnw - expectation
  fMinusMu = fnw - trueMean
  squareFMinusMu = fMinusMu ** 2

  variance = np.mean(squareFMinusMu)
  rmse = np.sqrt(variance/N)

  return (np.around(expectation, 6), np.around(variance , 6), np.around(rmse , 6))


def PlotExpectationVSm():
    means = np.linspace(-10,10,1000)
    expectations = np.zeros(means.size)
    for i in range(len(means)):
        m_20, v_20, r_20 = CalculateProbabilityExercise22(means[i],1,10000)
        expectations[i] = m_20

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    ax1.set_title('Scatter Plot')
    ax2.set_title('Continuous Curve')
    ax1.set_xlabel('m')
    ax1.set_ylabel('Expectation')
    ax2.set_xlabel('m')
    ax2.set_ylabel('Expectation')

    fig.suptitle('Plots with varying values of mean but std = 1')
    ax1.scatter(means, expectations)

    ax2.plot(means, expectations)
    groupName = '/Users/utkarsh/NYU/Monte Carlo Methods/Homeworks/homework2/' + 'exercise22comparisons' + '.png'
    #groupName = '/Users/utkarsh/NYU/Monte Carlo Methods/Homeworks/homework2/' + 'exercise20comparisons' + '.png'
    #plt.savefig(groupName)

def ExecuteExercise22():
    means = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    sigmas = [1.00, 1.50, 2.00, 2.50, 3.00]

    data = []
    for i in range(len(means)):
        for j in range(len(sigmas)):
            m_20, v_20, r_20 = CalculateProbabilityExercise20(means[i], sigmas[j], 10000)
            m_22, v_22, r_22 = CalculateProbabilityExercise22(means[i], sigmas[j], 10000)
            data.append((means[i], sigmas[j], m_20, r_20, m_22, r_22))

    print(tabulate(data, headers=['m', 'sigma', 'Ex 20: Mean', 'Ex 20: RMSE', 'Ex 22: Mean', 'Ex 22: RMSE']))

#Caller function for exercise 16
def Run():
    ExecuteExercise16()
    ExecuteExercise19()
    ExecuteExercise20()
    ExecuteExercise21()
    ExecuteExercise22()
    PlotExpectationVSm()


Run()