import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('emcee')
install('matplotlib')
install('numpy')
install('scipy')
#install('logging')
install('joblib')

import time
import math
import numpy as np
import matplotlib.pyplot as plt
from LangevinSchemes import GenerateSamplesFromOverdampled, GenerateSamplesFromHybrid, GenerateSampleFromUnderdamped

import logging
from joblib import Parallel, delayed

logging.basicConfig(level=logging.DEBUG,
                    format='%(message)s %(threadName)s %(processName)s',
                    )

#All of them working individually
def RunSimulationQuestion_64(beta, L, M, h, initialValues, scaleupConst):
    method_name = ['notM', 'M']
    #First one is without notMH and second one is with MH
    out = Parallel(n_jobs = 2, verbose = 10)(delayed(GenerateSamplesFromOverdampled)(beta, L, M, h, method, initialValues, scaleupConst) for method in method_name)
    return out

def RunSimulationQuestion_65(beta, L, M, h, initialValues, scaleupConst, n = 100):
    method_name = ['notM', 'M']
    #First one is without notMH and second one is with MH
    out = Parallel(n_jobs = 2, verbose = 10)(delayed(GenerateSamplesFromHybrid)(beta, L, M, h, method, initialValues, scaleupConst, n) for method in method_name)
    return out

def RunSimulationQuestion_66(beta, L, M, h, gamma, initialValues, scaleupConst):
    method_name = ['notM', 'M']
    #First one is without notMH and second one is with MH
    out = Parallel(n_jobs = 2, verbose = 10)(delayed(GenerateSampleFromUnderdamped)(beta, L, M, h, method, initialValues, scaleupConst, gamma) for method in method_name)
    return out


def RunTestsOnQuestion_64(beta, L, M, initialValues, hs, scaleupConst):
    out = Parallel(n_jobs = 4, verbose = 10)(delayed(RunSimulationQuestion_64)(beta, L, M, h, initialValues, scaleupConst) for h in hs)
    return out

def RunTestsOnQuestion_65(beta, L, M, initialValues, hs, scaleupConst, n = 100):
    out = Parallel(n_jobs = 4, verbose = 10)(delayed(RunSimulationQuestion_65)(beta, L, M, h, initialValues, scaleupConst, n) for h in hs)
    return out

def MidRunForGammas(beta, L, M, initalValues, h, gamas,scaleupConst):
    out = Parallel(n_jobs = 1, verbose = 10)(delayed(RunSimulationQuestion_66)(beta, L, M, h, gamma, initalValues, scaleupConst) for gamma in gamas)
    return out

def RunTestsOnQuestion_66(beta, L, M, initialValues, hs, gammas, scaleupConst):
    out = Parallel(n_jobs= 4, verbose= 10)(delayed(MidRunForGammas)(beta, L, M, initialValues, h, gammas, scaleupConst)for h in hs)
    return out

#exerciseName = '64' # '65' # '66'
def PlotGraphsForAllModelsIndividually(output, exerciseName):
    out = np.array(output)
    rows, cols = out.shape
    for r in range(rows):
        for c in range(cols):
            curr_model = out[r,c]
            totalNs = len(curr_model.IAT_sliceIndexOnePast)
            for i in range(totalNs):
                curr_model.GenerateIATGraphs(exerciseName, i)
                curr_model.GenerateHistograms(exerciseName, i)


def PlotGraphsOfMeanIATs(output, ModelName):
    out = np.array(output)
    rows, cols = out.shape
    #Every row is different value of h
    #Every col is two methods

    for row in range(rows):
        fig, ax = plt.subplots()  # fig : figure object, ax : Axes object
        ax.set_xlabel('Samples Count')
        ax.set_ylabel('Mean of IAT')
        titleString = ModelName + ' Mean_IAT ' + ' '+ " | h: " + str(out[row][0].h)
        saveFigName = ModelName + '_Mean_IAT_' + '_' + 'h_' + str(out[row][0].h)
        saveFigName = saveFigName.replace('.', 'd')
        for col in range(cols):
            N_array = out[row][col].IAT_sliceIndexOnePast
            mean_IATs = np.mean(out[row][col].IATs, axis=0)  # Should be of shape N,
            labelString = out[row][col].method
            ax.plot(N_array, mean_IATs, '-o', label=labelString);
        ax.legend()
        ax.set_title(titleString)
        plt.savefig(saveFigName)
        plt.close(fig)


def PlotGraphsOfMeanIATsForExercise66(output, ModelName):
    out = np.array(output)
    rows, cols, types = out.shape

    for row in range(rows):
        fig, ax = plt.subplots()  # fig : figure object, ax : Axes object
        ax.set_xlabel('Samples Count')
        ax.set_ylabel('Mean of IAT')
        titleString = ModelName + ' Mean_IAT ' + ' '+ " | h: " + str(out[row][0][0].h)
        saveFigName = ModelName + '_Mean_IAT_' + '_' + 'h_' + str(out[row][0][0].h)
        saveFigName = saveFigName.replace('.', 'd')
        for col in range(cols):
            for type in range(types):
                N_array = out[row][col][type].IAT_sliceIndexOnePast
                mean_IATs = np.mean(out[row][col][type].IATs, axis=0)  # Should be of shape N,
                labelString = out[row][col][type].method + '| gamma: ' + str(out[row][col][type].gamma)
                ax.plot(N_array, mean_IATs, '-o', label=labelString);
        ax.legend()
        ax.set_title(titleString)
        plt.savefig(saveFigName)
        plt.close(fig)

#Test All the Code
beta = 0.2
L = 3
M = 10
n = 100
initialValues = np.random.random_sample((L, 1))
ieee_remainder = np.vectorize(math.remainder)
initialValues = ieee_remainder(initialValues, math.tau)

hs = [1/2., 1/4., 1/8., 1/16., 1/32., 1/64.,1/128.]
hs = [1/2., 1/4., 1/8.]
gammas = [1e-1, 1, 1e1]
scaleupConst = 1e5
scaleupConst = 1e3

