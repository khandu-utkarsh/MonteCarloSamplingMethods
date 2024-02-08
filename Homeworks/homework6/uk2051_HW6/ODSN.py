import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

#install('emcee')
#install('matplotlib')
#install('numpy')
#install('scipy')
#install('logging')
#install('joblib')

from emcee.autocorr import AutocorrError, integrated_time
from AuxilaryFxns import *

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm
import math

#Base Class
class OverdampedStochasticNewtonSchemeBase:
    def __init__(self, M, h, method, scaleupConst = 1000000):
        self.M = M
        self.h = h
        self.scalingFactor = scaleupConst
        self.method = method
        self.N = int(scaleupConst/h)

        #Let's compute the x coordinate for visualization
        self.f = np.zeros((self.N, self.M))

        out = np.split(np.arange(self.N), 20)
        index = []
        for o in out:
            index.append(o[-1] + 1)
        self.IAT_sliceIndexOnePast = np.array(index)
        self.IATs = np.zeros((M,20))

    # x would be M, and y would be M,
    def Compute_f_field(self, x, y):
        return np.zeros((self.M,)) #Returning shape is M,

    def ComputeIATs(self):
        for listIndex, sliceIndex in enumerate(self.IAT_sliceIndexOnePast):
            taus = np.zeros((self.M,))
            for m in range(self.M):
                out = integrated_time(self.f[0:sliceIndex, m], c=5, tol=50, quiet=True)
                if np.isnan(out):
                    taus[m] = 0
                else:
                    taus[m] = out
            self.IATs[:, listIndex] = taus
        return

    def FindMeanOfIATs(self):
        return np.mean(self.IATs, axis = 0), self.IAT_sliceIndexOnePast

    def GenerateHistograms(self, exerciseName, index):
        titleString = exerciseName + '_' + self.method + " |h:" + str(self.h) + " |Samples:" + str(
            self.IAT_sliceIndexOnePast[index]) + " |beta:" + str(self.beta) + " |L:" + str(self.L)
        saveFigName = exerciseName + '_' + self.method + "_h_" + str(self.h) + "_Samples_" + str(
            self.IAT_sliceIndexOnePast[index]) + "_beta_" + str(self.beta) + "_L_" + str(self.L)
        saveFigName = saveFigName.replace('.', 'd')

        fig, ax = plt.subplots()  # fig : figure object, ax : Axes object
        #Make sure here that only one trajectory is remaining in self.cosines #TODO
        ax.hist(self.cosines[:self.IAT_sliceIndexOnePast[index]])
        ax.set_xlabel('Cosine of angle of Magnetization Vector')
        ax.set_title(titleString)
        fig.savefig('Histogram_' + saveFigName)
        plt.close(fig)
        return

    def GenerateIATGraphs(self, exerciseName, index):
        titleString = exerciseName + '_' + self.method + " |h:" + str(self.h) + " |Samples:" + str(self.IAT_sliceIndexOnePast[index]) + " |beta:" + str(self.beta) + " |L:" + str(self.L)
        saveFigName = exerciseName + '_' + self.method + "_h_" + str(self.h) + "_Samples_" + str(self.IAT_sliceIndexOnePast[index]) + "_beta_" + str(self.beta) + "_L_" + str(self.L)
        saveFigName = saveFigName.replace('.', 'd')

        x = np.arange(self.M)
        fig, ax = plt.subplots()  # fig : figure object, ax : Axes object

        ax.plot(x, self.IATs[:, index], '-o');
        ax.set_xlabel('Trajectories Index')
        ax.set_ylabel('Integrated Autocorrelation Time')

        ax.set_title(titleString)
        ax.set_xticks(x)
        fig.savefig(saveFigName)
        plt.close(fig)
        return

#Class for question 71
#   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -  #
#                                                                                                      #
#                                                                                                      #
#   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -  #
class OverdampedStochasticNewtonScheme(OverdampedStochasticNewtonSchemeBase):
    def __init__(self, M, h, method, x, y, scaleupConst=1000000):
        super().__init__(M, h, method, scaleupConst)
        self.xs = np.zeros((self.N, self.M, 2))
        self.xs[0,:,0] = x
        self.xs[0,:,1] = y


    def GetNextTimestampXs(self, k):
        #old_xs[:,0] #Shape will be (M, )
        old_xs = self.xs[k-1,:,:] #Shape will be (Mx2)

        grad = GetOutputOf_grad_of_log_pi(old_xs[:,0], old_xs[:,1]) # M x 2
        S_matrix = GetSMatrix(old_xs[:,0], old_xs[:,1], self.method) #Shape is M x 2 x 2

        SDotGrad = GetReqProduct(self.M, S_matrix, grad) #It would be M x 2

        root_S = GetSquareRootOfS(S_matrix, self.method) # shape is M x 2 x 2
        zetas = np.random.normal(0, 1, (self.M,2))  # M x 2 #This is standard normal

        root_SDotzeta = GetReqProduct(self.M, root_S, zetas) #It would be M x 2

        new_xs = old_xs + self.h * SDotGrad + np.sqrt(2 * self.h) * root_SDotzeta
        return new_xs #Shape should be M x 2


    def UpdateForMetropolization(self,boolArray, next_xs, old_xs):
        #old_xs are M x 2
        #new_xs are M x 2
        bools = boolArray.reshape(boolArray.shape[0], 1) # It will be M x 1
        z = np.where(bools, next_xs, old_xs)
        return z

    def ElongateChain(self):
        self.f[0,:] = self.xs[0,:,0] #Evaluating the first coordinate of the matrix
        for i in range(1, self.N):
            next_xs = self.GetNextTimestampXs(i)  # M x 2
            ratio_of_pi = GetRatiosOf_Fxn_pi(next_xs[:,0],next_xs[:,1], self.xs[i-1,:,0], self.xs[i-1,:,1])

            q_num = GetOutPutOfFxn_q(self.M, next_xs, self.xs[i-1,:,:], self.method, self.h)
            q_denom = GetOutPutOfFxn_q(self.M, self.xs[i-1,:,:],next_xs, self.method, self.h)

            ratio = q_num * ratio_of_pi/q_denom


            p_acc = np.minimum(np.ones((self.M,)), ratio)
            randNum = np.random.uniform(low=0, high=1.0, size=(self.M,))
            boolArray = randNum < p_acc
            updated_xs = self.UpdateForMetropolization(boolArray, next_xs, self.xs[i-1,:,:])
            self.xs[i,:,:] = updated_xs
            self.f[i,:] = self.xs[i,:,0]
        return

#Class for question 75
#   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -  #
#                                                                                                      #
#                                                                                                      #
#   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -  #
class EnsembleMCMC(OverdampedStochasticNewtonSchemeBase):
    def __init__(self, M, h, alpha, method, mean = 0, std = 1, scaleupConst=1000000):
        super().__init__(M, h, method, scaleupConst)
        self.xs = np.zeros((self.N, self.M, 2))
        self.xs[0,:,:] = np.random.multivariate_normal(mean = np.zeros(2,), cov= np.identity(2), size = M) #This shape should be M x 2
        self.alpha = alpha

    def GetNextTimestampXs(self, z, i_index, k):
        i_index_x = self.xs[k-1, i_index, :] #Shape will be (2,)

        j_index = i_index
        while j_index == i_index :
            j_index = np.random.randint(0,self.M)

        j_index_x = self.xs[k-1, j_index, :] #Shape will be (2,)
        y = j_index_x + z * (i_index_x - j_index_x) #Should be (2,)
        return y #(2, )

    def UpdateForMetropolization(self,boolArray, next_xs, old_xs):
        #old_xs are M x 2
        #new_xs are M x 2
        bools = boolArray.reshape(boolArray.shape[0], 1) # It will be M x 1
        z = np.where(bools, next_xs, old_xs)
        return z

    def ElongateChain(self):
        #F can't be computed same way,ask what to do here
        self.f[0,:] = self.xs[0,:,0] #Evaluating the first coordinate of the matrix

        for iteration_k in range(1, self.N):
            for i_traj in range(0, self.M):
                z = GenerateZ(self.alpha)
                old_x = self.xs[iteration_k - 1, i_traj, :] #Should be (2,)
                y = self.GetNextTimestampXs(z, i_traj, iteration_k) #Should be (2,)
                ratio_pi = GetRatiosOf_Fxn_pi(y[0], y[1], old_x[0], old_x[1])
                zTimesRatio = z * ratio_pi
                randNum = np.random.uniform(low=0, high=1.0)
                p_acc = np.minimum(1, zTimesRatio)
                boolValue = randNum < p_acc
                updated_xs = np.where(boolValue, y, old_x)
                self.xs[iteration_k,i_traj,:] = updated_xs
                self.f[iteration_k,i_traj] = self.xs[iteration_k, i_traj,0]
        return

def Simulation(model):
    model.ElongateChain()
    return model

def DoEnsemblingModel(alpha, scaleupConst = 100):
    plt.clf()
    model = EnsembleMCMC(10, 0.0005, alpha,'EnsembleMonteCarlo',mean = 0, std = 1, scaleupConst = scaleupConst)
    model = Simulation(model)
    model.ComputeIATs()
    IATs, Ns = model.FindMeanOfIATs()

    avg_x  = np.mean(model.xs[:,:,0], axis = 1)
    avg_y = np.mean(model.xs[:,:,1], axis= 1)
    N = avg_x.shape[0]

    fig, ax = plt.subplots()  # fig : figure object, ax : Axes object
    ax.scatter(avg_x, avg_y)
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.set_title("Scatter plot of points using Ensemble MCMC Scheme")
    save_name = 'ScatterPlotOf_Ensemble_alpha' + str(alpha)
    saveFigName = save_name.replace('.', 'd')
    fig.savefig(saveFigName)
    plt.close(fig)


    fig, ax = plt.subplots()  # fig : figure object, ax : Axes object
    ax.plot(np.arange(N) + 1, avg_x, '-o')
    ax.set_xlabel('Time')
    ax.set_ylabel('x_1')
    ax.set_title("Time Series of x_1 using Ensemble MCMC Scheme")
    save_name = 'TimeSeries_x1_EnsembleScheme_alpha' + str(alpha)
    saveFigName = save_name.replace('.', 'd')
    fig.savefig(saveFigName)
    plt.close(fig)

    fig, ax = plt.subplots()  # fig : figure object, ax : Axes object
    ax.plot(np.arange(N) + 1, avg_y, '-o')
    ax.set_xlabel('Time')
    ax.set_ylabel('x_2')
    ax.set_title("Time Series of x_2 using Ensemble MCMC Scheme")
    save_name = 'TimeSeries_x2_EnsembleScheme_alpha' + str(alpha)
    saveFigName = save_name.replace('.', 'd')
    fig.savefig(saveFigName)
    plt.close(fig)

    #Printing the IATs
    fig, ax = plt.subplots()  # fig : figure object, ax : Axes object
    ax.plot(Ns, IATs, '-o')
    ax.set_xlabel('N')
    ax.set_ylabel('IATs')
    ax.set_title("IATs vs N | Ensemble MCMC Scheme")
    save_name = 'IAT_EnsembleScheme_alpha' + str(alpha)
    saveFigName = save_name.replace('.','d')
    fig.savefig(saveFigName)
    plt.close(fig)


def DoOverdampedRun(type, scaleupConst = 100):
    plt.clf()
    x = np.random.random(1)
    y = np.random.random(1)
    if(type == 1):
        model = OverdampedStochasticNewtonScheme(10, 0.0005, 'newton', x, y, scaleupConst)#50000)
    else:
        model = OverdampedStochasticNewtonScheme(10, 0.0005, 'langevin', x, y, scaleupConst)#50000)

    model = Simulation(model)
    model.ComputeIATs()
    IATs, Ns = model.FindMeanOfIATs()


    avg_x  = np.mean(model.xs[:,:,0], axis = 1)
    avg_y = np.mean(model.xs[:,:,1], axis= 1)
    N = avg_x.shape[0]
    plt.clf()
    plt.scatter(avg_x, avg_y)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    if(type == 1):
        plt.title("Scatter plot of points using Overdamped Stochastic Newton Scheme")
        plt.savefig('ScatterPlotOf_StochasticNewton')
    else:
        plt.title("Scatter plot of points using Overdamped Langevin Scheme")
        plt.savefig('ScatterPlotOf_OverdampedLangevin')

    plt.clf()
    plt.plot(np.arange(N) + 1, avg_x, '-o')
    plt.xlabel('Time')
    plt.ylabel('x_1')
    if(type == 1):
        plt.title("Time Series of x_1 using Overdamped Stochastic Newton Scheme")
        plt.savefig('TimeSeries_x1_StochasticNewton')
    else:
        plt.title("Time Series of x_1 using Overdamped Langevin Scheme")
        plt.savefig('TimeSeries_x1_OverdampedLangevin')

    plt.clf()
    plt.plot(np.arange(N) + 1, avg_y, '-o')
    plt.xlabel('Time')
    plt.ylabel('x_2')
    if(type == 1):
        plt.title("Time Series of x_2 using Overdamped Stochastic Newton Scheme")
        plt.savefig('TimeSeries_x2_StochasticNewton')
    else:
        plt.title("Time Series of x_2 using Overdamped Langevin Scheme")
        plt.savefig('TimeSeries_x2_OverdampedLangevin')

    #Printing the IATs
    plt.clf()
    plt.plot(Ns, IATs, '-o')
    plt.xlabel('N')
    plt.ylabel('IATs')
    if(type == 1):
        plt.title("IATs vs N | Overdamped Stochastic Newton Scheme")
        plt.savefig('IAT_StochasticNewton')
    else:
        plt.title("IATs vs N | Overdamped Langevin Scheme")
        plt.savefig('IAT_OverdampedLangevin')

# 1 is for stochastic newton
# 2 is for overdamped langevin
# h is 0.0005, set scaleup Constant accordingly
DoOverdampedRun(1, scaleupConst = 1000)
print('1 Done')
DoOverdampedRun(2, scaleupConst = 1000)
print('2 Done')
#First one is alpha value
DoEnsemblingModel(1.5, scaleupConst = 1000)
print('3 Done')
DoEnsemblingModel(2.0, scaleupConst = 1000)
print('4 Done')
DoEnsemblingModel(2.5, scaleupConst = 1000)
print('5 Done')
DoEnsemblingModel(3.0, scaleupConst = 1000)
print('6 Done')