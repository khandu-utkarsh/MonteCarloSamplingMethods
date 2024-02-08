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

from emcee.autocorr import AutocorrError, integrated_time
from AuxilaryFxns import *

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm
import math

#Base Class
class LangevinBase:
    def __init__(self, beta, M, h, method, scaleupConst = 1000000):
        self.beta = beta
        self.M = M
        self.h = h
        self.scalingFactor = scaleupConst
        self.method = method
        self.N = int(scaleupConst/h)
        if self.N == 0:
            self.status = False
        else:
            self.status = True

        self.MagnetizationVector = np.zeros((self.N, self.M, 2)) # both x and y component of Magnetization
        self.cosines = None #This will be N x M
        self.bringBackThetaInDomain = lambda x : np.vectorize(math.remainder)(x, math.tau)

        if(self.N < 20):
            self.IAT_sliceIndexOnePast = np.arange(self.N) + 1
            self.IATs = np.zeros((M,self.N))
        else:
            out = np.split(np.arange(self.N), 20)
            index = []
            for o in out:
                index.append(o[-1] + 1)
            self.IAT_sliceIndexOnePast = np.array(index)
            self.IATs = np.zeros((M,20))



        #self.IAT_sliceIndexOnePast = np.zeros((self.intervalCounts,),dtype=int)
        #self.IATs = np.zeros((M,self.intervalCounts)) #20, considering 5 percent frequency of calculation


    def UpdateThetaForMH(self, boolArray, new, old):
        bools = boolArray.reshape(boolArray.shape[0], 1)
        updated = np.where(bools, new.T, old.T)
        return updated.T

    def GetMagnetizationComponents(self, thetas):
        # From the thetas, get the x components by summing up all the x projections and y components by summing
        # up all the y projections and store it
        # thetas will be L x M
        X_comp = np.sum(np.cos(thetas), axis=0)  # This should be M x 1 or M ,
        Y_comp = np.sum(np.sin(thetas), axis=0)  # This should be M x 1 or M ,

        initMagVector = np.zeros((self.M, 2))
        initMagVector[:, 0] = X_comp
        initMagVector[:, 1] = Y_comp
        return initMagVector  # This will be M x 2

    def ComputeCosineOfAngles(self):
        y_comps = self.MagnetizationVector[:, :, 1]
        x_comps = self.MagnetizationVector[:, :, 0]
        angles = np.arctan2(y_comps, x_comps)
        cosineComps = np.cos(angles)  # This should be N x M
        return cosineComps

    def ComputeIATs(self):
        # Cosine comps is N x M
        self.cosines = self.ComputeCosineOfAngles()
        for listIndex, sliceIndex in enumerate(self.IAT_sliceIndexOnePast):
            taus = np.zeros((self.M,))
            for m in range(self.M):
                out = integrated_time(self.cosines[0:sliceIndex, m], c=5, tol=50, quiet=True)
                if np.isnan(out):
                    taus[m] = 0
                else:
                    taus[m] = out
            self.IATs[:, listIndex] = taus
        return

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

#Class for question 64
#   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -  #
#                                                                                                      #
#                                                                                                      #
#   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -  #
class OverdampedLangevinUnMetropolized(LangevinBase):
    def __init__(self, beta, L, M, h, method, inputInitValues, scaleupConst=1000000):
        super().__init__(beta, M, h, method, scaleupConst)

        # Lattice is one dimension - periodic and 1D
        self.L = L
        # L x M means one column is basically theta vector for one trajectory
        self.initTheta = np.zeros((L, M))  # When slicing only one row out of it, it's shape will be (L, ) and not (L,1)
        self.initTheta[:] = inputInitValues #inputInitValues should be basically L x 1


    def GetNextTimestampThetas(self):
        old_theta = self.initTheta  # L x M

        #S matrix in the notes
        I = np.identity(self.L, dtype=int)
        grad = GetOutputOf_grad_of_log_pi(old_theta, self.beta)
        term2 = self.h * I.dot(grad) # L x M

        term3 = I * 2 * self.h  # L x L
        term3 = sqrtm(term3)  # L x L
        zetas = np.random.normal(0, 1, (self.L, self.M))  # L x M #This is standard normal
        term3 = term3.dot(zetas)  # L x M

        newTimestampThetas = old_theta + term2 + term3  # L x M
        return newTimestampThetas

    def ElongateChain(self):
        mags = self.GetMagnetizationComponents(self.initTheta) #M x 2
        #N x M x 2
        self.MagnetizationVector[0,:,:] = mags
        for i in range(1, self.N):
            nextThetas = self.GetNextTimestampThetas()  # L x M
            if self.method == 'M' :
                ratio = GetRatiosOf_Fxn_pi(nextThetas, self.initTheta, self.beta)
                p_acc = np.minimum(np.ones((self.M,)), ratio)
                randNum = np.random.uniform(low=0, high=1.0, size=(self.M,))
                boolArray = randNum < p_acc
                nextThetas = self.UpdateThetaForMH(boolArray, nextThetas, self.initTheta)

            nextThetas = self.bringBackThetaInDomain(nextThetas) #To make sure they are between -pi to pi
            self.initTheta = nextThetas
            self.MagnetizationVector[i, :, :] = self.GetMagnetizationComponents(nextThetas)
        return

#Class for question 65
#   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -  #
#                                                                                                      #
#                                                                                                      #
#   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -  #
class HybridMCMC(LangevinBase):
    def __init__(self, beta, d_hat, d_tilde, M, h, n, method, J_hat, init_d_hat, init_d_tilde, scaleupConst=1000000):
        super().__init__(beta, M, h, method, scaleupConst)
        self.d_hat = d_hat
        self.d_tilde = d_tilde
        self.J_hat = J_hat
        self.n = n
        self.L = d_hat
        self.initTheta = np.zeros((d_hat + d_tilde, M))  # When slicing only one row out of it, it's shape will be (L, ) and not (L,1)
        initialThetas = np.concatenate((init_d_hat, init_d_tilde), axis = 0) # (d_hat + d_tilde x M)
        self.initTheta[:] = initialThetas

        #mags = self.GetMagnetizationComponents(self.initTheta[:self.d_hat,:])  # M x 2
        #self.MagnetizationVector[0, :, :] = mags # N x M x 2

    def VelocityVerletScheme(self, x_hat, x_tilde, n):
        y_hat = x_hat
        y_tilde = x_tilde

        for i in range(n):
            #Step 1
            grad = GetOutputOf_grad_of_log_pi(y_hat, self.beta)
            y_tilde_prime = y_tilde + self.h/2. * self.J_hat.T.dot(grad)

            #Step 2
            grad_K = GetOutputOf_grad_of_Fxn_K(y_tilde_prime)
            y_hat_new = y_hat + self.h * self.J_hat.dot(grad_K)

            #Step 3
            grad_new = GetOutputOf_grad_of_log_pi(y_hat_new, self.beta)
            y_tilde_new = y_tilde_prime + self.h/2. * self.J_hat.dot(grad_new)

            y_hat = y_hat_new
            y_tilde = y_tilde_new

        y = np.concatenate((y_hat, y_tilde), axis = 0)
        return y

    def GetUpdateInThetaAndSampledThetas(self, n):
        Y_tilde_old = GetSamplesFromSomeDistribution_Of_K(self.d_tilde, self.M)
        Y_new = self.VelocityVerletScheme(self.initTheta[:self.d_hat, :], Y_tilde_old, n)
        return Y_tilde_old, Y_new


    def ElongateChain(self):
        mags = self.GetMagnetizationComponents(self.initTheta) #M x 2
        #N x M x 2
        self.MagnetizationVector[0,:,:] = mags
        for i in range(1, self.N):
            Y_tilde_prev, Y_new = self.GetUpdateInThetaAndSampledThetas(self.n)
            if(self.method == 'M'):
                ratio = GetRatios_Of_pi_H(Y_new[:self.d_hat, :],
                                          Y_new[self.d_hat: self.d_hat + self.d_tilde, :],
                                          self.initTheta[:self.d_hat, :],
                                          Y_tilde_prev,
                                          self.beta)

                p_acc = np.minimum(np.ones((self.M,)), ratio)
                randNum = np.random.uniform(low=0, high=1.0, size=(self.M,))
                boolArray = randNum < p_acc
                Y_new = self.UpdateThetaForMH(boolArray, Y_new, self.initTheta)

            Y_new = self.bringBackThetaInDomain(Y_new)
            self.initTheta = Y_new
            self.MagnetizationVector[i,:,:] = self.GetMagnetizationComponents(Y_new[:self.d_hat, :])
        return


# Class for question 66
#   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -  #
#                                                                                                      #
#                                                                                                      #
#   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -  #
class LangevianUnderdampledScheme(LangevinBase):
    def __init__(self, beta, d_hat, d_tilde, M, h, method, J_hat, gamma, init_d_hat, init_d_tilde, scaleupConst=1000000):
        super().__init__(beta, M, h, method, scaleupConst)
        self.d_hat = d_hat
        self.d_tilde = d_tilde
        self.J_hat = J_hat
        self.gamma = gamma
        self.L = d_hat

        self.initTheta = np.zeros((d_hat + d_tilde, M))  # When slicing only one row out of it, it's shape will be (L, ) and not (L,1)
        initialThetas = np.concatenate((init_d_hat, init_d_tilde), axis=0)  # (d_hat + d_tilde x M)
        self.initTheta[:] = initialThetas

        #mags = self.GetMagnetizationComponents(self.initTheta[:self.d_hat, :])  # M x 2
        #self.MagnetizationVector[0, :, :] = mags  # N x M x 2

    def DiscretiationScheme(self, y_hat, y_tilde):
        x_hat = y_hat
        x_tilde = y_tilde

        #Step 1
        grad = GetOutputOf_grad_of_log_pi(x_hat, self.beta)
        x_tilde_prime = x_tilde  + self.h/2. * self.J_hat.T.dot(grad)

        #Step 2
        x_hat_prime = x_hat + self.h/2. * self.J_hat.dot(x_tilde_prime)

        #Step 3
        zetas = (multivariate_normal.rvs(mean=np.zeros((self.d_tilde,)), cov=np.identity(self.d_tilde), size=self.M)).T
        x_tilde_double_prime = np.exp(-1 * self.gamma  * self.h) * x_tilde_prime + np.sqrt(1 - np.exp(-2 * self.gamma  * self.h)) * zetas

        #Step 4
        x_hat_new = x_hat_prime + self.h/2 * self.J_hat.dot(x_tilde_double_prime)

        #Step 5
        grad_new = GetOutputOf_grad_of_log_pi(x_hat_new, self.beta)
        x_tilde_new = x_tilde_double_prime + self.h/2 * self.J_hat.T.dot(grad_new)

        return x_hat_new, x_tilde_new

    def ElongateChain(self):
        mags = self.GetMagnetizationComponents(self.initTheta) #M x 2
        #N x M x 2
        self.MagnetizationVector[0,:,:] = mags
        for i in range(1, self.N):
            X_old_hat = self.initTheta[:self.d_hat, :]
            X_old_tilde = self.initTheta[self.d_hat:self.d_hat + self.d_tilde, :]
            Y_new_hat, Y_new_tilde = self.DiscretiationScheme(X_old_hat, X_old_tilde)

            if self.method == 'M':
                ratios_r = GetRatios_Of_Fxn_r(Y_new_hat, -1 * Y_new_tilde,
                                              X_old_hat, -1 * X_old_tilde,
                                              X_old_hat, X_old_tilde,
                                              Y_new_hat, Y_new_tilde,
                                              self.h, self.gamma, self.J_hat, self.beta)
                ratios_pi_H = GetRatios_Of_pi_H(Y_new_hat, Y_new_tilde, X_old_hat, X_old_tilde, self.beta)
                ratioProd = np.multiply(ratios_r, ratios_pi_H)
                p_acc = np.minimum(np.ones((self.M,)), ratioProd)
                randNum = np.random.uniform(low=0, high=1.0, size=(self.M,))
                boolArray = randNum < p_acc

                Y_new_hat = self.UpdateThetaForMH(boolArray, Y_new_hat, X_old_hat)
                Y_new_tilde = self.UpdateThetaForMH(boolArray, Y_new_tilde, -1 * X_old_tilde)

            Y_new = np.concatenate((Y_new_hat, Y_new_tilde), axis = 0)
            Y_new = self.bringBackThetaInDomain(Y_new)
            self.initTheta = Y_new
            self.MagnetizationVector[i, :, :] = self.GetMagnetizationComponents(Y_new)
        return

#   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -  #
#                                                                                                                      #
#                                                                                                                      #
#                                                                                                                      #
#                                                                                                                      #
#   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -  #

def Simulation(model):
    if model.status == False:
        return model

    model.ElongateChain()
    model.ComputeIATs()

    #Before returning, delete all the values which are not required to reduce the size of model
    model.initTheta = None
    model.MagnetizationVector = None
    model.cosines = model.cosines[:,0] #Remove all the trajectories apart from 0th index as this is one used
    return model


def GenerateSamplesFromOverdampled(beta, L, M, h, method, initialValues_d_hat, scaleupConst):
    model = OverdampedLangevinUnMetropolized(beta, L,M, h, method, initialValues_d_hat, scaleupConst)
    return Simulation(model)

def GenerateSamplesFromHybrid(beta, d_hat, M, h, method, init_d_hat, scaleupConst, n):
    d_tilde = d_hat
    init_d_tilde = init_d_hat #Considering d_tilde == d_hat
    #Since we are considering d_tilde == d_hat, hence J_hat would be a square matrix, hence
    J_hat = np.identity(d_hat)

    model = HybridMCMC(beta, d_hat, d_tilde, M, h, n, method, J_hat, init_d_hat, init_d_tilde, scaleupConst)
    return Simulation(model)

def GenerateSampleFromUnderdamped(beta, d_hat, M, h, method, init_d_hat, scaleupConst, gamma):
    d_tilde = d_hat
    init_d_tilde = init_d_hat #Considering d_tilde == d_hat
    #Since we are considering d_tilde == d_hat, hence J_hat would be a square matrix, hence
    J_hat = np.identity(d_hat)
    model = LangevianUnderdampledScheme(beta, d_hat, d_tilde, M, h, method, J_hat, gamma, init_d_hat, init_d_tilde, scaleupConst)
    return Simulation(model)

