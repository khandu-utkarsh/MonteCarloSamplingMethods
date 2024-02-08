import numpy as np
from scipy.stats import multivariate_normal

# x should be of the shape d_hat x M
def GetOutputOf_Fxn_pi(beta, x):
    d, M = x.shape
    newSum = 0.
    for i in range(d):
        newSum += np.cos(x[i, :] - x[(i + 1)%d, :])
    value = np.exp(beta * newSum)
    return value #Shape -> M,

def GetRatiosOf_Fxn_pi(n_x, d_x, beta):
    num = GetOutputOf_Fxn_pi(beta, n_x)
    denom = GetOutputOf_Fxn_pi(beta, d_x)
    ratio = np.divide(num,denom)
    return ratio #Shape -> M,

# x should be of shape d_hat x M
def GetOutputOf_grad_of_log_pi(x, beta):
    dims, M = x.shape

    grad = np.zeros(x.shape)
    for i in range(dims):
        grad[i,:] = np.sin(x[(i - 1)%dims,:] - x[i,:]) - np.sin(x[i,:] - x[(i + 1)%dims,:])
    grad = beta * grad
    return grad #Shape -> d_hat x M

# x should be of the shape d_tilde x M
def GetOutputOf_Fxn_K(x):
    value = 1/2. * np.sum(np.power(x, 2),axis = 0)
    return value #Shape -> M,

# x should be of d_tilde x M
def GetOutputOf_grad_of_Fxn_K(x):
    return x # Shape -> d_tilde x M

#d dimension, M # Number of samples
def GetSamplesFromSomeDistribution_Of_K(d, M):
    rands = multivariate_normal.rvs(mean=np.zeros((d,)), cov=np.identity(d), size=M)
    return rands.T #Shape -> d x M

def GetOutputOf_Fxn_H(x_hat, x_tilde, beta):
    term1 = -1 * np.log(GetOutputOf_Fxn_pi(beta, x_hat))
    term2 = GetOutputOf_Fxn_K(x_tilde)
    sum = term1 + term2
    return sum #Shape -> M,

def GetOutputOf_pi_H(x_hat, x_tilde, beta):
    value = np.exp(-1 * GetOutputOf_Fxn_H(x_hat, x_tilde, beta))
    return value #Shape -> M,

def GetRatios_Of_pi_H(n_hat, n_tilde, d_hat, d_tilde, beta):
    num = GetOutputOf_pi_H(n_hat, n_tilde, beta) # (M, )
    denom = GetOutputOf_pi_H(d_hat, d_tilde, beta)
    res = np.divide(num, denom)
    return res #Shape -> M,

# these will be of shape dims x M
def GetOutput_Of_Fxn_r(x_hat, x_tilde, y_hat, y_tilde, h, gamma, J_hat, beta):
    num_term1 = y_tilde # d_tilde x M
    exp_factor = np.exp(-1* gamma * h)
    num_term2 = exp_factor * x_tilde # d_tilde x M
    num_term3 = exp_factor * GetOutputOf_grad_of_log_pi(x_hat, beta) #d_hat x M
    num_term4 = GetOutputOf_grad_of_log_pi(y_hat, beta) # d_hat x M

    partialSum = num_term3 + num_term4
    simplifiedterm3Andterm4 = 1/2. * h * J_hat.T.dot(partialSum) #Shape should be d_tilde x M

    sum_num = num_term1 - num_term2 - simplifiedterm3Andterm4 #Shape should be d_tilde x M
    num_norm = np.sum(np.power(sum_num, 2),axis = 0)
    num = -1 * num_norm # 1 x M or M x 1 or M,
    denom = 2 * (1 - np.exp(-2 * h * gamma)) #Scaler
    ratio = num/denom

    #Fix
    returnRatio = np.exp(ratio)
    return returnRatio #Shape -> M,

def GetRatios_Of_Fxn_r(n_x_hat, n_x_tilde, n_y_hat, n_y_tilde,
                       d_x_hat, d_x_tilde, d_y_hat, d_y_tilde,
                       h, gamma, J_hat, beta):
    num = GetOutput_Of_Fxn_r(n_x_hat, n_x_tilde, n_y_hat, n_y_tilde, h, gamma, J_hat, beta)
    denom = GetOutput_Of_Fxn_r(d_x_hat, d_x_tilde, d_y_hat, d_y_tilde, h, gamma, J_hat, beta)
    ratio = np.divide(num, denom)
    return ratio #Shape -> M,


