import numpy as np

#x would be M, and y would be M,
def GetOutputOf_grad_of_log_pi(x, y):
    M = x.shape[0]

    grad = np.zeros((M, 2))
    grad[:,0] = 20. * x * (-x**2 + y) - x/10. + 1/10
    grad[:,1] = 10 * x**2 - 10 * y
    return grad #Shape -> M x 2


# x should be of the shape M, and y would be of the shape M,
def GetOutputOf_Fxn_pi(x, y):
    rational = -1 * (100 * (y - x**2)**2 + (1 - x)**2)/20
    final = np.exp(rational)
    return final #Should be of shape (M,)

def GetRatiosOf_Fxn_pi(n_x, n_y, d_x, d_y):
    num = GetOutputOf_Fxn_pi(n_x, n_y)
    denom = GetOutputOf_Fxn_pi(d_x, d_y)
    ratio = np.divide(num,denom)
    return ratio #Shape -> M,


#These x and y are basically coordinates of one entity
def GetSMatrix(x, y, methodName, small_coeff = 0.01):
    M = x.shape[0]
    if(methodName != "newton"):
        matrix = np.zeros((M, 2, 2))
        matrix[:] = np.identity(2)
        return matrix

    a_11 = 60 * x**2 - 20 * y + 1/10
    a_12 = -20 * x
    a_22 = 10 * np.ones(x.shape)

    matrix = np.zeros((M, 2, 2))
    for i in range(M):
        matLocal = np.array([[a_11[i], a_12[i]],[a_12[i], a_22[i]]])
        #Adding some small \alpha time identity just to make it work
        matLocal = matLocal + small_coeff * np.identity(2)

        matrix[i,:,:] = np.linalg.inv(matLocal)

    return matrix #Return shape is M x 2 x2



def GetReqProduct(M, ten, mat):
    #Shape of ten is M x 2 x 2
    #Shape of mat is M x 2
    #Output should be M x 2
    out = np.zeros((M, 2))
    for i in range(M):
        local_S = ten[i, :, :]  # Shape should be 2 x 2
        local_grad = mat[i, :]  # Shape should be 2,
        out[i, :] = local_S.dot(local_grad)

    return out

def GetSquareRootOfS(matrix, methodName):
    M = matrix.shape[0]
    if(methodName != "newton"): #If non -newton method, we could actually simply return the input matrix, it would be I itself, I forgot why I assigned identity to it's values
        matrix = np.zeros((M, 2, 2))
        matrix[:] = np.identity(2)
        return matrix

    root = np.zeros((M,2,2))
    for i in range(M):
        mat = matrix[i,:,:] # Shape would be (2 x 2)
        det = np.abs(np.linalg.det(mat))
        trace = np.abs(np.trace(mat))
        denom = np.sqrt(trace + 2 * np.sqrt(det))
        R = (mat + np.sqrt(det) * np.identity(2))/denom
        root[i,:,:] = R

    return root #Return shape is M x 2 x2

#These x and y are components
#xs should be M x 2
def ForqComputeAllThings_at_X(M, xs, h,method):
    grad = GetOutputOf_grad_of_log_pi(xs[:, 0], xs[:, 1])  # M x 2
    S_matrix = GetSMatrix(xs[:, 0], xs[:, 1], method)  # Shape is M x 2 x 2

    SDotGrad = GetReqProduct(M, S_matrix, grad)  # It would be M x 2

    #Do x plus SDotGrad
    mean =  xs + h * SDotGrad

    #Compute S inv
    inv_S = np.linalg.inv(S_matrix)
    return mean, inv_S

def GetOutPutOfFxn_q(M, xs, ys, method, h):
    mean, invS = ForqComputeAllThings_at_X(M, xs, h, method) #invS should be M x 2 x 2
    t1 = ys - mean #Should be M x 2

    laterProd = GetReqProduct(M, invS, t1) # It would be M x 2

    out = np.zeros((M,))
    for i in range(M):
        out[i] = np.dot(t1[i], laterProd[i],)

    out = out/(-4 * h)
    out = np.exp(out)
    return out #Shape shpuld be (M,)


# #Input would be (M x2) and (M x 2)
# def Get_OutputOf_Fxn_Conditionalq(M, h, num, denom, method):
#     grad = GetOutputOf_grad_of_log_pi(denom[:, 0], denom[:, 1])  # M x 2
#     S_matrix = GetSMatrix(denom[:, 0], denom[:, 1], method)  # Shape is M x 2 x 2
#     SDotGrad = GetReqProduct(M, S_matrix, grad)  # It would be M x 2
#
#     for i in range(M):
#
#
#     invS =
#
#



def InvProbabDisZ(x, alpha):
    return ((x + 2/np.sqrt(alpha))/2)**2


def GenerateZ(alpha):
    u = np.random.uniform(0,1,1)
    return InvProbabDisZ(u, alpha)