import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import numpy.random as random
from mpl_toolkits import mplot3d
from tabulate import tabulate

# Targer density \pi
def Get_p(x, sd, mean):
    return norm.pdf(x, mean, sd)


# Estimator density \tilde pi
def Get_q(x):
    return norm.pdf(x, 0, 1)


def Get_pByqRatio(x, sd, mean):
    return Get_p(x, sd, mean) / Get_q(x)


def GetNormalizedWeights(X, sd, mean):
    w = np.zeros(len(X))
    for i in range(len(w)):
        w[i] = Get_pByqRatio(X[i], sd, mean)

    sum = np.sum(w)
    w = w / sum
    return w


def DoMultinomialResampling(x, w):
    m = len(w)
    x_resampled = np.zeros(0)
    w_bar = np.sum(w) / m
    probab_vector = w / (m * w_bar)  # Success probab of MultiNomialTrail
    copyCounts = np.random.multinomial(m, probab_vector)

    for i in range(m):
        cc = copyCounts[i]
        for j in range(cc):
            x_resampled = np.append(x_resampled, x[i])

    return (copyCounts, x_resampled)


def DoBernaulliResampling(x, w):
    m = len(w)
    w_bar = np.sum(w) / m
    x_resampled = np.zeros(0)

    copyCounts = np.zeros(m)
    for k in range(m):
        curr_copy_count = np.floor(m * w[k])
        u = np.random.uniform(0, 1)
        if (u < m * w[k] - curr_copy_count):
            curr_copy_count = curr_copy_count + 1
        else:
            curr_copy_count = curr_copy_count
        copyCounts[k] = (curr_copy_count.astype(np.int64))
    # print(copyCounts)
    copyCounts = copyCounts.astype(np.int64)
    # print(copyCounts)

    for i in range(m):
        cc = copyCounts[i]
        for j in range(cc):
            x_resampled = np.append(x_resampled, x[i])

    return (copyCounts, x_resampled)


def DoSystematicSampling(x, w):
    m = len(w)
    cumW = np.cumsum(w)

    u_s = []
    u = np.random.uniform(0, 1 / m)
    for i in range(1, m + 1):
        u_s = np.append(u_s, i / m - u)

    copyCounts = np.zeros(m)
    for k in range(1, m + 1):
        if (k == 1):
            sw = 0  # Starting weight
            fw = cumW[k - 1]  # Final weight
            # Count the num of us between sw and fw
            satisfies = [u for u in u_s if u >= sw and u < fw]
            copyCounts[k - 1] = len(satisfies)
        else:
            sw = cumW[k - 2]
            fw = cumW[k - 1]
            # Count the num of us between sw and fw
            satisfies = [u for u in u_s if u >= sw and u < fw]
            copyCounts[k - 1] = len(satisfies)

    x_resampled = np.zeros(0)
    w_bar = np.sum(w) / m

    copyCounts = copyCounts.astype(np.int64)

    for i in range(m):
        cc = copyCounts[i]
        # print(cc)
        for j in range(cc):
            x_resampled = np.append(x_resampled, x[i])
    return (copyCounts, x_resampled)


def DoResampling(x, w, option=1):
    if (option == 1):
        return DoMultinomialResampling(x, w)
    elif (option == 2):
        return DoBernaulliResampling(x, w)
    elif (option == 3):
        return DoSystematicSampling(x, w)
    return ([], [], [])


def CheckSamplingAlgoriths(m = 5000, sigma2 = 1.5):

    mean = 0
    #sigma2 = 1.5
    sd = np.sqrt(sigma2)

    #m =  1000 #Points Count
    points = np.random.normal(0, 1, m)
    weights = GetNormalizedWeights(points, sd, mean)

    cts = np.arange(0, m)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
    ax1.hist(points)
    ax2.scatter(cts, weights);

    ax1.set_title('Histogram of samples generated from N(0,1)')


    ax2.set_title('Weights of samples')
    ax2.set_xlabel('Sample Points')
    ax2.set_xlabel('Weights')

    fig.suptitle('Weights of sample points')
    #plt.show()
    grpahName = '/Users/utkarsh/NYU/Monte Carlo Methods/Homeworks/homework3/' + 'question1_sample_example' + '.png'
    #plt.savefig(grpahName)
    plt.show()

    # Outputting the mean from each estimator
    cpc_m, points_re_m = DoResampling(points, weights, 1)  # For multinomial
    cpc_b, points_re_b = DoResampling(points, weights, 2)  # For bernoulli
    cpc_s, points_re_s = DoResampling(points, weights, 3)  # For systematic

    print('Multinomial, mean: ', np.mean(points_re_m))
    print('Bernoulli, mean: ', np.mean(points_re_b))
    print('Systematic, mean: ', np.mean(points_re_s))

def ComputeStasOfEstimator(pts, wts):
#Now, generate N = 1000 copies of each of the thing
  N = 500
  data_m = []
  data_b = []
  data_s = []
  for i in range(N):
    cpc_m, points_re_m  = DoResampling(pts, wts, 1) #For multinomial
    cpc_b, points_re_b  = DoResampling(pts, wts, 2) #For bernaulli
    cpc_s, points_re_s = DoResampling(pts, wts, 3) #For systematic

    data_m.append(cpc_m)
    data_b.append(cpc_b)
    data_s.append(cpc_s)
  return data_m, data_b, data_s

def GenerateVariancePlotsFor(points, weights, sd2):
  df_m, df_b, df_s = ComputeStasOfEstimator(points, weights)
  df_m = pd.DataFrame(df_m)
  var_num_m = df_m.var(ddof= 0)

  df_b = pd.DataFrame(df_b)
  var_num_b = df_b.var(ddof= 0)

  df_s = pd.DataFrame(df_s)
  var_num_s = df_s.var(ddof= 0)

  fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 12))
  axs[0].plot(var_num_m)
  axs[1].plot(var_num_b)
  axs[2].plot(var_num_s)
  axs[0].set_title('Multinomial')
  axs[1].set_title('Bernoulli')
  axs[2].set_xlabel('Systematic')
  fig.suptitle('Variances in counts of copies generated for each sample using different methods for sigma^2 = ' + str(sd2))
# plt.show()
  grpahName = '/Users/utkarsh/NYU/Monte Carlo Methods/Homeworks/homework3/' + 'question1_variance_' + str(sd2) + '.png'
  plt.savefig(grpahName)


def GenerateGraphsForDifferentVariances():
    mean = 0
    m = 500
    var = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]
    for i in range(len(var)):
        sd = np.sqrt(var[i])
        points = np.random.normal(0, 1, m)
        weights = GetNormalizedWeights(points, sd, mean)
        GenerateVariancePlotsFor(points, weights, var[i])


def GetAvailableDirections(curr_x, curr_y, curr_points_in_saw):
  free_neighbors_count = 0;
  directions = [(1,0), (0,1), (-1, 0), (0,-1)]
  availableDirections = []
  for dir in range(4):
    del_x, del_y = directions[dir]
    n_x = curr_x + del_x
    n_y = curr_y + del_y
    if(not((n_x,n_y) in curr_points_in_saw)):
      free_neighbors_count +=1
      availableDirections.append((del_x, del_y))
  return free_neighbors_count, availableDirections

def GetNextPointInSAW(curr_x, curr_y, availableDirections):
  dirIndex = random.randint(0, len(availableDirections))
  del_x, del_y = availableDirections[dirIndex]
  new_x = curr_x + del_x  #New point x coordinate for this sample
  new_y = curr_y + del_y  #New point y coordinate for this sample
  return new_x, new_y

def GetCopyCounts(m, weights):
  mean = weights.mean()   #This is w bar, weight to be used after resampling
  probabVector = ((weights/(m * mean)).to_numpy()).astype(np.float64)
  #print(probabVector)
  copyCounts = np.random.multinomial(m, probabVector)
  return copyCounts

def GetSelfAvoidingWalk(m, d, resamplingOn = True):
  x_frame = pd.DataFrame(index=np.arange(m), columns=np.arange(d + 1))
  y_frame = pd.DataFrame(index=np.arange(m), columns=np.arange(d + 1))
  w_frame = pd.DataFrame(index=np.arange(m), columns=np.arange(d + 1))

  #Dimensions will run from 0 to d. Setting col 0 value to 0 for coords and 1 for weights
  x_frame[0] = 0
  y_frame[0] = 0
  w_frame[0] = 1
  #List of dictionaries for each sample. Dict will contain tuple of all the points in SAW.
  coords_set_main = []
  for i in range(m):
    coords_set_main.append({(0,0)})

  for dim_iter in range(d):                                                       #All dimension
    for sample_iter in range(m):                                                  #All samples

      curr_x = x_frame.at[sample_iter,dim_iter]
      curr_y = y_frame.at[sample_iter,dim_iter]
      free_neighbors_count, availableDirections = GetAvailableDirections(curr_x, curr_y, coords_set_main[sample_iter])
      if(free_neighbors_count == 0):
        #print('Returning false')
        return False, [], [], []    #Return Status as of now. Not processing this any further

      new_x, new_y = GetNextPointInSAW(curr_x, curr_y, availableDirections)
      (coords_set_main[sample_iter]).add((new_x, new_y))  #Adding in the dictionary

      #Add it into the dataframes
      x_frame.at[sample_iter,dim_iter + 1] = new_x
      y_frame.at[sample_iter,dim_iter + 1] = new_y
      w_frame.at[sample_iter,dim_iter + 1] = w_frame.at[sample_iter,dim_iter] * free_neighbors_count

    #All samples computed for curr dim_iter and are stored in dim_iter + 1

    #Resampling code below
    if(dim_iter < d - 1 and resamplingOn):
      if(dim_iter == 1):
        print('Resampling On')
      copyCounts = GetCopyCounts(m, w_frame[dim_iter + 1])
      x_resampled = []
      y_resampled = []

      for i in range(m):
        cc = copyCounts[i]
        for j in range(cc):
          x_resampled.append(x_frame.loc[i])
          y_resampled.append(y_frame.loc[i])

      if(len(x_resampled) != m or len(y_resampled) != m):
        print('Size not equal')

      for i in range(m):
        x_frame.loc[i] = x_resampled[i]
        y_frame.loc[i] = y_resampled[i]

  #print('Code ran successfully')
  return True, x_frame, y_frame, w_frame

def CheckValidityOfCode():
    xFrames = []
    yFrames = []
    wFrames = []

    n = 1000
    m = 10
    d = 10
    for i in range(n):
        status, x_frame, y_frame, w_frame = GetSelfAvoidingWalk(m, d, False)
        if (status == True):
            xFrames.append(x_frame)
            yFrames.append(y_frame)
            wFrames.append(w_frame)

    # Appending the dataframes
    Xs = pd.concat(xFrames, ignore_index=True)
    Ys = pd.concat(yFrames, ignore_index=True)
    Ws = pd.concat(wFrames, ignore_index=True)

    rows, cols = Xs.shape

    for jCol in range(cols):
        xs = []
        ys = []
        for iRow in range(rows):
            curr_x = Xs.loc[iRow][jCol]
            curr_y = Ys.loc[iRow][jCol]
            xs.append(curr_x)
            ys.append(curr_y)

        countTable = {}
        for i in range(len(xs)):
            curr_x = xs[i]
            curr_y = ys[i]
            if ((curr_x, curr_y) not in countTable):
                countTable[(curr_x, curr_y)] = 1
            else:
                countTable[(curr_x, curr_y)] = countTable[(curr_x, curr_y)] + 1

        x_c = []
        y_c = []
        w_s = []
        table = []
        for (px, py) in countTable:
          x_c.append(px)
          y_c.append(py)
          w_s.append(countTable[(px, py)])
          table.append([(px, py), countTable[(px, py)]/len(xs)])

        print('Expectation for d = ', jCol)
        print(tabulate(table, headers=['(x,y)', 'Count']))
        print('\n')


    #constants = Ws.mean()

    #print(Ws.mean())

    #fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
#    ax = plt.axes(projection='3d')
#    ax.scatter3D(x_c, y_c, w_s, c = w_s);
#    ax.set_title('Counts of Visits to Latice site')
#    ax.set_xlabel('x')
#    ax.set_ylabel('y')
#    ax.set_zlabel('Absolute count')
#    grpahName = '/Users/utkarsh/NYU/Monte Carlo Methods/Homeworks/homework3/' + 'question2_validity' + '.png'
#    plt.savefig(grpahName)


def GetNormalizationConstantWithoutResampling():
    wFrames = []

    n = 1000
    m = 5
    d = 30
    for i in range(n):
        status, x_frame, y_frame, w_frame = GetSelfAvoidingWalk(m, d, False)
        if (status == True):
            wFrames.append(w_frame)

    Ws = pd.concat(wFrames, ignore_index=True)

    dims = np.arange(1, d + 1)
    constants = Ws.mean()
    constants = (constants[1:]).astype(np.int64)

    print(constants)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.plot(dims, constants, marker='o')
    ax.set_title('Normalization Constants v/s length of SAW')
    ax.set_xlabel('Number of edges in SAW')
    ax.set_ylabel('Normalization Constant')

    grpahName = '/Users/utkarsh/NYU/Monte Carlo Methods/Homeworks/homework3/' + 'question2_normalization' + '.png'
    plt.savefig(grpahName)

def PlotSAWs():
    for i in range(21):
        status, x_frame, y_frame, w_frame = GetSelfAvoidingWalk(1, i, False)
        if (status == True):
            min_x = min(x_frame.loc[0])
            max_x = max(x_frame.loc[0])
            min_y = min(y_frame.loc[0])
            max_y = max(y_frame.loc[0])
            max_a = max(max_x, max_y)
            min_a = min(min_x, min_y)
            a_ticks = np.linspace(min_a - 1, max_a + 1, max_a - min_a + 1 + 2)
            #x_ticks = np.linspace(min_x, max_x, max_x - min_x + 1)
            #y_ticks = np.linspace(min_y, max_y, max_y - min_y + 1)

            fig, ax = plt.subplots(nrows=1, ncols=1) #,figsize=(10, 10))
            ax.plot(x_frame.loc[0], y_frame.loc[0], '-ro')
            plt.xticks(a_ticks);
            plt.yticks(a_ticks);
            ax.grid()
            ax.set_title('SAW of size ' + str(i))
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            grpahName = '/Users/utkarsh/NYU/Monte Carlo Methods/Homeworks/homework3/' + 'question2_SAW_' + str(i) + '.png'
            plt.savefig(grpahName)
            plt.close(fig)


CheckSamplingAlgoriths()
GenerateGraphsForDifferentVariances()
CheckValidityOfCode()
GetNormalizationConstantWithoutResampling()
PlotSAWs()