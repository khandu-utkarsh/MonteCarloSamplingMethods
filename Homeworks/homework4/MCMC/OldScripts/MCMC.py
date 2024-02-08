import numpy as np

class MCMC:
#Intialize the lattice
  def __init__(self, beta, latticeSize):
      self.L = latticeSize
      self.initSpins = np.zeros((latticeSize, latticeSize), dtype=int)
      self.beta = beta

  def InitializeLattice(self):
    self.initSpins = np.random.randint(0, 2, size = (self.L,self.L)) * 2 - 1 #Randomly create int b/w 0 and 1 and then multiply by 2 and sub by 1

  def CalculateSumNeighborSpins(self, x_pos, y_pos, spins):
    sum = 0

    x_dirs = np.array([-1,1,0,0])
    x_dirs = x_dirs + x_pos
    y_dirs = np.array([0,0,1,-1])
    y_dirs = y_dirs + y_pos

    x_dirs = np.where(x_dirs >= self.L , 0, np.where(x_dirs < 0 , self.L -1, x_dirs))
    y_dirs = np.where(y_dirs >= self.L , 0, np.where(y_dirs < 0 , self.L -1, y_dirs))

    sum = 0
    for i in range(4):
      sum += spins[x_dirs[i]][y_dirs[i]]
    return sum

  def GetPositionsDeterministically(self, K):
    pos_list = np.zeros((K,1), dtype=[('x', 'int'), ('y', 'int')])
    nums = self.L * self.L
    count = 0
    while(count < K):
      for iNum in range(nums):
        if(count < K):
          x_pos = int(iNum//self.L)
          y_pos = int(iNum - x_pos * self.L)
          pos_list[count] = (x_pos, y_pos)
          count += 1
    return pos_list

  def GetPositionsRandomly(self, K):
    pos_list = np.zeros((K,1), dtype=[('x', 'int'), ('y', 'int')])
    nums = self.L * self.L
    count = 0
    while(count < K):
      iNum = np.random.randint(0,nums)
      x_pos = int(iNum//self.L)
      y_pos = int(iNum - x_pos * self.L)
      pos_list[count] = (x_pos, y_pos)
      count += 1
    return pos_list

  def CustomBernaulli(self,p):
    x = np.random.uniform()
    if x < p:
      return 1
    else:
      return 0
  	
  def SetParamBeta(self, beta):
    self.beta = beta

  def ChangeLatticeSize(self,L):
    self.L = L
    self.InitializeLattice()

class MCMC_Gibbs(MCMC):#class MCMC_Gibbs:
    def __init__(self, beta, latticeSize):
      super().__init__(beta, latticeSize)
      #print('temp')

    def UpdateSpins(self, x_pos, y_pos, spins):
      sum_neigh = self.CalculateSumNeighborSpins(x_pos, y_pos, spins)
      cond_1_succ = 1/(1 + np.exp(-2 * self.beta * sum_neigh))
      randNum = self.CustomBernaulli(cond_1_succ)
      updatedSpins = spins
      if(randNum == 1): #Success
        updatedSpins[x_pos][y_pos] = 1
      else:
        updatedSpins[x_pos][y_pos] = -1
      return updatedSpins

class MCMC_MH(MCMC):#class MCMC_Gibbs:
    def __init__(self, beta, latticeSize):
      super().__init__(beta, latticeSize)

    def FlipSpin(self, oldSpin):
      if oldSpin == 1:
        return -1
      else:
        return 1

    def UpdateSpins(self, x_pos, y_pos, spins):
      oldSpin = spins[x_pos][y_pos]
      newSpin = self.FlipSpin(oldSpin)
      newSum = self.CalculateSumNeighborSpins(x_pos, y_pos, spins)
      prod = np.exp(self.beta * (newSpin * newSum - oldSpin * newSum))
      succProbab = min(1.0, prod)
      #print(type(succProbab))
      randNum = self.CustomBernaulli(succProbab)
      updatedSpins = spins
      if randNum == 1:  # Success
        updatedSpins[x_pos][y_pos] = newSpin
      return updatedSpins


def GetMagnetization(model, N):
  #Deterministic Positions
  deterministicPos = model.GetPositionsDeterministically(N)
  spins = model.initSpins
  d_count = deterministicPos.shape[0]
  deter_mag = np.zeros((d_count,))
  for i in range(d_count):
    x_pos = deterministicPos[i][0][0]
    y_pos = deterministicPos[i][0][1]
    updatedSpins = model.UpdateSpins(x_pos, y_pos, spins)
    spins = updatedSpins
    magVal = np.sum(spins)
    deter_mag[i] = magVal

  #Random Positions
  randomPos = model.GetPositionsRandomly(N)
  spins = model.initSpins
  r_count = randomPos.shape[0]
  rand_mag = np.zeros((d_count,))
  for i in range(d_count):
    x_pos = randomPos[i][0][0]
    y_pos = randomPos[i][0][1]
    updatedSpins = model.UpdateSpins(x_pos, y_pos, spins)
    spins = updatedSpins
    magVal = np.sum(spins)
    rand_mag[i] = magVal

  return deter_mag, rand_mag
