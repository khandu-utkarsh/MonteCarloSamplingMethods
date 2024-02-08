import numpy as np
class MCMC_Gibbs:
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

    x_coords = np.where(x_dirs >= self.L , 0, np.where(x_dirs < 0 , self.L -1, x_dirs))
    y_coords = np.where(y_dirs >= self.L , 0, np.where(y_dirs < 0 , self.L -1, y_dirs))


    sum = 0
    for i in range(4):
      sum += spins[x_coords[i]][y_coords[i]]
    return sum

  def UpdateSpins(self, x_pos, y_pos, spins):
    sum_neigh = self.CalculateSumNeighborSpins(x_pos, y_pos, spins)
    cond_1_succ = 1/(1 + np.exp(-2 * self.beta * sum_neigh))
    randNum = np.random.binomial(1, cond_1_succ)
    updatedSpins = spins
    if(randNum == 1): #Success
      updatedSpins[x_pos][y_pos] = 1
    else:
      updatedSpins[x_pos][y_pos] = -1
    return updatedSpins

  def GetPositionsDeterministically(self, K):
    pos_list = []
    nums = self.L * self.L
    count = 0
    while(count < K):
      for iNum in range(nums):
        count += 1
        x_pos = int(iNum//self.L)
        y_pos = int(iNum - x_pos * self.L)
        pos_list.append((x_pos, y_pos))
    return pos_list

  def GetPositionsRandomly(self, K):
    pos_list = []
    nums = self.L * self.L
    count = 0
    while(count < K):
      iNum = np.random.randint(0,nums)
      count += 1
      x_pos = int(iNum//self.L)
      y_pos = int(iNum - x_pos * self.L)
      pos_list.append((x_pos, y_pos))
    return pos_list


  def SetParamBeta(self, beta):
    self.beta = beta

  def ChangeLatticeSize(self,L):
    self.L = L
    self.InitializeLattice()

class MCMC_MH:
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

    x_coords = np.where(x_dirs >= self.L , 0, np.where(x_dirs < 0 , self.L -1, x_dirs))
    y_coords = np.where(y_dirs >= self.L , 0, np.where(y_dirs < 0 , self.L -1, y_dirs))


    sum = 0
    for i in range(4):
      sum += spins[x_coords[i]][y_coords[i]]
    return sum

  def FlipSpin(self,oldSpin):
    if oldSpin == 1:
      return -1
    else:
      return 1

  def UpdateSpins(self, x_pos, y_pos, spins):
    oldSpin = spins[x_pos][y_pos]
    newSpin = self.FlipSpin(oldSpin)
    oldSum = self.CalculateSumNeighborSpins(x_pos, y_pos, spins)
    newSum = self.CalculateSumNeighborSpins(x_pos, y_pos, spins)
    prod = np.exp(self.beta * (newSpin * newSum - oldSpin * oldSum))
    succProbab = min(1, prod)
    randNum = np.random.binomial(1, succProbab)
    updatedSpins = spins
    if randNum == 1: #Success
      updatedSpins[x_pos][y_pos] = newSpin
    return updatedSpins

  def GetPositionsDeterministically(self, K):
    pos_list = []
    nums = self.L * self.L
    count = 0
    while(count < K):
      for iNum in range(nums):
        count += 1
        x_pos = int(iNum//self.L)
        y_pos = int(iNum - x_pos * self.L)
        pos_list.append((x_pos, y_pos))
    return pos_list

  def GetPositionsRandomly(self, K):
    pos_list = []
    nums = self.L * self.L
    count = 0
    while(count < K):
      iNum = np.random.randint(0,nums)
      count += 1
      x_pos = int(iNum//self.L)
      y_pos = int(iNum - x_pos * self.L)
      pos_list.append((x_pos, y_pos))
    return pos_list

  def SetParamBeta(self, beta):
    self.beta = beta

  def ChangeLatticeSize(self,L):
    self.L = L
    self.InitializeLattice()


def GetMagnetization(model, N):
  deter_mag = []
  rand_mag = []

  deterministicPos = model.GetPositionsDeterministically(N)
  randomPos = model.GetPositionsRandomly(N)

  #Deterministic Positions
  spins = model.initSpins
  for (x_pos, y_pos) in deterministicPos:
    updatedSpins = model.UpdateSpins(x_pos, y_pos, spins)
    spins = updatedSpins
    magVal = np.sum(spins)
    deter_mag.append(magVal)

  #Deterministic Positions
  spins = model.initSpins
  for (x_pos, y_pos) in randomPos:
    updatedSpins = model.UpdateSpins(x_pos, y_pos, spins)
    spins = updatedSpins
    magVal = np.sum(spins)
    rand_mag.append(magVal)

  return deter_mag, rand_mag
