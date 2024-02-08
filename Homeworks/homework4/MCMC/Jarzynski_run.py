import JarzynskiMCMC as jmc
#def Run(L,M,beta,resampling):
print('Without resampling started')
output = jmc.Run(50,30000,0.1,False) #Without resampling
print('Without resampling ended')
print('With resampling started')
output = jmc.Run(50,30000,0.1,True) #With resampling
print('With resampling ended')
#initializedValues = np.random.randint(0, 2, size=(50, 50)) * 2 - 1  # Randomly create int b/w 0 and 1 and then multiply by 2 and sub by 1
#print(GetVarianceOfSamples(50,10000,20000,initializedValues,0.05,False))

#print('Averages: ', avgs)
#print('Variances: ', vars)
#print('Time consumed: ', timeConsumed)
#print('Sample Counts: ',Ns)

#Have to compute variance vs N
#Have to compute effort (time taken) vs N

