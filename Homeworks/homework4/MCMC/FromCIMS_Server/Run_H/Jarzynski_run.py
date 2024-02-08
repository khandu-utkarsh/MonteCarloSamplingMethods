import JarzynskiMCMC as jmc
##def Run(L,M,beta,resampling):
print('Without resampling started')
output = jmc.Run(20,1000,0.2,False) #Without resampling
print('Without resampling ended')
print('With resampling started')
output = jmc.Run(20,1000,0.2,True) #With resampling
print('With resampling ended')
