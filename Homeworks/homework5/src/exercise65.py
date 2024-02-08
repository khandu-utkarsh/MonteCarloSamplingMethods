from driverScript import *


import time
startTime = time.time()
f1 = RunTestsOnQuestion_65(beta, L, M, initialValues, hs, scaleupConst, n)
PlotGraphsForAllModelsIndividually(f1,'Hybrid')
PlotGraphsOfMeanIATs(f1, 'Hybrid')
endTime = time.time()
print("Time Taken for the run (s): ", endTime - startTime)
