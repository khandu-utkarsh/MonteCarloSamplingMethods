from driverScript import *

import time
startTime = time.time()
fo = RunTestsOnQuestion_64(beta, L, M, initialValues, hs, scaleupConst)
PlotGraphsForAllModelsIndividually(fo,'Overdamped')
PlotGraphsOfMeanIATs(fo, 'Overdamped')
endTime = time.time()
print("Time Taken for the run (s): ", endTime - startTime)
