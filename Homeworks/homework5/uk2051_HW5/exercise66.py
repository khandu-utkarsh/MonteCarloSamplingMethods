from driverScript import *

import time
startTime = time.time()
f = RunTestsOnQuestion_66(beta, L, M, initialValues, hs, gammas, scaleupConst)
PlotGraphsForAllModelsIndividuallyForExercise66(f, 'Underdamped')
PlotGraphsOfMeanIATsForExercise66(f, 'Underdamped')

endTime = time.time()
print("Time Taken for the run (s): ", endTime - startTime)
