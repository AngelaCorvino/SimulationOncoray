
import scipy
import pandas as pd # to work with structured, tabulated data (.csv, .xlsx,..)
import numpy as np # to work fast with matrices, arrays and much more
from matplotlib import pyplot as plt # to visualize/plot your data
from scipy import interpolate
from tomograpic_viewer import Tomograpic_viewer


# read in the csv-file
outputfile_topas = '/home/corvin22/SimulationOncoray/data/SOBP/DoseToWater_150MeVproton_PVT_9PC_SOBP_3Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
dose=np.array(topas_datamatrix[:,3])
numberof_ybins= 186
numberof_xbins = 161
numberof_zbins = 170
dose_3D =dose.reshape(numberof_xbins,numberof_ybins,numberof_zbins)


#print(topas_datamatrix)
#print(np.shape(topas_datamatrix))
Tomograpic_viewer(dose_3D/dose_3D.max(),False,1)
