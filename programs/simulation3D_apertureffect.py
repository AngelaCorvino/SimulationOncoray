
import scipy
import pandas as pd # to work with structured, tabulated data (.csv, .xlsx,..)
import numpy as np # to work fast with matrices, arrays and much more
from matplotlib import pyplot as plt # to visualize/plot your data
from scipy import interpolate
from tomograpic_viewer import Tomograpic_viewer


# read in the csv-file
outputfile_topas = '/home/corvin22/SimulationOncoray/data/Single/Aperture/DoseToWater_9210KeVproton_Airbox_6PC1PMMA_3Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
dose=np.array(topas_datamatrix[:,3])
numberof_ybins= 200
numberof_xbins = 200
numberof_zbins = 200
dose_3D =dose.reshape(numberof_xbins,numberof_ybins,numberof_zbins)


#print(topas_datamatrix)
#print(np.shape(topas_datamatrix))
Tomograpic_viewer(dose_3D/dose_3D.max(),False,1)





def get_x_doseprofile_at_zy(data, z_binnumber):
    #z_binnumber = depth. The x-first value with e.g.
    #z=0 will be at x=0, the next with z=0 will be at x=1 etc. So increasing z-should be enough
    x_profile_dosevalues = data[data[:,2]==z_binnumber and data[:,2]==y_binnumber ][:,3] # get all doses in x-direction at given z
    return x_profile_dosevalues


def get_z_doseprofile_at_x(data, x_binnumber):

    z_profile_dosevalues = data[data[:,0]==x_binnumber][:,3] # get all doses in z-direction at given x
    return z_profile_dosevalues
