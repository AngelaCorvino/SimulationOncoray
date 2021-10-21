
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
#Tomograpic_viewer(dose_3D/dose_3D.max(),False,1)




def get_xy_doseprofile_at_z(data, z_binnumber):
    #z_binnumber = depth. The x-first value with e.g.
    #z=0 will be at x=0, the next with z=0 will be at x=1 etc. So increasing z-should be enough
    xy_profile_dosevalues = data[data[:,2]==z_binnumber][:,3] # get all doses in x-direction at given z
    return xy_profile_dosevalues


def get_yz_doseprofile_at_x(data, x_binnumber):

    yz_profile_dosevalues = data[data[:,0]==x_binnumber][:,3] # get all doses in z-direction at given x
    return yz_profile_dosevalues





outputfile_topas = '/home/corvin22/SimulationOncoray/data/SOBP/DoseToWater_150MeVproton_PVT_9PC_SOBP_3Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array






numberof_ybins= 186
numberof_xbins = 161
numberof_zbins = 170
a=np
xyprofiles = np.zeros((numberof_zbins, numberof_xbins*numberof_ybins)) # create empty array to be filled
#yprofiles = np.zeros((numberof_zbins, numberof_ybins)) # create empty array to be filled
yzprofiles = np.zeros((numberof_xbins, numberof_zbins*numberof_ybins)) # create empty array to be filled
#zprofiles2 = np.zeros((numberof_xbins, numberof_zbins)) # create empty array to be filled
#zprofiles_energy = np.zeros((numberof_xbins, numberof_zbins)) # create empty array to be filled

daten = topas_datamatrix








for i in range(numberof_zbins):
    xyprofile_at_z  = get_xy_doseprofile_at_z(daten, i)
    xyprofiles[i,:] = xyprofile_at_z
    i += 1

xymeanprofile=np.sum(xyprofiles,axis=0)/numberof_zbins
print(len(xymeanprofile))

for i in range(numberof_xbins):
    yzprofile_at_x  = get_yz_doseprofile_at_x(daten, i)
    yzprofiles[i,:] = yzprofile_at_x
    i += 1

yzmeanprofile=np.sum(yzprofiles,axis=0)/numberof_xbins
print(yzmeanprofile)







def get_x_doseprofile_at_y(data, y_binnumber):
    #z_binnumber = depth. The x-first value with e.g.
    #z=0 will be at x=0, the next with z=0 will be at x=1 etc. So increasing z-should be enough
    x_profile_dosevalues = data[data[:,1]==y_binnumber][:,2] # get all doses in x-direction at given y
    return x_profile_dosevalues


def get_y_doseprofile_at_x(data, x_binnumber):

    x_profile_dosevalues = data[data[:,0]==x_binnumber][:,2] # get all doses in y-direction at given x
    return y_profile_dosevalues



data=xymeanprofile


xprofiles = np.zeros((numberof_ybins, numberof_xbins))
yprofiles = np.zeros((numberof_zbins, numberof_ybins))




for i in range(numberof_ybins):
    xprofile_at_y  = get_x_doseprofile_at_y(data, i)
    xprofiles[i,:] = xprofile_at_y
    i += 1

xmeanprofile=np.sum(xprofiles,axis=0)/numberof_ybins
print(len(xmeanprofile))

for i in range(numberof_xbins):
    yprofile_at_x  = get_y_doseprofile_at_x(data, i)
    yprofiles[i,:] = yprofile_at_x
    i += 1

ymeanprofile=np.sum(yprofiles,axis=0)/numberof_xbins
print(ymeanprofile)
