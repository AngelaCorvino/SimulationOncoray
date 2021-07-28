
import scipy
import pandas as pd # to work with structured, tabulated data (.csv, .xlsx,..)
import numpy as np # to work fast with matrices, arrays and much more
from matplotlib import pyplot as plt # to visualize/plot your data
from scipy import interpolate



# read in the csv-file
outputfile_topas = '/home/corvin22/SimulationOncoray/data/SOBP/DoseToWater_150MeVproton_PVT_12PC_SOBP.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array


def get_y_doseprofile_at_z(data, z_binnumber):

    y_profile_dosevalues = data[data[:,1]==z_binnumber][:,3] # get all doses in y-direction at given z
    return y_profile_dosevalues


def get_x_doseprofile_at_z(data, z_binnumber):
    #z_binnumber = depth. The x-first value with e.g.
    #z=0 will be at x=0, the next with z=0 will be at x=1 etc. So increasing z-should be enough
    x_profile_dosevalues = data[data[:,2]==z_binnumber][:,3] # get all doses in x-direction at given z
    return x_profile_dosevalues


def get_z_doseprofile_at_x(data, x_binnumber):

    z_profile_dosevalues = data[data[:,0]==x_binnumber][:,3] # get all doses in z-direction at given x
    return z_profile_dosevalues


numberof_ybins= 186
numberof_xbins = 161
numberof_zbins = 170

daten = topas_datamatrix

yprofiles = np.zeros((numberof_zbins, numberof_ybins)) # create empty array to be filled
xprofiles = np.zeros((numberof_zbins, numberof_xbins)) # create empty array to be filled
zprofiles = np.zeros((numberof_xbins, numberof_zbins)) # create empty array to be filled





# write each x- profile at depth z in row i in the array outside the loop
for i in range(numberof_zbins):
    yprofile_at_z  = get_y_doseprofile_at_z(daten, i)
    yprofiles[i,:] = yprofile_at_z
    i += 1

ymeanprofile=np.sum(yprofiles,axis=0)/numberof_zbins




for i in range(numberof_zbins):
    xprofile_at_z  = get_x_doseprofile_at_z(daten, i)
    xprofiles[i,:] = xprofile_at_z
    i += 1

xmeanprofile=np.sum(xprofiles,axis=0)/numberof_zbins



for i in range(numberof_xbins):
    zprofile_at_x  = get_z_doseprofile_at_x(daten, i)
    zprofiles[i,:] = zprofile_at_x
    i += 1

zmeanprofile=np.sum(zprofiles,axis=0)/numberof_xbins









plt.figure(1) # creates a figure in which we plot
plt.plot(np.arange(0,numberof_xbins,1)*0.0634,xprofiles[10,:] ,'.-',label='{x profile at z = 0,634 mm inside PVT}') # plots depth on x, dose on y, and uses default layout
plt.plot(np.arange(0,numberof_xbins,1)*0.0634,xprofiles[50,:] ,'.-',label='{x profile at z = 3,17 mm inside PVT}') # plots depth on x, dose on y, and uses default layout
#plt.plot(np.arange(0,numberof_xbins,1)*0.0634,xprofiles[100,:] ,'.-',label='{x profile at z = 6,34 mm inside PVT}') # plots depth on x, dose on y, and uses default layout
plt.plot(np.arange(0,numberof_xbins,1)*0.0634,xprofiles[150,:] ,'.-',label='{x profile at z = 9,51 mm inside PVT}') # plots depth on x, dose on y, and uses default layout
#plt.plot(depthPVT, dosePVT,'.-',label='{70 MeV protons ,0cm distance}') # plots depth on x, dose on y, and uses default layout
plt.title('Scoring in PVT  in PVT(aperture diameter= 7mm, 12 PC)') # Title
plt.xlabel('Depth in Water x direction [mm]') # label for x-axis
plt.ylabel(' Dose in water ') # label for y axis


plt.legend()
plt.minorticks_on()
plt.grid()








plt.figure(3) # creates a figure in which we plot
plt.plot(np.arange(0,numberof_xbins,1)*0.0634, xmeanprofile/np.max( xmeanprofile),'.-',label='{mean value of 170 bins along z}') # plots depth on x, dose on y, and uses default layout


#plt.plot(depthPVT, dosePVT,'.-',label='{70 MeV protons ,0cm distance}') # plots depth on x, dose on y, and uses default layout
plt.title('Scoring in PVT(aperture diameter= 7mm, 12PC)') # Title
plt.xlabel('Depth in Water x direction [mm]') # label for x-axis
plt.ylabel('Relative Dose in water ') # label for y axis


plt.legend()
plt.minorticks_on()
plt.grid()



plt.figure(4) # creates a figure in which we plot
plt.plot(np.arange(0,numberof_zbins,1)*0.0634, zmeanprofile,'.-',label='{ mean value of 161 bin along x}')

#plt.plot(depthPVT, dosePVT,'.-',label='{70 MeV protons ,0cm distance}') # plots depth on x, dose on y, and uses default layout
plt.title('Scoring in PVT (aperture diameter=7mm ,12 PC ) ') # Title
plt.xlabel('Depth in water z direction [mm]') # label for x-axis
plt.ylabel(' Dose in water ') # label for y axis


plt.legend()
plt.minorticks_on()
plt.grid()





























plt.show()
