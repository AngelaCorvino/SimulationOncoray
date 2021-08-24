import scipy
import pandas as pd # to work with structured, tabulated data (.csv, .xlsx,..)
import numpy as np # to work fast with matrices, arrays and much more
from matplotlib import pyplot as plt # to visualize/plot your data
from scipy import interpolate



# read in the csv-file
#outputfile_topas = '/home/corvin22/SimulationOncoray/data/DoseToWater_150MeVproton_8PCprova.csv'

outputfile_topas = '/home/corvin22/SimulationOncoray/data/WaterPhantomdoseafteraperture.csv'
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
depth=np.array(topas_datamatrix[:,2])*0.4 #[mm]
dose1=np.array(topas_datamatrix[:,3])
dose1=dose1/np.max(dose1) #normalization

outputfile_topas = '/home/corvin22/SimulationOncoray/data/WaterPhantomdose.csv'
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
depth=np.array(topas_datamatrix[:,2])*0.4 #[mm]
dose2=np.array(topas_datamatrix[:,3])
dose2=dose2/np.max(dose2) #normalization



"""
outputfile_topas = '/home/corvin22/SimulationOncoray/data/SOBP/DoseToWater_150MeVproton_PVT_9PC_SOBP_3Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array

# read in the csv-file
outputfile_topas = '/home/corvin22/SimulationOncoray/data/SOBP/EnergyDeposit_150MeVproton_9PC.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix_energy = np.array(df)# convert dataframe df to array



def get_x_doseprofile_at_z(data, z_binnumber):
    #z_binnumber = depth. The x-first value with e.g.
    #z=0 will be at x=0, the next with z=0 will be at x=1 etc. So increasing z-should be enough
    x_profile_dosevalues = data[data[:,2]==z_binnumber][:,3] # get all doses in x-direction at given z
    return x_profile_dosevalues


def get_z_doseprofile_at_x(data, x_binnumber):

    z_profile_dosevalues = data[data[:,0]==x_binnumber][:,3] # get all doses in z-direction at given x
    return z_profile_dosevalues

def get_z_energydeposition_at_x(data, x_binnumber):

    z_profile_energyvalues = data[data[:,0]==x_binnumber][:,3] # get all energies in z-direction at given x
    return z_profile_energyvalues



numberof_xbins = 161
numberof_zbins = 170

daten = topas_datamatrix

xprofiles = np.zeros((numberof_zbins, numberof_xbins)) # create empty array to be filled
zprofiles = np.zeros((numberof_xbins, numberof_zbins)) # create empty array to be filled
zprofiles_energy = np.zeros((numberof_xbins, numberof_zbins)) # create empty array to be filled




# write each x- profile at depth z in row i in the array outside the loop

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


daten = topas_datamatrix_energy

for i in range(numberof_xbins):
    zprofile_energy_at_x  = get_z_energydeposition_at_x(daten, i)
    zprofiles_energy[i,:] = zprofile_energy_at_x
    i += 1

zmeanprofile_energy=np.sum(zprofiles_energy,axis=0)/numberof_xbins
"""






plt.figure(2) # creates a figure in which we plot
plt.plot(depth,dose1,'.-',label='Phantom after the aperture ')
plt.plot(depth,dose2,'.-',label='Phantom before the aperture ')
#plt.plot(np.arange(0,numberof_zbins,1)*0.6, zmeanprofile/np.max(zmeanprofile),'.-',label='{Minisicidom }')
plt.title('Dose scored in  water phantom (120mm diametr cylinder) ') # Title
plt.xlabel('Depth z direction [mm]') # label for x-axis
plt.ylabel('Relative dose ') # label for y axis


plt.legend()
plt.minorticks_on()
plt.grid()
"""
plt.figure(3) # creates a figure in which we plot
plt.plot(np.arange(0,numberof_xbins,1)*0.0634, xmeanprofile,'.-',label='{mean value of 200 bins along z}') # plots depth on x, dose on y, and uses default layout


#plt.plot(depthPVT, dosePVT,'.-',label='{70 MeV protons ,0cm distance}') # plots depth on x, dose on y, and uses default layout
plt.title('Scoring PVT(aperture diameter= 7mm, 8PC 10⁷ protons)') # Title
plt.xlabel('Depth in Water x direction [mm]') # label for x-axis
plt.ylabel('Dose in water ') # label for y axis


plt.legend()
plt.minorticks_on()
plt.grid()



plt.figure(4) # creates a figure in which we plot
plt.plot(np.arange(0,numberof_zbins,1)*0.3, zmeanprofile,'.-',label='{ mean value of 161 bin along x}')

#plt.plot(depthPVT, dosePVT,'.-',label='{70 MeV protons ,0cm distance}') # plots depth on x, dose on y, and uses default layout
plt.title('Scoring PVT(aperture diameter=7mm,8PC 10⁷ protons) ') # Title
plt.xlabel('Depth in water z direction [mm]') # label for x-axis
plt.ylabel(' Dose in water [Gy]') # label for y axis


plt.legend()
plt.minorticks_on()
plt.grid()




"""
























plt.show()
