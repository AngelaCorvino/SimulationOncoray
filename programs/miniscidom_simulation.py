import scipy
import pandas as pd # to work with structured, tabulated data (.csv, .xlsx,..)
import numpy as np # to work fast with matrices, arrays and much more
from matplotlib import pyplot as plt # to visualize/plot your data
from scipy import interpolate




def get_x_doseprofile_at_z(data, z_binnumber):
    #z_binnumber = depth. The x-first value with e.g.
    #z=0 will be at x=0, the next with z=0 will be at x=1 etc. So increasing z-should be enough
    x_profile_dosevalues = data[data[:,2]==z_binnumber][:,3] # get all doses in x-direction at given z
    return x_profile_dosevalues


def get_z_doseprofile_at_x(data, x_binnumber):

    z_profile_dosevalues = data[data[:,0]==x_binnumber][:,3] # get all doses in z-direction at given x
    return z_profile_dosevalues
"""
def get_z_energydeposition_at_x(data, x_binnumber):

    z_profile_energyvalues = data[data[:,0]==x_binnumber][:,3] # get all energies in z-direction at given x
    return z_profile_energyvalues
"""








# read in the csv-fille
# read in the csv-file
outputfile_topas = '/home/corvin22/SimulationOncoray/data/DoseToWater_90MeVproton_PVT_7PC.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix1 = np.array(df)# convert dataframe df to array

# read in the csv-file
outputfile_topas = '/home/corvin22/SimulationOncoray/data/DoseToWater_90MeVproton_PVT_7PC.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix2 = np.array(df)# convert dataframe df to array





numberof_xbins = 161
numberof_zbins = 170
xprofiles = np.zeros((numberof_zbins, numberof_xbins)) # create empty array to be filled
xprofiles1 = np.zeros((numberof_zbins, numberof_xbins)) # create empty array to be filled
zprofiles1 = np.zeros((numberof_xbins, numberof_zbins)) # create empty array to be filled
#zprofiles2 = np.zeros((numberof_xbins, numberof_zbins)) # create empty array to be filled
#zprofiles_energy = np.zeros((numberof_xbins, numberof_zbins)) # create empty array to be filled

daten = topas_datamatrix1








for i in range(numberof_zbins):
    xprofile_at_z  = get_x_doseprofile_at_z(daten, i)
    xprofiles[i,:] = xprofile_at_z
    i += 1

xmeanprofile1=np.sum(xprofiles,axis=0)/numberof_zbins


for i in range(numberof_xbins):
    zprofile_at_x  = get_z_doseprofile_at_x(daten, i)
    zprofiles1[i,:] = zprofile_at_x
    i += 1

zmeanprofile1=np.sum(zprofiles1,axis=0)/numberof_xbins













"""
daten = topas_datamatrix_energy

for i in range(numberof_xbins):
    zprofile_energy_at_x  = get_z_energydeposition_at_x(daten, i)
    zprofiles_energy[i,:] = zprofile_energy_at_x
    i += 1

zmeanprofile_energy=np.sum(zprofiles_energy,axis=0)/numberof_xbins

"""





plt.figure(3) # creates a figure in which we plot
plt.plot(np.arange(0,numberof_xbins,1)*0.0634, xmeanprofile1/np.max(xmeanprofile1),'.-',label='6 PC 1 PMMA') # plots depth on x, dose on y, and uses default layout
#plt.plot(np.arange(0,numberof_xbins,1)*0.0634, xmeanprofile2/np.max(xmeanprofile2),'.-',label='7 PC') # plots depth on x, dose on y, and uses default layout

#plt.plot(depthPVT, dosePVT,'.-',label='{70 MeV protons ,0cm distance}') # plots depth on x, dose on y, and uses default layout
plt.title('Dose scored  in Miniscidom( aluminum aperture diameter= 7m)') # Title
plt.xlabel('Depth in Water x direction [mm]') # label for x-axis
plt.ylabel('Relative Dose in water ') # label for y axis


plt.legend()
plt.minorticks_on()
plt.grid()



plt.figure(4) # creates a figure in which we plot
plt.plot(np.arange(0,numberof_zbins,1)*0.0634, zmeanprofile1/np.max(zmeanprofile1),'.-',label='{ 6 PC 1 PMMA }')
#plt.plot(np.arange(0,numberof_zbins,1)*0.0634, zmeanprofile2/np.max(zmeanprofile2),'.-',label='{7PC}')
#plt.plot(depthPVT, dosePVT,'.-',label='{70 MeV protons ,0cm distance}') # plots depth on x, dose on y, and uses default layout
plt.title('Dose Scored  in Minisicdom (aperture diameter= 7 mm) ') # Title
plt.xlabel('Depth in water z direction [mm]') # label for x-axis
plt.ylabel(' Relative Dose in water ') # label for y axis


plt.legend()
plt.minorticks_on()
plt.grid()



directory='/home/corvin22/Desktop/miniscidom/pictures/2021-09-02/notnormalized/'


filename1=directory+'notnormalizedmean_array19.npy'
data1= np.load(filename1)
dose=data1[0:len(data1)-3]

#tof1=data1[len(data1)-3:len(data1)]
unidose=0
data2= np.load(directory+'notnormalizederr19.npy')
err=data2[0:len(data2)]
#top_projection_dose1= np.load(directory+'1Dtopprojection/'+'top1Dmean_array42.npy')


plt.figure(5)
#plt.plot( np.arange(0,len(dose),1)*0.0634,
#                                                           top_projection_dose,
#                                                                           '.',
#                                                                  Markersize=11,
#                                               label='Top projection measured ')
#plt.plot( np.arange(0,len(dose),1)*0.0634,
#                                                                    dose[::-1],
#                                                                           '.',
#                                                                  markersize=7,
#                                                        label='{}'.format(tof))
plt.errorbar(  np.arange(0,len(dose),1)*0.074,                         dose/dose.max() ,
                                                                      yerr=err/dose.max(),
                                                                      xerr=None,
                                                                        fmt='.',
                                                                   ecolor='gray',
                                                                elinewidth=None,
                                label=' reconstruction '.format(unidose))
plt.plot(np.arange(0,numberof_zbins,1)*0.0634, zmeanprofile1/np.max(zmeanprofile1),'.',label='{simulation }')




plt.title('Dose scored  in Miniscidom( 7PC,aluminum aperture diameter= 7mm)') # Title
plt.xlabel('Depth in Water x direction [mm]') # label for x-axis
plt.ylabel('Relative Dose in water ') # label for y axis


plt.legend()
plt.minorticks_on()
plt.grid()


















plt.show()
