import scipy
import pandas as pd # to work with structured, tabulated data (.csv, .xlsx,..)
import numpy as np # to work fast with matrices, arrays and much more
from matplotlib import pyplot as plt # to visualize/plot your data
from scipy import interpolate
import csv

from readcsv import read_datarcf
from getprofile import  Get_profile

directory='/home/corvin22/Desktop/miniscidom/pictures/2021-09-01/'
filename=directory+'RCF21CR.csv'
depth,rcfdose=read_datarcf(filename)
rcferr=rcfdose*(546/10000)
filename1=directory+'RCF21CP.csv'
depth1,rcfdose1=read_datarcf(filename)
rcferr1=rcfdose1*(546/10000)
# read in the csv-file
outputfile_topas = '/home/corvin22/SimulationOncoray/data/SOBP/DoseToWater_150MeVproton_PVT_11PC_SOBP_2Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array

# read in the csv-file
outputfile_topas = '/home/corvin22/SimulationOncoray/data/SOBP/EnergyDeposit_150MeVproton_PVT_11PC_SOBP_2Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix_energy = np.array(df)# convert dataframe df to array
print(np.shape(topas_datamatrix))

numberof_xbins = 161
numberof_zbins = 170
profile=Get_profile(topas_datamatrix, 170,161)
xmeanprofile=profile.xmeanprofile
zmeanprofile=profile.zmeanprofile
zmeanprofile=zmeanprofile[::-1]




plt.figure(1) # creates a figure in which we plot
#plt.plot(np.arange(0,numberof_xbins,1)*0.0634, xmeanprofile/np.max(xmeanprofile),'.-',label='{mean value of 170 bins along z}') # plots depth on x, dose on y, and uses default layout
plt.plot(np.arange(0,numberof_zbins,1)*0.0634,
                                            zmeanprofile/np.max(zmeanprofile),
                                                                            '.',
                                                                            markersize=9,
                                        label='{ TOPAS Simulation}')


"""
plt.errorbar(  np.arange(0,len(dose),1)*0.074,                         dose/dose.max() ,
                                                                      yerr=err/dose.max(),
                                                                      xerr=None,
                                                                        fmt='.',
                                                                   ecolor='gray',
                                                                elinewidth=None,
                                            label=' reconstruction '.format(unidose))
"""
plt.errorbar( depth,                      rcfdose/rcfdose.max() ,
                                                                  yerr=rcferr/rcfdose.max(),
                                                                      xerr=None,
                                                                        fmt='.',
                                                                        color='orange',
                                                                        markersize=9,
                                                                   ecolor='darkorange',
                                                                elinewidth=None,
                                                                label=' RCF 21CR measured dose')

plt.fill_between(depth,
                                                                rcfdose/rcfdose.max()-rcferr/rcfdose.max(),
                                                                rcfdose/rcfdose.max()+rcferr/rcfdose.max(),
                                                        color='orange', alpha=0.3)
#plt.plot(depthPVT, dosePVT,'.-',label='{70 MeV protons ,0cm distance}') # plots depth on x, dose on y, and uses default layout
plt.title('Depht dose distribution(aperture diameter= 7mm, 11PC)') # Title
plt.xlabel('Depth in Water x direction [mm]') # label for x-axis
plt.ylabel('Relative Dose in water ') # label for y axis


plt.legend()
plt.minorticks_on()
plt.grid()
plt.show()
