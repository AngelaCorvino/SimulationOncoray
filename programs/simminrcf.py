import scipy
import pandas as pd # to work with structured, tabulated data (.csv, .xlsx,..)
import numpy as np # to work fast with matrices, arrays and much more
from matplotlib import pyplot as plt # to visualize/plot your data
from scipy import interpolate
import csv

from readcsv import read_datarcf
from getprofile import  Get_profile

directory='/home/corvin22/Desktop/miniscidom/pictures/2021-09-02/'
filename=directory+'RCF21CX.csv'
depth,rcfdose=read_datarcf(filename)
rcferr=rcfdose*(546/10000)

#filename1=directory+'RCF21CQ.csv'
#depth1,rcfdose1=read_datarcf(filename1)
#rcferr1=rcfdose1*(546/10000)

#normalization

#rcfdose1=rcfdose1/rcfdose1.max()
#rcferr1=rcferr1/rcfdose1.max()




directory='/home/corvin22/Desktop/miniscidom/pictures/2021-09-02/notnormalized/'

data= np.load(directory+'notnormalizedmean_array17.npy')
dose=data[0:len(data)-3]

#tof1=data1[len(data1)-3:len(data1)]
unidose=0
data0= np.load(directory+'notnormalizederr17.npy')
err=data0[0:len(data0)]
#top_projection_dose= np.load(directory+'1Dtopprojection/'+'top1Dmean_array1.npy')


"""
directory='/home/corvin22/Desktop/miniscidom/pictures/2021-09-01/notnormalized/'

data1= np.load(directory+'notnormalizedmean_array35.npy')
dose1=data1[0:len(data1)-3]

#tof1=data1[len(data1)-3:len(data1)]
unidose=0
data2= np.load(directory+'notnormalizederr35.npy')
err2=data2[0:len(data2)]
#top_projection_dose= np.load(directory+'1Dtopprojection/'+'top1Dmean_array1.npy')

"""







# read in the csv-file
outputfile_topas = '/home/corvin22/SimulationOncoray/data/Single/DoseToWater_90MeVproton_PVT_6PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array

# read in the csv-file
outputfile_topas = '/home/corvin22/SimulationOncoray/data/Single/EnergyDeposit_90MeVproton_PVT_6PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix_energy = np.array(df)# convert dataframe df to array
print(np.shape(topas_datamatrix))

numberof_xbins = 1
numberof_zbins = 149
profile=Get_profile(topas_datamatrix, 149,1)
xmeanprofile=profile.xmeanprofile
zmeanprofile=profile.zmeanprofile
zmeanprofile=zmeanprofile[::-1]




plt.figure(1) # creates a figure in which we plot
#plt.plot(np.arange(0,numberof_xbins,1)*0.0634, xmeanprofile/np.max(xmeanprofile),'.-',label='{mean value of 170 bins along z}') # plots depth on x, dose on y, and uses default layout
plt.plot(np.arange(0,numberof_zbins,1)*0.0738255,
                                            zmeanprofile/np.max(zmeanprofile),
                                                                            '.',
                                                                            markersize=8,
                                        label='{ TOPAS Simulation}')



plt.errorbar(  np.arange(0,len(dose),1)*0.074,                         dose/dose.max() ,
                                                                      yerr=err/dose.max(),
                                                                      xerr=None,
                                                                        fmt='.',
                                                                        color='green',
                                                                        markersize=8,
                                                                   ecolor='green',
                                                                elinewidth=None,
                                            label=' reconstruction '.format(unidose))
"""

plt.errorbar(  np.arange(0,len(dose1),1)*0.074,                         dose1/dose1.max() ,
                                                                      yerr=err2/dose1.max(),
                                                                      xerr=None,
                                                                        fmt='.',
                                                                        color='purple',
                                                                        markersize=8,
                                                                   ecolor='purple',
                                                                elinewidth=None,
                                            label=' reconstruction '.format(unidose))



plt.errorbar( depth,                      rcfdose1 ,
                                                                  yerr=rcferr1,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                        color='orange',
                                                                        markersize=8,
                                                                   ecolor='orange',
                                                                elinewidth=None,
                                                                label=' RCF 21CW measured dose')

plt.fill_between(depth,
                                                                rcfdose1-rcferr1,
                                                                rcfdose1+rcferr1,
                                                                color='orange', alpha=0.1)
"""
plt.errorbar( depth,                      rcfdose/rcfdose.max() ,
                                                                  yerr=rcferr/rcfdose.max(),
                                                                      xerr=None,
                                                                        fmt='.',
                                                                        color='red',
                                                                        markersize=8,
                                                                   ecolor='red',
                                                                elinewidth=None,
                                                                label=' RCF 21CX measured dose')

plt.fill_between(depth,
                                                                rcfdose/rcfdose.max()-rcferr/rcfdose.max(),
                                                                rcfdose/rcfdose.max()+rcferr/rcfdose.max(),
                                                        color='red', alpha=0.1)












#plt.plot(depthPVT, dosePVT,'.-',label='{70 MeV protons ,0cm distance}') # plots depth on x, dose on y, and uses default layout
plt.title('Depht dose distribution(aperture diameter= 7mm, 6PC)') # Title
plt.xlabel('Depth in Water x direction [mm]') # label for x-axis
plt.ylabel('Relative Dose in water ') # label for y axis


plt.legend()
plt.minorticks_on()
plt.grid()
plt.show()
