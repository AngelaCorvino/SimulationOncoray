import scipy
import pandas as pd # to work with structured, tabulated data (.csv, .xlsx,..)
import numpy as np # to work fast with matrices, arrays and much more
from matplotlib import pyplot as plt # to visualize/plot your data
from scipy import interpolate
import csv

from readcsv import read_datarcf
from getprofile import  Get_profile

directory='/home/corvin22/Desktop/miniscidom/pictures/2021-09-02/'
filename=directory+'RCF21CW.csv'
depth,rcfdose=read_datarcf(filename)
rcferr=rcfdose*(546/10000)
area_rcf=np.trapz(rcfdose[0:len(rcfdose)], depth[0:len(depth)])
#filename1=directory+'RCF21CQ.csv'
#depth1,rcfdose1=read_datarcf(filename1)
#rcferr1=rcfdose1*(546/10000)

#normalization

#rcfdose1=rcfdose1/rcfdose1.max()
#rcferr1=rcferr1/rcfdose1.max()




directory='/home/corvin22/Desktop/miniscidom/pictures/2021-09-02/notnormalized/'

data= np.load(directory+'notnormalizedmean_array1.npy')
dose=data[0:len(data)-3]

#tof1=data1[len(data1)-3:len(data1)]
unidose=0
data0= np.load(directory+'notnormalizederr1.npy')
err=data0[0:len(data0)]
#top_projection_dose= np.load(directory+'1Dtopprojection/'+'top1Dmean_array1.npy')
x= np.arange(0,len(dose),1)*0.0738255
area=np.trapz(dose[3:len(dose)],x[3:len(x)]) #eliminate last 3 points

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
#outputfile_topas = '/home/corvin22/SimulationOncoray/data/SOBP/DoseToWater_150MeVproton_PVT_12PC_SOBP_1Dscorer.csv'
outputfile_topas = '/home/corvin22/SimulationOncoray/data/Single/DoseToWater_90MeVproton_PVT_6PC1PMMA_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array

numberof_xbins = 1
numberof_zbins = 149
doseprofile=Get_profile(topas_datamatrix, 149,1)
xdoseprofile=doseprofile.xmeanprofile
zdoseprofile=doseprofile.zmeanprofile
zdoseprofile=zdoseprofile[::-1]
depth_sci= np.arange(0,len(zdoseprofile),1)*0.0738255
area_sci=np.trapz(zdoseprofile[3:len(zdoseprofile)],depth_sci[3:len(depth_sci)]) #eliminate last 3 points


"""
# read in the csv-file
outputfile_topas = '/home/corvin22/SimulationOncoray/data/SOBP/EnergyDeposit_150MeVproton_PVT_12PC_SOBP_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix_energy = np.array(df)# convert dataframe df to array
numberof_xbins = 1
numberof_zbins = 149
energyprofile=Get_profile(topas_datamatrix_energy, 149,1)
xenergyprofile=energyprofile.xmeanprofile
zenergyprofile=energyprofile.zmeanprofile
zenergyprofile=zenergyprofile[::-1]

#area_sci=np.trapz(zenergprofile[3:len(zdoseprofile)],depth_sci[3:len(depth_sci)]) #eliminate last 3 points
"""

plt.figure(1) # creates a figure in which we plot
#plt.plot(np.arange(0,numberof_xbins,1)*0.0634, xmeanprofile/np.max(xmeanprofile),'.-',label='{mean value of 170 bins along z}') # plots depth on x, dose on y, and uses default layout
plt.plot(np.arange(0,numberof_zbins,1)*0.0738255,
                                                          zdoseprofile/area_sci,
                                                                            '.',
                                                                    markersize=8,
                                                        label='Simulation in miniscidom ')




plt.errorbar(np.arange(0,len(dose),1)*0.0738255,                         dose/area ,
                                                                      yerr=err/area,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                        color='green',
                                                                        markersize=8,
                                                                   ecolor='green',
                                                                elinewidth=None,
                                            label=' Reconstruction '.format(unidose))
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
plt.errorbar(depth,                      rcfdose/area_rcf,
                                                                  yerr=rcferr/area_rcf,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                        color='red',
                                                                        markersize=8,
                                                                   ecolor='red',
                                                                elinewidth=None,
                                                                label=' RCF 21CW')

plt.fill_between(depth,
                                                                rcfdose/area_rcf-rcferr/area_rcf,
                                                                rcfdose/area_rcf+rcferr/area_rcf,
                                                        color='red', alpha=0.1)












#plt.plot(depthPVT, dosePVT,'.-',label='{70 MeV protons ,0cm distance}') # plots depth on x, dose on y, and uses default layout
plt.title('Depht dose distribution(aperture diameter= 7mm, 6PC+1PMMA)') # Title
plt.xlabel('Depth in Water x direction [mm]') # label for x-axis
plt.ylabel('Relative Dose in water ') # label for y axis


plt.legend()
plt.minorticks_on()
plt.grid()
plt.show()
