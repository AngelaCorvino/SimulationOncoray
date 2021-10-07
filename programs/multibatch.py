import scipy
import pandas as pd # to work with structured, tabulated data (.csv, .xlsx,..)
import numpy as np # to work fast with matrices, arrays and much more
from matplotlib import pyplot as plt # to visualize/plot your data
from scipy import interpolate
import csv

from readcsv import read_datarcf
from getprofile import  Get_profile

directory='/home/corvin22/Desktop/miniscidom/pictures/2021-09-02/'
filename=directory+'RCF21CY.csv'
depth,rcfdose=read_datarcf(filename)
rcferr=rcfdose*(546/10000)

#filename1=directory+'RCF21CQ.csv'
#depth1,rcfdose1=read_datarcf(filename1)
#rcferr1=rcfdose1*(546/10000)

#normalization

#rcfdose1=rcfdose1/rcfdose1.max()
#rcferr1=rcferr1/rcfdose1.max()




directory='/home/corvin22/Desktop/miniscidom/pictures/2021-09-02/notnormalized/'

data= np.load(directory+'notnormalizedmean_array19.npy')
dose=data[0:len(data)-3]

#tof1=data1[len(data1)-3:len(data1)]
unidose=0
data0= np.load(directory+'notnormalizederr19.npy')
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



# read in the csv-file
outputfile_topas = '/home/corvin22/Desktop/Simulationhemera/data/SOBP/DoseToWater_150MeVproton_PVT_10PC_SOBP_2Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrixold = np.array(df)# convert dataframe df to array
profileold=Get_profile(topas_datamatrixold, 170,161)
xmeanprofileold=profileold.xmeanprofile
zmeanprofileold=profileold.zmeanprofile




numberof_zbins=149

# read in the csv-file
outputfile_topas = '/home/corvin22/Desktop/Simulationhemera/data/SOBP/Job4Output_Dose10PC.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix4 = np.array(df)# convert dataframe df to array
profile4=Get_profile(topas_datamatrix4, 149,1)
xmeanprofile4=profile4.xmeanprofile
zmeanprofile4=profile4.zmeanprofile
zmeanprofile4=zmeanprofile4[::-1]


# read in the csv-file
outputfile_topas = '/home/corvin22/Desktop/Simulationhemera/data/SOBP/Job1Output_Dose10PC.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix1 = np.array(df)# convert dataframe df to array
profile1=Get_profile(topas_datamatrix1, 149,1)
xmeanprofile1=profile1.xmeanprofile
zmeanprofile1=profile1.zmeanprofile
zmeanprofile1=zmeanprofile1[::-1]



# read in the csv-file
outputfile_topas = '/home/corvin22/Desktop/Simulationhemera/data/SOBP/Job2Output_Dose10PC.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix2 = np.array(df)# convert dataframe df to array
profile2=Get_profile(topas_datamatrix2, 149,1)
xmeanprofile2=profile2.xmeanprofile
zmeanprofile2=profile2.zmeanprofile
zmeanprofile2=zmeanprofile2[::-1]








# read in the csv-file
outputfile_topas = '/home/corvin22/Desktop/Simulationhemera/data/SOBP/Job3Output_Dose10PC.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix3 = np.array(df)# convert dataframe df to array
profile3=Get_profile(topas_datamatrix3, 149,1)
xmeanprofile3=profile3.xmeanprofile
zmeanprofile3=profile3.zmeanprofile
zmeanprofile3=zmeanprofile3[::-1]




# read in the csv-file
outputfile_topas = '/home/corvin22/Desktop/Simulationhemera/data/SOBP/Job5Output_Dose10PC.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix5 = np.array(df)# convert dataframe df to array
profile5=Get_profile(topas_datamatrix5, 149,1)

zmeanprofile5=profile5.zmeanprofile
zmeanprofile5=zmeanprofile5[::-1]






# read in the csv-file
outputfile_topas = '/home/corvin22/Desktop/Simulationhemera/data/SOBP/Job6Output_Dose10PC.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix6 = np.array(df)# convert dataframe df to array
profile6=Get_profile(topas_datamatrix6, 149,1)

zmeanprofile6=profile6.zmeanprofile
zmeanprofile6=zmeanprofile6[::-1]






# read in the csv-file
outputfile_topas = '/home/corvin22/Desktop/Simulationhemera/data/SOBP/Job7Output_Dose10PC.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix7 = np.array(df)# convert dataframe df to array
profile7=Get_profile(topas_datamatrix7, 149,1)

zmeanprofile7=profile7.zmeanprofile
zmeanprofile7=zmeanprofile7[::-1]


# read in the csv-file
outputfile_topas = '/home/corvin22/Desktop/Simulationhemera/data/SOBP/Job8Output_Dose10PC.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix8 = np.array(df)# convert dataframe df to array
profile8=Get_profile(topas_datamatrix8, 149,1)

zmeanprofile8=profile8.zmeanprofile
zmeanprofile8=zmeanprofile8[::-1]




# read in the csv-file
outputfile_topas = '/home/corvin22/Desktop/Simulationhemera/data/SOBP/Job9Output_Dose10PC.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix9 = np.array(df)# convert dataframe df to array
profile9=Get_profile(topas_datamatrix9, 149,1)

zmeanprofile9=profile9.zmeanprofile
zmeanprofile9=zmeanprofile9[::-1]






# read in the csv-file
outputfile_topas = '/home/corvin22/Desktop/Simulationhemera/data/SOBP/Job10Output_Dose10PC.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix10 = np.array(df)# convert dataframe df to array
profile10=Get_profile(topas_datamatrix10, 149,1)

zmeanprofile10=profile10.zmeanprofile
zmeanprofile10=zmeanprofile10[::-1]


zmeanprofiletot=zmeanprofile1+zmeanprofile2+zmeanprofile3+zmeanprofile4+zmeanprofile5+zmeanprofile6+zmeanprofile7+zmeanprofile8+zmeanprofile9+zmeanprofile10
"""


# read in the csv-file
outputfile_topas = '/home/corvin22/Desktop/Simulationhemera/data/Single/DoseToWater_90MeVproton_PVT_7PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix10 = np.array(df)# convert dataframe df to array
profile10=Get_profile(topas_datamatrix10, 149,1)

zmeanprofile=profile10.zmeanprofile
zmeanprofile=zmeanprofile[::-1]

plt.figure(1) # creates a figure in which we plot
#plt.plot(np.arange(0,numberof_xbins,1)*0.0634, xmeanprofile/np.max(xmeanprofile),'.-',label='{mean value of 170 bins along z}') # plots depth on x, dose on y, and uses default layout
plt.plot(np.arange(0,149,1)*0.0738255,
                                            zmeanprofile/np.max(zmeanprofile),
                                                                            '.',
                                                                            markersize=8,
                                        label='TOPAS Simulation')
"""
plt.plot(np.arange(0,170,1)*0.0634,
                                            zmeanprofileold/np.max(zmeanprofileold),
                                                                            '.',
                                                                            markersize=8,
                                        label='{ TOPAS Simulation}')
"""



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
                                                                label=' RCF 21CY measured dose')

plt.fill_between(depth,
                                                                rcfdose/rcfdose.max()-rcferr/rcfdose.max(),
                                                                rcfdose/rcfdose.max()+rcferr/rcfdose.max(),
                                                        color='red', alpha=0.1)












#plt.plot(depthPVT, dosePVT,'.-',label='{70 MeV protons ,0cm distance}') # plots depth on x, dose on y, and uses default layout
plt.title('Depht dose distribution(aperture diameter= 7mm, 7PC, scoring in 5mm diameter cylinder )') # Title
plt.xlabel('Depth in Water x direction [mm]') # label for x-axis
plt.ylabel('Relative Dose in water ') # label for y axis


plt.legend()
plt.minorticks_on()
plt.grid()
plt.show()
