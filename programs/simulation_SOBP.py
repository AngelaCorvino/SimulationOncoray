
import scipy
import pandas as pd # to work with structured, tabulated data (.csv, .xlsx,..)
import numpy as np # to work fast with matrices, arrays and much more
from matplotlib import pyplot as plt # to visualize/plot your data
from scipy import interpolate
import csv
from getprofile import  Get_profile
from readcsv import read_datarcf
from readnpy import read_dose
from readnpy import read_doserr





m=0.073825 #[mm/pixel ]
directory='/Users/angelacorvino/Desktop/HZDR/miniscidom/pictures/2021-09-01/notnormalized/'
#directory='/home/corvin22/Desktop/miniscidom/pictures/2021-09-01/notnormalized/'
dose,depth,tof,shotnumber=read_dose(directory,'notnormalizedmean_array9.npy',m)

#dose= dose- dose.min()/100 #background subtaction

err=read_doserr(directory,'notnormalizederr9.npy') #standard deviation divided by radical n
area=np.trapz(dose[3:len(dose)-1], depth[3:len(depth)-1])





# read in the csv-file
outputfile_topas='/Users/angelacorvino/Desktop/GitHub/SimulationOncoray/data/SOBP/10PC/DoseToWater_152MeVproton_PVT_10PC_1Dscorer.csv'
#outputfile_topas = '/home/corvin22/SimulationOncoray/data/SOBP/DoseToWater_14960KeVproton_PVT_12PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array


(directory,energy,scoringvolume,PC,dimension)= outputfile_topas.split('_')
(energy,particle)=energy.split('MeV')




numberof_xbins = 1
numberof_zbins = 149
doseprofile=Get_profile(topas_datamatrix, 149,1)
xmeanprofile=doseprofile.xmeanprofile
zmeanprofile=doseprofile.zmeanprofile
zmeanprofile=zmeanprofile[::-1]


"""
#m1=0.0634
m1=m
# read in the csv-file
outputfile_topas = '/home/corvin22/SimulationOncoray/data/SOBP/DoseToWater_150MeVproton_PVT_12PC_SOBP_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array


(directory,energy1,scoringvolume,PC,SOBP,dimension)= outputfile_topas.split('_')
(energy1,particle)=energy1.split('MeV')



numberof_xbins1 = 1
numberof_zbins1 = 149
doseprofile=Get_profile(topas_datamatrix, 149,1)
xprofiles=doseprofile.xprofiles
xmeanprofile=doseprofile.xmeanprofile
zmeanprofile1=doseprofile.zmeanprofile
zmeanprofile1=zmeanprofile1[::-1]

"""




"""


plt.figure(1) # creates a figure in which we plot
plt.plot(np.arange(0,numberof_xbins1,1)*m,xprofiles[10,:]/np.nanmax(xprofiles[10,:]) ,'.-',
                                    label='{x profile at z = 0,634 mm inside PVT}')
plt.plot(np.arange(0,numberof_xbins1,1)*m,
                                    xprofiles[50,:]/np.nanmax(xprofiles[50,:]) ,
                                                                            '.-',
                                    label='{x profile at z = 3,17 mm inside PVT}')

plt.plot(np.arange(0,numberof_xbins1,1)*m,
                                    xprofiles[120,:]/np.nanmax(xprofiles[120,:]),
                                                                            '.-',
                                    label='{x profile at z = 9,51 mm inside PVT}')


plt.title('Scoring in Miniscidom {} '.format(PC),
                                                                  fontdict=None,
                                                                  fontsize=24,
                                                                  loc='center',
                                                                       pad=None)

plt.tick_params(axis='x', which='major', labelsize=16)
plt.tick_params(axis='y', which='major', labelsize=16)

plt.xlabel('Depth x direction [mm]',fontsize=24) # label for x-axis
plt.ylabel('Relative dose ',fontsize=24) # label for y axis
plt.legend( title='',fontsize=22,loc=1)
plt.minorticks_on()
plt.grid()





plt.figure(3) # creates a figure in which we plot
plt.plot(np.arange(0,numberof_xbins,1)*m,
                                                xmeanprofile/np.max( xmeanprofile),
                                                                            '.-',
                                                                            markersize=12,
                                        label='Simulated Dose Eo={} KeV '.format(energy)) # plots depth on x, dose on y, and uses default layout


#plt.plot(depthPVT, dosePVT,'.-',label='{70 MeV protons ,0cm distance}') # plots depth on x, dose on y, and uses default layout
plt.title('Scoring in Miniscidom {} '.format(PC),
                                                                  fontdict=None,
                                                                  fontsize=24,
                                                                  loc='center',
                                                                       pad=None)

plt.tick_params(axis='x', which='major', labelsize=16)
plt.tick_params(axis='y', which='major', labelsize=16)

plt.xlabel('Depth x direction [mm]',fontsize=24) # label for x-axis
plt.ylabel('Relative dose ',fontsize=24) # label for y axis
plt.legend( title='',fontsize=22,loc=1)
plt.minorticks_on()
plt.grid()

"""

plt.figure(4) # creates a figure in which we plot

plt.plot(np.arange(0,numberof_zbins,1)*m, zmeanprofile/np.nanmax(zmeanprofile),
                                                                            '.',
                                                                markersize=12,
                               label='Simulated Dose Eo={} MeV '.format(energy))


"""

plt.plot(np.arange(0,numberof_zbins1,1)*m1, zmeanprofile1/np.nanmax(zmeanprofile1),
                                                                            '.-',
                                                                    markersize=12,
                                             label='Simulated Dose Eo=150 MeV ')
#                                                    label='Simulated Dose Eo={} KeV '.format(energy1))

"""

plt.title('Scoring in Miniscidom {} '.format(PC),
                                                                  fontdict=None,
                                                                    fontsize=24,
                                                                   loc='center',
                                                                       pad=None)
plt.ylim([0,1.1])
plt.tick_params(axis='x', which='major', labelsize=16)
plt.tick_params(axis='y', which='major', labelsize=16)

plt.xlabel('Depth z direction [mm]',fontsize=24) # label for x-axis
plt.ylabel('Relative dose ',fontsize=24) # label for y axis
plt.legend( title='',fontsize=22,loc=1)
plt.minorticks_on()
plt.grid()




"""




directory='/home/corvin22/Desktop/miniscidom/pictures/2021-09-01/'
filename='RCF21CQ.csv'
rcfdepth,rcfdose,rcferr,area_rcf,rcfname=read_datarcf(directory,filename)


filename1='RCF21CP.csv'
rcfdepth1,rcfdose1,rcferr1,area_rcf1,rcfname1=read_datarcf(directory,filename1)

plt.figure(6)
#plt.plot( np.arange(0,len(dose2),1)*0.074,
#                                                           dose2/np.max(dose2),
#                                                                           '.',
#                                                                  Markersize=11,
#                                               label='recobnstruction (different filters settings) ')
#plt.plot( np.arange(0,len(dose),1)*0.074,
#                                                                    dosenotmasked/np.max(dosenotmasked),
#                                                                          '.',
#                                                                  markersize=7,
#                                                        label='{notmasked}')
plt.errorbar(  np.arange(0,len(dose),1)*m,                         dose/dose.max() ,
                                                                      yerr=err/dose.max(),
                                                                      xerr=None,
                                                                        fmt='.',
                                                                   ecolor='gray',
                                                                elinewidth=None,
                                            label=' reconstruction '.format(unidose))
plt.plot(np.arange(0,numberof_zbins,1)*0.0634, zmeanprofile/np.max(zmeanprofile),'.',label='{simulation }')



plt.errorbar( rcfdepth,                      rcfdose/rcfdose.max() ,
                                                                  yerr=rcferr/rcfdose.max(),
                                                                      xerr=None,
                                                                        fmt='.',
                                                                        markersize=9,
                                                                   ecolor='gray',
                                                                elinewidth=None,
                                                                label='{} measured dose'.format(rcfname))

plt.fill_between(rcfdepth,
                                                                rcfdose/rcfdose.max()-rcferr/rcfdose.max(),
                                                                rcfdose/rcfdose.max()+rcferr/rcfdose.max(),
                                                        color='gray', alpha=0.5)

plt.errorbar( rcfdepth,                      rcfdose1/rcfdose1.max() ,
                                                                  yerr=rcferr1/rcfdose1.max(),
                                                                      xerr=None,
                                                                        fmt='.',
                                                                        markersize=9,
                                                                   ecolor='gray',
                                                                elinewidth=None,
                                                                label='{} measured dose'.format(rcfname1))

plt.fill_between(rcfdepth,
                                                                rcfdose1/rcfdose1.max()-rcferr1/rcfdose1.max(),
                                                                rcfdose1/rcfdose1.max()+rcferr1/rcfdose1.max(),
                                                        color='gray', alpha=0.5)


plt.title('Dose scored  in Miniscidom( 10PC,aluminum aperture diameter= 7mm)') # Title
plt.xlabel('Depth in Water x direction [mm]') # label for x-axis
plt.ylabel('Relative Dose in water ') # label for y axis


plt.legend()
plt.minorticks_on()
plt.grid()





"""









plt.show()
