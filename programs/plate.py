
"""DEFINE NUMBER OF PC PLATES"""


import scipy
import pandas as pd # to work with structured, tabulated data (.csv, .xlsx,..)
import numpy as np # to work fast with matrices, arrays and much more
from matplotlib import pyplot as plt # to visualize/plot your data
from scipy import interpolate
from getprofile import Get_profile
from mpl_toolkits.axes_grid1.inset_locator import mark_inset




#Simulation
outputfile_topas = '/Users/angelacorvino/Desktop/TOPASpreciousimulation/waterphantom/rightcollimator/DoseToWater_1500KeVproton_phantomafteraperture_1Dscorer.csv'


header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
depth=np.array(topas_datamatrix[:,2])*0.2 #[mm]
doseprofile=Get_profile(topas_datamatrix, 900,1)
zdoseprofile=doseprofile.zmeanprofile
zdoseprofile=zdoseprofile[::-1]
dose=zdoseprofile
dose=dose/np.nanmax(dose)
area=np.trapz(dose[0:len(dose)], depth[0:len(depth)])
(directory,energy,scoringvolume,dimension)= outputfile_topas.split('_')
(energy,particle)=energy.split('KeV')





#Simulation
outputfile_topas = '/Users/angelacorvino/Desktop/TOPASpreciousimulation/waterphantom/rightcollimator/DoseToWater_1500KeVproton_phantombeforeaperture_1Dscorer.csv'
#outputfile_topas = '/home/corvin22/Simulationhemera/data/SOBP/DoseToWater_1500KeVproton_phantombeforeaperture_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix1= np.array(df)# convert dataframe df to array
depth=np.array(topas_datamatrix1[:,2])*0.2 #[mm]
doseprofile1=Get_profile(topas_datamatrix1, 900,1)
zdoseprofile1=doseprofile1.zmeanprofile
zdoseprofile1=zdoseprofile1[::-1]
dose1=zdoseprofile1
dose1=dose1/np.nanmax(dose1)
area1=np.trapz(dose1[0:len(dose1)], depth[0:len(depth)])

(directory,energy,scoringvolume,dimension)= outputfile_topas.split('_')
(energy,particle)=energy.split('KeV')



print(len(depth),len(dose))


xmin=[87.2,95.93,104]
dx=11
fig, ax = plt.subplots()
plt.title('Dose scored in water phantom (diameter=12 mm) ',fontsize=24) # Title
#ax2 = ax.twinx()


sub_axes = plt.axes([.54, .52, .33, .33])

sub_axes.plot(depth[450:600],
                                                                         (np.divide(dose,dose1)*100)[450:600],
                                                                          '.',
                                                                           color='green',
                                                                            markersize=10,
                                                                label='Ratio')
sub_axes1 = plt.axes([.16, .35, .3, .3])
sub_axes1.plot(depth[450:600],
                                                                          dose1[450:600],
                                                                           '.',
                                                                           color='orange',
                                                                            markersize=9)


sub_axes1.plot(depth[450:600],
                                                                          dose[450:600],
                                                                           '.',
                                                                           color='blue',
                                                                            markersize=9)



ax.plot(depth,
                                                                          dose1,
                                                                           '.',
                                                                           color='orange',
                                                                            markersize=11,
                                            label='Phantom before aperture')


ax.plot(depth,
                                                                          dose,
                                                                           '.',
                                                                           markersize=11,
                                            label='Phantom after aperture')
#ax.plot(depth[450:600],
#                                                                         (np.divide(dose,dose1)*100)[450:600],
#                                                                          '.',
#                                                                           color='green',
#                                                                            markersize=11,
#                                                                label='Ratio')




#plt.plot(depth,dose2,'.-',label='Phantom before the aperture ')
#plt.plot(np.arange(0,numberof_zbins,1)*0.6, zmeanprofile/np.max(zmeanprofile),'.-',label='{Minisicidom }')
"""
ax.axvspan(xmin[0], xmin[0]+dx, facecolor='green', alpha=0.3, label='  10 PC')
ax.axvspan(xmin[1], xmin[1]+dx,
                                                         facecolor='turquoise',
                                                                      alpha=0.3,
                                                                label='  11 PC')
#plt.axvspan(xmin[2],xmin[2]+dx, facecolor='deepskyblue', alpha=0.3, label='12 PC')



ax.axvspan(xmin[2],xmin[2]+dx,
                                                              facecolor='orange',
                                                                      alpha=0.3,
                                                                    label='  12 PC')

"""
#mark_inset(ax, sub_axes, loc1=3, loc2=4, fc="none", ec="0.5")
mark_inset(ax, sub_axes1, loc1=2, loc2=3, fc="none", ec="0.5")
sub_axes1.set_ylim([0,0.01])
ax.set_xlabel('Depth z [mm]',fontsize=20) # label for x-axis
ax.set_ylabel('Relative dose ',fontsize=24) # label for y axis
sub_axes.set_ylabel('$D_{a}/ D_{b}$[%]',fontsize=20) # label for y axis
sub_axes.set_xlabel('Depth z [mm]',fontsize=20)
ax.legend( title='',fontsize=22,loc=3,
markerscale=3)
sub_axes.legend( title='',fontsize=22,loc=1,markerscale=3)
#ax2.grid(b=True,color='k',linestyle='-',alpha=0.2)
#ax.grid(b=True,color='k',linestyle='dotted',alpha=0.2)
ax.tick_params(axis='x', which='major', labelsize=22)
sub_axes.tick_params(axis='x', which='major',colors='green', labelsize=18)
sub_axes.tick_params(axis='y', which='major',colors='green', labelsize=18)
ax.tick_params(axis='y', which='major',colors='black', labelsize=22)

plt.figure(2) # creates a figure in which we plot
#plt.plot(np.arange(0,numberof_xbins,1)*0.0634, xmeanprofile,'.-',label='{mean value of 200 bins along z}') # plots depth on x, dose on y, and uses default layout
plt.plot(depth,
                                                                          dose,
                                                                           '.',
                                                                           markersize=10,
                                            label='Phantom after the aperture ')

plt.plot(depth,
                                                                          dose1,
                                                                           '.',
                                                                            markersize=10,
                                            label='Phantom before the aperture ')






plt.tick_params(axis='x', which='major', labelsize=16)
plt.tick_params(axis='y', which='major', labelsize=16)

plt.title('Dose scored in  water phantom (120mm diametr cylinder) ',fontsize=24) # Title
plt.xlabel('Depth z direction [mm]',fontsize=24) # label for x-axis
plt.ylabel('Relative dose ',fontsize=24) # label for y axis

plt.legend( title='',fontsize=22,loc=1)

plt.minorticks_on()
plt.grid()



plt.show()
