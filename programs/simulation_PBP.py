import scipy
import pandas as pd # to work with structured, tabulated data (.csv, .xlsx,..)
import numpy as np # to work fast with matrices, arrays and much more
from matplotlib import pyplot as plt # to visualize/plot your data
from scipy import interpolate
from getprofile import Get_profile
import seaborn as sns



outputfile_topas = '/home/corvin22/SimulationOncoray/data/Single/7PC/DoseToWater_90MeVproton_PVT_7PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
doseprofile=Get_profile(topas_datamatrix, 149,1)
zdoseprofile=doseprofile.zmeanprofile
xdoseprofile=doseprofile.xmeanprofile
zdoseprofile=zdoseprofile[::-1]


(directory,energy,scoringvolume,PC,scorer)= outputfile_topas.split('_')
#(PC,format)=PC.split('.csv')
(energy,particle)=energy.split('MeV')



outputfile_topas = '/home/corvin22/SimulationOncoray/data/Single/6PC/DoseToWater_90MeVproton_PVT_6PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
doseprofile1=Get_profile(topas_datamatrix, 149,1)
zdoseprofile1=doseprofile1.zmeanprofile
xdoseprofile1=doseprofile1.xmeanprofile
zdoseprofile1=zdoseprofile1[::-1]
(directory,energy,scoringvolume,PC1,scorer)= outputfile_topas.split('_')
#(PC1,format)=PC1.split('.csv')
(energy1,particle)=energy.split('MeV')




outputfile_topas = '/home/corvin22/SimulationOncoray/data/DoseToWater_90MeVproton_PVT_6PC.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
doseprofileprova=Get_profile(topas_datamatrix, 170,161)
zdoseprofileprova=doseprofileprova.zmeanprofile
#xdoseprofile1=doseprofile1.xmeanprofile
#zdoseprofile1=zdoseprofile1[::-1]
#(directory,energy,scoringvolume,PC1)= outputfile_topas.split('_')
#(PC1,format)=PC1.split('.csv')
#(energy1,particle)=energy.split('MeV')











outputfile_topas = '/home/corvin22/SimulationOncoray/data/DoseToWater_90MeVproton_PVT_6PC_1PMMA.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
doseprofile2=Get_profile(topas_datamatrix, 170,161)
zdoseprofile2=doseprofile2.zmeanprofile
xdoseprofile2=doseprofile2.xmeanprofile
#zdoseprofile=zdoseprofile[::-1]
(directory,energy,scoringvolume,PC2,PMMA)= outputfile_topas.split('_')
(PMMA,format)=PMMA.split('.csv')
(energy2,particle)=energy.split('MeV')



s=0.0634
s1=0.073825
numberof_zbins=170
numberof_xbins=161






################################################################################







plt.figure(4) # creates a figure in which we plot
plt.plot(np.arange(0,149,1)*s1, zdoseprofile/np.max(zdoseprofile),
                                                                            '.',
                                                                            markersize=12,
                                                                color=sns.color_palette(  "Paired")[1],
                                                            label='{} '.format(PC))


plt.plot(np.arange(0,149,1)*s1, zdoseprofile1/np.max(zdoseprofile1),
                                                                            '.',
                                                                            markersize=12,
                                                                    color=sns.color_palette(  "Paired")[2],
                                                            label='{} '.format(PC1))

plt.plot(np.arange(0,numberof_zbins,1)*s, zdoseprofile2/np.max(zdoseprofile2),
                                                                        '.',
                                                                        markersize=12,
                                                                    color=sns.color_palette(  "Paired")[3],
                                                            label=' {}{}'.format(PC2,PMMA))






"""
ax.errorbar(  np.arange(0,len(dose),1)*s,                              dose*norm,
                                                                   yerr=err*norm,
                                                                       xerr=None,
                                                                         fmt='.',
                                          color=sns.color_palette(  "Paired")[3],
                                         ecolor=sns.color_palette(  "Paired")[3],
                                                                 elinewidth=None,
                                             label='  ')
"""
#plt.plot(np.arange(0,numberof_zbins,1)*s, zmeanprofile2/np.max(zmeanprofile2),'.-',label='{7PC}')
#plt.plot(depthPVT, dosePVT,'.-',label='{70 MeV protons ,0cm distance}') # plots depth on x, dose on y, and uses default layout

plt.title('Dose scored in Miniscidom',fontsize=26)
plt.xlabel(' z direction [mm]',fontsize=24)
plt.ylabel('Relative Dose in water ',fontsize=24)




plt.legend(title='E_{primary}=92.40 MeV',fontsize=24,loc=4,markerscale=2,title_fontsize=24)
plt.grid(b=True, which='major', color='k', linestyle='-',alpha=0.2)
plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
plt.minorticks_on()
plt.tick_params(axis='x', which='major', labelsize=22)
plt.tick_params(axis='y', which='major', labelsize=22)







outputfile_topas = '/home/corvin22/SimulationOncoray/data/DoseToWater_90MeVproton_PVT_7PC.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
doseprofile=Get_profile(topas_datamatrix, 170,161)
zdoseprofile=doseprofile.zmeanprofile
xdoseprofile=doseprofile.xmeanprofile
zdoseprofile=zdoseprofile[::-1]

(directory,energy,scoringvolume,PC)= outputfile_topas.split('_')
(PC,format)=PC.split('.csv')
(energy,particle)=energy.split('MeV')



outputfile_topas = '/home/corvin22/SimulationOncoray/data/DoseToWater_90MeVproton_PVT_6PC.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
doseprofile1=Get_profile(topas_datamatrix, 170,161)
zdoseprofile1=doseprofile1.zmeanprofile
xdoseprofile1=doseprofile1.xmeanprofile
zdoseprofile1=zdoseprofile1[::-1]
(directory,energy,scoringvolume,PC1)= outputfile_topas.split('_')
(PC1,format)=PC1.split('.csv')
(energy1,particle)=energy.split('MeV')





outputfile_topas = '/home/corvin22/SimulationOncoray/data/DoseToWater_90MeVproton_PVT_6PC_1PMMA.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
doseprofile2=Get_profile(topas_datamatrix, 170,161)
zdoseprofile2=doseprofile2.zmeanprofile
xdoseprofile2=doseprofile2.xmeanprofile
#zdoseprofile=zdoseprofile[::-1]
(directory,energy,scoringvolume,PC2,PMMA)= outputfile_topas.split('_')
(PMMA,format)=PMMA.split('.csv')
(energy2,particle)=energy.split('MeV')







plt.figure(5) # creates a figure in which we plot
plt.plot(np.arange(0,161,1)*s, xdoseprofile/np.max(xdoseprofile),
                                                                            '.',
                                                                            markersize=12,
                                                                color=sns.color_palette(  "Paired")[1],
                                                            label='{} '.format(PC))


plt.plot(np.arange(0,161,1)*s, xdoseprofile1/np.max(xdoseprofile1),
                                                                            '.',
                                                                            markersize=12,
                                                                    color=sns.color_palette(  "Paired")[2],
                                                            label='{} '.format(PC1))

plt.plot(np.arange(0,161,1)*s, xdoseprofile2/np.max(xdoseprofile2),
                                                                        '.',
                                                                        markersize=12,
                                                                    color=sns.color_palette(  "Paired")[3],
                                                            label=' {}{}'.format(PC2,PMMA))








#plt.plot(np.arange(0,numberof_zbins,1)*s, zmeanprofile2/np.max(zmeanprofile2),'.-',label='{7PC}')
#plt.plot(depthPVT, dosePVT,'.-',label='{70 MeV protons ,0cm distance}') # plots depth on x, dose on y, and uses default layout

plt.title('Dose scored in Miniscidom',fontsize=26)
plt.xlabel('x direction [mm]',fontsize=24)
plt.ylabel('Relative Dose in water ',fontsize=24)




plt.legend(title='E_{primary}=92.40 MeV',fontsize=24,loc=4,markerscale=2,title_fontsize=24)
plt.grid(b=True, which='major', color='k', linestyle='-',alpha=0.2)
plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
plt.minorticks_on()
plt.tick_params(axis='x', which='major', labelsize=22)
plt.tick_params(axis='y', which='major', labelsize=22)


































"""

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
#plt.plot( np.arange(0,len(dose),1)*s,
#                                                           top_projection_dose,
#                                                                           '.',
#                                                                  Markersize=11,
#                                               label='Top projection measured ')
#plt.plot( np.arange(0,len(dose),1)*s,
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
plt.plot(np.arange(0,numberof_zbins,1)*s, zmeanprofile1/np.max(zmeanprofile1),'.',label='{simulation }')




plt.title('Dose scored  in Miniscidom( 7PC,aluminum aperture diameter= 7mm)') # Title
plt.xlabel('Depth in Water x direction [mm]') # label for x-axis
plt.ylabel('Relative Dose in water ') # label for y axis


plt.legend()
plt.minorticks_on()
plt.grid()


"""















plt.show()
