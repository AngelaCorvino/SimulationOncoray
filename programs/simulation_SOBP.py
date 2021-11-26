
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
import seaborn as sns




"""
m=0.073825 #[mm/pixel ]
directory='/Users/angelacorvino/Desktop/HZDR/miniscidom/pictures/2021-09-01/notnormalized/'
#directory='/home/corvin22/Desktop/miniscidom/pictures/2021-09-01/notnormalized/'
dose,depth,tof,shotnumber=read_dose(directory,'notnormalizedmean_array9.npy',m)

#dose= dose- dose.min()/100 #background subtaction

err=read_doserr(directory,'notnormalizederr9.npy') #standard deviation divided by radical n
area=np.trapz(dose[3:len(dose)-1], depth[3:len(depth)-1])

"""




outputfile_topas = '/home/corvin22/SimulationOncoray/data/SOBP/DoseToWater_150MeVproton_PVT_10PC_SOBP_2Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
doseprofile=Get_profile(topas_datamatrix, 170,161)
zdoseprofile=doseprofile.zmeanprofile
xdoseprofile=doseprofile.xmeanprofile
#zdoseprofile=zdoseprofile[::-1]

(directory,energy,scoringvolume,PC,SOBP,scorer)= outputfile_topas.split('_')
#(PC,format)=PC.split('.csv')
(energy,particle)=energy.split('MeV')



outputfile_topas ='/home/corvin22/SimulationOncoray/data/SOBP/DoseToWater_150MeVproton_PVT_12PC_SOBP_2Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
doseprofile1=Get_profile(topas_datamatrix, 170,161)
zdoseprofile1=doseprofile1.zmeanprofile
xdoseprofile1=doseprofile1.xmeanprofile
#zdoseprofile1=zdoseprofile1[::-1]
(directory,energy,scoringvolume,PC1,SOBP,scorer)= outputfile_topas.split('_')
#(PC1,format)=PC1.split('.csv')
(energy1,particle)=energy.split('MeV')



outputfile_topas = '/home/corvin22/SimulationOncoray/data/SOBP/DoseToWater_150MeVproton_PVT_11PC_SOBP_2Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
doseprofile2=Get_profile(topas_datamatrix, 170,161)
zdoseprofile2=doseprofile2.zmeanprofile
xdoseprofile2=doseprofile2.xmeanprofile
zdoseprofile2=zdoseprofile2[::-1]
(directory,energy,scoringvolume,PC2,SOBP,scorer)= outputfile_topas.split('_')
#(PMMA,format)=PMMA.split('.csv')
(energy2,particle)=energy.split('MeV')



"""
outputfile_topas = '/home/corvin22/SimulationOncoray/data/SOBP/DoseToWater_150MeVproton_PVT_11PC_SOBP._2Dscorerbis.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
doseprofile3=Get_profile(topas_datamatrix, 170,161)
zdoseprofile3=doseprofile3.zmeanprofile
xdoseprofile3=doseprofile3.xmeanprofile
#zdoseprofile3=zdoseprofile3[::-1]
(directory,energy,scoringvolume,PC3,SOBP,scorer)= outputfile_topas.split('_')
#(PMMA,format)=PMMA.split('.csv')
(energy3,particle)=energy.split('MeV')


"""

















s=0.0634






plt.figure(1) # creates a figure in which we plot
plt.plot(np.arange(0,170,1)*s, zdoseprofile/np.max(zdoseprofile),
                                                                            '.',
                                                                            markersize=12,
                                                                color=sns.color_palette(  "Paired")[1],
                                                            label='{} '.format(PC))


plt.plot(np.arange(0,170,1)*s,np.true_divide(zdoseprofile1,np.max(zdoseprofile1)),
                                                                            '.',
                                                                            markersize=12,
                                                                    color=sns.color_palette(  "Paired")[2],
                                                            label='{} '.format(PC1))

plt.plot(np.arange(0,170,1)*s, zdoseprofile2/np.max(zdoseprofile2),
                                                                        '.',
                                                                        markersize=12,
                                                                    color=sns.color_palette(  "Paired")[3],
                                                            label=' {}'.format(PC2))



"""
plt.plot(np.arange(0,170,1)*s, zdoseprofile3/np.max(zdoseprofile3),
                                                                        '.',
                                                                        markersize=12,
                                                                    color=sns.color_palette(  "Paired")[4],
                                                            label=' {}'.format(PC3))


"""



#plt.plot(np.arange(0,numberof_zbins,1)*s, zmeanprofile2/np.max(zmeanprofile2),'.-',label='{7PC}')
#plt.plot(depthPVT, dosePVT,'.-',label='{70 MeV protons ,0cm distance}') # plots depth on x, dose on y, and uses default layout

plt.title('Dose scored in Miniscidom',fontsize=26)
plt.xlabel('z direction [mm]',fontsize=24)
plt.ylabel('Relative Dose in water ',fontsize=24)




plt.legend(title='E_{primary}=150 MeV',fontsize=24,loc=4,markerscale=2,title_fontsize=24)
plt.grid(b=True, which='major', color='k', linestyle='-',alpha=0.2)
plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
plt.minorticks_on()
plt.tick_params(axis='x', which='major', labelsize=22)
plt.tick_params(axis='y', which='major', labelsize=22)












plt.figure(2) # creates a figure in which we plot
plt.plot(np.arange(0,161,1)*s, xdoseprofile/np.max(xdoseprofile),
                                                                            '.',
                                                                            markersize=12,
                                                                color=sns.color_palette(  "Paired")[1],
                                                            label='{} '.format(PC))


plt.plot(np.arange(0,161,1)*s,np.true_divide(xdoseprofile1,np.max(xdoseprofile1)),
                                                                            '.',
                                                                            markersize=12,
                                                                    color=sns.color_palette(  "Paired")[2],
                                                            label='{} '.format(PC1))

plt.plot(np.arange(0,161,1)*s, xdoseprofile2/np.max(xdoseprofile2),
                                                                        '.',
                                                                        markersize=12,
                                                                    color=sns.color_palette(  "Paired")[3],
                                                            label=' {}'.format(PC2))








#plt.plot(np.arange(0,numberof_zbins,1)*s, zmeanprofile2/np.max(zmeanprofile2),'.-',label='{7PC}')
#plt.plot(depthPVT, dosePVT,'.-',label='{70 MeV protons ,0cm distance}') # plots depth on x, dose on y, and uses default layout

plt.title('Dose scored in Miniscidom',fontsize=26)
plt.xlabel('x direction [mm]',fontsize=24)
plt.ylabel('Relative Dose in water ',fontsize=24)




plt.legend(title='E_{primary}=150 MeV',fontsize=24,loc=4,markerscale=2,title_fontsize=24)
plt.grid(b=True, which='major', color='k', linestyle='-',alpha=0.2)
plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
plt.minorticks_on()
plt.tick_params(axis='x', which='major', labelsize=22)
plt.tick_params(axis='y', which='major', labelsize=22)








s1=0.073825

outputfile_topas = '/home/corvin22/Desktop/precoiusthings/SOBP/10PCLET/150.5MeV/DoseToWater_151500KeVproton_PVT_10PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
doseprofile=Get_profile(topas_datamatrix, 149,1)
zdoseprofile=doseprofile.zmeanprofile
xdoseprofile=doseprofile.xmeanprofile
zdoseprofile=zdoseprofile[::-1]
(directory,energy,scoringvolume,PC,scorer)= outputfile_topas.split('_')
#(PC,format)=PC.split('.csv')
(energy,particle)=energy.split('KeV')



outputfile_topas = '/home/corvin22/Desktop/precoiusthings/SOBP/11PCLET/150.5MeV108/DoseToWater_150500KeVproton_PVT_11PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
doseprofile1=Get_profile(topas_datamatrix, 149,1)
zdoseprofile1=doseprofile1.zmeanprofile
xdoseprofile1=doseprofile1.xmeanprofile
zdoseprofile1=zdoseprofile1[::-1]
(directory,energy,scoringvolume,PC1,scorer)= outputfile_topas.split('_')
#(PC1,format)=PC1.split('.csv')
(energy1,particle)=energy.split('KeV')





outputfile_topas = '/home/corvin22/Desktop/precoiusthings/SOBP/12PC/150.5MeV/DoseToWater_150500KeVproton_PVT_12PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
doseprofile2=Get_profile(topas_datamatrix, 149,1)
zdoseprofile2=doseprofile2.zmeanprofile
xdoseprofile2=doseprofile2.xmeanprofile
zdoseprofile2=zdoseprofile2[::-1]
(directory,energy,scoringvolume,PC2,scorer)= outputfile_topas.split('_')
#(PMMA,format)=PMMA.split('.csv')
(energy2,particle)=energy.split('KeV')









plt.figure(3) # creates a figure in which we plot
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

plt.plot(np.arange(0,149,1)*s1, zdoseprofile2/np.max(zdoseprofile2),
                                                                        '.',
                                                                        markersize=12,
                                                                    color=sns.color_palette(  "Paired")[3],
                                                            label=' {}'.format(PC2))






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




plt.legend(title='E_{primary}=150 MeV',fontsize=24,loc=4,markerscale=2,title_fontsize=24)
plt.grid(b=True, which='major', color='k', linestyle='-',alpha=0.2)
plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
plt.minorticks_on()
plt.tick_params(axis='x', which='major', labelsize=22)
plt.tick_params(axis='y', which='major', labelsize=22)




plt.show()



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
