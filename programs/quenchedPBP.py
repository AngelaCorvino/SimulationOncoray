'''
COMPARISON BETWEEN TOPAS SIMULATED QUENCHED CURVE DATA IN SCINTILLATOR  AND SCINTILLATOR MEASUREMENTS
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
from scipy import optimize
from scipy.optimize import curve_fit
import csv
import pandas as pd
from readnpy import read_dose
from readnpy import read_doserr
from readcsv import read_datarcf
from readcsv import read_data_let
from readcsv import read_data_let_scintillator
from readcsv import read_data_let_mini
from readcsv import read_data_scintillatorsimulateddose
from readcsv import read_data_scintillatorsimulateddose_it
from readcsv import read_data_mini
from Birkmodel import lightcorrection
from Birkmodel import dosecorrection
from getprofile import Get_profile
from RCFresponse import rcfdosecorrection
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

###############################################################################
#September 2021
s=0.073825 #[mm/pixel ]
directory='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2021-09-02/notnormalized/'
dose,depth,tof,shotnumber=read_dose(directory,'notnormalizedmean_array1.npy',s)
dose1,depth1,tof1,shotnumber1=read_dose(directory,'notnormalizedmean_array19.npy',s)
dose2,depth2,tof2,shotnumber2=read_dose(directory,'notnormalizedmean_array17.npy',s)

err=read_doserr(directory,'notnormalizederr1.npy') #standard deviation divided by radical n
err1=read_doserr(directory,'notnormalizederr19.npy')
err2=read_doserr(directory,'notnormalizederr17.npy')
area=np.trapz(dose[3:len(dose)-7], depth[3:len(depth)-7])
area1=np.trapz(dose1[3:len(dose1)-7], depth[3:len(depth1)-7])
area2=np.trapz(dose2[3:len(dose2)-7], depth[3:len(depth2)-7])



outputfile_topas = '/Users/angelacorvino/Desktop/TOPASpreciousimulation/Single/10°7/secondcollimator/DoseToWater_92100KeVproton_PVT_7PC_1Dscorer.csv'
#outputfile_topas = '/Users/angelacorvino/Documents/GitHub/SimulationOncoray/data/Single/DoseToWater_9210KeVproton_PVT_7PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
doseprofile1=Get_profile(topas_datamatrix, 149,1)
zdoseprofile1=doseprofile1.zmeanprofile
zdoseprofile1=zdoseprofile1[::-1]
dose_sci1=zdoseprofile1
(directory,energy1,scoringvolume1,PC1,dimension)= outputfile_topas.split('_')
(energy1,particle)=energy1.split('KeV')




outputfile_topas = '/Users/angelacorvino/Documents/GitHub/SimulationOncoray/data/Single/DoseToWater_9210KeVproton_PVT_6PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
doseprofile2=Get_profile(topas_datamatrix, 149,1)
zdoseprofile2=doseprofile2.zmeanprofile
zdoseprofile2=zdoseprofile2[::-1]
dose_sci2=zdoseprofile2
(directory,energy2,scoringvolume2,PC2,dimension)= outputfile_topas.split('_')
(energy2,particle)=energy2.split('KeV')




#Simulation

outputfile_topas = '/Users/angelacorvino/Documents/GitHub/SimulationOncoray/data/Single/DoseToWater_9160KeVproton_PVT_6PC1PMMA_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
doseprofile=Get_profile(topas_datamatrix, 149,1)
zdoseprofile=doseprofile.zmeanprofile
zdoseprofile=zdoseprofile[::-1]
dose_sci=zdoseprofile
(directory,energy,scoringvolume,PC,dimension)= outputfile_topas.split('_')
(energy,particle)=energy.split('KeV')




#outputfile_topas = '/home/corvin22/Simulationhemera/data/Single/LET_fluenceweighted_9160KeVproton_PVT_6PC1PMMA_1Dscorer.csv'
outputfile_topas = '/Users/angelacorvino/Documents/GitHub/SimulationOncoray/data/Single/LET_doseweighted_9160KeVproton_PVT_6PC1PMMA_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix= np.array(df)# convert dataframe df to array
LET_doseprofile=Get_profile(topas_datamatrix, 149,1)
LET_zdoseprofile=LET_doseprofile.zmeanprofile
LET_zdoseprofile=LET_zdoseprofile[::-1]
LET_zdoseprofile[0]=LET_zdoseprofile[1]

#outputfile_topas = '/home/corvin22/Simulationhemera/data/Single/LET_fluenceweighted_9160KeVproton_PVT_6PC1PMMA_1Dscorer.csv'
outputfile_topas = '/Users/angelacorvino/Documents/GitHub/SimulationOncoray/data/Single/LET_fluenceweighted_9160KeVproton_PVT_6PC1PMMA_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix= np.array(df)# convert dataframe df to array
LET_fluenceprofile=Get_profile(topas_datamatrix, 149,1)
LET_zfluenceprofile=LET_fluenceprofile.zmeanprofile
LET_zfluenceprofile=LET_zfluenceprofile[::-1]
LET_zfluenceprofile[0]=LET_zfluenceprofile[1]




outputfile_topas = '/Users/angelacorvino/Desktop/TOPASpreciousimulation/Single/10°7/secondcollimator/LET_fluenceweighted_92100KeVproton_PVT_7PC_1Dscorer.csv'
#outputfile_topas = '/Users/angelacorvino/Documents/GitHub/SimulationOncoray/data/Single/LET_fluenceweighted_9210KeVproton_PVT_7PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix= np.array(df)# convert dataframe df to array
LET_fluenceprofile1=Get_profile(topas_datamatrix, 149,1)
LET_zfluenceprofile1=LET_fluenceprofile1.zmeanprofile
LET_zfluenceprofile1=LET_zfluenceprofile1[::-1]
LET_zfluenceprofile1[0]=LET_zfluenceprofile1[1]

outputfile_topas = '/Users/angelacorvino/Desktop/TOPASpreciousimulation/Single/10°7/secondcollimator/LET_fluenceweighted_92100KeVproton_PVT_7PC_1Dscorer.csv'
#outputfile_topas = '/Users/angelacorvino/Documents/GitHub/SimulationOncoray/data/Single/LET_fluenceweighted_9210KeVproton_PVT_7PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix= np.array(df)# convert dataframe df to array
LET_doseprofile1=Get_profile(topas_datamatrix, 149,1)
LET_zdoseprofile1=LET_doseprofile1.zmeanprofile
LET_zdoseprofile1=LET_zdoseprofile1[::-1]
LET_zdoseprofile1[0]=LET_zdoseprofile1[1]

outputfile_topas = '/Users/angelacorvino/Documents/GitHub/SimulationOncoray/data/Single/LET_fluenceweighted_9210KeVproton_PVT_6PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix= np.array(df)# convert dataframe df to array
LET_fluenceprofile2=Get_profile(topas_datamatrix, 149,1)
LET_zfluenceprofile2=LET_fluenceprofile2.zmeanprofile
LET_zfluenceprofile2=LET_zfluenceprofile2[::-1]
LET_zfluenceprofile2[0]=LET_zfluenceprofile2[1]
outputfile_topas = '/Users/angelacorvino/Documents/GitHub/SimulationOncoray/data/Single/LET_fluenceweighted_9210KeVproton_PVT_6PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix= np.array(df)# convert dataframe df to array
LET_doseprofile2=Get_profile(topas_datamatrix, 149,1)
LET_zdoseprofile2=LET_doseprofile2.zmeanprofile
LET_zdoseprofile2=LET_zdoseprofile2[::-1]
LET_zdoseprofile2[0]=LET_zdoseprofile2[1]


depth_sci= np.arange(0,len(zdoseprofile),1)*s
area_sim=np.trapz(zdoseprofile[3:len(zdoseprofile)-3],depth_sci[3:len(depth_sci)-3]) #eliminate last 3 points
area_LET=np.trapz(LET_zdoseprofile,depth_sci)
area_f_LET=np.trapz(LET_zfluenceprofile,depth_sci)


norm=1/area
norm1=1/area1
norm2=1/area2
norm_sim2=1/area_sim2
norm_sim=1/area_sim
norm_sim1=1/area_sim1


print(len(depth_sci))
print(len(LET_zfluenceprofile))
print(len(dose_sci))
"""Qyenching simulated curves  in scintillator """

dS=0 #theoretical values
dscintillator= 1.023 #[g/cm^3] scintillator density
dactivelayer=1.2 #[g/cm^3]
k=207.0/10000 #[g/MeV cm^2]
a=0.9 #scintillator efficiency
dx= 65 #µm scintillator spatial resolution
ddx=1 #spatial resolution error
k=k*10#[micrometers/kev]
#SIMULATED LET IN Scintillator starting from TOF of a scidom shot


def quenching(dose,s,dose_sci,depth_sci,LET_zfluenceprofile,k):

    dosequenched_sci=np.divide(dose_sci, 1+k*LET_zfluenceprofile, out=np.zeros_like(dose_sci),
                                            where= (1+k*LET_zfluenceprofile)!=0)
    dosequenched_sci=np.interp(np.arange(0,len(dose),1)*s,depth_sci,dosequenched_sci)
    norm_que= 1/ dosequenched_sci.max()
    return dosequenched_sci,norm_que

dosequenched_sci,norm_que=quenching(dose,s,dose_sci,depth_sci,LET_zfluenceprofile,k)
dosequenched_sci1,norm1_que=quenching(dose1,s,dose_sci1,depth_sci,LET_zfluenceprofile1,k)
dosequenched_sci2,norm2_que=quenching(dose2,s,dose_sci2,depth_sci,LET_zfluenceprofile2,k)
print(len(dosequenched_sci))


def plotfunction(depth,dose,err,norm,depth_sci,dosequenched_sci,norm_que,color1,color2,PC,s):
    plt.plot(np.arange(0,len(dose),1)*s,
                                        np.abs((dose*norm)-(dosequenched_sci*norm_que)),
                                                                           '.',
                                                                            markersize=12,
                                                                     color='red',
                                                    label='$|D_{MS}$ - $D_{Sim,quen}|$',
                                                    zorder=3)

    plt.errorbar(depth,                              dose*norm,
                                                                   yerr=err*norm,
                                                                       xerr=None,
                                                                         fmt='.',
                                                                         markersize=12,
                                                                   color=color1,
                                                                  ecolor=color1,
                                                                 elinewidth=None,
                                                        label='$D_{MS}$' ,
                                                        zorder=2)


    plt.fill_between(depth,
                                                             dose*norm-err*norm,
                                                           dose*norm + err*norm,
                                          color=sns.color_palette(  "Paired")[0],
                                                                      alpha=0.5,
                                                                      zorder=2)







    plt.plot(np.arange(0,len(dose),1)*s,
                                                      dosequenched_sci*norm_que,

                                                                        '.',
                                                                    color=color2,
                                                                    markersize=12,
                                                 label='$D_{Sim,quen}$',
                                                 zorder=1)




    return

plt.figure()
plotfunction(depth,dose2,err2,norm2,depth_sci,dosequenched_sci2,norm2_que,sns.color_palette("Paired")[3],sns.color_palette(  "Paired")[6],PC,s)


plt.title('Depth-dose distribution ',fontsize=26)
plt.xlabel(" Depth[mm] ",fontsize=24)
plt.ylabel('Relative Dose in water ',fontsize=24)
#plt.legend(title='E_{primary}=92.10 MeV',fontsize=24,loc=3,markerscale=2,title_fontsize=24)
plt.legend(title='7PC',fontsize=24,loc=1,markerscale=3,title_fontsize=20)
plt.grid(b=True, which='major', color='k', linestyle='-',alpha=0.2)
plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
plt.minorticks_on()
plt.tick_params(axis='x', which='major', labelsize=22)
plt.tick_params(axis='y', which='major', labelsize=22)
plt.show()
