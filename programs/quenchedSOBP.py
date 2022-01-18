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

directory='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2021-09-01/notnormalized/'
dose2,depth2,tof,shotnumber2=read_dose(directory,'notnormalizedmean_array29.npy',s)
#dose2=dose2-dose2.min()
dose,depth,tof,shotnumber=read_dose(directory,'notnormalizedmean_array34.npy',s)
dose1,depth1,tof,shotnumber1=read_dose(directory,'notnormalizedmean_array9.npy',s)
#dose= dose- dose.min()/100 #background subtaction

err=read_doserr(directory,'notnormalizederr34.npy')
err1=read_doserr(directory,'notnormalizederr9.npy')
err2=read_doserr(directory,'notnormalizederr29.npy') #standard deviation divided by radical n
area=np.trapz(dose[5:len(dose)-12], depth[5:len(depth)-12])
area1=np.trapz(dose1[1:len(dose1)-1], depth1[1:len(depth1)-1])
area2=np.trapz(dose2[1:len(dose2)-1], depth2[1:len(depth2)-1])

outputfile_topas='/Users/angelacorvino/Desktop/TOPASpreciousimulation/SOBP/11PC/152.5MeV107/secondcollimator/DoseToWater_152500KeVproton_PVT_11PC_1Dscorer.csv'
#outputfile_topas='/Users/angelacorvino/Desktop/GitHub/SimulationOncoray/data/SOBP/11PC/DoseToWater_153MeVproton_PVT_11PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
doseprofile1=Get_profile(topas_datamatrix, 149,1)
zdoseprofile1=doseprofile1.zmeanprofile
zdoseprofile1=zdoseprofile1[::-1]
dose_sci1=zdoseprofile1
(directory,energy1,scoringvolume1,PC1,dimension)= outputfile_topas.split('_')
(energy1,particle)=energy1.split('KeV')


#outputfile_topas = '/home/corvin22/Simulationhemera/data/SOBP/LET_doseweighted_14960KeVproton_PVT_11PC_1Dscorer.csv'
outputfile_topas='/Users/angelacorvino/Desktop/TOPASpreciousimulation/SOBP/11PC/152.5MeV107/secondcollimator/LET_doseweighted_152500KeVproton_PVT_11PC_1Dscorer.csv'
#outputfile_topas='/Users/angelacorvino/Desktop/GitHub/SimulationOncoray/data/SOBP/11PC/LET_doseweighted_153MeVproton_PVT_11PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix= np.array(df)# convert dataframe df to array
LET_doseprofile=Get_profile(topas_datamatrix, 149,1)
LET_zdoseprofile1=LET_doseprofile.zmeanprofile
LET_zdoseprofile1=LET_zdoseprofile1[::-1]
LET_zdoseprofile1[0]=LET_zdoseprofile1[1]

#outputfile_topas = '/home/corvin22/Simulationhemera/data/SOBP/LET_fluenceweighted_14960KeVproton_PVT_11PC_1Dscorer.csv'
outputfile_topas='/Users/angelacorvino/Desktop/TOPASpreciousimulation/SOBP/11PC/152.5MeV107/secondcollimator/LET_fluenceweighted_152500KeVproton_PVT_11PC_1Dscorer.csv'
#outputfile_topas='/Users/angelacorvino/Desktop/GitHub/SimulationOncoray/data/SOBP/11PC/LET_fluenceweighted_152500KeVproton_PVT_11PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix= np.array(df)# convert dataframe df to array
LET_fluenceprofile=Get_profile(topas_datamatrix, 149,1)
LET_zfluenceprofile1=LET_fluenceprofile.zmeanprofile
LET_zfluenceprofile1=LET_zfluenceprofile1[::-1]
LET_zfluenceprofile1[0]=LET_zfluenceprofile1[1]

depth_sci1= np.arange(0,len(zdoseprofile1),1)*s
area_sim1=np.trapz(zdoseprofile1[0:len(zdoseprofile1)-8],depth_sci1[0:len(depth_sci1)-8]) #eliminate last 3 points
area_f_LET1=np.trapz(LET_zfluenceprofile1,depth_sci1)


########################################################################################################################################
#outputfile_topas='/Users/angelacorvino/Desktop/GitHub/SimulationOncoray/data/SOBP/12PC/DoseToWater_152500KeVproton_PVT_12PC_1Dscorer.csv'
outputfile_topas='/Users/angelacorvino/Desktop/TOPASpreciousimulation/SOBP/12PC/152.5MeV/secondcollimator/DoseToWater_152500KeVproton_PVT_12PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
doseprofile2=Get_profile(topas_datamatrix, 149,1)
zdoseprofile2=doseprofile2.zmeanprofile
zdoseprofile2=zdoseprofile2[::-1]
dose_sci2=zdoseprofile2
(directory,energy2,scoringvolume2,PC2,dimension)= outputfile_topas.split('_')
(energy2,particle)=energy2.split('KeV')


#outputfile_topas = '/home/corvin22/Simulationhemera/data/SOBP/LET_doseweighted_14960KeVproton_PVT_11PC_1Dscorer.csv'
outputfile_topas='/Users/angelacorvino/Desktop/TOPASpreciousimulation/SOBP/12PC/152.5MeV/secondcollimator/LET_doseweighted_152500KeVproton_PVT_12PC_1Dscorer.csv'
#outputfile_topas='/Users/angelacorvino/Desktop/GitHub/SimulationOncoray/data/SOBP/12PC/LET_doseweighted_153MeVproton_PVT_12PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix= np.array(df)# convert dataframe df to array
LET_doseprofile=Get_profile(topas_datamatrix, 149,1)
LET_zdoseprofile2=LET_doseprofile.zmeanprofile
LET_zdoseprofile2=LET_zdoseprofile2[::-1]
LET_zdoseprofile2[0]=LET_zdoseprofile2[1]

#outputfile_topas = '/home/corvin22/Simulationhemera/data/SOBP/LET_fluenceweighted_14960KeVproton_PVT_11PC_1Dscorer.csv'
outputfile_topas='/Users/angelacorvino/Desktop/TOPASpreciousimulation/SOBP/12PC/152.5MeV/secondcollimator/LET_fluenceweighted_152500KeVproton_PVT_12PC_1Dscorer.csv'
#outputfile_topas='/Users/angelacorvino/Desktop/GitHub/SimulationOncoray/data/SOBP/12PC/LET_fluenceweighted_152500KeVproton_PVT_12PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix= np.array(df)# convert dataframe df to array
LET_fluenceprofile=Get_profile(topas_datamatrix, 149,1)
LET_zfluenceprofile2=LET_fluenceprofile.zmeanprofile
LET_zfluenceprofile2=LET_zfluenceprofile2[::-1]
LET_zfluenceprofile2[0]=LET_zfluenceprofile2[1]

depth_sci2= np.arange(0,len(zdoseprofile2),1)*s
area_sim2=np.trapz(zdoseprofile2[0:len(zdoseprofile2)-8],depth_sci2[0:len(depth_sci2)-8]) #eliminate last 3 points
area_f_LET2=np.trapz(LET_zfluenceprofile2,depth_sci2)



#######################################################################################################################################


outputfile_topas='/Users/angelacorvino/Desktop/TOPASpreciousimulation/SOBP/10PC/152.5MeV/DoseToWater_152500KeVproton_PVT_10PC_1Dscorer.csv'
#outputfile_topas='/Users/angelacorvino/Desktop/GitHub/SimulationOncoray/data/SOBP/10PC/DoseToWater_152500KeVproton_PVT_10PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
doseprofile=Get_profile(topas_datamatrix, 149,1)
zdoseprofile=doseprofile.zmeanprofile
zdoseprofile=zdoseprofile[::-1]
dose_sci=zdoseprofile
(directory,energy,scoringvolume,PC,dimension)= outputfile_topas.split('_')
(energy,particle)=energy.split('KeV')
depth_sci= np.arange(0,len(zdoseprofile),1)*s
area_sim=np.trapz(zdoseprofile[3:len(zdoseprofile)-8],depth_sci[3:len(depth_sci)-8])



outputfile_topas='/Users/angelacorvino/Desktop/TOPASpreciousimulation/SOBP/10PC/152.5MeV/LET_fluenceweighted_152500KeVproton_PVT_10PC_1Dscorer.csv'
#outputfile_topas='/Users/angelacorvino/Desktop/GitHub/SimulationOncoray/data/SOBP/10PC/LET_fluenceweighted_152500KeVproton_PVT_10PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix= np.array(df)# convert dataframe df to array
LET_fluenceprofile=Get_profile(topas_datamatrix, 149,1)
LET_zfluenceprofile=LET_fluenceprofile.zmeanprofile
LET_zfluenceprofile=LET_zfluenceprofile[::-1]
LET_zfluenceprofile[0]=LET_zfluenceprofile[2]


area_f_LET=np.trapz(LET_zfluenceprofile,depth_sci)



outputfile_topas='/Users/angelacorvino/Desktop/TOPASpreciousimulation/SOBP/10PC/152.5MeV/LET_doseweighted_152500KeVproton_PVT_10PC_1Dscorer.csv'
#outputfile_topas='/Users/angelacorvino/Desktop/GitHub/SimulationOncoray/data/SOBP/10PC/LET_doseweighted_153MeVproton_PVT_10PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix= np.array(df)# convert dataframe df to array
LET_doseprofile=Get_profile(topas_datamatrix, 149,1)
LET_zdoseprofile=LET_doseprofile.zmeanprofile
LET_zdoseprofile=LET_zdoseprofile[::-1]
LET_zdoseprofile[0]=LET_zdoseprofile[1]



################################################################################

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
    x=np.arange(0,len(dose),1)*s
    dosequenched_sci=np.divide(dose_sci, 1+k*LET_zfluenceprofile, out=np.zeros_like(dose_sci),
                                            where= (1+k*LET_zfluenceprofile)!=0)
    dosequenched_sci=np.interp(np.arange(0,len(dose),1)*s,depth_sci,dosequenched_sci)
    area_que=np.trapz(dosequenched_sci[1:len(dosequenched_sci)-1], x[1:len(x)-1])
    norm_que= 1/ area_que
    return dosequenched_sci,norm_que

dosequenched_sci,norm_que=quenching(dose,s,dose_sci,depth_sci,LET_zfluenceprofile,k)
dosequenched_sci1,norm1_que=quenching(dose1,s,dose_sci1,depth_sci,LET_zfluenceprofile1,k)
dosequenched_sci2,norm2_que=quenching(dose2,s,dose_sci2,depth_sci,LET_zfluenceprofile2,k)
print(len(dosequenched_sci))


def plotfunction(depth,dose,err,norm,depth_sci,dosequenched_sci,norm_que,color1,color2,PC,s):
    fig, ax = plt.subplots()
    twin1 = ax.twinx()
    plt.rc('text', usetex=True)
    twin1.plot(np.arange(0,len(dose),1)*s,
                                        (np.abs((dose*norm)-(dosequenched_sci*norm_que))/(dose*norm))*100,
                                                                           '.',
                                                                            markersize=12,
                                                                     color='red',
                                                    label=r'$| \frac{D_{MS} - D_{Sim,quen}}{D_{MS}}|$',
                                                    zorder=3)

    ax.errorbar(depth,                              dose*norm,
                                                                   yerr=err*norm,
                                                                       xerr=None,
                                                                         fmt='.',
                                                                         markersize=12,
                                                                   color=color1,
                                                                  ecolor=color1,
                                                                 elinewidth=None,
                                                        label='$D_{MS}$' ,
                                                        zorder=2)


    ax.fill_between(depth,
                                                             dose*norm-err*norm,
                                                           dose*norm + err*norm,
                                          color=sns.color_palette(  "Paired")[0],
                                                                      alpha=0.5,
                                                                      zorder=2)







    ax.plot(np.arange(0,len(dose),1)*s,
                                                      dosequenched_sci*norm_que,

                                                                        '.',
                                                                    color=color2,
                                                                    markersize=12,
                                                 label='$D_{Sim,quen}$',
                                                 zorder=1)
    twin1.set_ylabel("Normalized background [%]",color='red',fontsize=22)
    twin1.tick_params(axis='x', which='major', labelsize=22)
    twin1.tick_params(axis='y', which='major', color='red',labelsize=22)
    ax.set_ylabel(" Relative Dose in water ",color="green",fontsize=22)
    ax.set_xlabel(" Depth[mm] ",color="black",fontsize=22)
    ax.legend( title='12 PC',fontsize=22,loc=1,markerscale=3,title_fontsize=24)
    twin1.legend( title='',fontsize=24,loc=2,markerscale=3,title_fontsize=24)
    twin1.grid(b=True,color='k',linestyle='-',alpha=0.2)
    ax.grid(b=True, which='major', color='k', linestyle='-',alpha=0.2)
    ax.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
    ax.tick_params(axis='x', which='major', labelsize=22)
    ax.tick_params(axis='y', which='major',  color='green',labelsize=22)
    twin1.tick_params(axis='y', which='major',colors='red', labelsize=22)
    ax.tick_params(axis='y', which='major',colors='green', labelsize=22)

    return


plotfunction(depth,dose2,err2,norm2,depth_sci,dosequenched_sci2,norm2_que,sns.color_palette("Paired")[3],sns.color_palette(  "Paired")[6],PC,s)

plt.title('Depth-dose distribution ',fontsize=26)


plt.show()
