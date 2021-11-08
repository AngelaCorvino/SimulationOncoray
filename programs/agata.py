'''
COMPARISON BETWEEN TOF SIMULATED DATA IN SCINTILLATOR  AND SCINTILLATOR MEASUREMENTS
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

#RCF data

directory=''
filename='RCF21CW.csv'
rcfdepth,rcfdose,rcferr,area_rcf,rcfname=read_datarcf(directory,filename)

#September 2021
m=0.073825 #[mm/pixel ]
directory=''
dose,depth,tof,shotnumber=read_dose(directory,'notnormalizedmean_array1.npy',m)

#dose= dose- dose.min()/100 #background subtaction

err=read_doserr(directory,'notnormalizederr1.npy') #standard deviation divided by radical n
area=np.trapz(dose[3:len(dose)], depth[3:len(depth)])

#Simulation
outputfile_topas = 'DoseToWater_9160KeVproton_PVT_6PC1PMMA_1Dscorer.csv'
#outputfile_topas = '/home/corvin22/Simulationhemera/data/SOBP/DoseToWater_14960KeVproton_PVT_12PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
doseprofile=Get_profile(topas_datamatrix, 149,1)
zdoseprofile=doseprofile.zmeanprofile
zdoseprofile=zdoseprofile[::-1]
dose_sci=zdoseprofile

(directory,energy,scoringvolume,PC,dimension)= outputfile_topas.split('_')
(energy,particle)=energy.split('KeV')

outputfile_topas = 'LET_fluenceweighted_9160KeVproton_PVT_6PC1PMMA_1Dscorer.csv'
#outputfile_topas = '/home/corvin22/Simulationhemera/data/SOBP/LET_doseweighted_14960KeVproton_PVT_12PC.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix= np.array(df)# convert dataframe df to array
LET_doseprofile=Get_profile(topas_datamatrix, 149,1)
LET_zdoseprofile=LET_doseprofile.zmeanprofile
LET_zdoseprofile=LET_zdoseprofile[::-1]
LET_zdoseprofile[0]=LET_zdoseprofile[1]

#outputfile_topas = '/home/corvin22/Simulationhemera/data/SOBP/LET_fluenceweighted_14960KeVproton_PVT_12PC.csv'
outputfile_topas = 'LET_fluenceweighted_9160KeVproton_PVT_6PC1PMMA_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix= np.array(df)# convert dataframe df to array
LET_fluenceprofile=Get_profile(topas_datamatrix, 149,1)
LET_zfluenceprofile=LET_fluenceprofile.zmeanprofile
LET_zfluenceprofile=LET_zfluenceprofile[::-1]
LET_zfluenceprofile[0]=LET_zfluenceprofile[1]


depth_sci= np.arange(0,len(zdoseprofile),1)*m
area_sim=np.trapz(zdoseprofile[3:len(zdoseprofile)-6],depth_sci[3:len(depth_sci)-6]) #eliminate last 3 points
area_LET=np.trapz(LET_zdoseprofile,depth_sci)
area_f_LET=np.trapz(LET_zfluenceprofile,depth_sci)


normrcf=1/area_rcf
norm=1/area
norm_sim=1/area_sim



""" correction of RCf measure with LET (doseweighted ) simulations in scintillator"""

a=-0.0251
b= 1.02


ys=LET_zdoseprofile
S=np.interp(np.arange(0,len(rcfdose),1)*m,depth_sci,ys)

D_rcf=rcfdosecorrection(rcfdose,S,a,b)
Derr_rcf=rcfdosecorrection(rcferr,S,a,b)
area_rcfcorrected=np.trapz(rcfdosecorrection(rcfdose,S,a,b),rcfdepth)
####################################################################################
normrcf=1
normrcfcorrecrted=1



"""LET correction simulations in scintillator """

dS=0 #theoretical values
dscintillator= 1.023 #[g/cm^3] scintillator density
dactivelayer=1.2 #[g/cm^3]
k=207/10000 #[g/MeV cm^2]
a=0.9 #scintillator efficiency
dx= 65 #µm scintillator spatial resolution
ddx=1 #spatial resolution error
k=k/dscintillator*10#[micrometers/kev]
#SIMULATED LET IN Scintillator starting from TOF of a scidom shot


ys_ana_mini=LET_zdoseprofile
ys_fluence_mini=LET_zfluenceprofile
S_a_mini=np.interp(np.arange(0,len(dose),1)*m,depth_sci,ys_ana_mini)
#S_a_low_mini=np.interp(np.arange(0,len(dose),1)*m,depth_sci,ys_ana_lower_mini)
#S_a_up_mini=np.interp(np.arange(0,len(dose),1)*m,depth_sci,ys_ana_upper_mini)

S_fluence_mini=np.interp(np.arange(0,len(dose),1)*m,depth_sci,ys_fluence_mini)



#NORMALIZATION
area_corrected=np.trapz(dosecorrection(dose,S_a_mini,a,k,dx)[3:len(dosecorrection(dose,S_a_mini,a,k,dx))],
                                                                    depth[3:len(depth)]) #eliminate first 4 points
area_f_corrected=np.trapz(dosecorrection(dose,S_fluence_mini,a,k,dx)[3:len(dosecorrection(dose,S_fluence_mini,a,k,dx))],
                                                                    depth[3:len(depth)]) #eliminate first 4 points
norm_dose_mini=area_rcfcorrected/area_corrected
norm_fluence_mini=area_rcfcorrected/area_f_corrected
#CORRECTED DOSE

D_a_mini=dosecorrection(dose,S_a_mini,a,k,dx)

#D_a_up_mini=dosecorrection(dose,S_a_up_mini,a,k,dx)*norm_a_mini
#D_a_low_mini=dosecorrection(dose,S_a_low_mini,a,k,dx)*norm_a_mini
D_fluence_mini=dosecorrection(dose,S_fluence_mini,a,k,dx)
Dabsolute_a_mini=dosecorrection(dose,S_a_mini,a,k,dx)*norm_dose_mini* lightcorrection(S_a_mini,a,k,dx)
#L_a_up_mini=lightcorrection(S_a_up_mini,a,k,dx)
#L_a_low_mini=lightcorrection(S_a_low_mini,a,k,dx)

Dabsolute_fluence_mini=dosecorrection(dose,S_fluence_mini,a,k,dx)*norm_dose_mini* lightcorrection(S_fluence_mini,a,k,dx)

###############################################################################



fig, ax = plt.subplots()
plt.title('Shape comparison {} '.format(rcfname),fontsize=22)


ax2 = ax.twinx()



ax.errorbar( rcfdepth,                                          rcfdose/area_rcf,
                                                             yerr=rcferr/area_rcf,
                                                                      xerr=None,
                                                                        fmt='.',
                                            color=sns.color_palette(  "Paired")[1],
                                                                    markersize=8,
                                         ecolor=sns.color_palette(  "Paired")[1],
                                                                elinewidth=None,
                                                label='{} measured dose'.format(rcfname))

ax.fill_between(rcfdepth,
                                                                rcfdose/area_rcf-rcferr/area_rcf,
                                                                rcfdose/area_rcf+rcferr/area_rcf,
                                                        color=sns.color_palette(  "Paired")[1],
                                                         alpha=0.3)


ax.errorbar( rcfdepth,                                  D_rcf/area_rcfcorrected ,
                                                 yerr=Derr_rcf/area_rcfcorrected,
                                                                      xerr=None,
                                                                        fmt='.',
                                          color=sns.color_palette(  "Paired")[5],
                                                                   markersize=8,
                                        ecolor=sns.color_palette(  "Paired")[5],
                                                                 elinewidth=None,
                              label='LET(dose weighted) corrected {} dose'.format(rcfname))


ax.fill_between(rcfdepth,
                              D_rcf/area_rcfcorrected-Derr_rcf/area_rcfcorrected,
                              D_rcf/area_rcfcorrected+Derr_rcf/area_rcfcorrected,
                                          color=sns.color_palette( "Paired")[5],
                                                                       alpha=0.1)

sub_axes = plt.axes([.6, .6, .25, .25])

# plot the zoomed portion





sub_axes.errorbar( rcfdepth[5:10],                                          (rcfdose/area_rcf)[5:10],
                                                             yerr=(rcferr/area_rcf)[5:10],
                                                                      xerr=None,
                                                                        fmt='.',
                                            color=sns.color_palette(  "Paired")[1],
                                                                    markersize=8,
                                         ecolor='lightblue',

                                                                elinewidth=None)




sub_axes.errorbar( rcfdepth[5:10],                                  (D_rcf/area_rcfcorrected)[5:10] ,
                                                 yerr=(Derr_rcf/area_rcfcorrected)[5:10],
                                                                      xerr=None,
                                                                        fmt='.',
                                          color=sns.color_palette(  "Paired")[5],
                                                                   markersize=8,
                                        ecolor='lightcoral',

                                                                 elinewidth=None)
sub_axes.grid(b=True,color='k',linestyle='-',alpha=0.2)










ax2.plot(depth_sci[0:110],LET_zdoseprofile[0:110],
                                                                            '.',
                                         color=sns.color_palette(  "Paired")[2],
                                                                  Markersize=10,
                                                                   label='LET ')



ax.set_xlabel(" Depth [mm] ",color="black",fontsize=16)
ax2.set_ylabel("LET [KeV/um]",color=sns.color_palette(  "Paired")[5],fontsize=16)
ax.set_ylabel(" Relative Dose ",color="Green",fontsize=16)
ax.legend( title='',fontsize=20,loc=2)
ax2.legend( title='',fontsize=20,loc=5)

ax2.grid(b=True,color='k',linestyle='-',alpha=0.2)
ax.grid(b=True,color='k',linestyle='dotted',alpha=0.2)

ax.tick_params(axis='x', which='major', labelsize=16)
ax2.tick_params(axis='y', which='major',colors=sns.color_palette(  "Paired")[5],
                                                                   labelsize=16)
ax.tick_params(axis='y', which='major',colors='green', labelsize=16)

mark_inset(ax, sub_axes, loc1=2, loc2=3, fc="none", ec="0.5")


plt.show()