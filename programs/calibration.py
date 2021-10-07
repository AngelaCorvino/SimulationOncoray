'''
COMPARISON BETWEEN TOF SIMULATED DATA IN SCINTILLATOR  AND SCINTILLATOR MEASUREMENTS
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from scipy import optimize
from scipy.optimize import curve_fit
import csv
import pandas as pd
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
###############################################################################

#RCF data

directory='/home/corvin22/Desktop/miniscidom/pictures/2021-09-02/'
filename=directory+'RCF21CX.csv'
depth,rcfdose=read_datarcf(filename)
rcferr=rcfdose*(546/10000)
area_rcf=np.trapz(rcfdose[0:len(rcfdose)], depth[0:len(depth)])



#September 2021

directory='/home/corvin22/Desktop/miniscidom/pictures/2021-09-02/notnormalized/'
data= np.load(directory+'notnormalizedmean_array17.npy')
dose=data[0:len(data)-3]
tof=data[len(data)-3:len(data)]
data0= np.load(directory+'notnormalizederr17.npy')
err=data0[0:len(data0)]  #standard deviation divided by radical n


x= np.arange(0,len(dose),1)*0.0738255
area=np.trapz(dose, x)


#Simulation
outputfile_topas = '/home/corvin22/Simulationhemera/data/Single/DoseToWater_90MeVproton_PVT_6PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
doseprofile=Get_profile(topas_datamatrix, 149,1)
zdoseprofile=doseprofile.zmeanprofile
zdoseprofile=zdoseprofile[::-1]

dose_sci=zdoseprofile

outputfile_topas = '/home/corvin22/Simulationhemera/data/Single/LET_doseweighted_90MeVproton_PVT_6PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix= np.array(df)# convert dataframe df to array
LET_doseprofile=Get_profile(topas_datamatrix, 149,1)
LET_zdoseprofile=LET_doseprofile.zmeanprofile
LET_zdoseprofile=LET_zdoseprofile[::-1]

outputfile_topas = '/home/corvin22/Simulationhemera/data/Single/LET_fluenceweighted_90MeVproton_PVT_6PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix= np.array(df)# convert dataframe df to array
LET_fluenceprofile=Get_profile(topas_datamatrix, 149,1)
LET_zfluenceprofile=LET_fluenceprofile.zmeanprofile
LET_zfluenceprofile=LET_zfluenceprofile[::-1]



depth_sci= np.arange(0,len(zdoseprofile),1)*0.0738255
area_sci=np.trapz(zdoseprofile[1:len(zdoseprofile)],depth_sci[1:len(depth_sci)]) #eliminate last 3 points
area_LET=np.trapz(LET_zdoseprofile[1:len(LET_zdoseprofile)],depth_sci[1:len(depth_sci)])

area_f_LET=np.trapz(LET_zfluenceprofile[1:len(LET_zfluenceprofile)],depth_sci[1:len(depth_sci)])


normrcf=1
norm=area_rcf/area
norm_sim=area_rcf/area_sci




#PLOT RAW DATA

plt.figure(1)

plt.errorbar( depth,                      rcfdose/area_rcf,
                                                                  yerr=rcferr/area_rcf,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                        color='red',
                                                                        markersize=8,
                                                                   ecolor='red',
                                                                elinewidth=None,
                                                                label=' RCF 21CX measured dose')

plt.fill_between(depth,
                                                                rcfdose/area_rcf-rcferr/area_rcf,
                                                                rcfdose/area_rcf+rcferr/area_rcf,
                                                        color='red', alpha=0.1)
plt.errorbar(  np.arange(0,len(dose),1)*0.0738255,                      dose/area,
                                                                  yerr=err/area,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                   ecolor='gray',
                                                                elinewidth=None,
                                                         label=' reconstructed')

plt.fill_between(np.arange(0,len(dose),1)*0.0738255,
                                                             dose/area-err/area,
                                                           dose/area + err/area,
                                                        color='gray', alpha=0.5)


plt.grid(True)

plt.plot( np.arange(0,149,1)*0.0738255,zdoseprofile/area_sci,
                                                                            'g.',
                                                                  Markersize=11,
                                        label='Dose simulated in scintillator ')

"""
plt.fill_between(depth_sci,
                                                             dose_sci_lower*norm_sim,
                                                            dose_sci_upper*norm_sim,
                                                        color='lightgreen', alpha=0.5)
"""

plt.legend( title='Unidose',fontsize='large')
plt.title('Depth-dose distribution 6PC  shot n17',
                                                                  fontdict=None,
                                                                  loc='center',
                                                                       pad=None)
plt.xlabel('Depth[mm]')
plt.ylabel('Dose[Gy]')





"LET correction simulations in scintillator"

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
print(ys_ana_mini)
S_a_mini=np.interp(np.arange(0,len(dose),1)*0.0738255,depth_sci,ys_ana_mini)
#S_a_low_mini=np.interp(np.arange(0,len(dose),1)*0.0738255,depth_sci,ys_ana_lower_mini)
#S_a_up_mini=np.interp(np.arange(0,len(dose),1)*0.0738255,depth_sci,ys_ana_upper_mini)

S_fluence_mini=np.interp(np.arange(0,len(dose),1)*0.0738255,depth_sci,ys_fluence_mini)



#NORMALIZATION
area_corrected=np.trapz(dosecorrection(dose,S_a_mini,a,k,dx)[3:len(dosecorrection(dose,S_a_mini,a,k,dx))],
                                                                    x[3:len(x)]) #eliminate first 4 points
area_f_corrected=np.trapz(dosecorrection(dose,S_fluence_mini,a,k,dx)[3:len(dosecorrection(dose,S_fluence_mini,a,k,dx))],
                                                                    x[3:len(x)]) #eliminate first 4 points
norm_a_mini=1
norm_fluence_mini=1
#CORRECTED DOSE



D_a_mini=dosecorrection(dose,S_a_mini,a,k,dx)*norm_a_mini
print(D_a_mini)
#D_a_up_mini=dosecorrection(dose,S_a_up_mini,a,k,dx)*norm_a_mini
#D_a_low_mini=dosecorrection(dose,S_a_low_mini,a,k,dx)*norm_a_mini
D_fluence_mini=dosecorrection(dose,S_fluence_mini,a,k,dx)*norm_fluence_mini
Dabsolute_a_mini=dosecorrection(dose,S_a_mini,a,k,dx)*norm_a_mini* lightcorrection(S_a_mini,a,k,dx)
#L_a_up_mini=lightcorrection(S_a_up_mini,a,k,dx)
#L_a_low_mini=lightcorrection(S_a_low_mini,a,k,dx)

Dabsolute_fluence_mini=dosecorrection(dose,S_fluence_mini,a,k,dx)*norm_a_mini* lightcorrection(S_fluence_mini,a,k,dx)

###############################################################################
#PLOT
plt.figure(2)


plt.errorbar( depth,                      rcfdose/area_rcf ,
                                                                  yerr=rcferr/area_rcf,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                        color='red',
                                                                        markersize=8,
                                                                   ecolor='red',
                                                                elinewidth=None,
                                                                label=' RCF 21CX measured dose')

plt.fill_between(depth,
                                                                rcfdose/area_rcf-rcferr/area_rcf,
                                                                rcfdose/area_rcf+rcferr/area_rcf,
                                                        color='red', alpha=0.1)






plt.errorbar(np.arange(0,len(dose),1)*0.0738255,                      dose/area ,
                                                                  yerr=err/area,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                  ecolor='gray',
                                                                elinewidth=None,
                                                label='Measured Dose Miniscidom ')

plt.plot(x[1:len(x)],
                                                        D_a_mini[1:len(D_a_mini)]/area_corrected,
                                                                            '.',
                                                                 color='orange',
                                            label='LET(dose weighted) corrected dose',
                                                                  Markersize=11)

plt.plot(x[1:len(x)],
                                                        D_fluence_mini[1:len(D_fluence_mini)]/area_f_corrected,
                                                                            '.',
                                                                 color='purple',
                                            label='LET(fluence weighted) corrected dose',
                                                                  Markersize=11)


plt.plot(depth_sci,dose_sci/area_sci,
                                                                            'g.',
                                                                  Markersize=11,
                                                    label='Simulated Dose in scintillator ')
plt.plot(depth_sci[1:len(depth_sci)],LET_zdoseprofile[1:len(LET_zdoseprofile)]/area_LET,
                                                                            'y.',
                                                                  Markersize=8,
                                                                   label='LET ')
plt.plot(depth_sci[1:len(depth_sci)],LET_zfluenceprofile[1:len(LET_zfluenceprofile)]/area_f_LET,
                                                                            'b.',
                                                                  Markersize=8,
                                                                   label='LET ')


plt.legend( title='',fontsize='large')

plt.grid(b=True, which='major', color='k', linestyle='-',alpha=0.2)
plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
plt.minorticks_on()
plt.tick_params(axis='x', which='major', labelsize=18)
plt.tick_params(axis='y', which='major', labelsize=18)
plt.ylabel('Relative Dose ',fontsize=16)
plt.xlabel('Depth in PVT[mm]',fontsize=16)
plt.title('Dose shape comparison 6PC',fontsize=18)





plt.show()
