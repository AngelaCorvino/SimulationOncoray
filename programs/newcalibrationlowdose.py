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

###############################################################################

#DATA

directory='/home/corvin22/Desktop/miniscidom/pictures/2021-09-02/'
filename=directory+'RCF21CX.csv'
depth,rcfdose=read_datarcf(filename)
rcferr=rcfdose*(546/10000)


area_rcf=np.trapz(rcfdose[0:len(rcfdose)], depth[0:len(depth)])



#low dose september 2021



directory='/home/corvin22/Desktop/miniscidom/pictures/2021-09-02/notnormalized/'

data= np.load(directory+'notnormalizedmean_array17.npy')
dose=data[0:len(data)-3]

#tof1=data1[len(data1)-3:len(data1)]
unidose=0
data0= np.load(directory+'notnormalizederr17.npy')
err=data0[0:len(data0)]
#top_projection_dose= np.load(directory+'1Dtopprojection/'+'top1Dmean_array1.npy')


x= np.arange(0,len(dose),1)*0.0738255
area=np.trapz(dose, x)


#Simulation
outputfile_topas = '/home/corvin22/Simulationhemera/data/Single/DoseToWater_90MeVproton_PVT_6PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix10 = np.array(df)# convert dataframe df to array
doseprofile=Get_profile(topas_datamatrix, 149,1)
zdoseprofile=doseprofile.zmeanprofile
zdoseprofile=zdoseprofile[::-1]



outputfile_topas = '/home/corvin22/Simulationhemera/data/Single/LET_doseweighted_90MeVproton_PVT_6PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix10 = np.array(df)# convert dataframe df to array
LET_doseprofile=Get_profile(topas_datamatrix, 149,1)
LET_zdoseprofile=LET_doseprofile.zmeanprofile
LET_zdoseprofile=LET_zdoseprofile[::-1]



area_sci=np.trapz(zdoseprofile,x) #eliminate last 3 points




norm=area_sci/area
norm_sim=1








###############################################################################
#PLOT RAW DATA

plt.figure(1)


plt.errorbar(  np.arange(0,len(dose),1)*0.0738255,                      dose*norm ,
                                                                  yerr=err*norm,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                   ecolor='gray',
                                                                elinewidth=None,
                                     label=' reconstructed')

plt.fill_between(np.arange(0,len(dose),1)*0.0634,
                                                             dose*norm-err*norm,
                                                           dose*norm + err*norm,
                                                        color='gray', alpha=0.5)


plt.grid(True)
plt.figure(1) # creates a figure in which we plot


plt.plot( np.arange(0,149,1)*0.0738255,zdoseprofile*norm_sim,
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



S_a_mini=np.interp(np.arange(0,len(dose),1)*0.0634,depth_sci,ys_ana_mini)
S_a_low_mini=np.interp(np.arange(0,len(dose),1)*0.0634,depth_sci,ys_ana_lower_mini)
S_a_up_mini=np.interp(np.arange(0,len(dose),1)*0.0634,depth_sci,ys_ana_upper_mini)

#NORMALIZATION
area_corrected=np.trapz(dosecorrection(dose,S_a_mini,a,k,dx)[3:len(dosecorrection(dose,S_a_mini,a,k,dx))],
                                                                    x[3:len(x)]) #eliminate first 4 points

norm_a_mini=1

#CORRECTED DOSE



D_a_mini=dosecorrection(dose,S_a_mini,a,k,dx)*norm_a_mini
D_a_up_mini=dosecorrection(dose,S_a_up_mini,a,k,dx)*norm_a_mini
D_a_low_mini=dosecorrection(dose,S_a_low_mini,a,k,dx)*norm_a_mini

Dabsolute_a_mini=dosecorrection(dose,S_a_mini,a,k,dx)*norm_a_mini* lightcorrection(S_a_mini,a,k,dx)
#L_a_up_mini=lightcorrection(S_a_up_mini,a,k,dx)
#L_a_low_mini=lightcorrection(S_a_low_mini,a,k,dx)



###############################################################################
#PLOT
plt.figure(2)
#plt.plot( np.arange(0,len(dose),1)*0.0634*1.03,top_projection_dose*norm_i,'.'
#                                          ,Markersize=11,label='Top projection')
plt.errorbar(  np.arange(0,len(dose),1)*0.0634,                      dose/area ,
                                                                  yerr=err/area,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                  ecolor='gray',
                                                                elinewidth=None,
                                                label='Measured Dose Miniscidom '
                                                                ,Markersize=11)

plt.plot(  np.arange(0,len(dose),1)*0.0634,
                                                                        D_a_mini/area_corrected,
                                                                            '.',
                                                                 color='orange',
              label='LET corrected dose Miniscidom',
                                                                  Markersize=11)

plt.fill_between( np.arange(0,len(dose),1)*0.0634,
                                                                        D_a_mini/area_corrected,
                                                                     D_a_up_mini/area_corrected,
                                                                   color='orange',
                                                                      alpha=0.3)

plt.fill_between( np.arange(0,len(dose),1)*0.0634,
                                                                    D_a_low_mini/area_corrected,
                                                                        D_a_mini/area_corrected,
                                                                 color='orange',
                                                                      alpha=0.3)


plt.plot( depth_sci,dose_sci/area_sci,
                                                                            'g.',
                                                                  Markersize=11,
                                                    label='Simulated Dose TOF ')
plt.fill_between(depth_sci,
                                                        dose_sci_lower/area_sci,
                                                        dose_sci_upper/area_sci,
                                                  color='lightgreen', alpha=0.2)




plt.legend( title='',fontsize='large')

plt.grid(b=True, which='major', color='k', linestyle='-',alpha=0.2)
plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
plt.minorticks_on()
plt.tick_params(axis='x', which='major', labelsize=18)
plt.tick_params(axis='y', which='major', labelsize=18)
plt.ylabel('Dose [Gy]',fontsize=16)
plt.xlabel('Depth in PVT[mm]',fontsize=16)
plt.title('Dose shape comparison  11-05 low dose shot n45 ',fontsize=18)





#PLOT
norm_a_mini=area_sci/area_corrected #area normalization

D_a_mini=dosecorrection(dose,S_a_mini,a,k,dx)*norm_a_mini
D_a_up_mini=dosecorrection(dose,S_a_up_mini,a,k,dx)*norm_a_mini
D_a_low_mini=dosecorrection(dose,S_a_low_mini,a,k,dx)*norm_a_mini

Dabsolute_a_mini=dosecorrection(dose,S_a_mini,a,k,dx)*norm_a_mini* lightcorrection(S_a_mini,a,k,dx)


plt.figure(3)

plt.plot( depth_sci,dose_sci*norm_sim,
                                                                            'g.',
                                                                  Markersize=11,
                                                    label='Simulated Dose TOF ')
plt.fill_between(depth_sci,
                                                        dose_sci_lower*norm_sim,
                                                        dose_sci_upper*norm_sim,
                                                  color='lightgreen', alpha=0.2)



plt.plot(  np.arange(0,len(dose),1)*0.0634,
                                                                        D_a_mini,
                                                                            '.',
                                                                 color='orange',
              label='LET corrected dose Miniscidom',
                                                                  Markersize=11)

plt.fill_between( np.arange(0,len(dose),1)*0.0634,
                                                                        D_a_mini,
                                                                     D_a_up_mini,
                                                                 color='orange',
                                                                      alpha=0.3)

plt.fill_between( np.arange(0,len(dose),1)*0.0634,
                                                                    D_a_low_mini,
                                                                       D_a_mini,
                                                                 color='orange',
                                                                      alpha=0.3)
plt.plot(  np.arange(0,len(dose),1)*0.0634,
                                                                Dabsolute_a_mini,
                                                                            '.',
                                                                   color='blue',
                                                       label='Dose Miniscidom ',
                                                                Markersize=11)

#plt.fill_between( np.arange(0,len(dose),1)*0.0634,
#                                                                        L_a_mini,
#                                                                    L_a_up_mini,
#                                                                    color='blue',
#                                                                      alpha=0.3)

#plt.fill_between( np.arange(0,len(dose),1)*0.0634,
#                                                                    L_a_low_mini,
#                                                                        L_a_mini,
#                                                                    color='blue',
#                                                                      alpha=0.3)

plt.legend(title='',fontsize='large')

plt.grid(b=True, which='major', color='k', linestyle='-',alpha=0.2)
plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
plt.minorticks_on()
plt.tick_params(axis='x', which='major', labelsize=18)
plt.tick_params(axis='y', which='major', labelsize=18)
plt.ylabel('Dose [Gy]',fontsize=16)
plt.xlabel('Depth in PVT[mm]',fontsize=16)
plt.title('Absolute dose comparison  11-05 low dose shot n45 ',fontsize=18)


plt.show()

"PIXEL MEAN VALUE "


P_a_mini=dosecorrection(dose,S_a_mini,a,k,dx)      #mean pixel value corrected withLET value
P_a_up_mini=dosecorrection(dose,S_a_up_mini,a,k,dx)
P_a_low_mini=dosecorrection(dose,S_a_low_mini,a,k,dx)


"CALIBRATION FACTOR "
#Step 1 : interpolate dosce_sci in order to divide it by P_a_mini
dose_sci_interp=np.interp(np.arange(0,len(dose),1)*0.0634,depth_sci,dose_sci)
#Step 2: divide
c=dose_sci_interp/P_a_mini         #this is an arry (simulate dose in scintillatore/ mean pixel value corrected withLET value )
#Step 3 : average the vector
c_mean=np.mean(c)

print(c_mean)
###############################################################################
#PLOT

# create figure and axis objects with subplots()





fig, ax = plt.subplots()
ax2 = ax.twinx()



ax.errorbar(  np.arange(0,len(dose),1)*0.0634,                           dose ,
                                                                       yerr=err,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                   ecolor='gray',
                                                                elinewidth=None,
                                       label='Reconstruction mean pixel value  ')

ax.plot(  np.arange(0,len(dose),1)*0.0634,
                                                                        P_a_mini,
                                                                            '.',
                                                                 color='orange',
                              label='LET corrected mean pixel value Miniscidom',
                                                                  Markersize=11)

ax.fill_between( np.arange(0,len(dose),1)*0.0634,
                                                                        P_a_mini,
                                                                     P_a_up_mini,
                                                                 color='orange',
                                                                      alpha=0.3)

ax.fill_between( np.arange(0,len(dose),1)*0.0634,
                                                                    P_a_low_mini,
                                                                        P_a_mini,
                                                                 color='orange',
                                                                      alpha=0.3)


ax2.plot( depth_sci,dose_sci,
                                                                            'g.',
                                                                  Markersize=11,
                                                    label='Simulated Dose TOF ')
ax2.fill_between(depth_sci,
                                                        dose_sci_lower*norm_sim,
                                                        dose_sci_upper*norm_sim,
                                                color='lightgreen', alpha=0.2)






#ax2.set_ylim([0,dose_sci.max()])
#ax.set_ylim([0,P_a_mini.max()])
ax.set_ylabel("Mean Pixel  [a.u.]",color="blue",fontsize=16)
ax2.set_ylabel("Dose [Gy]",color="Green",fontsize=16)
plt.title('Mean pixel value  11-05 low dose shot n45',fontsize=18)

ax.legend( title='',fontsize='large',loc=2)
ax2.legend( title='',fontsize='large',loc=1)

ax2.grid(b=True,color='k',linestyle='-',alpha=0.2)
ax.grid(b=True,color='k',linestyle='dotted',alpha=0.2)

ax.tick_params(axis='x', which='major', labelsize=16)
ax.tick_params(axis='y', which='major',colors='blue', labelsize=16)
ax2.tick_params(axis='y', which='major',colors='green', labelsize=16)
"""

"TOF SIMULATED DATA IN SCINTILLATOR with Quenching effect "
path='pictures/2020-12-03/lowdose/quenchingsimulation_shot3.csv'   #data ha alreay normalized to their own area
depth_sci,realdose,quenchedose,c,ys=read_data_mini(path)

quenchedarea=np.trapz(quenchedose[0:len(quenchedose)], depth_sci[0:len(depth_sci)])
norm=quenchedarea/area



plt.figure(4)
"TOF SIMULATED DATA IN SCINTILLATOR with Quenching effect "

plt.errorbar(  np.arange(0,len(dose),1)*0.0634,                       dose*norm,
                                                                  yerr=err*norm,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                   ecolor='gray',
                                                                elinewidth=None,
                                     label=' measured')




plt.plot( depth_sci,quenchedose,
                                                                            'g.',
                                                                   Markersize=8,
                                              label=' simulated quenched curve')
plt.plot( depth_sci,realdose,
                                                                            'r.',
                                                                    Markersize=8,
                                                      label=' simulated  curve')

plt.title('Shape comparsison low dose setting 12-03 shot n3',
                                                                  fontdict=None,
                                                                   loc='center',
                                                                       pad=None)
plt.xlabel('Depth[mm]')
plt.ylabel('Dose[Gy]')
plt.grid(True)
plt.legend()
"""
plt.show()
