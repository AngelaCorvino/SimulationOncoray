'''
COMPARISON BETWEEN SIMULATED DATA IN SCINTILLATOR  AND RCF TO UNDERSTAND THE CORRECT INITIAL ENERGY
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

directory= '~/Documents/GitHub/SimulationOncoray/pictures/2021-09-02/'
filename='RCF21CY.csv'
depth,rcfdose,rcferr,area_rcf,rcfname=read_datarcf(directory,filename)


"""

#September 2021

directory= '~/Documents/GitHub/SimulationOncoray/pictures/2021-09-02/notnormalized/'
data= np.load(directory+'notnormalizedmean_array17.npy')
dose=data[0:len(data)-3]
tof=data[len(data)-3:len(data)]
data0= np.load(directory+'notnormalizederr17.npy')
err=data0[0:len(data0)]  #standard deviation divided by radical n


x= np.arange(0,len(dose),1)*0.0738255
area=np.trapz(dose, x)


"""
outputfile_topas = '~/Documents/GitHub/SimulationOncoray/data/Single/DoseToWater_9140MeVproton_PVT_7PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
doseprofile=Get_profile(topas_datamatrix, 149,1)
zdoseprofile=doseprofile.zmeanprofile
zdoseprofile=zdoseprofile[::-1]

dose_sci=zdoseprofile


depth_sci= np.arange(0,len(zdoseprofile),1)*0.0738255
area_sci=np.trapz(zdoseprofile,depth_sci)
#area_sci=np.trapz(zdoseprofile[1:len(zdoseprofile)],depth_sci[1:len(depth_sci)]) #eliminate last 3 pointsà
#area_LET=np.trapz(LET_zdoseprofile[1:len(LET_zdoseprofile)],depth_sci[1:len(depth_sci)])

#area_f_LET=np.trapz(LET_zfluenceprofile[1:len(LET_zfluenceprofile)],depth_sci[1:len(depth_sci)])


normrcf=1
#norm=area_rcf/area
#norm_sim=area_rcf/area_sci

outputfile_topas = '~/Documents/GitHub/SimulationOncoray/data/Single/DoseToWater_92MeVproton_PVT_7PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
doseprofile_medium=Get_profile(topas_datamatrix, 149,1)
zdoseprofile_medium=doseprofile_medium.zmeanprofile
zdoseprofile_medium=zdoseprofile_medium[::-1]

area_scimedium=np.trapz(zdoseprofile_medium,depth_sci)


outputfile_topas = '~/Documents/GitHub/SimulationOncoray/data/Single/DoseToWater_92M10eVproton_PVT_7PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
doseprofile_m=Get_profile(topas_datamatrix, 149,1)
zdoseprofile_m=doseprofile_m.zmeanprofile
zdoseprofile_m=zdoseprofile_m[::-1]

area_scim=np.trapz(zdoseprofile_m,depth_sci)


outputfile_topas = '~/Documents/GitHub/SimulationOncoray/data/Single/DoseToWater_9240MeVproton_PVT_7PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix = np.array(df)# convert dataframe df to array
doseprofile_high=Get_profile(topas_datamatrix, 149,1)
zdoseprofile_high=doseprofile_high.zmeanprofile
zdoseprofile_high=zdoseprofile_high[::-1]

#area_scihigh=np.trapz(zdoseprofile_high[1:len(zdoseprofile_high)],depth_sci[1:len(depth_sci)])
area_scihigh=np.trapz(zdoseprofile_high,depth_sci)

#eliminate last 3 pointsà
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
"""
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

"""
plt.grid(True)

plt.plot( np.arange(0,149,1)*0.0738255,zdoseprofile/area_sci,
                                                                            'g.',
                                                                  Markersize=10,
                                        label='Simulation Eo = 91.40 MeV ')
plt.plot( np.arange(0,149,1)*0.0738255,zdoseprofile_m/area_scim,
                                                                            'm.',
                                                                  Markersize=10,
                                        label='Simulation Eo = 92.10 MeV ')
plt.plot( np.arange(0,149,1)*0.0738255,zdoseprofile_high/area_scihigh,
                                                                            'y.',
                                                                 Markersize=10,
                                        label='Simulation Eo = 92.40 MeV ')
plt.plot( np.arange(0,149,1)*0.0738255,zdoseprofile_medium/area_scimedium,
                                                                            'b.',
                                                                 Markersize=10,
                                        label='Simulation Eo = 92.00 MeV ')


plt.legend( title='Unidose',fontsize='large')
plt.title('Depth-dose distribution 7PC ',
                                                                  fontdict=None,
                                                                  loc='center',
                                                                       pad=None)
plt.xlabel('Depth[mm]')
plt.ylabel('Relative Dose')
plt.show()
