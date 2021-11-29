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

###############################################################################

#RCF data

directory='/home/corvin22/Desktop/miniscidom/pictures/2021-09-01/'
#directory='/Users/angelacorvino/Desktop/HZDR/miniscidom/pictures/2021-09-01/'
filename2='RCF21CT.csv'
filename1='RCF21CR.csv'
filename='RCF21CP.csv'
rcfdepth,rcfdose,rcferr,area_rcf,rcfname=read_datarcf(directory,filename)
rcfdepth1,rcfdose1,rcferr1,area_rcf1,rcfname1=read_datarcf(directory,filename1)
rcfdepth2,rcfdose2,rcferr2,area_rcf2,rcfname2=read_datarcf(directory,filename2)





#September 2021
s=0.073825 #[mm/pixel ]
directory='/home/corvin22/Desktop/miniscidom/pictures/2021-09-01/notnormalized/'
#directory='/Users/angelacorvino/Desktop/HZDR/miniscidom/pictures/2021-09-01/notnormalized/'
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





outputfile_topas='/home/corvin22/Desktop/precoiusthings/SOBP/11PC/151.5MeV/DoseToWater_151500KeVproton_PVT_11PC_1Dscorer.csv'
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
outputfile_topas='/home/corvin22/Desktop/precoiusthings/SOBP/11PC/151.5MeV/LET_doseweighted_151500KeVproton_PVT_11PC_1Dscorer.csv'
#outputfile_topas='/Users/angelacorvino/Desktop/GitHub/SimulationOncoray/data/SOBP/11PC/LET_doseweighted_153MeVproton_PVT_11PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix= np.array(df)# convert dataframe df to array
LET_doseprofile=Get_profile(topas_datamatrix, 149,1)
LET_zdoseprofile1=LET_doseprofile.zmeanprofile
LET_zdoseprofile1=LET_zdoseprofile1[::-1]
LET_zdoseprofile1[0]=LET_zdoseprofile1[1]

#outputfile_topas = '/home/corvin22/Simulationhemera/data/SOBP/LET_fluenceweighted_14960KeVproton_PVT_11PC_1Dscorer.csv'
outputfile_topas='/home/corvin22/Desktop/precoiusthings/SOBP/11PC/151.5MeV/LET_fluenceweighted_151500KeVproton_PVT_11PC_1Dscorer.csv'
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
outputfile_topas='/home/corvin22/Desktop/precoiusthings/SOBP/12PC/152.5MeV/DoseToWater_152500KeVproton_PVT_12PC_1Dscorer.csv'
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
outputfile_topas='/home/corvin22/Desktop/precoiusthings/SOBP/12PC/152.5MeV/LET_doseweighted_152500KeVproton_PVT_12PC_1Dscorer.csv'
#outputfile_topas='/Users/angelacorvino/Desktop/GitHub/SimulationOncoray/data/SOBP/12PC/LET_doseweighted_153MeVproton_PVT_12PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix= np.array(df)# convert dataframe df to array
LET_doseprofile=Get_profile(topas_datamatrix, 149,1)
LET_zdoseprofile2=LET_doseprofile.zmeanprofile
LET_zdoseprofile2=LET_zdoseprofile2[::-1]
LET_zdoseprofile2[0]=LET_zdoseprofile2[1]

#outputfile_topas = '/home/corvin22/Simulationhemera/data/SOBP/LET_fluenceweighted_14960KeVproton_PVT_11PC_1Dscorer.csv'
outputfile_topas='/home/corvin22/Desktop/precoiusthings/SOBP/12PC/152.5MeV/LET_fluenceweighted_152500KeVproton_PVT_12PC_1Dscorer.csv'
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









outputfile_topas='/home/corvin22/Desktop/precoiusthings/SOBP/10PC/152.5MeV/DoseToWater_152500KeVproton_PVT_10PC_1Dscorer.csv'
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



outputfile_topas='/home/corvin22/Desktop/precoiusthings/SOBP/10PC/152.5MeV/LET_fluenceweighted_152500KeVproton_PVT_10PC_1Dscorer.csv'
#outputfile_topas='/Users/angelacorvino/Desktop/GitHub/SimulationOncoray/data/SOBP/10PC/LET_fluenceweighted_152500KeVproton_PVT_10PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix= np.array(df)# convert dataframe df to array
LET_fluenceprofile=Get_profile(topas_datamatrix, 149,1)
LET_zfluenceprofile=LET_fluenceprofile.zmeanprofile
LET_zfluenceprofile=LET_zfluenceprofile[::-1]
LET_zfluenceprofile[0]=LET_zfluenceprofile[1]


area_f_LET=np.trapz(LET_zfluenceprofile,depth_sci)



outputfile_topas='/home/corvin22/Desktop/precoiusthings/SOBP/10PC/152.5MeV/LET_doseweighted_152500KeVproton_PVT_10PC_1Dscorer.csv'
#outputfile_topas='/Users/angelacorvino/Desktop/GitHub/SimulationOncoray/data/SOBP/10PC/LET_doseweighted_153MeVproton_PVT_10PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix= np.array(df)# convert dataframe df to array
LET_doseprofile=Get_profile(topas_datamatrix, 149,1)
LET_zdoseprofile=LET_doseprofile.zmeanprofile
LET_zdoseprofile=LET_zdoseprofile[::-1]
LET_zdoseprofile[0]=LET_zdoseprofile[1]










################################################################################
norm_rcf=1/rcfdose.max()
norm_rcf1=1/rcfdose1.max()    #area_rcf
norm_rcf2=1/rcfdose2.max()
norm=1/dose.max()#area
norm1=1/dose1.max()#area
norm2=1/dose2.max()#area
norm_sim=1/zdoseprofile.max() #area_sim
norm_sim1=1/zdoseprofile1.max() #area_sim
norm_sim2=1/zdoseprofile2.max() #area_sim

#norm_rcf=1/area_rcf
#norm_rcf1=1/area_rcf1
#norm_rcf2=1/area_rcf2
#norm=1/area
#norm1=1/area1
#norm2=1/area2
#norm_sim2=1/area_sim2
#norm_sim=1/area_sim
#norm_sim1=1/area_sim1
##############################################################################


""" correction of RCf measure with LET (doseweighted ) simulations in scintillator"""

a=-0.0251
b= 1.02

def function1(rcfdepth,rcfdose,rcferr,depth_sci,LET_zdoseprofile,a,b):
    S=np.interp(np.arange(0,len(rcfdose),1)*s,depth_sci,LET_zdoseprofile)
    D_rcf=rcfdosecorrection(rcfdose,S,a,b)
    Derr_rcf=rcfdosecorrection(rcferr,S,a,b)
    area_rcfcorrected=np.trapz(rcfdosecorrection(rcfdose,S,a,b),rcfdepth)
    return D_rcf,Derr_rcf,area_rcfcorrected

D_rcf,Derr_rcf,area_rcfcorrected=function1(rcfdepth,rcfdose,rcferr,depth_sci,LET_zdoseprofile,a,b)
D_rcf1,Derr_rcf1,area_rcfcorrected1=function1(rcfdepth,rcfdose,rcferr,depth_sci,LET_zdoseprofile,a,b)
D_rcf2,Derr_rcf2,area_rcfcorrected2=function1(rcfdepth2,rcfdose2,rcferr2,depth_sci2,LET_zdoseprofile2,a,b)
####################################################################################



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

def function2(dose,depth,depth_sci,LET_zdoseprofile,LET_zfluenceprofile,s,area_rcfcorrected):


    ys_ana_mini=LET_zdoseprofile
    ys_fluence_mini=LET_zfluenceprofile

    S_a_mini=np.interp(np.arange(0,len(dose),1)*s,depth_sci,ys_ana_mini)
    #S_a_low_mini=np.interp(np.arange(0,len(dose),1)*s,depth_sci,ys_ana_lower_mini)
    #S_a_up_mini=np.interp(np.arange(0,len(dose),1)*s,depth_sci,ys_ana_upper_mini)
    S_fluence_mini=np.interp(np.arange(0,len(dose),1)*s,depth_sci,ys_fluence_mini)


    area_corrected=np.trapz(dosecorrection(dose,S_a_mini,a,k,dx)[3:len(dosecorrection(dose,S_a_mini,a,k,dx))],
                                                                        depth[3:len(depth)]) #eliminate first 4 points
    area_f_corrected=np.trapz(dosecorrection(dose,S_fluence_mini,a,k,dx)[3:len(dosecorrection(dose,S_fluence_mini,a,k,dx))-12],
                                                                        depth[3:len(depth)-12]) #eliminate first 4 points
    norm_dose_mini=area_rcfcorrected/area_corrected
    #norm_fluence_mini=1/area_f_corrected
    norm_fluence_mini=1/dosecorrection(dose,S_fluence_mini,a,k,dx).max()




    D_a_mini=dosecorrection(dose,S_a_mini,a,k,dx)
    #D_a_up_mini=dosecorrection(dose,S_a_up_mini,a,k,dx)*norm_a_mini
    #D_a_low_mini=dosecorrection(dose,S_a_low_mini,a,k,dx)*norm_a_mini
    D_fluence_mini=dosecorrection(dose,S_fluence_mini,a,k,dx)
    D_fluence_mini=D_fluence_mini*norm_fluence_mini
    D_a_mini=D_a_mini*norm_dose_mini
    #Dabsolute_a_mini=dosecorrection(dose,S_a_mini,a,k,dx)*norm_dose_mini* lightcorrection(S_a_mini,a,k,dx)
    #L_a_up_mini=lightcorrection(S_a_up_mini,a,k,dx)
    #L_a_low_mini=lightcorrection(S_a_low_mini,a,k,dx)

    Dabsolute_fluence_mini=dosecorrection(dose,S_fluence_mini,a,k,dx)*norm_dose_mini* lightcorrection(S_fluence_mini,a,k,dx)


    return D_a_mini,D_fluence_mini

D_a_mini2,D_fluence_mini2=function2(dose2,depth2,depth_sci2,LET_zdoseprofile2,LET_zfluenceprofile2,s,area_rcfcorrected2)
D_a_mini1,D_fluence_mini1=function2(dose1,depth1,depth_sci1,LET_zdoseprofile1,LET_zfluenceprofile1,s,area_rcfcorrected1)
D_a_mini,D_fluence_mini=function2(dose,depth,depth_sci,LET_zdoseprofile,LET_zfluenceprofile,s,area_rcfcorrected)




###############################################################################


plt.figure(1) # creates a figure in which we plot
plt.plot( np.arange(0,149,1)*s,zdoseprofile*norm_sim,

                                                                    '.',
                                                                  markersize=12,
                                          color=sns.color_palette(  "Paired")[3],
                                                       label=' {}'.format(PC))

plt.errorbar( rcfdepth,                                          rcfdose*norm_rcf,
                                                            yerr=rcferr*norm_rcf,
                                                                      xerr=None,
                                                                        fmt='.',
                                          color=sns.color_palette(  "Paired")[2],
                                                                    markersize=12,
                                         ecolor=sns.color_palette(  "Paired")[2],
                                                                 elinewidth=None,
                                                 label='{} measured dose'.format(rcfname),
                                                 zorder=2)

plt.fill_between(rcfdepth,
                                                rcfdose*norm_rcf-rcferr*norm_rcf,
                                                rcfdose*norm_rcf+rcferr*norm_rcf,
                                          color=sns.color_palette(  "Paired")[2],
                                                                       alpha=0.1,
                                                                       zorder=2)




plt.plot( np.arange(0,149,1)*s,zdoseprofile1*norm_sim1,
                                                                        '.',
                                                                  markersize=12,
                                          color=sns.color_palette(  "Paired")[5],
                                                       label=' {}'.format(PC1),
                                                       zorder=1)


plt.errorbar( rcfdepth1,                                      rcfdose1*norm_rcf1,
                                                            yerr=rcferr1*norm_rcf1,
                                                                      xerr=None,
                                                                        fmt='.',
                                          color=sns.color_palette(  "Paired")[4],
                                                                    markersize=12,
                                         ecolor=sns.color_palette(  "Paired")[4],
                                                                 elinewidth=None,
                                                 label='{} measured dose'.format(rcfname1),
                                                 zorder=2)

plt.fill_between(rcfdepth1,
                                                rcfdose1*norm_rcf1-rcferr1*norm_rcf1,
                                                rcfdose1*norm_rcf1+rcferr1*norm_rcf1,
                                          color=sns.color_palette(  "Paired")[4],
                                                                       alpha=0.1,
                                                                       zorder=2)






plt.plot( np.arange(0,149,1)*s,zdoseprofile2*norm_sim2,
                                                                        '.',
                                                                  markersize=12,
                                          color=sns.color_palette(  "Paired")[7],
                                                       label=' {}'.format(PC2),
                                                       zorder=1)



plt.errorbar( rcfdepth2,                                          rcfdose2*norm_rcf2,
                                                            yerr=rcferr2*norm_rcf2,
                                                                      xerr=None,
                                                                        fmt='.',
                                          color=sns.color_palette(  "Paired")[6],
                                                                    markersize=12,
                                         ecolor=sns.color_palette(  "Paired")[6],
                                                                 elinewidth=None,
                                                 label='{} measured dose'.format(rcfname2),
                                                 zorder=2)

plt.fill_between(rcfdepth,
                                                rcfdose2*norm_rcf2-rcferr2*norm_rcf2,
                                                rcfdose2*norm_rcf2+rcferr2*norm_rcf2,
                                          color=sns.color_palette(  "Paired")[6],
                                                                       alpha=0.1,
                                                                       zorder=2)



plt.title('Depth-dose distribution shape comparison',fontsize=26)
plt.xlabel(" Depth[mm] ",fontsize=24)
plt.ylabel('Relative Dose in water ',fontsize=24)
plt.legend(title='E_{primary}=152.5 MeV',fontsize=24,loc=3,markerscale=2,title_fontsize=24)
plt.grid(b=True, which='major', color='k', linestyle='-',alpha=0.2)
plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
plt.minorticks_on()
plt.tick_params(axis='x', which='major', labelsize=22)
plt.tick_params(axis='y', which='major', labelsize=22)

plt.show()





def function4(rcfdepth,rcfdose,norm_rcf,rcferr,rcfname,dose,depth,err,norm,depth_sci2,zdoseprofile,LET_zfluenceprofile,norm_sim,D_fluence_mini,shotnumber,PC):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()



    ax.plot(depth,
                                                                D_fluence_mini,
                                                                                '.',
                                            color=sns.color_palette(  "Paired")[10],
                                        label='LET(fluence weighted) corrected dose',
                                                                      Markersize=12,
                                                                      zorder=10)



    ax.errorbar( rcfdepth,                                          rcfdose*norm_rcf,
                                                            yerr=rcferr*norm_rcf,
                                                                      xerr=None,
                                                                        fmt='.',
                                          color=sns.color_palette(  "Paired")[1],
                                                                    markersize=12,
                                         ecolor=sns.color_palette(  "Paired")[1],
                                                                 elinewidth=None,
                                                 label='{} measured dose'.format(rcfname),
                                                                                 zorder=4)

    ax.fill_between(rcfdepth,
                                                rcfdose*norm_rcf-rcferr*norm_rcf,
                                                rcfdose*norm_rcf+rcferr*norm_rcf,
                                          color=sns.color_palette(  "Paired")[1],
                                                                       alpha=0.1)

    ax.errorbar(  np.arange(0,len(dose),1)*s,                              dose*norm,
                                                                   yerr=err*norm,
                                                                       xerr=None,
                                                                         fmt='.',
                                                                         markersize=12,
                                          color=sns.color_palette(  "Paired")[3],
                                         ecolor=sns.color_palette(  "Paired")[3],
                                                                 elinewidth=None,
                                             label=' Miniscidom reconstruction ',
                                              zorder=3)

    ax.fill_between(np.arange(0,len(dose),1)*s,
                                                             dose*norm-err*norm,
                                                           dose*norm + err*norm,
                                          color=sns.color_palette(  "Paired")[3],
                                                                      alpha=0.5)

    """
ax.errorbar( rcfdepth,                                  D_rcf/area_rcfcorrected ,
                                                 yerr=Derr_rcf/area_rcfcorrected,
                                                                      xerr=None,
                                                                        fmt='.',
                                          color=sns.color_palette(  "Paired")[2],
                                                                   markersize=8,
                                        ecolor=sns.color_palette(  "Paired")[2],
                                                                 elinewidth=None,
                              label='LET(dose weighted)corrected {} dose'.format(rcfname))


ax.fill_between(rcfdepth,
                              D_rcf/area_rcfcorrected-Derr_rcf/area_rcfcorrected,
                              D_rcf/area_rcfcorrected+Derr_rcf/area_rcfcorrected,
                                          color=sns.color_palette(  "Paired")[2],
                                                                       alpha=0.1)

ax2.plot(depth_sci,LET_zdoseprofile,
                                                                            '.',
                                         color=sns.color_palette(  "Paired")[4],
                                                                   Markersize=8,
                                                    label='LET (doseweighted) ')

  """

    ax2.plot(depth_sci2,LET_zfluenceprofile,
                                                                            '.',
                                            color=sns.color_palette(  "Paired")[4],
                                                                   Markersize=12,
                                               label='LET (fluence weighted) ',
                                                                 zorder=1)




    ax.plot( np.arange(0,149,1)*s,zdoseprofile*norm_sim,
                                                                            '.',
                                         color=sns.color_palette(  "Paired")[7],
                                                                  Markersize=12,
                label='Simulated Dose in scintillator Eo={} KeV '.format(energy),
                                                                    zorder=2)





    """
ax.plot(depth,
                                                        D_a_mini/D_a_mini.max(),
                                                                            '.',
                                          color=sns.color_palette(  "Paired")[9],
                                        label='LET(dose weighted) corrected dose',
                                                                  Markersize=11)


    """






    plt.title('Depth-dose distribution shape comparison {} shot n.{}'.format(PC,shotnumber),
                                                                    fontsize=24,
                                                                  fontdict=None,
                                                                  loc='center',
                                                                       pad=None)

    #ax.set_ylim([0,0.2])
    ax2.set_ylabel("LET [KeV/um]",color=sns.color_palette(  "Paired")[5],fontsize=16)
    ax.set_ylabel(" Relative Dose ",color="Green",fontsize=22)
    ax.set_xlabel(" Depth[mm] ",color="black",fontsize=22)
    ax.legend( title='',fontsize=20,loc=4,markerscale=3)
    ax2.legend( title='',fontsize=20,loc=6,markerscale=3)
    ax2.grid(b=True,color='k',linestyle='-',alpha=0.2)
    ax.grid(b=True,color='k',linestyle='dotted',alpha=0.2)
    ax.tick_params(axis='x', which='major', labelsize=22)
    ax2.tick_params(axis='y', which='major',colors=sns.color_palette(  "Paired")[5], labelsize=22)
    ax.tick_params(axis='y', which='major',colors='green', labelsize=22)


    return


function4(rcfdepth2,rcfdose2,norm_rcf2,rcferr2,rcfname2,dose2,depth2,err2,norm2,depth_sci2,zdoseprofile2,LET_zfluenceprofile2,norm_sim2,D_fluence_mini2,shotnumber2,PC2)
function4(rcfdepth1,rcfdose1,norm_rcf1,rcferr1,rcfname1,dose1,depth1,err1,norm1,depth_sci1,zdoseprofile1,LET_zfluenceprofile1,norm_sim1,D_fluence_mini1,shotnumber1,PC1)
function4(rcfdepth,rcfdose,norm_rcf,rcferr,rcfname,dose,depth,err,norm,depth_sci,zdoseprofile,LET_zfluenceprofile,norm_sim,D_fluence_mini,shotnumber,PC)
#PLOT







def function3(rcfdose,rcfdepth,rcferr,rcfname,area_rcf,area_rcfcorrected,D_rcf,Derr_rcf,depth_sci,LET_zdoseprofile):


    fig, ax = plt.subplots()
    plt.title('Shape comparison {} '.format(rcfname),fontsize=22)
    ax2 = ax.twinx()



    ax.errorbar( rcfdepth,                                          rcfdose/area_rcf,
                                                             yerr=rcferr/area_rcf,
                                                                      xerr=None,
                                                                        fmt='.',
                                            color=sns.color_palette(  "Paired")[1],
                                                                    markersize=12,
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
                                                                   markersize=12,
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


    """
    sub_axes.plot( rcfdepth[0:5],                      (rcfdose/area_rcf)[0:5],
                                                                        '.',
                                            color=sns.color_palette(  "Paired")[1],
                                                                    markersize=8)


    sub_axes.plot( rcfdepth[0:5],                       (D_rcf/area_rcfcorrected)[0:5],
                                                                        '.',
                                            color=sns.color_palette(  "Paired")[5],
                                                                    markersize=8)


    sub_axes.grid(b=True,color='k',linestyle='-',alpha=0.2)





    """

    sub_axes.errorbar(rcfdepth[0:5],                      (rcfdose/area_rcf)[0:5],
                                                        yerr=(rcferr/area_rcf)[0:5],
                                                                      xerr=None,
                                                                        fmt='.',
                                            color=sns.color_palette(  "Paired")[1],
                                                                    markersize=12,
                                                                ecolor='lightblue',
                                                                elinewidth=None)






    sub_axes.errorbar( rcfdepth[0:5],               (D_rcf/area_rcfcorrected)[0:5] ,
                                        yerr=(Derr_rcf/area_rcfcorrected)[0:5],
                                                                      xerr=None,
                                                                        fmt='.',
                                          color=sns.color_palette(  "Paired")[5],
                                                                   markersize=12,
                                                            ecolor='lightcoral',
                                                                 elinewidth=None)
    sub_axes.grid(b=True,color='k',linestyle='-',alpha=0.2)









    """"
ax.plot( np.arange(0,149,1)*s,zdoseprofileprova1/zdoseprofileprova1.max(),
                                                                            '.',
                                        color=sns.color_palette(  "Paired")[8],
                                                                 Markersize=11,
                            label='Simulated Dose in scintillator Eo= KeV '.format(energy))



ax.plot(depth_sci,zdoseprofile/area_sim,
                                                                            'g.',
                                                                  Markersize=11,
                            label='Simulated Dose in scintillator Eo={} KeV '.format(energy))
    """




    ax2.plot(depth_sci,LET_zdoseprofile,
                                                                            '.',
                                         color=sns.color_palette(  "Paired")[2],
                                                                  Markersize=12,
                                                                   label='LET  dose-weighted')


    ax.set_ylim([0.02,0.15])
    ax.set_xlabel(" Depth [mm] ",color="black",fontsize=16)
    ax2.set_ylabel("LET [KeV/um]",color=sns.color_palette(  "Paired")[2],fontsize=16)
    ax.set_ylabel(" Relative Dose ",color="blue",fontsize=16)
    ax.legend( title='',fontsize=20,loc=4,markerscale=3)
    ax2.legend( title='',fontsize=20,loc=6,markerscale=3)
    ax2.grid(b=True,color='k',linestyle='-',alpha=0.2)
    ax.grid(b=True,color='k',linestyle='dotted',alpha=0.2)

    ax.tick_params(axis='x', which='major', labelsize=22)
    ax2.tick_params(axis='y', which='major',colors=sns.color_palette(  "Paired")[2],
                                                                   labelsize=22)
    ax.tick_params(axis='y', which='major',colors='blue', labelsize=22)

#plt.title('Shape comparison {} '.format(rcfname),fontsize=22)

    mark_inset(ax, sub_axes, loc1=2, loc2=3, fc="none", ec="0.5")

    plt.show()
    return

function3(rcfdose2,rcfdepth2,rcferr2,rcfname2,area_rcf2,area_rcfcorrected2,D_rcf2,Derr_rcf2,depth_sci2,LET_zdoseprofile2)
function3(rcfdose1,rcfdepth1,rcferr1,rcfname1,area_rcf1,area_rcfcorrected1,D_rcf1,Derr_rcf1,depth_sci1,LET_zdoseprofile1)
function3(rcfdose,rcfdepth,rcferr,rcfname,area_rcf,area_rcfcorrected,D_rcf,Derr_rcf,depth_sci,LET_zdoseprofile)
