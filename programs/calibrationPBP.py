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
directory='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2021-09-02/'
filename='RCF21CW.csv'
rcfdepth,rcfdose,rcferr,area_rcf,rcfname=read_datarcf(directory,filename)

filename1='RCF21CY.csv'
rcfdepth1,rcfdose1,rcferr1,area_rcf1,rcfname1=read_datarcf(directory,filename1)

filename2='RCF21CX.csv'
rcfdepth2,rcfdose2,rcferr2,area_rcf2,rcfname2=read_datarcf(directory,filename2)

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





outputfile_topas = '/Users/angelacorvino/Documents/GitHub/SimulationOncoray/data/Single/DoseToWater_9210KeVproton_PVT_7PC_1Dscorer.csv'
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





outputfile_topas = '/Users/angelacorvino/Documents/GitHub/SimulationOncoray/data/Single/LET_fluenceweighted_9210KeVproton_PVT_7PC_1Dscorer.csv'
header = pd.read_csv(outputfile_topas, nrows = 7)
df = pd.read_csv(outputfile_topas, comment='#', header=None)
topas_datamatrix= np.array(df)# convert dataframe df to array
LET_fluenceprofile1=Get_profile(topas_datamatrix, 149,1)
LET_zfluenceprofile1=LET_fluenceprofile1.zmeanprofile
LET_zfluenceprofile1=LET_zfluenceprofile1[::-1]
LET_zfluenceprofile1[0]=LET_zfluenceprofile1[1]


outputfile_topas = '/Users/angelacorvino/Documents/GitHub/SimulationOncoray/data/Single/LET_fluenceweighted_9210KeVproton_PVT_7PC_1Dscorer.csv'
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
#norm_sim=1/area_sim




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
D_rcf2,Derr_rcf2,area_rcfcorrected2=function1(rcfdepth2,rcfdose2,rcferr2,depth_sci,LET_zdoseprofile2,a,b)


"""LET correction simulations in scintillator """

dS=0 #theoretical values
dscintillator= 1.023 #[g/cm^3] scintillator density
dactivelayer=1.2 #[g/cm^3]
k=207/10000 #[g/MeV cm^2]
a=0.9 #scintillator efficiency
dx= 65 #µm scintillator spatial resolution
ddx=1 #spatial resolution error
k=k*10#[micrometers/kev]
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

D_a_mini2,D_fluence_mini2=function2(dose2,depth2,depth_sci,LET_zdoseprofile2,LET_zfluenceprofile2,s,area_rcfcorrected2)
D_a_mini1,D_fluence_mini1=function2(dose1,depth1,depth_sci,LET_zdoseprofile1,LET_zfluenceprofile1,s,area_rcfcorrected1)
D_a_mini,D_fluence_mini=function2(dose,depth,depth_sci,LET_zdoseprofile,LET_zfluenceprofile,s,area_rcfcorrected)



###############################################################################


def plotfunction(depth,dose,err,norm,rcfdepth,rcfdose,rcferr,norm_rcf,color1,color2,PC,rcfname,s):


    plt.errorbar(depth,                              dose*norm,
                                                                   yerr=err*norm,
                                                                       xerr=None,
                                                                         fmt='.',
                                                                         markersize=12,
                                                                   color=color1,
                                                                  ecolor=color1,
                                                                 elinewidth=None,
                                                        label=' Miniscidom {}'.format(PC))


    plt.fill_between(depth,
                                                             dose*norm-err*norm,
                                                           dose*norm + err*norm,
                                          color=sns.color_palette(  "Paired")[0],
                                                                      alpha=0.5)







    plt.errorbar( rcfdepth,                                          rcfdose*norm_rcf,
                                                            yerr=rcferr*norm_rcf,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                    color=color2,
                                                                    markersize=12,
                                                                  ecolor=color2,
                                                                 elinewidth=None,
                                                 label='{}'.format(rcfname))

    plt.fill_between(rcfdepth,
                                                rcfdose*norm_rcf-rcferr*norm_rcf,
                                                rcfdose*norm_rcf+rcferr*norm_rcf,
                                          color=color2,
                                                                       alpha=0.1)


    return

def plotfunction1(depth_sci,dose_sci,norm_sim,rcfdepth,rcfdose,rcferr,norm_rcf,color1,color2,PC,rcfname,s):


    plt.plot(depth_sci,dose_sci*norm_sim,
                                                                                '.',
                                                                     color=color1,
                                                                     Markersize=12,
                                                            label=' {}'.format(PC))



    plt.errorbar( rcfdepth,                                          rcfdose*norm_rcf,
                                                            yerr=rcferr*norm_rcf,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                    color=color2,
                                                                    markersize=12,
                                                                  ecolor=color2,
                                                                 elinewidth=None,
                                                 label='{}'.format(rcfname))

    plt.fill_between(rcfdepth,
                                                rcfdose*norm_rcf-rcferr*norm_rcf,
                                                rcfdose*norm_rcf+rcferr*norm_rcf,
                                          color=color2,
                                                                       alpha=0.1)
    return


def plotfunction2(depth_sci,dose_sci,norm_sim,LET_zfluenceprofile,color1,PC,zlim):

    plt.plot(depth_sci[0:zlim],LET_zfluenceprofile[0:zlim],
                                                                            '.',
                                                                   color=color1,
                                                                   Markersize=12,
                                               label='{} '.format(PC))

    return







plt.figure(1)

plotfunction(depth,dose,err,norm,rcfdepth,rcfdose,rcferr,norm_rcf,sns.color_palette("Paired")[0],sns.color_palette(  "Paired")[1],PC,rcfname,s)
plotfunction(depth1,dose1,err1,norm1,rcfdepth1,rcfdose1,rcferr1,norm_rcf1,sns.color_palette(  "Paired")[2],sns.color_palette(  "Paired")[3],PC1,rcfname1,s)
plotfunction(depth2,dose2,err2,norm2,rcfdepth2,rcfdose2,rcferr2,norm_rcf2,sns.color_palette(  "Paired")[8],sns.color_palette(  "Paired")[9],PC2,rcfname2,s)



plt.title('Depth-dose distribution shape comparison',fontsize=26)
plt.xlabel(" Depth[mm] ",fontsize=24)
plt.ylabel('Relative Dose in water ',fontsize=24)
#plt.legend(title='E_{primary}=92.10 MeV',fontsize=24,loc=3,markerscale=2,title_fontsize=24)
plt.legend(title='',fontsize=24,loc=3,markerscale=3,title_fontsize=20)
plt.grid(b=True, which='major', color='k', linestyle='-',alpha=0.2)
plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
plt.minorticks_on()
plt.tick_params(axis='x', which='major', labelsize=22)
plt.tick_params(axis='y', which='major', labelsize=22)







plt.figure(2)


plotfunction1(depth_sci,dose_sci,norm_sim,rcfdepth,rcfdose,rcferr,norm_rcf,sns.color_palette(  "Paired")[0],sns.color_palette(  "Paired")[1],PC,rcfname,s)
plotfunction1(depth_sci,dose_sci1,norm_sim1,rcfdepth1,rcfdose1,rcferr1,norm_rcf1,sns.color_palette(  "Paired")[2],sns.color_palette(  "Paired")[3],PC1,rcfname1,s)
plotfunction1(depth_sci,dose_sci2,norm_sim2,rcfdepth2,rcfdose2,rcferr2,norm_rcf2,sns.color_palette(  "Paired")[8],sns.color_palette(  "Paired")[9],PC2,rcfname2,s)


plt.legend(fontsize=20,loc=3,markerscale=3)
plt.title('Simulated Dose in scintillator',fontsize=20)
plt.xlabel(" Depth[mm] ",fontsize=20)
plt.ylabel('Relative Dose in water ',fontsize=20)
plt.grid(b=True, which='major', color='k', linestyle='-',alpha=0.2)
plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)

plt.tick_params(axis='x', which='major', labelsize=20)
plt.tick_params(axis='y', which='major', labelsize=20)

plt.figure(3)
plotfunction2(depth_sci,dose_sci,norm_sim,LET_zfluenceprofile,sns.color_palette(  "Paired")[1],PC,190)
plotfunction2(depth_sci,dose_sci1,norm_sim1,LET_zfluenceprofile1,sns.color_palette(  "Paired")[3],PC1,190)
plotfunction2(depth_sci,dose_sci2,norm_sim2,LET_zfluenceprofile2,sns.color_palette(  "Paired")[9],PC2,150)


plt.title('Simulated fluence-weighted LET',fontsize=20)
plt.legend(fontsize=20,loc=2,markerscale=3)
plt.ylabel("LET [KeV/um]",fontsize=20)
plt.xlabel(" Depth[mm] ",color="black",fontsize=20)
plt.gca().yaxis.set_label_position("right")
plt.ylim([-1,18])
plt.grid(b=True, which='major', color='k', linestyle='-',alpha=0.2)
plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
plt.tick_params(axis='y', labelsize=20,which='both', labelleft=False, labelright=True)
plt.tick_params(axis='x',which='both', labelsize=20)

plt.show()
####################################################################################





norm_rcf=1/rcfdose.max()
norm_rcf1=1/rcfdose1.max()    #area_rcf
norm_rcf2=1/rcfdose2.max()
norm=1/dose.max()#area
norm1=1/dose1.max()
norm2=1/dose2.max()#
norm_sim=1/zdoseprofile.max() #area_sim
norm_sim1=1/zdoseprofile1.max() #area_sim
norm_sim2=1/zdoseprofile2.max() #area_sim













def function4(rcfdepth,rcfdose,norm_rcf,rcferr,rcfname,dose,depth,err,norm,depth_sci,zdoseprofile,LET_zfluenceprofile,norm_sim,D_fluence_mini,shotnumber,PC):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()



    ax.plot(depth,
                                                                D_fluence_mini,
                                                                                '.',
                                            color=sns.color_palette(  "Paired")[10],
                                                     label='LET corrected dose',
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



    ax2.plot(depth_sci[0:150],LET_zfluenceprofile[0:150],
                                                                            '.',
                                            color=sns.color_palette(  "Paired")[4],
                                                                   Markersize=12,
                                               label='LET(fluence weighted) ',
                                                                 zorder=3)




    ax.plot( np.arange(0,149,1)*s,zdoseprofile*norm_sim,
                                                                            '.',
                                         color=sns.color_palette(  "Paired")[7],
                                                                  Markersize=12,
                label='Simulated Dose Eo={} KeV '.format(energy),
                                                                    zorder=2)






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
    #ax2.set_ylim([0,20])
    #ax2.set_ylabel("LET [KeV/um]",color=sns.color_palette(  "Paired")[5],fontsize=16)
    ax.set_ylabel(" Relative Dose ",color="Green",fontsize=22)
    ax.set_xlabel(" Depth[mm] ",color="black",fontsize=22)
    ax.legend( title='',fontsize=20,loc=4,markerscale=3).set_zorder(0)
    #ax2.legend( title='',fontsize=20,loc=6,markerscale=3).set_zorder(0)
    #ax2.grid(b=True,color='k',linestyle='-',alpha=0.2)
    ax.grid(b=True,color='k',linestyle='dotted',alpha=0.2)
    ax.tick_params(axis='x', which='major', labelsize=22)
    #ax2.tick_params(axis='y', which='major',colors=sns.color_palette(  "Paired")[5], labelsize=22)
    ax.tick_params(axis='y', which='major',colors='green', labelsize=22)

    plt.show()
    return

function4(rcfdepth2,rcfdose2,norm_rcf2,rcferr2,rcfname2,dose2,depth2,err2,norm2,depth_sci,zdoseprofile2,LET_zfluenceprofile2,norm_sim2,D_fluence_mini2,shotnumber2,PC2)
function4(rcfdepth1,rcfdose1,norm_rcf1,rcferr1,rcfname1,dose1,depth1,err1,norm1,depth_sci,zdoseprofile1,LET_zfluenceprofile1,norm_sim1,D_fluence_mini1,shotnumber1,PC1)
function4(rcfdepth,rcfdose,norm_rcf,rcferr,rcfname,dose,depth,err,norm,depth_sci,zdoseprofile,LET_zfluenceprofile,norm_sim,D_fluence_mini,shotnumber,PC)



#PLOT



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









#sub_axes = plt.axes([.6, .6, .25, .25])

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
"""
sub_axes.errorbar( rcfdepth[0:5],                      (rcfdose/area_rcf)[0:5],
                                                        yerr=(rcferr/area_rcf)[0:5],
                                                                      xerr=None,
                                                                        fmt='.',
                                            color=sns.color_palette(  "Paired")[1],
                                                                    markersize=8,
                                                                ecolor='lightblue',
                                                                elinewidth=None)






sub_axes.errorbar( rcfdepth[0:5],               (D_rcf/area_rcfcorrected)[0:5] ,
                                        yerr=(Derr_rcf/area_rcfcorrected)[0:5],
                                                                      xerr=None,
                                                                        fmt='.',
                                          color=sns.color_palette(  "Paired")[5],
                                                                   markersize=8,
                                                            ecolor='lightcoral',
                                                                 elinewidth=None)
sub_axes.grid(b=True,color='k',linestyle='-',alpha=0.2)










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




ax2.plot(depth_sci[0:150],LET_zdoseprofile[0:150],
                                                                            '.',
                                         color=sns.color_palette(  "Paired")[2],
                                                                  Markersize=10,
                                                                   label='LET  dose-weighted')


#ax.set_ylim([0,0.4])
ax.set_xlabel(" Depth [mm] ",color="black",fontsize=16)
ax2.set_ylabel("LET [KeV/um]",color=sns.color_palette(  "Paired")[2],fontsize=16)
ax.set_ylabel(" Relative Dose ",color="blue",fontsize=16)
ax.legend( title='',fontsize=20,loc=4)
ax2.legend( title='',fontsize=20,loc=7)
ax2.grid(b=True,color='k',linestyle='-',alpha=0.2)
ax.grid(b=True,color='k',linestyle='dotted',alpha=0.2)
ax.tick_params(axis='x', which='major', labelsize=16)
ax2.tick_params(axis='y', which='major',colors=sns.color_palette(  "Paired")[2],                                                                   labelsize=16)
ax.tick_params(axis='y', which='major',colors='blue', labelsize=16)




plt.show()


""""DOSE RATE """
MU21CW=1.30
calib6PC1PMMA=3078.69 #mGy/MU
dose21CW=calib6PC1PMMA*MU21CW/1000
ratio=rcfdose.max()/dose21CW
doseMU=np.array([4.01,4.04,4.12,4.26,4.57,5.14])
irrtime=np.array([50.37,24.75,12.53,6.433,3.453,1.942])
doserate=doseMU*ratio/irrtime

directory='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2021-09-02/notnormalized/'
#directory='/home/corvin22/Desktop/miniscidom/pictures/2021-08-13/notnormalized/'
filename0 ='notnormalizedmean_array1.npy'
dose0,depth,tof,numbe0r=read_dose(directory,filename0,s)

err0=read_doserr(directory,'notnormalizederr1.npy') #standard deviation divided by radical n
area0=np.trapz(dose0[3:len(dose0)], depth[3:len(depth)])
#norm0=area_rcf/area0
norm0=1/dose0.max()
doserate0=doserate[0]

filename1 ='notnormalizedmean_array3.npy'
dose1,depth,tof1,number1=read_dose(directory,filename1,s)

err1=read_doserr(directory,'notnormalizederr3.npy') #standard deviation divided by radical n
area1=np.trapz(dose1[3:len(dose1)], depth[3:len(depth)])
#norm1=area_rcf/area1
norm1=1/dose1.max()
doserate1=doserate[1]



filename2 ='notnormalizedmean_array5.npy'
dose2,depth,tof2,number2=read_dose(directory,filename2,s)

err2=read_doserr(directory,'notnormalizederr5.npy') #standard deviation divided by radical n
area2=np.trapz(dose2[3:len(dose2)], depth[3:len(depth)])
#norm2=area_rcf/area2
norm2=1/dose2.max()
doserate2=doserate[2]



filename3 ='notnormalizedmean_array7.npy'
dose3,depth,tof3,number3=read_dose(directory,filename3,s)

err3=read_doserr(directory,'notnormalizederr7.npy') #standard deviation divided by radical n
area3=np.trapz(dose3[3:len(dose3)], depth[3:len(depth)])
#norm3=area_rcf/area3
norm3=1/dose3.max()
doserate3=doserate[3]


filename4 ='notnormalizedmean_array9.npy'
dose4,depth,tof4,number4=read_dose(directory,filename4,s)

err4=read_doserr(directory,'notnormalizederr9.npy') #standard deviation divided by radical n
area4=np.trapz(dose4[4:len(dose4)], depth[4:len(depth)])
#norm4=area_rcf/area4
norm4=1/dose4.max()
doserate4=doserate[4]

err4=read_doserr(directory,'notnormalizederr9.npy') #standard deviation divided by radical n
area4=np.trapz(dose4[4:len(dose4)], depth[4:len(depth)])

norm4=1/dose4.max()
doserate4=doserate[4]

filename5 ='notnormalizedmean_array15.npy'
dose5,depth,tof5,number5=read_dose(directory,filename5,s)

err5=read_doserr(directory,'notnormalizederr15.npy') #standard deviation divided by radical n
area5=np.trapz(dose5[4:len(dose5)], depth[4:len(depth)])
#norm5=area_rcf/area5
norm5=1/dose5.max()
doserate5=doserate[5]


D_rcf,Derr_rcf,area_rcfcorrected=function1(rcfdepth,rcfdose,rcferr,depth_sci,LET_zdoseprofile,a,b)

D_a_mini0,D_fluence_mini0=function2(dose0,depth,depth_sci,LET_zdoseprofile,LET_zfluenceprofile,s,area_rcfcorrected)
D_a_mini1,D_fluence_mini1=function2(dose1,depth,depth_sci,LET_zdoseprofile,LET_zfluenceprofile,s,area_rcfcorrected)
D_a_mini2,D_fluence_mini2=function2(dose2,depth,depth_sci,LET_zdoseprofile,LET_zfluenceprofile,s,area_rcfcorrected)
D_a_mini3,D_fluence_mini3=function2(dose3,depth,depth_sci,LET_zdoseprofile,LET_zfluenceprofile,s,area_rcfcorrected)
D_a_mini4,D_fluence_mini4=function2(dose4,depth,depth_sci,LET_zdoseprofile,LET_zfluenceprofile,s,area_rcfcorrected)
D_a_mini5,D_fluence_mini5=function2(dose5,depth,depth_sci,LET_zdoseprofile,LET_zfluenceprofile,s,area_rcfcorrected)
def plotfunction(depth,dose,err,norm,doserate,D_fluence_mini,color1,s):

    ax.errorbar(np.arange(0,len(dose),1)*s,                              dose*norm,
                                                                   yerr=err*norm,
                                                                       xerr=None,
                                                                         fmt='.',
                                                                         markersize=9,
                                                                         color=color1,
                                                                         ecolor=color1,
                                                                 elinewidth=None,
                                             label=f' Doserate={doserate:.2f}Gy/s')
    ax.plot(depth,
                                                                        D_fluence_mini,
                                                                                '-',
                                            color=color1,
                                        label='LET corrected dose',
                                                                      #Markersize=9
                                                                      )


    return

plt.figure(1)
fig, ax = plt.subplots()
plt.title('Does quenching depend on dose rate ?(6PC 1PMMA)',fontsize=24)
plotfunction(depth,dose0,err0,norm0,doserate0,D_fluence_mini0,sns.color_palette(  "Paired")[3],s)
plotfunction(depth,dose1,err1,norm1,doserate1,D_fluence_mini1,sns.color_palette(  "Paired")[10],s)
plotfunction(depth,dose2,err2,norm2,doserate2,D_fluence_mini2,sns.color_palette(  "Paired")[5],s)
plotfunction(depth,dose3,err3,norm3,doserate3,D_fluence_mini3,sns.color_palette(  "Paired")[6],s)
plotfunction(depth,dose4,err4,norm4,doserate4,D_fluence_mini4,sns.color_palette(  "Paired")[7],s)
plotfunction(depth,dose5,err5,norm5,doserate5,D_fluence_mini5,sns.color_palette(  "Paired")[8],s)
#plotfunction(dose1,err1,norm1,doserate1,sns.color_palette(  "Paired")[4],s)
normrcf=1/rcfdose.max()
ax.errorbar( rcfdepth,                      rcfdose*normrcf ,
                                                                  yerr=rcferr*normrcf ,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                        color=sns.color_palette(  "Paired")[1],
                                                                        markersize=9,
                                                                   ecolor=sns.color_palette(  "Paired")[0],
                                                                elinewidth=None,
                                                                label=' RCF 21CW measured dose')
ax.fill_between(rcfdepth,
                                                                rcfdose*normrcf -rcferr*normrcf ,
                                                                rcfdose*normrcf +rcferr*normrcf ,
                                                        color=sns.color_palette(  "Paired")[0], alpha=0.2)


sub_axes = plt.axes([.3, .3, .25, .25])

sub_axes.errorbar( np.arange(0,len(dose0),1)[10:40]*s,
                                                            (dose0*norm0)[10:40],
                                                        yerr=(err0*norm0)[10:40],
                                                                       xerr=None,
                                                                         fmt='.',
                                          color=sns.color_palette(  "Paired")[3],
                                         ecolor=sns.color_palette(  "Paired")[3],
                                                                 elinewidth=None)






sub_axes.errorbar( np.arange(0,len(dose1),1)[10:40]*s,
                                                            (dose1*norm1)[10:40],
                                                        yerr=(err1*norm1)[10:40],
                                                                       xerr=None,
                                                                         fmt='.',
                                          color=sns.color_palette(  "Paired")[1],
                                         ecolor=sns.color_palette(  "Paired")[1],
                                                                 elinewidth=None)

sub_axes.errorbar( np.arange(0,len(dose2),1)[10:40]*s,
                                                            (dose2*norm2)[10:40],
                                                        yerr=(err2*norm2)[10:40],
                                                                       xerr=None,
                                                                         fmt='.',
                                          color=sns.color_palette(  "Paired")[5],
                                         ecolor=sns.color_palette(  "Paired")[5],
                                                                 elinewidth=None)

sub_axes.errorbar( np.arange(0,len(dose3),1)[10:40]*s,
                                                            (dose3*norm3)[10:40],
                                                        yerr=(err3*norm3)[10:40],
                                                                       xerr=None,
                                                                         fmt='.',
                                          color=sns.color_palette(  "Paired")[6],
                                         ecolor=sns.color_palette(  "Paired")[6],
                                                                 elinewidth=None)

sub_axes.errorbar( np.arange(0,len(dose4),1)[10:40]*s,
                                                            (dose4*norm4)[10:40],
                                                        yerr=(err4*norm4)[10:40],
                                                                       xerr=None,
                                                                         fmt='.',
                                          color=sns.color_palette(  "Paired")[7],
                                         ecolor=sns.color_palette(  "Paired")[7],
                                                                 elinewidth=None)




sub_axes.errorbar( np.arange(0,len(dose5),1)[10:40]*s,
                                                            (dose5*norm5)[10:40],
                                                        yerr=(err5*norm5)[10:40],
                                                                       xerr=None,
                                                                         fmt='.',
                                          color=sns.color_palette(  "Paired")[8],
                                         ecolor=sns.color_palette(  "Paired")[8],
                                                                 elinewidth=None)








sub_axes.grid(b=True,color='k',linestyle='-',alpha=0.2)
mark_inset(ax, sub_axes, loc1=2, loc2=3, fc="none", ec="0.5")





ax.set_xlim([0,9])
ax.set_xlabel('mm',fontsize=20)
ax.set_ylabel('Relative Dose',fontsize=20)
ax.grid(b=True,color='k',linestyle='dotted',alpha=0.2)
ax.tick_params(axis='x', which='major', labelsize=20)
ax.tick_params(axis='y', which='major', labelsize=20)
ax.legend( title='doserate=doseMU*0.7/irrtime',fontsize=18,loc=4,markerscale=3,title_fontsize=20)
plt.show()
