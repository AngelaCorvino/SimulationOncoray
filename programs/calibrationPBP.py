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


def readtopasoutput(directory,filename,filenameLETdose,filenameLETfluence,nz,nx):
    outputfile_topas=directory+filename
    header = pd.read_csv(outputfile_topas, nrows = 7)
    df = pd.read_csv(outputfile_topas, comment='#', header=None)
    topas_datamatrix = np.array(df)# convert dataframe df to array
    doseprofile=Get_profile(topas_datamatrix, nz,nx)
    zdoseprofile=doseprofile.zmeanprofile
    zdoseprofile=zdoseprofile[::-1]
    (dir,energy,scoringvolume,PC,dimension)= outputfile_topas.split('_')
    (energy,particle)=energy.split('KeV')
    depth_sci= np.arange(0,len(zdoseprofile),1)*s
    area_sim=np.trapz(zdoseprofile[3:len(zdoseprofile)-3],depth_sci[3:len(depth_sci)-3])
    norm_sim=1/zdoseprofile.max()
    #norm_sim=1/area_sim

    outputfile_topasLET=directory+filenameLETdose
    header = pd.read_csv(outputfile_topasLET, nrows = 7)
    df = pd.read_csv(outputfile_topasLET, comment='#', header=None)
    topas_datamatrix= np.array(df)# convert dataframe df to array
    LET_doseprofile=Get_profile(topas_datamatrix, 149,1)
    LET_zdoseprofile=LET_doseprofile.zmeanprofile
    LET_zdoseprofile=LET_zdoseprofile[::-1]
    LET_zdoseprofile[0]=LET_zdoseprofile[1]
    area_LET=np.trapz(LET_zdoseprofile,depth_sci)

    outputfile_topasLETfluence=directory+filenameLETfluence
    header = pd.read_csv(outputfile_topasLETfluence, nrows = 7)
    df = pd.read_csv(outputfile_topasLETfluence, comment='#', header=None)
    topas_datamatrix= np.array(df)# convert dataframe df to array
    LET_fluenceprofile=Get_profile(topas_datamatrix, 149,1)
    LET_zfluenceprofile=LET_fluenceprofile.zmeanprofile
    LET_zfluenceprofile=LET_zfluenceprofile[::-1]
    LET_zfluenceprofile[0]=LET_zfluenceprofile[1]
    area_f_LET=np.trapz(LET_zfluenceprofile,depth_sci)


    return depth_sci,zdoseprofile,LET_zdoseprofile,LET_zfluenceprofile,area_LET,area_f_LET,norm_sim,energy,PC

directory="/Users/angelacorvino/Documents/GitHub/SimulationOncoray/data/Single/"
depth_sci,zdoseprofile,LET_zdoseprofile,LET_zfluenceprofile,area_LET,area_f_LET,norm_sim,energy,PC=readtopasoutput(directory,
                            'DoseToWater_9160KeVproton_PVT_6PC1PMMA_1Dscorer.csv',
                       'LET_doseweighted_9160KeVproton_PVT_6PC1PMMA_1Dscorer.csv',
                    'LET_fluenceweighted_9160KeVproton_PVT_6PC1PMMA_1Dscorer.csv',
                                                                            149,1)
dose_sci=zdoseprofile


directory1='/Users/angelacorvino/Desktop/TOPASpreciousimulation/Single/10°7/secondcollimator/'
depth_sci1,zdoseprofile1,LET_zdoseprofile1,LET_zfluenceprofile1,area_LET1,area_f_LET1,norm_sim1,energy1,PC1=readtopasoutput(directory1,
                               'DoseToWater_92100KeVproton_PVT_7PC_1Dscorer.csv',
                          'LET_doseweighted_92100KeVproton_PVT_7PC_1Dscorer.csv',
                       'LET_fluenceweighted_92100KeVproton_PVT_7PC_1Dscorer.csv',
                                                                           149,1)
dose_sci1=zdoseprofile1

directory2='/Users/angelacorvino/Documents/GitHub/SimulationOncoray/data/Single/'
depth_sci2,zdoseprofile2,LET_zdoseprofile2,LET_zfluenceprofile2,area_LET2,area_f_LET2,norm_sim2,energy2,PC2=readtopasoutput(directory2,
                               'DoseToWater_9210KeVproton_PVT_6PC_1Dscorer.csv',
                               'LET_doseweighted_9210KeVproton_PVT_6PC_1Dscorer.csv',
                               'LET_fluenceweighted_9210KeVproton_PVT_6PC_1Dscorer.csv',
                                                                          149,1)
dose_sci2=zdoseprofile2


#############################################################################
"""Normalization"""

norm_rcf=1/rcfdose.max()
norm_rcf1=1/rcfdose1.max()    #area_rcf
norm_rcf2=1/rcfdose2.max()
norm=1/dose.max()#area
norm1=1/dose1.max()#area
norm2=1/dose2.max()#area


#norm_rcf=1/area_rcf
#norm_rcf1=1/area_rcf1
#norm_rcf2=1/area_rcf2







""" correction of RCF measure with LET (doseweighted ) simulations in scintillator"""
a=-0.0251
b= 1.02

def rcfcorrection(rcfdepth,rcfdose,rcferr,depth_sci,LET_zdoseprofile,a,b):
    S=np.interp(np.arange(0,len(rcfdose),1)*s,depth_sci,LET_zdoseprofile)
    D_rcf=rcfdosecorrection(rcfdose,S,a,b)
    Derr_rcf=rcfdosecorrection(rcferr,S,a,b)
    area_rcfcorrected=np.trapz(rcfdosecorrection(rcfdose,S,a,b),rcfdepth)
    return D_rcf,Derr_rcf,area_rcfcorrected

D_rcf,Derr_rcf,area_rcfcorrected=rcfcorrection(rcfdepth,rcfdose,rcferr,depth_sci,LET_zdoseprofile,a,b)
D_rcf1,Derr_rcf1,area_rcfcorrected1=rcfcorrection(rcfdepth1,rcfdose1,rcferr1,depth_sci,LET_zdoseprofile1,a,b)
D_rcf2,Derr_rcf2,area_rcfcorrected2=rcfcorrection(rcfdepth2,rcfdose2,rcferr2,depth_sci,LET_zdoseprofile2,a,b)

#norm_rcfcorrected=1./D_rcf.max()
#norm_rcfcorrected1=1./D_rcf1.max()
#norm_rcfcorrected2=1./D_rcf2.max()

norm_rcfcorrected=1./area_rcfcorrected
norm_rcfcorrected1=1./area_rcfcorrected1
norm_rcfcorrected2=1./area_rcfcorrected2
"""LET correction simulations in scintillator """

dS=0 #theoretical values
dscintillator= 1.023 #[g/cm^3] scintillator density
dactivelayer=1.2 #[g/cm^3]
k=207.0/10000 #[g/MeV cm^2]
a=0.9 #scintillator efficiency
dx= 65 #µm scintillator spatial resolution
ddx=1 #spatial resolution error
k=k*10#[micrometers/kev]
#SIMULATED LET IN Scintillator starting from TOF of a scidom shot

def birkcorrection(dose,depth,depth_sci,LET_zdoseprofile,LET_zfluenceprofile,s,area_rcfcorrected):
    ys_ana_mini=LET_zdoseprofile
    ys_fluence_mini=LET_zfluenceprofile

    S_dose_mini=np.interp(np.arange(0,len(dose),1)*s,depth_sci,ys_ana_mini)
    #S_a_low_mini=np.interp(np.arange(0,len(dose),1)*s,depth_sci,ys_ana_lower_mini)
    #S_a_up_mini=np.interp(np.arange(0,len(dose),1)*s,depth_sci,ys_ana_upper_mini)
    S_fluence_mini=np.interp(np.arange(0,len(dose),1)*s,depth_sci,ys_fluence_mini)


    area_corrected=np.trapz(dosecorrection(dose,S_dose_mini,a,k,dx)[3:len(dosecorrection(dose,S_dose_mini,a,k,dx))],
                                                                        depth[3:len(depth)]) #eliminate first 4 points
    area_f_corrected=np.trapz(dosecorrection(dose,S_fluence_mini,a,k,dx)[3:len(dosecorrection(dose,S_fluence_mini,a,k,dx))-12],
                                                                        depth[3:len(depth)-12]) #eliminate first 4 points
    norm_dose_mini=area_rcfcorrected/area_corrected
    #norm_fluence_mini=1/area_f_corrected
    norm_fluence_mini=1/dosecorrection(dose,S_fluence_mini,a,k,dx).max()
    #norm_fluence_mini=1/dosecorrection(dose,S_fluence_mini,a,k,dx)[10]

    D_dose_mini=dosecorrection(dose,S_dose_mini,a,k,dx)
    #D_a_up_mini=dosecorrection(dose,S_a_up_mini,a,k,dx)*norm_a_mini
    #D_a_low_mini=dosecorrection(dose,S_a_low_mini,a,k,dx)*norm_a_mini
    D_fluence_mini=dosecorrection(dose,S_fluence_mini,a,k,dx)
    D_fluence_mini=D_fluence_mini*norm_fluence_mini
    D_dose_mini=D_dose_mini*norm_dose_mini
    #Dabsolute_a_mini=dosecorrection(dose,S_dose_mini,a,k,dx)*norm_dose_mini* L_fluence_mini(S_dose_mini,a,k,dx)
    L_fluence_mini=lightcorrection(S_fluence_mini,a,k,dx)
    #L_a_low_mini=lightcorrection(S_a_low_mini,a,k,dx)

    Dabsolute_fluence_mini=dosecorrection(dose,S_fluence_mini,a,k,dx)*norm_dose_mini* lightcorrection(S_fluence_mini,a,k,dx)


    return D_dose_mini,D_fluence_mini,L_fluence_mini

D_dose_mini2,D_fluence_mini2,L_fluence_mini2=birkcorrection(dose2,depth2,depth_sci,LET_zdoseprofile2,LET_zfluenceprofile2,s,area_rcfcorrected2)
D_dose_mini1,D_fluence_mini1,L_fluence_mini1=birkcorrection(dose1,depth1,depth_sci,LET_zdoseprofile1,LET_zfluenceprofile1,s,area_rcfcorrected1)
D_dose_mini,D_fluence_mini,L_fluence_mini=birkcorrection(dose,depth,depth_sci,LET_zdoseprofile,LET_zfluenceprofile,s,area_rcfcorrected)






###############################################################################
"""Plot Functions"""

def plotfunction(depth,dose,err,norm,rcfdepth,rcfdose,rcferr,norm_rcf,color1,color2,PC,rcfname,s):


    plt.errorbar(depth,                                               dose*norm,
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


def plotfunction4(rcfdepth,rcfdose,norm_rcf,rcferr,rcfname,dose,depth,err,norm,depth_sci,zdoseprofile,LET_zfluenceprofile,norm_sim,D_fluence_mini,shotnumber,PC,L_fluence_mini):
    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.70)

    twin1 = ax.twinx()
    twin2 = ax.twinx()

    twin2.spines['right'].set_position(("axes", 1.2))
    plt.rc('text', usetex=True)

    ax.plot(depth,
                                                                D_fluence_mini,
                                                                                '.',
                                            color=sns.color_palette(  "Paired")[10],
                                                     label=r'$D_{quenchingcorrection}$',
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
                                                 label=r'$D_{}$'.format(rcfname),
                                                                                 zorder=4)

    ax.fill_between(rcfdepth,
                                                rcfdose*norm_rcf-rcferr*norm_rcf,
                                                rcfdose*norm_rcf+rcferr*norm_rcf,
                                          color=sns.color_palette("Paired")[1],
                                                                       alpha=0.1)

    ax.errorbar(  np.arange(0,len(dose),1)*s,                              dose*norm,
                                                                   yerr=err*norm,
                                                                       xerr=None,
                                                                         fmt='.',
                                                                   markersize=12,
                                           color=sns.color_palette("Paired")[3],
                                          ecolor=sns.color_palette("Paired")[3],
                                                                 elinewidth=None,
                                             label=r'$D_{miniscidom} $',
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

    twin1.plot(depth_sci[0:100],LET_zfluenceprofile[0:100],
                                                                            '.',
                                            color=sns.color_palette(  "Paired")[4],
                                                                   Markersize=13,
                                               label=r'$LET_{fluence}$',
                                                                 zorder=0)


    ax.plot( np.arange(0,149,1)*s,zdoseprofile*norm_sim,
                                                                            '.',
                                         color=sns.color_palette(  "Paired")[7],
                                                                  Markersize=12,
                                                         label=r'$D_{simulated}$',
                                                                    zorder=2)


    twin2.plot(depth_sci[0:100],(1+k*LET_zfluenceprofile)[0:100],
                                                                            '^',
                                                                   markersize=5,
                                            color=sns.color_palette("Paired")[9],
                                                             label='(1+KB*LET)',
                                                                     zorder=1)



    """
    ax.plot(depth,
                                                        D_dose_mini/D_dose_mini.max(),
                                                                            '.',
                                          color=sns.color_palette(  "Paired")[9],
                                        label='LET(dose weighted) corrected dose',
                                                                  Markersize=11)
    """






    plt.rc('text', usetex=False)


    plt.title('Depth-dose distribution shape comparison {} shot n.{}'.format(PC,shotnumber),
                                                                    fontsize=24,
                                                                  fontdict=None,
                                                                  loc='center',
                                                                       pad=None)
    #twin2.set_ylim([LET_zfluenceprofile.min()[0:100],LET_zfluenceprofile[0:100].max()])

    #twin2.set_yticks(np.arange(min((1+k*LET_zfluenceprofile)[0:100]), max((1+k*LET_zfluenceprofile)[0:100]), 30))
    twin2.set_yticks([1.8,2,2.2,2.4,2.6,2.8,3,3.2,3.4,3.6,3.8])
    #twin2.set_ylim([2,4])
    twin2.set_ylabel("a.u.",color=sns.color_palette(  "Paired")[9],fontsize=22)
    twin2.tick_params(axis='y', which='major',colors=sns.color_palette(  "Paired")[9], labelsize=22)
    twin2.legend( title='',fontsize=20,loc=1,markerscale=3).set_zorder(0)

    twin1.set_ylabel("LET [KeV/um]",color=sns.color_palette(  "Paired")[5],fontsize=22)
    twin1.legend( title='',fontsize=20,loc=6,markerscale=3)
    twin1.grid(b=True,color='k',linestyle='-',alpha=0.2)
    twin1.tick_params(axis='y', which='major',colors=sns.color_palette(  "Paired")[5], labelsize=22)

    ax.set_ylabel(" Relative Dose ",color="Green",fontsize=22)
    ax.set_xlabel(" Depth[mm] ",color="black",fontsize=22)
    ax.legend( title='',fontsize=20,loc=5,markerscale=3)
    ax.grid(b=True,color='k',linestyle='dotted',alpha=0.2)
    ax.tick_params(axis='x', which='major', labelsize=22)
    ax.tick_params(axis='y', which='major',colors='green', labelsize=22)

    plt.show()
    return


def plotfunction5(rcfdepth,rcfdose,rcferr,norm_rcf,D_rcf,Derr_rcf,norm_rcfcorrected,depth_sci,LET_zdoseprofile,rcfname):
    fig, ax = plt.subplots()
    plt.title('Shape comparison {} '.format(rcfname),fontsize=22)
    ax2 = ax.twinx()



    ax.errorbar( rcfdepth,                                          rcfdose*norm_rcf,
                                                             yerr=rcferr*norm_rcf,
                                                                      xerr=None,
                                                                        fmt='.',
                                            color=sns.color_palette(  "Paired")[1],
                                                                    markersize=10,
                                         ecolor=sns.color_palette(  "Paired")[1],
                                                                elinewidth=None,
                                                label='{} measured dose'.format(rcfname))

    ax.fill_between(rcfdepth,
                                                                rcfdose*norm_rcf-rcferr*norm_rcf,
                                                                rcfdose*norm_rcf+rcferr*norm_rcf,
                                                        color=sns.color_palette(  "Paired")[1],
                                                         alpha=0.3)


    ax.errorbar(rcfdepth,                                  D_rcf*norm_rcfcorrected ,
                                                 yerr=Derr_rcf*norm_rcfcorrected,
                                                                      xerr=None,
                                                                        fmt='.',
                                          color=sns.color_palette(  "Paired")[5],
                                                                   markersize=10,
                                        ecolor=sns.color_palette(  "Paired")[5],
                                                                 elinewidth=None,
                              label='LET(dose weighted) corrected {} dose'.format(rcfname))


    ax.fill_between(rcfdepth,
                              D_rcf*norm_rcfcorrected-Derr_rcf*norm_rcfcorrected,
                              D_rcf*norm_rcfcorrected+Derr_rcf*norm_rcfcorrected,
                                          color=sns.color_palette( "Paired")[5],
                                                                       alpha=0.1)



    sub_axes = plt.axes([.6, .6, .25, .25])

# plot the zoomed portion


    """
    sub_axes.plot( rcfdepth[0:5],                      (rcfdose*norm_rcf)[0:5],
                                                                        '.',
                                            color=sns.color_palette(  "Paired")[1],
                                                                    markersize=8)


    sub_axes.plot( rcfdepth[0:5],                       (D_rcf*norm_rcfcorrected)[0:5],
                                                                        '.',
                                            color=sns.color_palette(  "Paired")[5],
                                                                    markersize=8)

    """
    sub_axes.grid(b=True,color='k',linestyle='-',alpha=0.2)



    sub_axes.errorbar( rcfdepth[0:5],                      (rcfdose*norm_rcf)[0:5],
                                                        yerr=(rcferr*norm_rcf)[0:5],
                                                                      xerr=None,
                                                                        fmt='.',
                                            color=sns.color_palette(  "Paired")[1],
                                                                    markersize=8,
                                                                ecolor='lightblue',
                                                                elinewidth=None)


    sub_axes.errorbar( rcfdepth[0:5],               (D_rcf*norm_rcfcorrected)[0:5] ,
                                        yerr=(Derr_rcf*norm_rcfcorrected)[0:5],
                                                                      xerr=None,
                                                                        fmt='.',
                                          color=sns.color_palette(  "Paired")[5],
                                                                   markersize=8,
                                                            ecolor='lightcoral',
                                                                 elinewidth=None)


    """
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




    ax2.plot(depth_sci[0:140],LET_zdoseprofile[0:140],
                                                                            '.',
                                         color=sns.color_palette(  "Paired")[2],
                                                                  Markersize=12,
                                                                   label='LET  dose-weighted')

    ax2.set_ylim([0,20])
    mark_inset(ax, sub_axes, loc1=2, loc2=3, fc="none", ec="0.5")
    #ax.set_ylim([0,0.4])
    ax.set_xlabel(" Depth [mm] ",color="black",fontsize=20)
    ax2.set_ylabel("LET [KeV/um]",color=sns.color_palette(  "Paired")[3],fontsize=16)
    ax.set_ylabel(" Relative Dose ",color="blue",fontsize=20)
    ax.legend( title='',fontsize=20,loc=4,markerscale=3)
    ax2.legend( title='',fontsize=20,loc=7,markerscale=3)
    ax2.grid(b=True,color='k',linestyle='-',alpha=0.2)
    ax.grid(b=True,color='k',linestyle='dotted',alpha=0.2)
    ax.tick_params(axis='x', which='major', labelsize=20)
    ax2.tick_params(axis='y', which='major',colors=sns.color_palette(  "Paired")[3], labelsize=16)
    ax.tick_params(axis='y', which='major',colors='blue', labelsize=18)
    plt.show()
    return

def plotbackground(rcfdepth,rcfdose,norm_rcf,rcferr,rcfname,dose,depth,err,norm,depth_sci,zdoseprofile,LET_zfluenceprofile,norm_sim,D_fluence_mini,shotnumber,PC,L_fluence_mini):

    D_fluence_mini_int=np.interp(rcfdepth,depth,D_fluence_mini)

    fig, ax = plt.subplots()


    twin1 = ax.twinx()

    plt.rc('text', usetex=True)

    ax.plot(depth,
                                                                D_fluence_mini,
                                                                                '.',
                                            color=sns.color_palette(  "Paired")[10],
                                                     label=r'$D_{quenchingcorrection}$',
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
                                                 label=r'$D_{RCF}$',
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
                                             label=r'$D_{miniscidom} $',
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




    """
    ax.plot(depth,
                                                        D_dose_mini/D_dose_mini.max(),
                                                                            '.',
                                          color=sns.color_palette(  "Paired")[9],
                                        label='LET(dose weighted) corrected dose',
                                                                  Markersize=11)
    """



    twin1.plot(rcfdepth,np.abs((D_fluence_mini_int-rcfdose*norm_rcf)/rcfdose*norm_rcf)*100,
                                                                            '.',
                                            color=sns.color_palette(  "Paired")[4],
                                                                   Markersize=13,
                                               label=r'|$\frac{D_{quenchingcorrection}-D_{RCF}}{D_{RCF}}$|',
                                                                 zorder=0)


    plt.rc('text', usetex=False)


    plt.title('Depth-dose distribution shape comparison {}'.format(PC,shotnumber),
                                                                    fontsize=24,
                                                                  fontdict=None,
                                                                  loc='center',
                                                                       pad=None)


    ax.set_ylabel(" Relative Dose ",color="Green",fontsize=22)
    ax.set_xlabel(" Depth[mm] ",color="black",fontsize=22)
    ax.legend( title='',fontsize=20,loc=2,markerscale=3)
    ax.grid(b=True,color='k',linestyle='dotted',alpha=0.2)
    ax.tick_params(axis='x', which='major', labelsize=22)
    ax.tick_params(axis='y', which='major',colors='green', labelsize=22)
    twin1.set_ylabel("Deviation [$\%$]",color=sns.color_palette(  "Paired")[5],fontsize=22)
    twin1.legend( title='',fontsize=20,loc=6,markerscale=3)
    twin1.grid(b=True,color='k',linestyle='-',alpha=0.2)
    twin1.tick_params(axis='y', which='major',colors=sns.color_palette(  "Paired")[5], labelsize=22)


    plt.show()
    return
###############################################################################
"""PLOTS"""


plt.figure(1)

plotfunction(depth,dose,err,norm,rcfdepth,rcfdose,rcferr,norm_rcf,sns.color_palette("Paired")[0],sns.color_palette(  "Paired")[1],PC,rcfname,s)
plotfunction(depth1,dose1,err1,norm1,rcfdepth1,rcfdose1,rcferr1,norm_rcf1,sns.color_palette(  "Paired")[2],sns.color_palette(  "Paired")[3],PC1,rcfname1,s)
plotfunction(depth2,dose2,err2,norm2,rcfdepth2,rcfdose2,rcferr2,norm_rcf2,sns.color_palette(  "Paired")[8],sns.color_palette(  "Paired")[9],PC2,rcfname2,s)

plt.title('Depth-dose distribution shape comparison',fontsize=26)
plt.xlabel(" Depth[mm] ",fontsize=24)
plt.ylabel('Relative Dose in water ',fontsize=24)
#plt.legend(title='E_{primary}=92.10 MeV',fontsize=24,loc=3,markerscale=2,title_fontsize=24)
plt.legend(title='',fontsize=24,loc=3,markerscale=3,title_fontsize=20)
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

plotfunction2(depth_sci,dose_sci,norm_sim,LET_zfluenceprofile,sns.color_palette("Paired")[1],PC,190)
plotfunction2(depth_sci,dose_sci1,norm_sim1,LET_zfluenceprofile1,sns.color_palette("Paired")[3],PC1,190)
plotfunction2(depth_sci,dose_sci2,norm_sim2,LET_zfluenceprofile2,sns.color_palette("Paired")[9],PC2,150)

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

plotfunction4(rcfdepth2,rcfdose2,norm_rcf2,rcferr2,rcfname2,dose2,depth2,err2,norm2,depth_sci,zdoseprofile2,LET_zfluenceprofile2,norm_sim2,D_fluence_mini2,shotnumber2,PC2,L_fluence_mini2)
plotfunction4(rcfdepth1,rcfdose1,norm_rcf1,rcferr1,rcfname1,dose1,depth1,err1,norm1,depth_sci,zdoseprofile1,LET_zfluenceprofile1,norm_sim1,D_fluence_mini1,shotnumber1,PC1,L_fluence_mini1)
plotfunction4(rcfdepth,rcfdose,norm_rcf,rcferr,rcfname,dose,depth,err,norm,depth_sci,zdoseprofile,LET_zfluenceprofile,norm_sim,D_fluence_mini,shotnumber,PC,L_fluence_mini)

################### PLOT 6PC BACKGROUND ISSUE
"""

norm_rcf=1/rcfdose[0]
norm_rcf1=1/rcfdose1[0]    #area_rcf
norm_rcf2=1/rcfdose2[0]
norm=1/dose[10]#area
norm1=1/dose1[10]#area
norm2=1/dose2[10]#area


plotbackground(rcfdepth2,rcfdose2,norm_rcf2,rcferr2,rcfname2,dose2,depth2,err2,norm2,depth_sci,zdoseprofile2,LET_zfluenceprofile2,norm_sim2,D_fluence_mini2,shotnumber2,PC2,L_fluence_mini2)
plotbackground(rcfdepth1,rcfdose1,norm_rcf1,rcferr1,rcfname1,dose1,depth1,err1,norm1,depth_sci,zdoseprofile1,LET_zfluenceprofile1,norm_sim1,D_fluence_mini1,shotnumber1,PC1,L_fluence_mini1)
plotbackground(rcfdepth,rcfdose,norm_rcf,rcferr,rcfname,dose,depth,err,norm,depth_sci,zdoseprofile,LET_zfluenceprofile,norm_sim,D_fluence_mini,shotnumber,PC,L_fluence_mini)




##########PLOT RCF LET CORRECTION

plotfunction5(rcfdepth,rcfdose,rcferr,norm_rcf,D_rcf,Derr_rcf,norm_rcfcorrected,depth_sci,LET_zdoseprofile,rcfname)
plotfunction5(rcfdepth1,rcfdose1,rcferr1,norm_rcf1,D_rcf1,Derr_rcf1,norm_rcfcorrected1,depth_sci,LET_zdoseprofile1,rcfname1)
plotfunction5(rcfdepth2,rcfdose2,rcferr2,norm_rcf2,D_rcf2,Derr_rcf2,norm_rcfcorrected2,depth_sci,LET_zdoseprofile2,rcfname2)
"""
