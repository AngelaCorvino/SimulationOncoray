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
from RCFresponse import rcfcorrection
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from read2Dtopasoutput import readtopasoutput

###############################################################################

#RCF data


directory='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2021-09-01/'
filename2='RCF21CT.csv'
filename1='RCF21CR.csv'
filename='RCF21CP.csv'
rcfdepth,rcfdose,rcferr,area_rcf,rcfname=read_datarcf(directory,filename)
rcfdepth1,rcfdose1,rcferr1,area_rcf1,rcfname1=read_datarcf(directory,filename1)
rcfdepth2,rcfdose2,rcferr2,area_rcf2,rcfname2=read_datarcf(directory,filename2)





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



########################################################################################################################################


directory='/Users/angelacorvino/Desktop/TOPASpreciousimulation/SOBP/10PC/152.5MeV/'
depth_sci,zdoseprofile,LET_zdoseprofile,LET_zfluenceprofile,area_LET,area_f_LET,norm_sim,energy,PC=readtopasoutput(directory,
                            'DoseToWater_152500KeVproton_PVT_10PC_1Dscorer.csv',
                       'LET_fluenceweighted_152500KeVproton_PVT_10PC_1Dscorer.csv',
                    'LET_fluenceweighted_152500KeVproton_PVT_10PC_1Dscorer.csv',
                                                                            149,1,
                                                                            'KeV')
dose_sci=zdoseprofile
area_sim=np.trapz(zdoseprofile[3:len(zdoseprofile)-8],depth_sci[3:len(depth_sci)-8])
area_f_LET=np.trapz(LET_zfluenceprofile,depth_sci)




directory='/Users/angelacorvino/Desktop/TOPASpreciousimulation/SOBP/11PC/152.5MeV107/secondcollimator/'
depth_sci1,zdoseprofile1,LET_zdoseprofile1,LET_zfluenceprofile1,area_LET1,area_f_LET1,norm_sim1,energy1,PC1=readtopasoutput(directory,
                                'DoseToWater_152500KeVproton_PVT_11PC_1Dscorer.csv',
                                'LET_doseweighted_152500KeVproton_PVT_11PC_1Dscorer.csv',
                                'LET_fluenceweighted_152500KeVproton_PVT_11PC_1Dscorer.csv',
                                                                        149,1,
                                                                        'KeV')
dose_sci1=zdoseprofile1
depth_sci1= np.arange(0,len(zdoseprofile1),1)*s
area_sim1=np.trapz(zdoseprofile1[0:len(zdoseprofile1)-8],depth_sci1[0:len(depth_sci1)-8]) #eliminate last 3 points
area_f_LET1=np.trapz(LET_zfluenceprofile1,depth_sci1)





directory='/Users/angelacorvino/Desktop/TOPASpreciousimulation/SOBP/12PC/152.5MeV/secondcollimator/'
depth_sci2,zdoseprofile2,LET_zdoseprofile2,LET_zfluenceprofile2,area_LET2,area_f_LET2,norm_sim2,energy2,PC2=readtopasoutput(directory,
                                'DoseToWater_152500KeVproton_PVT_12PC_1Dscorer.csv',
                                'LET_doseweighted_152500KeVproton_PVT_12PC_1Dscorer.csv',
                                'LET_fluenceweighted_152500KeVproton_PVT_12PC_1Dscorer.csv',
                                                                        149,1,
                                                                        'KeV')

dose_sci2=zdoseprofile2
depth_sci2= np.arange(0,len(zdoseprofile2),1)*s
area_sim2=np.trapz(zdoseprofile2[0:len(zdoseprofile2)-8],depth_sci2[0:len(depth_sci2)-8]) #eliminate last 3 points
area_f_LET2=np.trapz(LET_zfluenceprofile2,depth_sci2)


################################################################################
norm_rcf=1/rcfdose.max()
norm_rcf1=1/rcfdose1.max()    #area_rcf
norm_rcf2=1/rcfdose2.max()
norm=1/dose.max()#area
norm1=1/dose1.max()
norm2=1/dose2.max()#
norm_sim=1/zdoseprofile.max() #area_sim
norm_sim1=1/zdoseprofile1[7] #area_sim
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


""" correction of RCF measurements with LET (doseweighted ) simulations in scintillator"""

a=-0.0251
b= 1.02

D_rcf,Derr_rcf,area_rcfcorrected=rcfcorrection(rcfdepth,rcfdose,rcferr,depth_sci,LET_zdoseprofile,a,b,s)
D_rcf1,Derr_rcf1,area_rcfcorrected1=rcfcorrection(rcfdepth,rcfdose1,rcferr1,depth_sci1,LET_zdoseprofile1,a,b,s)
D_rcf2,Derr_rcf2,area_rcfcorrected2=rcfcorrection(rcfdepth2,rcfdose2,rcferr2,depth_sci2,LET_zdoseprofile2,a,b,s)
####################################################################################



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
    S_a_mini=np.interp(np.arange(0,len(dose),1)*s,depth_sci,ys_ana_mini)
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

D_a_mini2,D_fluence_mini2=birkcorrection(dose2,depth2,depth_sci2,LET_zdoseprofile2,LET_zfluenceprofile2,s,area_rcfcorrected2)
D_a_mini1,D_fluence_mini1=birkcorrection(dose1,depth1,depth_sci1,LET_zdoseprofile1,LET_zfluenceprofile1,s,area_rcfcorrected1)
D_a_mini,D_fluence_mini=birkcorrection(dose,depth,depth_sci,LET_zdoseprofile,LET_zfluenceprofile,s,area_rcfcorrected)




################################################################################
""" PLOTTING FUNCTIONS """

def plotfunction1(depth_sci,dose_sci,norm_sim,rcfdepth,rcfdose,rcferr,norm_rcf,color1,color2,PC,rcfname,s):


    plt.plot(depth_sci,dose_sci*norm_sim,
                                                                            '.',
                                                                     color=color1,
                                                                     Markersize=12,
                                                            label='{}'.format(PC))



    plt.errorbar( rcfdepth,                                     rcfdose*norm_rcf,
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


def plotfunction4(rcfdepth,rcfdose,norm_rcf,rcferr,rcfname,dose,depth,err,norm,depth_sci2,zdoseprofile,LET_zfluenceprofile,norm_sim,D_fluence_mini,shotnumber,PC):
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



    ax.errorbar( rcfdepth,                                       rcfdose*norm_rcf,
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

    twin1.plot(depth_sci,LET_zfluenceprofile,
                                                                            '.',
                                            color=sns.color_palette(  "Paired")[4],
                                                                   Markersize=12,
                                               label=r'$LET_{fluence}$',
                                                                 zorder=1)


    twin2.plot(depth_sci,(1+k*LET_zfluenceprofile),
                                                                                '^',
                                                                       markersize=5,
                                                color=sns.color_palette("Paired")[9],
                                                                 label='(1+KB*LET)',
                                                                         zorder=1)




    ax.plot( np.arange(0,149,1)*s,zdoseprofile*norm_sim,
                                                                            '.',
                                         color=sns.color_palette(  "Paired")[7],
                                                                  Markersize=12,
                                                      label=r'$D_{simulated}$',
                                                                    zorder=2)





    plt.rc('text', usetex=False)
    plt.title('Depth-dose distribution shape comparison {} shot n.{}'.format(PC,shotnumber),
                                                                    fontsize=24,
                                                                  fontdict=None,
                                                                  loc='center',
                                                                       pad=None)


    twin2.set_ylabel("a.u.",color=sns.color_palette(  "Paired")[9],fontsize=22)
    twin2.tick_params(axis='y', which='major',colors=sns.color_palette(  "Paired")[9], labelsize=22)
    twin2.legend( title='',fontsize=20,loc=9,markerscale=3).set_zorder(0)
    twin1.set_ylabel("LET [KeV/um]",color=sns.color_palette(  "Paired")[5],fontsize=22)
    ax.set_ylabel(" Relative Dose ",color="Green",fontsize=22)
    ax.set_xlabel(" Depth[mm] ",color="black",fontsize=22)
    ax.legend( title='',fontsize=20,loc=4,markerscale=3)
    twin1.legend( title='',fontsize=20,loc=6,markerscale=3)
    ax.tick_params(axis='x', which='major', labelsize=22)
    twin1.tick_params(axis='y', which='major',colors=sns.color_palette(  "Paired")[5], labelsize=22)
    ax.tick_params(axis='y', which='major',colors='green', labelsize=22)
    plt.show()

    return


def plotfunction3(rcfdose,rcfdepth,rcferr,rcfname,area_rcf,area_rcfcorrected,D_rcf,Derr_rcf,depth_sci,LET_zdoseprofile):


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



    mark_inset(ax, sub_axes, loc1=2, loc2=3, fc="none", ec="0.5")
    plt.show()
    return
##############################################################################
"""PLOTS"""



plt.figure(1) # creates a figure in which we plot

plotfunction1(np.arange(0,149,1)*s,zdoseprofile,norm_sim,rcfdepth,rcfdose,rcferr,norm_rcf,sns.color_palette("Paired")[3],sns.color_palette("Paired")[2],PC,rcfname,s)
plotfunction1(np.arange(0,149,1)*s,zdoseprofile1,norm_sim1,rcfdepth1,rcfdose1,rcferr1,norm_rcf1,sns.color_palette("Paired")[5],sns.color_palette("Paired")[4],PC1,rcfname1,s)
plotfunction1(np.arange(0,149,1)*s,zdoseprofile2,norm_sim2,rcfdepth2,rcfdose2,rcferr2,norm_rcf2,sns.color_palette("Paired")[7],sns.color_palette("Paired")[6],PC2,rcfname2,s)

plt.title('Depth-dose distribution shape comparison',fontsize=20)
plt.xlabel(" Depth[mm] ",fontsize=20)
plt.ylabel('Relative Dose in water ',fontsize=20)
plt.legend(title='$E_{primary}$=152.5 MeV',fontsize=24,loc=3,markerscale=2,title_fontsize=24)
plt.tick_params(axis='x', which='major', labelsize=20)
plt.tick_params(axis='y', which='major', labelsize=20)



plt.figure(2)

plotfunction2(depth_sci,dose_sci,norm_sim,LET_zfluenceprofile,sns.color_palette(  "Paired")[3],PC,190)
plotfunction2(depth_sci,dose_sci1,norm_sim1,LET_zfluenceprofile1,sns.color_palette(  "Paired")[5],PC1,190)
plotfunction2(depth_sci,dose_sci2,norm_sim2,LET_zfluenceprofile2,sns.color_palette(  "Paired")[7],PC2,190)

plt.title('Simulated fluence-weighted LET',fontsize=20)
plt.legend(fontsize=20,loc=2,markerscale=3)
plt.ylabel("LET [KeV/um]",fontsize=20)
plt.xlabel(" Depth[mm] ",color="black",fontsize=20)
plt.gca().yaxis.set_label_position("right")
plt.tick_params(axis='y', labelsize=20,which='both', labelleft=False, labelright=True)
plt.tick_params(axis='x',which='both', labelsize=20)
plt.show()


plotfunction3(rcfdose2,rcfdepth2,rcferr2,rcfname2,area_rcf2,area_rcfcorrected2,D_rcf2,Derr_rcf2,depth_sci2,LET_zdoseprofile2)
plotfunction3(rcfdose1,rcfdepth1,rcferr1,rcfname1,area_rcf1,area_rcfcorrected1,D_rcf1,Derr_rcf1,depth_sci1,LET_zdoseprofile1)
plotfunction3(rcfdose,rcfdepth,rcferr,rcfname,area_rcf,area_rcfcorrected,D_rcf,Derr_rcf,depth_sci,LET_zdoseprofile)


plotfunction4(rcfdepth2,rcfdose2,norm_rcf2,rcferr2,rcfname2,dose2,depth2,err2,norm2,depth_sci2,zdoseprofile2,LET_zfluenceprofile2,norm_sim2,D_fluence_mini2,shotnumber2,PC2)
plotfunction4(rcfdepth1,rcfdose1,norm_rcf1,rcferr1,rcfname1,dose1,depth1,err1,norm1,depth_sci1,zdoseprofile1,LET_zfluenceprofile1,norm_sim1,D_fluence_mini1,shotnumber1,PC1)
plotfunction4(rcfdepth,rcfdose,norm_rcf,rcferr,rcfname,dose,depth,err,norm,depth_sci,zdoseprofile,LET_zfluenceprofile,norm_sim,D_fluence_mini,shotnumber,PC)
