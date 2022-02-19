
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
from calibrationPBP import birkcorrection
from calibrationPBP import rcfcorrection
from calibrationPBP import readtopasoutput
#RCF data
directory='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2021-09-02/'
filename='RCF21CW.csv'
rcfdepth,rcfdose,rcferr,area_rcf,rcfname=read_datarcf(directory,filename)
###############SIMULATION
s=0.073825 #[mm/pixel ]
directory="/Users/angelacorvino/Documents/GitHub/SimulationOncoray/data/Single/"
depth_sci,zdoseprofile,LET_zdoseprofile,LET_zfluenceprofile,area_LET,area_f_LET,energy,PC=readtopasoutput(directory,
                            'DoseToWater_9160KeVproton_PVT_6PC1PMMA_1Dscorer.csv',
                       'LET_doseweighted_9160KeVproton_PVT_6PC1PMMA_1Dscorer.csv',
                    'LET_fluenceweighted_9160KeVproton_PVT_6PC1PMMA_1Dscorer.csv',
                                                                            149,1)
dose_sci=zdoseprofile
area_sim=np.trapz(zdoseprofile[3:len(zdoseprofile)-3],depth_sci[3:len(depth_sci)-3])




norm_rcf=1/rcfdose.max()
norm_sim=1/zdoseprofile.max() #area_sim



################################################################################
""" correction of RCf measure with LET (doseweighted ) simulations in scintillator"""

a=-0.0251
b= 1.02
D_rcf,Derr_rcf,area_rcfcorrected=rcfcorrection(rcfdepth,rcfdose,rcferr,depth_sci,LET_zdoseprofile,a,b)



###########DOSE RATE############################################################
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


D_dose_mini0,D_fluence_mini0,L_fluence_mini0=birkcorrection(dose0,depth,depth_sci,LET_zdoseprofile,LET_zfluenceprofile,s,area_rcfcorrected)
D_dose_mini1,D_fluence_mini1,L_fluence_mini1=birkcorrection(dose1,depth,depth_sci,LET_zdoseprofile,LET_zfluenceprofile,s,area_rcfcorrected)
D_dose_mini2,D_fluence_mini2,L_fluence_mini2=birkcorrection(dose2,depth,depth_sci,LET_zdoseprofile,LET_zfluenceprofile,s,area_rcfcorrected)
D_dose_mini3,D_fluence_mini3,L_fluence_mini3=birkcorrection(dose3,depth,depth_sci,LET_zdoseprofile,LET_zfluenceprofile,s,area_rcfcorrected)
D_dose_mini4,D_fluence_mini4,L_fluence_mini4=birkcorrection(dose4,depth,depth_sci,LET_zdoseprofile,LET_zfluenceprofile,s,area_rcfcorrected)
D_dose_mini5,D_fluence_mini5,L_fluence_mini5=birkcorrection(dose5,depth,depth_sci,LET_zdoseprofile,LET_zfluenceprofile,s,area_rcfcorrected)

def plotfunction6(depth,dose,err,norm,doserate,D_fluence_mini,color1,s):

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
                                                       label='LET corrected dose')
    return


fig, ax = plt.subplots()
plt.title('Does quenching depend on dose rate ?(6PC 1PMMA)',fontsize=24)
plotfunction6(depth,dose0,err0,norm0,doserate0,D_fluence_mini0,sns.color_palette("Paired")[3],s)
plotfunction6(depth,dose1,err1,norm1,doserate1,D_fluence_mini1,sns.color_palette("Paired")[10],s)
plotfunction6(depth,dose2,err2,norm2,doserate2,D_fluence_mini2,sns.color_palette("Paired")[5],s)
plotfunction6(depth,dose3,err3,norm3,doserate3,D_fluence_mini3,sns.color_palette("Paired")[6],s)
plotfunction6(depth,dose4,err4,norm4,doserate4,D_fluence_mini4,sns.color_palette("Paired")[7],s)
plotfunction6(depth,dose5,err5,norm5,doserate5,D_fluence_mini5,sns.color_palette("Paired")[8],s)

ax.errorbar( rcfdepth,                      rcfdose*norm_rcf ,
                                                                  yerr=rcferr*norm_rcf ,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                        color=sns.color_palette(  "Paired")[1],
                                                                        markersize=9,
                                                                   ecolor=sns.color_palette(  "Paired")[0],
                                                                elinewidth=None,
                                                                label=' RCF 21CW measured dose')
ax.fill_between(rcfdepth,
                                                                rcfdose*norm_rcf -rcferr*norm_rcf ,
                                                                rcfdose*norm_rcf +rcferr*norm_rcf ,
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
