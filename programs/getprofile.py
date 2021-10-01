import scipy
import pandas as pd # to work with structured, tabulated data (.csv, .xlsx,..)
import numpy as np # to work fast with matrices, arrays and much more
from matplotlib import pyplot as plt # to visualize/plot your data
from scipy import interpolate


def get_x_doseprofile_at_z(data, z_binnumber):
    #z_binnumber = depth. The x-first value with e.g.
    #z=0 will be at x=0, the next with z=0 will be at x=1 etc. So increasing z-should be enough
    x_profile_dosevalues = data[data[:,2]==z_binnumber][:,3] # get all doses in x-direction at given z
    return x_profile_dosevalues


def get_z_doseprofile_at_x(data, x_binnumber):

    z_profile_dosevalues = data[data[:,0]==x_binnumber][:,3] # get all doses in z-direction at given x
    return z_profile_dosevalues
"""
def get_z_energydeposition_at_x(data, x_binnumber):

    z_profile_energyvalues = data[data[:,0]==x_binnumber][:,3] # get all energies in z-direction at given x
    return z_profile_energyvalues
"""



class Get_profile(object):

    #Initialization of class
    def __init__(self,data, z_binnumber,x_binnumber):

        #3D data matrix
        self.data = data
        self.z_binnumber= z_binnumber
        self.x_binnumberl=x_binnumber

        self.xprofiles = np.zeros((self.z_binnumber, self.x_binnumber)) # create empty array to be filled
        self.zprofiles = np.zeros((self.x_binnumber, self.z_binnumber)) # create empty array to be filled
        #zprofiles_energy = np.zeros((numberof_xbins, numberof_zbins)) # create empty array to be filled




        #dimension of 3d data matrix
        self.ind_x=self.slices_x//2
        self.ind_y=self.slices_y//2
        self.ind_z=0



        # write each x- profile at depth z in row i in the array outside the loop

        for i in range(self.z_binnumber):
            xprofile_at_z  = get_x_doseprofile_at_z(self.data, i)
            xprofiles[i,:] = xprofile_at_z
            i += 1

        self.xmeanprofile=np.sum(self.xprofiles,axis=0)/self.z_binnumber



        for i in range(self.x_binnumber):
            zprofile_at_x  = get_z_doseprofile_at_x(daten, i)
            zprofiles[i,:] = zprofile_at_x
            i += 1

        self.zmeanprofile=np.sum(self.zprofiles,axis=0)/self.x_binnumber
