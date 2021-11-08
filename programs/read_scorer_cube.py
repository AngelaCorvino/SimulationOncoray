import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

#Class for Topas cube scorer reading
class Read_Scorer_Cube():

############################### Constructor of class ###################################
                                                                                       #
    def __init__(             self,
                              path,
                          skiprows ):
                                                                                       #
############################ Read in parameters #############################
                                                                            #
        self.path = path
        self.x_line = 4
        self.y_line = 5
        self.z_line = 6
                                                                            #
        self.skiprows = skiprows
                                                                            #
#############################################################################
                                                                                       #
############################ Read in data ###################################
                                                                            #
        (  self.matrix_3D,
         self.x_axis_vals,
         self.y_axis_vals,
         self.z_axis_vals ) = self.read_data()
                                                                            #
#############################################################################
                                                                                       #
########################################################################################


############################# Function for read DOSE ###################################
                                                                                       #
    def read_data(self):
                                                                                       #
        with open(self.path) as f:
              content = f.readlines()
                                                                                       #
        #Read in Depth information
        x_string_result = re.findall(  '[-+]?\d*\.\d+|\d+',
                                      content[self.x_line] )
        y_string_result = re.findall(  '[-+]?\d*\.\d+|\d+',
                                      content[self.y_line] )
        z_string_result = re.findall(  '[-+]?\d*\.\d+|\d+',
                                      content[self.z_line] )
                                                                                       #
        x_bin_number = int(x_string_result[0])
        y_bin_number = int(y_string_result[0])
        z_bin_number = int(z_string_result[0])
                                                                                       #
        x_bin_size = float(x_string_result[1])
        y_bin_size = float(y_string_result[1])
        z_bin_size = float(z_string_result[1])
                                                                                       #
        x_axis_vals = (np.arange(0,x_bin_number,1)+0.5)*x_bin_size # in cm
        y_axis_vals = (np.arange(0,y_bin_number,1)+0.5)*y_bin_size # in cm
        z_axis_vals = (np.arange(0,z_bin_number,1)+0.5)*z_bin_size # in cm
                                                                                       #
        data = pd.read_csv(             self.path,
                                      header=None,
                                    delimiter=',',
                           skiprows=self.skiprows )
                                                                                       #
        data_3D = np.array(data[3]).reshape(x_bin_number,
                                             y_bin_number,
                                             z_bin_number )
                                                                                       #
        return (    data_3D,
                x_axis_vals,
                y_axis_vals,
                z_axis_vals )
                                                                                       #
########################################################################################



if __name__ == "__main__":


    #directory ='/Users/angelacorvino/Documents/GitHub/SimulationOncoray/data/'
    directory ='/home/corvin22/SimulationOncoray/data/Single/Aperture/'
    file_name='DoseToWater_91600KeVproton_Airbox_6PC1PMMA_3Dscorer.csv'
    #='EnergyDeposit_90MeVproton_PVT_6PC_2Dscorer.csv'
    path=directory+file_name


    Scored_values = Read_Scorer_Cube(path,8)

    #3D Matrix
    matrix_3D = Scored_values.matrix_3D
    x_axis_vals = Scored_values.x_axis_vals
    y_axis_vals = Scored_values.y_axis_vals
    z_axis_vals = Scored_values.z_axis_vals

    #2D Cutplane
    y_idx = 100
    cutplane = matrix_3D[:,y_idx,:]

    plt.title('y = {} cm'.format(y_axis_vals[y_idx]))
    plt.imshow(cutplane,extent=(z_axis_vals[0],z_axis_vals[-1],x_axis_vals[0],x_axis_vals[-1]))
    plt.ylabel('x in cm')
    plt.xlabel('z in cm')

    plt.show()


    #1D Cutplane in x
    x_idx = 80
    y_idx = 0
    cutplane = matrix_3D[x_idx,y_idx,:]

    plt.title('y = {} cm ; x = {} cm'.format(y_axis_vals[y_idx],x_axis_vals[x_idx]))
    plt.plot(z_axis_vals,cutplane)
    plt.ylabel('Energy in MeV')
    plt.xlabel('z in cm')

    plt.show()


    #1D Cutplane in z
    z_idx = 80
    y_idx = 0
    cutplane = matrix_3D[:,y_idx,z_idx]

    plt.title('y = {} cm ; z = {}'.format(y_axis_vals[y_idx],z_axis_vals[z_idx]))
    plt.plot(x_axis_vals,cutplane)
    plt.ylabel('Energy in MeV')
    plt.xlabel('z in cm')

    plt.show()
