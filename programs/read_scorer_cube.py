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


    directory ='/Users/angelacorvino/Documents/GitHub/SimulationOncoray/data/Single/Aperture/'
    #directory ='/home/corvin22/SimulationOncoray/data/Single/Aperture/'
    file_name='DoseToWater_91600KeVproton_Airbox_6PC1PMMA_3Dscorer.csv'
    #file_name='LET_fluenceweighted_91600KeVproton_Airbox_6PC1PMMA_3Dscorer.csv'
    #='EnergyDeposit_90MeVproton_PVT_6PC_2Dscorer.csv'
    path=directory+file_name


    Scored_values = Read_Scorer_Cube(path,8)

    #3D Matrix
    matrix_3D = Scored_values.matrix_3D
    x_axis_vals = Scored_values.x_axis_vals
    y_axis_vals = Scored_values.y_axis_vals
    z_axis_vals = Scored_values.z_axis_vals

    #2D Cutplane
    plt.figure(1)
    y_idx = 100
    cutplane = matrix_3D[:,y_idx,:]

    plt.title('y = {} cm'.format(y_axis_vals[y_idx]))
    plt.imshow(cutplane/cutplane.max(),extent=(z_axis_vals[0],z_axis_vals[-1],x_axis_vals[0],x_axis_vals[-1]),
                                                            vmin=0,vmax=0.5)
    plt.ylabel('x in cm')
    plt.xlabel('z in cm')



    #2D Cutplane
    plt.figure(2)
    y_idx = 50
    cutplane = matrix_3D[:,y_idx,:]

    plt.title('y = {} cm'.format(y_axis_vals[y_idx]))
    plt.imshow(cutplane/cutplane.max(),
            extent=(z_axis_vals[0],z_axis_vals[-1],x_axis_vals[0],x_axis_vals[-1]),
                                                                    vmin=0,vmax=0.5)
    plt.ylabel('x in cm')
    plt.xlabel('z in cm')



    #2D Cutplane
    plt.figure(3)
    x_idx = 50
    cutplane = matrix_3D[x_idx,:,:]

    plt.title('x = {} cm'.format(x_axis_vals[x_idx]))
    plt.imshow(cutplane/cutplane.max(),extent=(z_axis_vals[0],z_axis_vals[-1],y_axis_vals[0],y_axis_vals[-1]),vmin=0,vmax=0.5)
    plt.ylabel('y in cm')
    plt.xlabel('z in cm')
    plt.show()


    #2D Cutplane
    plt.figure(1)
    z_idx = 10
    cutplane = matrix_3D[:,:,z_idx ]

    plt.title('z = {} cm'.format(z_axis_vals[z_idx]))
    plt.imshow(cutplane/cutplane.max(),extent=(x_axis_vals[0],x_axis_vals[-1],y_axis_vals[0],y_axis_vals[-1]),vmin=0,vmax=0.5)
    plt.ylabel('y in cm')
    plt.xlabel('x in cm')




    #2D Cutplane
    plt.figure(2)
    z_idx = 50
    cutplane = matrix_3D[:,:,z_idx ]

    plt.title('z = {} cm'.format(z_axis_vals[z_idx]))
    im=plt.imshow(cutplane/cutplane.max(),extent=(x_axis_vals[0],x_axis_vals[-1],y_axis_vals[0],y_axis_vals[-1]),vmin=0,vmax=0.5)
    plt.colorbar(im,label='Relative Intensity')
    plt.ylabel('y in cm')
    plt.xlabel('x in cm')




    #2D Cutplane
    plt.figure(3)
    z_idx = 70
    cutplane = matrix_3D[:,:,z_idx ]

    plt.title('z = {} cm'.format(z_axis_vals[z_idx]))
    im=plt.imshow(cutplane/cutplane.max(),extent=(x_axis_vals[0],x_axis_vals[-1],y_axis_vals[0],y_axis_vals[-1]),vmin=0,
                                                                        vmax=0.5)
    plt.colorbar(im,label='Relative Intensity')
    plt.ylabel('y in cm')
    plt.xlabel('x in cm')

    #2D Cutplane
    plt.figure(4)
    z_idx = 80
    cutplane = matrix_3D[:,:,z_idx ]

    plt.title('z = {} cm'.format(z_axis_vals[z_idx]))

    im=plt.imshow(cutplane/cutplane.max(),extent=(x_axis_vals[0],x_axis_vals[-1],y_axis_vals[0],y_axis_vals[-1]),vmin=0,vmax=0.5)
    plt.colorbar(im,label='Relative Intensity')
    plt.ylabel('y in cm')
    plt.xlabel('x in cm')



    #2D Cutplane
    plt.figure(5)
    z_idx = 90
    cutplane = matrix_3D[:,:,z_idx ]

    plt.title('z = {} cm'.format(z_axis_vals[z_idx]))
    plt.imshow(cutplane/cutplane.max(),extent=(x_axis_vals[0],x_axis_vals[-1],y_axis_vals[0],y_axis_vals[-1]),vmin=0,vmax=0.5)
    plt.ylabel('y in cm')
    plt.xlabel('x in cm')

    #2D Cutplane
    plt.figure(6)
    z_idx = 110
    cutplane = matrix_3D[:,:,z_idx ]

    plt.title('z = {} cm'.format(z_axis_vals[z_idx]))
    plt.imshow(cutplane/cutplane.max(),extent=(x_axis_vals[0],x_axis_vals[-1],y_axis_vals[0],y_axis_vals[-1]),vmin=0,vmax=0.5)
    plt.ylabel('y in cm')
    plt.xlabel('x in cm')




    plt.show()


    #1D Cutplane in x
    x1_idx = 50
    x2_idx = 100
    x3_idx = 150
    y_idx = 100
    cutplane1 = matrix_3D[x1_idx,y_idx,:]
    cutplane2 = matrix_3D[x2_idx,y_idx,:]
    cutplane3 = matrix_3D[x3_idx,y_idx,:]

    plt.title('y = {} cm '.format(y_axis_vals[y_idx]))
    plt.plot(z_axis_vals,cutplane1,'.',label='x = {}'.format(x_axis_vals[x1_idx]))
    plt.plot(z_axis_vals,cutplane2,'.',label='x = {}'.format(x_axis_vals[x2_idx]))
    plt.plot(z_axis_vals,cutplane3,'.',label='x = {}'.format(x_axis_vals[x3_idx]))
    #plt.ylabel('Dose to Water')
    #plt.ylabel('LET fluece weighted')
    plt.xlabel('z in cm')
    plt.legend()
    plt.show()


    #1D Cutplane in z
    z1_idx = 50
    z2_idx = 100
    z3_idx = 150
    y_idx = 100
    cutplane1 = matrix_3D[:,y_idx,z1_idx]
    cutplane2 = matrix_3D[:,y_idx,z2_idx]
    cutplane3 = matrix_3D[:,y_idx,z3_idx]

    plt.title('y = {} cm '.format(y_axis_vals[y_idx]))
    plt.plot(x_axis_vals,cutplane1,'.',label='z = {}'.format(z_axis_vals[z1_idx]))
    plt.plot(x_axis_vals,cutplane2,'.',label='z = {}'.format(z_axis_vals[z2_idx]))
    plt.plot(x_axis_vals,cutplane3,'.',label='z = {}'.format(z_axis_vals[z3_idx]))
    plt.ylabel('Dose to Water')
    #plt.ylabel('LET fluece weighted')
    plt.xlabel('x in cm')
    plt.legend()

    plt.show()
