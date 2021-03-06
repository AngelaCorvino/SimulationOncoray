import numpy as np
"LET correction simulations in scintillator"




def lightout(S,a,k,dx):
    ''' Birks'Model'''
    return ((a*S)/(1+(k*S)))*dx


def lightcorrection(S,a,k,dx):
    '''Birk's Model quenching correction / linear trend'''
    return np.divide(lightout(S,a,k,dx), lightout(S,a,0,dx), out=np.zeros_like(lightout(S,a,k,dx)),
                                                where= lightout(S,a,0,dx)!=0)



def dosecorrection(dose,S,a,k,dx):
    return np.divide(dose, lightcorrection(S,a,k,dx), out=np.zeros_like(dose),
                                                where= lightcorrection(S,a,k,dx)!=0)  #S1 should be a function of depth, i can not make a fit but a ican interpolate, data are not normalized
