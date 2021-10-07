
"LET correction simulations in scintillator"




def lightout(S,a,k,dx):
    ''' Birks'Model'''
    return ((a*S)/(1+(k*S)))*dx


def lightcorrection(S,a,k,dx):
    '''Birk's Model quenching correction / linear trend'''
    return lightout(S,a,k,dx)/lightout(S,a,0,dx)



def dosecorrection(dose,S,a,k,dx):
    return (dose/lightcorrection(S,a,k,dx) )  #S1 should be a function of depth, i can not make a fit but a ican interpolate, data are not normalized
