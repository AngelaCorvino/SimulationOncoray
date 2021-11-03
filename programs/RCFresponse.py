import numpy as np
"LET correction simulations in scintillator"




def underesponse(S,a,b):  #we need the dose weighted LET
    ''' linear model for under responde '''
    return (a*S)+b


def rcfdosecorrection(rcfdose,S,a,b):
    '''rcf measurements / linear trend '''

    return np.divide(rcfdose, underesponse(S,a,b), out=np.zeros_like(rcfdose),
                                                where= underesponse(S,a,b)!=0 )
