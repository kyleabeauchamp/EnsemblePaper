"""
"""

import numpy as np

def assign_grid(phi, psi, num_bins):
    divisor = 360 / num_bins
    x = (phi % 360.) / divisor
    y = (psi % 360.) / divisor
    x = x.astype('int')
    y = y.astype('int')
    ass = x + y * num_bins
    ass = np.unique(ass, return_inverse=True)[1]
    return x, y, ass

def assign(phi,psi):
    """
    Notes:
    State 0: PPII
    State 1: beta
    State 2: alpha
    State 3: gamma, alpha_l
    """
    ass = (0*phi).astype('int') + 3
    
    #ass[(phi <= 0)&(phi>=-100)] = 0
    #ass[(phi <= -100)] = 1
    #ass[(phi <= 0)&(psi>=-100)&(psi<=50)] = 2
    #ass[phi > 0 ] = 3

    #States from Tobin
    ass[(phi <= 0)&(phi>=-100)&((psi>=50.)|(psi<= -100))] = 0
    ass[(phi <= -100)&((psi>=50.)|(psi<= -100))] = 1
    ass[(phi <= 0)&((psi<=50.)&(psi>= -100))] = 2
    ass[(phi > 0)] = 3
    
    """
    ass[(phi>=-90)&(phi<=-25)&(psi>=80)&(psi<=160)] = 0
    ass[(phi>=-150)&(phi<=-90)&(psi>=80)&(psi<=160)] = 1
    ass[(phi>=-150)&(phi<=-25)&(psi>=-150)&(psi<=0)] = 2
    """
    
    return ass
