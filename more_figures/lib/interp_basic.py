# -*- coding: utf-8 -*-
"""
create function-like object for interpolation.

use np.interp for speed as opposed to interp1d. Too much overhead in interp1d.
"""

import numpy as np
from scipy.interpolate import interp1d,Akima1DInterpolator,CubicSpline
from numpy.core.multiarray import interp as compiled_interp

from numba import njit

class interp_basic(object):
    
    def __init__(self,x,y,T):
        
        self.x = x
        self.y = y
        self.T = T
        
        #self.interp = interp1d(self.x,self.y,kind='linear',fill_value='extrapolate')
        #self.interp = interp1d(self.x,self.y,kind='cubic')
        #self.interp = np.interp(self.x,self.y,kind='cubic')
        #self.interp = Akima1DInterpolator(self.x,self.y)
        self.interp = CubicSpline(self.x,self.y)
    
    def __call__(self, x_new):
        x_new = np.mod(x_new,self.T)
        
        return self.interp(x_new)
        #return compiled_interp(x_new, self.x, self.y,
        #                       left=False, right=False)
