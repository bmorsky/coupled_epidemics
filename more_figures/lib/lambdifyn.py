# -*- coding: utf-8 -*-
"""
create function-like object for lambidfying numpy stuff
"""

import numpy as np
import numexpr as ne

class lambdifyn(object):
    
    def __init__(self,coeffs,fqs):
        """
        coeffs - ndim array of coefficients.
        fqs should be 1d array of frequencies already trimmed.
        """

        self.coeffs = coeffs
        
        #idxs = np.where(np.abs(coeffs)>0)
        
        self.dim = len(np.shape(self.coeffs))

        self.fq_mats = []
        for i in range(self.dim):

            slices = ()
            for j in range(self.dim):
                if j == i:
                    slices += (len(fqs),)
                else:
                    slices += (1,)

            fq = fqs.reshape(slices) + np.zeros(self.dim*(len(fqs),))
            
            self.fq_mats.append(fq)
            
    #@profile
    def __call__(self, x):
    
        #print('ldn cs',np.sort(self.coeffs[np.where(np.abs(self.coeffs)>0)]))

        tot = 0
        for i in range(len(x)):

            tot += self.fq_mats[i]*x[i]
        
        c = self.coeffs
        #print('ldn tot',np.sum(np.abs(tot)))
        out = ne.evaluate('c*exp(1j*tot)').real
        
        return np.sum(out.flatten())
