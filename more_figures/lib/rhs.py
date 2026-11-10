
"""
library functions for right-hand sides
"""

import numpy as np
import matplotlib.pyplot as plt

def rhs_avg_2d(t,y,a,eps=0,del1=0,miter=None):
    """ for forcing only. 2d."""

    if miter is None:
        nn = a.system1.miter
    else:
        nn = miter
    
    th,ps = y
    dth = 0
    dps = a.system1.kappa_val*ps
    
    for i in range(nn):
        dth += eps*ps**i*a.hz_lam[i](th)
        dps += eps*ps**i*a.hi_lam[i](th)
    dth -= del1/a._m[1]
    return np.array([dth*a._n[1],dps*a._m[1]])

def rhs_avg_1df(t,th,a,eps=0,del1=0,miter=None):
    """ for forcing only. 1d"""

    if miter is None:
        nn = a.system1.miter
    else:
        nn = miter
    
    dth = 0
    
    for i in range(nn):
        dth += eps**(i+1)*a.system1.h['lam'][i](th/a._n[1])

    dth -= del1/a._m[1]
    return a._n[1]*dth


def _redu_c(t,y,a,eps=0,del1=None,miter=None):
    """ for coupling only. 1d"""
    system1 = a.system1; system2 = a.system2
    n = a._n[1]
    
    if miter is None:
        nn = a.system1.miter
    else:
        nn = miter
    
    h = 0

    in1 = y
    for k in range(nn):
            
        h1 = system1.h['lam'][k](in1)
        h2 = system2.h['lam'][k](in1)
        d = h1 - h2

        h += eps**(k+1)*d
    return h


def _redu_c2(t,y,a,eps=0,b=0,miter=None,n_factor=False):
    """
    for coupling only
    with explicit heterogeneity.
    """
    s1 = a.system1; s2 = a.system2
    n = a._n[1]
    
    if miter is None:
        nn = a.system1.miter
    else:
        nn = miter
    
    h = 0
    om = a.om

    in1 = y/n
    if nn >= 1:
        s1_h1_hom = s1.h['lam_hom'][0](in1)
        s1_h1_het = s1.h['lam_het'][0][0](in1)*b

        s2_h1_hom = s2.h['lam_hom'][0](in1)
        s2_h1_het = s2.h['lam_het'][0][0](in1)*b

        h += eps*(s1_h1_hom+s1_h1_het)
        h -= eps*(s2_h1_hom+s2_h1_het)

    if nn >= 2:
        s1_h2_hom = s1.h['lam_hom'][1](in1)
        s1_h2_het1 = s1.h['lam_het'][1][0](in1)*b
        s1_h2_het2 = s1.h['lam_het'][1][1](in1)*b**2

        s2_h2_hom = s2.h['lam_hom'][1](in1)
        s2_h2_het1 = s2.h['lam_het'][1][0](in1)*b
        s2_h2_het2 = s2.h['lam_het'][1][1](in1)*b**2

        h += eps**2*(s1_h2_hom+s1_h2_het1+s1_h2_het2)
        h -= eps**2*(s2_h2_hom+s2_h2_het1+s2_h2_het2)

    if nn >= 3:
        s1_h3_hom = s1.h['lam_hom'][2](in1)
        s1_h3_het1 = s1.h['lam_het'][2][0](in1)*b
        s1_h3_het2 = s1.h['lam_het'][2][1](in1)*b**2
        s1_h3_het3 = s1.h['lam_het'][2][2](in1)*b**3

        s2_h3_hom = s2.h['lam_hom'][2](in1)
        s2_h3_het1 = s2.h['lam_het'][2][0](in1)*b
        s2_h3_het2 = s2.h['lam_het'][2][1](in1)*b**2
        s2_h3_het3 = s2.h['lam_het'][2][2](in1)*b**3

        h += eps**3*(s1_h3_hom+s1_h3_het1+s1_h3_het2+s1_h3_het3)
        h -= eps**3*(s2_h3_hom+s2_h3_het1+s2_h3_het2+s2_h3_het3)

    if nn >= 4:
        s1_h4_hom = s1.h['lam_hom'][3](in1)
        s1_h4_het1 = s1.h['lam_het'][3][0](in1)*b
        s1_h4_het2 = s1.h['lam_het'][3][1](in1)*b**2
        s1_h4_het3 = s1.h['lam_het'][3][2](in1)*b**3
        s1_h4_het4 = s1.h['lam_het'][3][3](in1)*b**4

        s2_h4_hom = s2.h['lam_hom'][3](in1)
        s2_h4_het1 = s2.h['lam_het'][3][0](in1)*b
        s2_h4_het2 = s2.h['lam_het'][3][1](in1)*b**2
        s2_h4_het3 = s2.h['lam_het'][3][2](in1)*b**3
        s2_h4_het4 = s2.h['lam_het'][3][3](in1)*b**4

        h += eps**4*(s1_h4_hom+s1_h4_het1+s1_h4_het2+s1_h4_het3+s1_h4_het4)
        h -= eps**4*(s2_h4_hom+s2_h4_het1+s2_h4_het2+s2_h4_het3+s2_h4_het4)

    
    if n_factor:
        
        return h*a._n[1]
    else:
        return h


def _redu(t,y,a,eps=.01,del1=0):
    th,ps,tf = y
    
    u = a.system1.forcing_fn(tf)
    
    dth = 1 + eps*a.th_lam(th,ps)*u
    dps = a.system1.kappa_val*ps + eps*a.ps_lam(th,ps)*u
    dtf = 1 + del1/a._m[1]
    return np.array([dth*a._n[1],dps*a._n[1],dtf*a._m[1]])


def _full(t,y,a,eps,b,option='val'):
    """del1 in a pardict"""
    pd1 = a.system1.pardict;pd2 = a.system2.pardict
    y1 = y[:len(a.system1.var_names)];y2 = y[len(a.system1.var_names):]

    pd1['del0'] = 0
    for i in range(len(a.het_coeffs)):
        pd1['del0'] += eps**i*b**(i+1)*a.het_coeffs[i]

    
    c1 = a.system1.coupling(y,pd1,option,0)
    c2 = a.system2.coupling(list(y2)+list(y1),pd2,option,1)
    
    out1 = (a.system1.rhs(t,y1,pd1,option,0) + eps*c1)
    out2 = (a.system2.rhs(t,y2,pd2,option,1) + eps*c2)
    return np.array(list(out1)+list(out2))



def _full_mono_rhs(t,y,pdict,option='val',idx=None):
    """del1 in a pardict"""
    a = pdict['a']
    y1 = y[:len(a.system1.var_names)];y2 = y[len(a.system1.var_names):]

    eps = pdict['eps']

    pdict['del0'] = 0
    b = pdict['b']
    for i in range(len(a.het_coeffs)):
        pdict['del0'] += eps**i*b**(i+1)*a.het_coeffs[i]
        
    

    c1 = a.system1.coupling(y,pdict,option,0)
    c2 = a.system2.coupling(list(y2)+list(y1),pdict,option,1)
    
    out1 = a.system1.rhs(t,y1,pdict,option,0) + eps*c1
    out2 = a.system2.rhs(t,y2,pdict,option,1) + eps*c2
    return np.array(list(out1)+list(out2))



def _redu_full(t,y,a,eps:float=.01,del1=None):
    """
    Original reduction without averaging
    """
    s1 = a.system1;s2 = a.system2
    thA,psA,thB,psB = y    
    k1 = s1.kappa_val;k2 = s2.kappa_val

    in1 = thA-thB+t;in2 = t

    dthA = eps*s1.gz_lam(in1,in2,psA,psB)
    dthB = eps*s2.gz_lam(in1,in2,psA,psB)
    
    dpsA = k1*psA + eps*s1.gi_lam(in1,in2,psA,psB)
    dpsB = k2*psB + eps*s2.gi_lam(in1,in2,psA,psB)

    return np.array([dthA,dpsA,dthB,dpsB])


def _redu_3dc_gw(t,y,a,eps:float=.01,del1=None):
    """
    Goodwin-specific 3d (1 phase diff, 2 isostables)
    """
    system1 = a.system1;system2 = a.system2
    pdA = a.system1.pardict;pdB = a.system2.pardict

    s,ds = np.linspace(0,2*np.pi*a._m[1],5000,retstep=True)
    phi,psA,psB = y

    k1 = system1.kappa_val;k2 = system2.kappa_val
    Tm = (2*np.pi*a._m[1])

    om = a._n[1]/a._m[1]
    in1 = phi+a.om*s;in2 = s

    zA = a.th_lam(in1,psA);iA = a.ps_lam(in1,psA)
    zB = a.th_lam(in2,psB);iB = a.ps_lam(in2,psB)

    # coupling terms
    v1 = system1.lc['lam_v0'](in1) + a.g_lam(in1,psA)
    v2 = system2.lc['lam_v1'](in2) + a.g_lam(in2,psB)

    d1 = a.system1.pardict['del0']

    GA = (pdA['K0']*(v1+v2)/2/pdA['kc0']+d1)*pdA['om_fix0']
    GB = (pdB['K1']*(v2+v1)/2/pdB['kc1'])*pdB['om_fix1']

    dthA = eps*np.sum(zA*GA)*ds/Tm
    dthB = eps*np.sum(zB*GB)*ds/Tm
    dphi = (dthA - dthB)
    
    dpsA = (k1*psA + eps*np.sum(iA*GA)*ds/Tm)
    dpsB =  k2*psB + eps*np.sum(iB*GB)*ds/Tm
    
    return np.array([dphi,dpsA,dpsB])


def _redu_3dc_thal(t,y,a,eps:float=.01,del1=None):
    """
    thal-specific 3d (1 phase diff, 2 isostables)
    """
    
    s1 = a.system1;s2 = a.system2
    phi,psA,psB = y
    k1 = s1.kappa_val;k2 = s2.kappa_val

    s,ds = np.linspace(0,2*np.pi*a._m[1],5000,retstep=True,endpoint=False)
    Tm = s[-1]
    om = a._n[1]/a._m[1]
    in1 = phi+a.om*s;in2 = s

    dthA = np.sum(s1.gz_lam(in1,in2,psA,psB))*ds/Tm
    dthB = np.sum(s2.gz_lam(in1,in2,psA,psB))*ds/Tm
    dphi = eps*(dthA - dthB)
    
    dpsA = k1*psA + eps*np.sum(s1.gi_lam(in1,in2,psA,psB))*ds/Tm
    dpsB = k2*psB + eps*np.sum(s2.gi_lam(in1,in2,psA,psB))*ds/Tm
    
    return np.array([dphi,dpsA,dpsB])


def _redu_3dc_gwt(t,y,a,eps:float=.01,del1=None):
    """
    thal-specific 3d (1 phase diff, 2 isostables)
    """
    
    s1 = a.system1;s2 = a.system2
    phi,psA,psB = y
    k1 = s1.kappa_val;k2 = s2.kappa_val

    s,ds = np.linspace(0,2*np.pi*a._m[1],3000,retstep=True,endpoint=False)
    Tm = s[-1]
    om = a._n[1]/a._m[1]
    in1 = phi+a.om*s;in2 = s

    dthA = np.sum(s1.gz_lam(in1,in2,psA,psB))*ds/Tm
    dthB = np.sum(s2.gz_lam(in1,in2,psA,psB))*ds/Tm
    dphi = eps*(dthA - dthB)
    
    dpsA = k1*psA + eps*np.sum(s1.gi_lam(in1,in2,psA,psB))*ds/Tm
    dpsB = k2*psB + eps*np.sum(s2.gi_lam(in1,in2,psA,psB))*ds/Tm
    
    return np.array([dphi,dpsA,dpsB])


def _redu_4dc_thal(t,y,a,eps=.01,del1=None):
    """
    thal-specific 3d (1 phase diff, 2 isostables)
    """
    
    s1 = a.system1;s2 = a.system2
    thA,psA,thB,psB = y
    k1 = s1.kappa_val;k2 = s2.kappa_val

    #dper = eps*np.mean(a.system1.z['dat'][0][:,0])
    #dper += eps**2*np.mean(a.system1.z['dat'][1][:,0])

    s,ds = np.linspace(0,2*np.pi*a._m[1],5000,retstep=True,endpoint=False)
    
    Tm = s[-1]
    om = a._n[1]/a._m[1]
    in1 = thA-om*thB + om*s;in2 = s
    
    dthA = eps*np.sum(s1.gz_lam(in1,in2,psA,psB))*ds/Tm
    dthB = eps*np.sum(s2.gz_lam(in1,in2,psA,psB))*ds/Tm
    
    dpsA = k1*psA + eps*np.sum(s1.gi_lam(in1,in2,psA,psB))*ds/Tm
    dpsB = k2*psB + eps*np.sum(s2.gi_lam(in1,in2,psA,psB))*ds/Tm
    
    return np.array([dthA,dpsA,dthB,dpsB])
