import numpy as np
import sympy as sym

Matrix = sym.Matrix

def rhs(t, y, pdict, option='val', idx=''):

    idx = str(idx)
    alpha = pdict['alpha'+idx];beta=pdict['beta'+idx]
    gamma = pdict['gamma'+idx];eta = pdict['eta'+idx]
    mu=pdict['mu'+idx];kappa = pdict['kappa'+idx];
    rho = pdict['rho'+idx];
    # psi = pdict['psi'+idx]
    om = pdict['om'+idx]; om_fix = pdict['om_fix'+idx]

    if option in ['value', 'val']:
        exp = np.exp
    elif option in ['sym', 'symbolic']:
        exp = sym.exp
    
    S, I, v = y

    S *= 0.5
    I *= .04
    q = -kappa*(beta * I - rho - mu * (1 - 2 * v))
    
    dS = alpha * (1 - S - I) - beta * S * I - eta * v * S
    dI = -gamma * I + beta * S * I
    dv = -v + 1 / (1 + exp(q))

    if option in ['value', 'val']:
        return pdict['om_fix'+str(idx)]*np.array([dS/0.5, dI/.04, dv])
    elif option in ['sym', 'symbolic']:
        return pdict['om_fix'+str(idx)]*Matrix([dS/0.5, dI/.04, dv])
    else:
        raise ValueError('Invalid option for right-hand side', option)


def coupling_social_eta(vars_pair,pdict,hetdict,option='val',idx=''):
    r"""
    """
    
    s0,i0,v0 ,  s1,i1,v1 = vars_pair

    s0 *= 0.5
    s1 *= 0.5
    
    i0 *= .04
    i1 *= .04

    idx = str(idx)
    # parameter aliases for readability
    alpha = pdict['alpha'+idx];beta=pdict['beta'+idx]
    gamma = pdict['gamma'+idx];eta = pdict['eta'+idx]
    mu=pdict['mu'+idx];kappa = pdict['kappa'+idx];
    rho = pdict['rho'+idx];
    
    psi = pdict['psi'+idx]
    om = pdict['om'+idx]; om_fix = pdict['om_fix'+idx]

    eps = hetdict['eps'+idx]
    d0 = hetdict['del0']
    aa = pdict['aa'+idx]

    if option in ['value', 'val']:
        exp = np.exp
    elif option in ['sym', 'symbolic']:
        exp = sym.exp
    
    xi0 = -kappa*(beta * i0 - rho - mu * (1 - 2 * v0))
    
    dsigma = -exp(xi0)/(1+exp(xi0))**2

    xi1 = -kappa*2*mu*(v1-v0)

    if idx in [0,'0']:
        aterm = 1+aa
    else:
        aterm = 1-aa
    out1 = [0,0,eps*aterm*dsigma*xi1]
    
    if idx in [0,'0']:
        out1[0] -= d0*v0*s0

    out1[0] /= 0.5
    out1[1] /= .04
    
    if option in ['val','value']:
        return om_fix*om*np.array(out1)
    
    elif option in ['sym','symbolic']:
        return om_fix*om*Matrix(out1)


def coupling_physical_eta(vars_pair,pdict,hetdict,option='val',idx=''):
    r"""
    """
    
    s0,i0,v0 ,  s1,i1,v1 = vars_pair

    s0 *= 0.5
    s1 *= 0.5

    i0 *= .04
    i1 *= .04

    idx = str(idx)
    # parameter aliases for readability
    alpha = pdict['alpha'+idx];beta=pdict['beta'+idx]
    gamma = pdict['gamma'+idx];eta = pdict['eta'+idx]
    mu=pdict['mu'+idx];kappa = pdict['kappa'+idx];
    rho = pdict['rho'+idx];
    
    psi = pdict['psi'+idx]
    om = pdict['om'+idx]; om_fix = pdict['om_fix'+idx]

    eps = hetdict['eps'+idx]
    d0 = hetdict['del0']
    aa = pdict['aa'+idx]

    if option in ['value', 'val']:
        exp = np.exp
    elif option in ['sym', 'symbolic']:
        exp = sym.exp
    
    xi0 = -kappa*(beta * i0 - rho - mu * (1 - 2 * v0))
    
    dsigma = -exp(xi0)/(1+exp(xi0))**2

    xi1 = -kappa*beta*(i1-i0)
    if idx in [0,'0']:
        aterm = 1+aa
    else:
        aterm = 1-aa
    out1 = [eps*aterm*beta*s0*(i0-i1),
            eps*aterm*beta*s0*(i1-i0),
            eps*aterm*dsigma*xi1]

    if idx in [0,'0']:
        out1[0] -= d0*v0*s0

    out1[0] /= 0.5
    out1[1] /= .04
    
    if option in ['val','value']:
        return om_fix*om*np.array(out1)
    
    elif option in ['sym','symbolic']:
        return om_fix*om*Matrix(out1)

def coupling_both_eta(vars_pair,pdict,hetdict,option='val',idx=''):
    r"""
    """
    
    s0,i0,v0 ,  s1,i1,v1 = vars_pair
    s0 *= 0.5
    s1 *= 0.5

    i0 *= .04
    i1 *= .04

    idx = str(idx)
    # parameter aliases for readability
    alpha = pdict['alpha'+idx];beta=pdict['beta'+idx]
    gamma = pdict['gamma'+idx];eta = pdict['eta'+idx]
    mu=pdict['mu'+idx];kappa = pdict['kappa'+idx];
    rho = pdict['rho'+idx];
    
    psi = pdict['psi'+idx]
    om = pdict['om'+idx]; om_fix = pdict['om_fix'+idx]

    eps = hetdict['eps'+idx]
    d0 = hetdict['del0']
    aa = pdict['aa'+idx]

    if option in ['value', 'val']:
        exp = np.exp
    elif option in ['sym', 'symbolic']:
        exp = sym.exp
    
    xi0 = -kappa*(beta * i0 - rho - mu * (1 - 2 * v0))
    
    dsigma = -exp(xi0)/(1+exp(xi0))**2
    
    xi1 = -kappa*(beta*(i1-i0) + 2*mu*(v1-v0))
    
    if idx in [0,'0']:
        aterm = 1+aa
    else:
        aterm = 1-aa
        
    out1 = [eps*aterm*beta*s0*(i0-i1),
            eps*aterm*beta*s0*(i1-i0),
            eps*aterm*dsigma*xi1]

    
    if idx in [0,'0']:
        out1[0] -= d0*v0*s0

    out1[0] /= 0.5
    out1[1] /= .04
    
    if option in ['val','value']:
        return om_fix*om*np.array(out1)
    
    elif option in ['sym','symbolic']:
        return om_fix*om*Matrix(out1)


def coupling_social_gamma(vars_pair,pdict,hetdict,option='val',idx=''):
    r"""
    """
    
    s0,i0,v0 ,  s1,i1,v1 = vars_pair

    s0 *= 0.5
    s1 *= 0.5
    
    i0 *= .04
    i1 *= .04

    idx = str(idx)
    # parameter aliases for readability
    alpha = pdict['alpha'+idx];beta=pdict['beta'+idx]
    gamma = pdict['gamma'+idx];eta = pdict['eta'+idx]
    mu=pdict['mu'+idx];kappa = pdict['kappa'+idx];
    rho = pdict['rho'+idx];
    
    psi = pdict['psi'+idx]
    om = pdict['om'+idx]; om_fix = pdict['om_fix'+idx]

    eps = hetdict['eps'+idx]
    d0 = hetdict['del0']
    aa = pdict['aa'+idx]

    if option in ['value', 'val']:
        exp = np.exp
    elif option in ['sym', 'symbolic']:
        exp = sym.exp
    
    xi0 = -kappa*(beta * i0 - rho - mu * (1 - 2 * v0))
    
    dsigma = -exp(xi0)/(1+exp(xi0))**2

    xi1 = -kappa*2*mu*(v1-v0)
    
    if idx in [0,'0']:
        aterm = 1+aa
    else:
        aterm = 1-aa
    out1 = [0,0,eps*aterm*dsigma*xi1]
    
    if idx in [0,'0']:
        out1[1] -= d0*i0

    out1[0] /= 0.5
    out1[1] /= .04
    
    if option in ['val','value']:
        return om_fix*om*np.array(out1)
    
    elif option in ['sym','symbolic']:
        return om_fix*om*Matrix(out1)

def coupling_physical_gamma(vars_pair,pdict,hetdict,option='val',idx=''):
    r"""
    """
    
    s0,i0,v0 ,  s1,i1,v1 = vars_pair

    s0 *= 0.5
    s1 *= 0.5

    i0 *= .04
    i1 *= .04

    idx = str(idx)
    # parameter aliases for readability
    alpha = pdict['alpha'+idx];beta=pdict['beta'+idx]
    gamma = pdict['gamma'+idx];eta = pdict['eta'+idx]
    mu=pdict['mu'+idx];kappa = pdict['kappa'+idx];
    rho = pdict['rho'+idx];
    
    psi = pdict['psi'+idx]
    om = pdict['om'+idx]; om_fix = pdict['om_fix'+idx]

    eps = hetdict['eps'+idx]
    d0 = hetdict['del0']
    aa = pdict['aa'+idx]

    if option in ['value', 'val']:
        exp = np.exp
    elif option in ['sym', 'symbolic']:
        exp = sym.exp
    
    xi0 = -kappa*(beta * i0 - rho - mu * (1 - 2 * v0))
    
    dsigma = -exp(xi0)/(1+exp(xi0))**2

    xi1 = -kappa*beta*(i1-i0)

    if idx in [0,'0']:
        aterm = 1+aa
    else:
        aterm = 1-aa
        
    out1 = [eps*aterm*beta*s0*(i0-i1),
            eps*aterm*beta*s0*(i1-i0),
            eps*aterm*dsigma*xi1]

    if idx in [0,'0']:
        out1[1] -= d0*i0

    out1[0] /= 0.5
    out1[1] /= .04
    
    if option in ['val','value']:
        return om_fix*om*np.array(out1)
    
    elif option in ['sym','symbolic']:
        return om_fix*om*Matrix(out1)

def coupling_both_gamma(vars_pair,pdict,hetdict,option='val',idx=''):
    r"""
    """
    
    s0,i0,v0 ,  s1,i1,v1 = vars_pair
    s0 *= 0.5
    s1 *= 0.5

    i0 *= .04
    i1 *= .04

    idx = str(idx)
    # parameter aliases for readability
    alpha = pdict['alpha'+idx];beta=pdict['beta'+idx]
    gamma = pdict['gamma'+idx];eta = pdict['eta'+idx]
    mu=pdict['mu'+idx];kappa = pdict['kappa'+idx];
    rho = pdict['rho'+idx];
    
    psi = pdict['psi'+idx]
    om = pdict['om'+idx]; om_fix = pdict['om_fix'+idx]

    eps = hetdict['eps'+idx]
    d0 = hetdict['del0']
    aa = pdict['aa'+idx]

    if option in ['value', 'val']:
        exp = np.exp
    elif option in ['sym', 'symbolic']:
        exp = sym.exp
    
    xi0 = -kappa*(beta * i0 - rho - mu * (1 - 2 * v0))
    
    dsigma = -exp(xi0)/(1+exp(xi0))**2
    
    xi1 = -kappa*(beta*(i1-i0) + 2*mu*(v1-v0))

    if idx in [0,'0']:
        aterm = 1+aa
    else:
        aterm = 1-aa
        
    out1 = [eps*aterm*beta*s0*(i0-i1),
            eps*aterm*beta*s0*(i1-i0),
            eps*aterm*dsigma*xi1]
    
    if idx in [0,'0']:
        out1[1] -= d0*i0

    out1[0] /= 0.5
    out1[1] /= .04
    
    if option in ['val','value']:
        return om_fix*om*np.array(out1)
    
    elif option in ['sym','symbolic']:
        return om_fix*om*Matrix(out1)
