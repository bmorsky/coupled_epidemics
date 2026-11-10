from sirv2 import rhs, coupling_social_eta, coupling_physical_eta, coupling_both_eta, coupling_social_gamma, coupling_physical_gamma, coupling_both_gamma

import couplingPI as cpi
import response as rp

import sympy as sym

import matplotlib.pyplot as plt

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import copy
from copy import deepcopy

Matrix = sym.Matrix
CPI = cpi.couplingPI
rsp = rp.Response

plt.rcParams.update({
    "text.usetex": True,
    #"font.family": "Helvetica"
})

def _redu(t, y, a, eps, delta, aa=0):
    """Averaged phase-isostable equations (eq. 20, Nicks et al. 2024), N=2,
    with heterogeneity delta in oscillator 0 (system1).

    phi = th1 - th0. H_k(chi) are stored with chi = th_j - th_i.
    Oscillator i pairs its own psi with H2/H5 and the other's psi with H3/H6.

    Heterogeneous terms (system1 only):
      h['lam'][6] : J_theta, coefficient of delta in the theta equation
      h['lam'][7] : J_psi,   coefficient of delta in the psi equation
    Both depend only on oscillator 0's own phase, so after averaging they are
    constants; they are evaluated once at 0.

    Assumes delta enters at order eps^0, i.e. load_h_sym key (0,0,0,1).
    If h['sym'][6] is zero, delta was multiplied by eps in the coupling
    function: change the key to (1,0,0,1) in load_h_sym and replace delta
    with eps*delta below.
    """
    phi = y

    h1 = a.system1.h['lam'][0]

    J_th = a.system1.h['lam'][6](0.)   # constant theta drift per unit delta
    # J_ps = a.system1.h['lam'][7](0.)   # constant psi forcing per unit delta


    dphi = eps*(1-aa)*h1(-phi) - eps*(1+aa)*h1(phi) - delta*J_th

    return dphi

def get_kw_sim():
    return {'rtol':1e-7,'atol':1e-7,'method':'LSODA'}

def get_pd():
    """model parameters for a single oscillator"""
    pd = dict(alpha = .01, beta = 0.5, gamma = 1/7, eta = 1/7,mu=0.01 , kappa = 300, rho = 0.01,psi = 0.5,  zeta=1,om=1,om_fix=1,aa=0)
    return pd

def load_system(kap=300, load_data_only=False, type1='social', par='eta', recompute_list=['']):

    kw_sim = get_kw_sim()

    # all het. parameters must be named, indexed, and ordered as shown!!
    # for oscillator 1: del_0, del_1, ...
    # for oscillator 2: sig_0, sig_1, ...
    # eps_i is created inside rsp.
    het_pars1 = ['del0','del1']
    het_pars2 = ['sig0','sig1']

    pardict = get_pd()
    pardict['kappa'] = kap
    # print(pardict)
    init0 = [0.17603663,
      0.829848,
      0.99917684]
    t = np.arange(0,1000,.01)
    out = solve_ivp(rhs,[t[0],t[-1]],init0,t_eval=t,args=(pardict,),**kw_sim)
    idxs = find_peaks(out.y[0])[0]
    per = np.diff(t[idxs])[-1]
    init = out.y.T[-1,:]
    init = np.append(init,per)

    if type1 in ['social']:
        if par in ['eta']:
            cc = coupling_social_eta
        elif par in ['gamma','gm']:
            cc = coupling_social_gamma
    elif type1 in ['physical']:
        if par in ['eta']:
            cc = coupling_physical_eta
        elif par in ['gamma','gm']:
            cc = coupling_physical_gamma
    elif type1 in ['both']:
        if par in ['eta']:
            cc = coupling_both_eta
        elif par in ['gamma','gm']:
            cc = coupling_both_gamma
    else:
        raise ValueError('tf are you doing?')
    
    kwargs_jr = dict(var_names=['S', 'I', 'R'], 
                     pardict=copy.deepcopy(pardict), 
                     rhs=rhs,
                     coupling=cc,
                     TN=2000,
                     trunc_order=0,
                     init=init,
                     recompute_list=[],
                     z_forward=False,
                     i_forward=False,
                     i_bad_dx=[False,True,False,False],
                     lc_prominence=0.02,
                     max_iter=5,
                     save_fig=False,
                     max_time_lc=10000,
                     init_is_precise=False)
    
    # always use idx=0 for oscillator 1 and idx=1 for oscillator 2.
    system1=rsp(idx=0, model_name='sirv_0_'+type1+'_k='+str(kap)+'_par='+str(par), het_pars=het_pars1, **copy.deepcopy(kwargs_jr))
    system2=rsp(idx=1, model_name='sirv_1_'+type1+'_k='+str(kap)+'_par='+str(par), het_pars=het_pars2, **copy.deepcopy(kwargs_jr))
    
    kw_nm = {'system1':system1, 'system2':system2, 'NH':1024, 'save_fig':False,'recompute_list':recompute_list}
    a = CPI(**kw_nm)
    return a


def periodic_zero_crossings(x):
    """
    Zero crossings of a periodic sampled signal whose last sample duplicates the first.

    Returns
    -------
    idx : int array
        Left index i of each crossing. The crossing lies between y[i] and y[(i+1) % n].
    t : float array
        Fractional index of each crossing from linear interpolation, in [0, n).
    up : bool array
        True for a rising crossing (- to +), False for a falling one.

    Notes
    -----
    np.signbit counts an exact 0 as positive. A sample that is exactly zero
    therefore gives one crossing instead of two.
    """
    y = np.asarray(x, dtype=float)[:-1]  # drop the duplicate endpoint
    n = y.size

    neg = np.signbit(y)                  # True where y < 0
    neg_next = np.roll(neg, -1)          # sign of y[(i+1) % n]
    idx = np.flatnonzero(neg != neg_next)

    y0 = y[idx]
    y1 = y[(idx + 1) % n]
    t = (idx + y0 / (y0 - y1)) % n       # interpolated position, wrapped into [0, n)
    up = neg[idx]                        # left sample negative means rising

    return idx, t, up


def plot_crossings(eps, delta, kappas_in, systems, M=100, axs=None, aa=0, legend=False, label=None, label_size=12, type1=None):
    """Scatter the zeros of dphi/dt against kappa and return the axes.

    eps, delta : coupling strength and heterogeneity passed to _redu
    M          : samples per period (the endpoint is included and duplicates the first)
    kappas_in  : kappa values
    systems    : CPI objects in the same order as kappas_in
    axs        : existing axes to draw on; a new figure is made if None
    legend     : draw the stable/unstable legend if True
    label      : panel label string, e.g. 'A' or upper_alphabet[i]; None for no label.
                 Drawn as a left-aligned title, so it sits beside the centered title.
    label_size : font size of the label

    Falling crossings (up=False) are stable and rising ones unstable, since dphi/dt = f(phi).
    """
    if axs is None:
        fig, axs = plt.subplots(figsize=(4, 2))

    K_pts, PH_pts, UP_pts = [], [], []
    for kap, a in zip(kappas_in, systems):
        x1 = np.linspace(0, a.T, M)            # phi grid; last point duplicates the first
        y = _redu(0, x1, a, eps, delta, aa=aa) # dphi/dt sampled on one period

        _, t, up = periodic_zero_crossings(y)  # t: fractional index in [0, M-1)
        ph = t * 2*np.pi / (M - 1)             # fractional index -> phase in [0, 2*pi)

        K_pts.append(np.full(t.size, kap))     # one kappa per crossing
        PH_pts.append(ph)
        UP_pts.append(up)

    K_pts  = np.concatenate(K_pts)
    PH_pts = np.concatenate(PH_pts)
    UP_pts = np.concatenate(UP_pts)

    axs.scatter(K_pts[~UP_pts], PH_pts[~UP_pts], s=10, c='k', label='Stable')
    axs.scatter(K_pts[UP_pts],  PH_pts[UP_pts],  s=10, facecolors='none', edgecolors='r', label='Unstable')
    axs.set_xlim(kappas_in[0], kappas_in[-1])
    axs.set_xlabel(r'$\kappa$')
    axs.set_ylabel(r'$\phi$')

    # y axis: ticks only at 0 and 2*pi, with 0.1 padding so points on the boundary aren't clipped
    axs.set_ylim(-0.1, 2*np.pi + 0.1)
    axs.set_yticks([0, 2*np.pi])
    axs.set_yticklabels([r'$0$', r'$2\pi$'])

    # Centered title; matplotlib keeps separate left/center/right title slots, so the label below doesn't replace it.
    axs.set_title(rf'$b={tex_num(delta)}$; $a={tex_num(aa)}$')

    if legend:
        axs.legend()

    if label is not None:

        axs.set_title(label, loc='left', fontsize=label_size,usetex=True)

    return axs




def tex_num(x, sig=3, lo=1e-3, hi=1e4):
    """Format a number as a TeX string (no surrounding $).

    Plain decimal if lo <= |x| < hi, otherwise scientific notation.
      tex_num(0.1)    -> '0.1'
      tex_num(1e-7)   -> '10^{-7}'
      tex_num(2.5e-7) -> '2.5\\times 10^{-7}'
      tex_num(0)      -> '0'
    sig: significant digits kept in the mantissa (trailing zeros dropped).
    """
    if x == 0:
        return '0'
    if lo <= abs(x) < hi:
        return f'{x:.{sig}g}'
    mant, exp = f'{x:.{sig-1}e}'.split('e')  # rounds before splitting, so 9.999e-8 -> 1.00e-07
    mant = mant.rstrip('0').rstrip('.')
    exp = int(exp)
    if mant == '1':
        return f'10^{{{exp}}}'
    if mant == '-1':
        return f'-10^{{{exp}}}'
    return f'{mant}\\times 10^{{{exp}}}'
