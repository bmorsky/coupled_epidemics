"""
Utility library functions
"""
from . import rhs

import time
import os
import dill
import sys

import numpy as np
import scipy as sp
import sympy as sym
import matplotlib.pyplot as plt

import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from scipy.optimize import bisect

from copy import deepcopy

from .rhs import _redu_c

rhs_avg_ld = rhs.rhs_avg_1df
rhs_avg = rhs.rhs_avg_2d

#from scipy.interpolate import interp1d
from sympy.physics.quantum import TensorProduct as kp
#from sympy.utilities.lambdify import lambdify, implemented_function

from scipy.integrate import solve_ivp
kw_bif = {'method':'LSODA','dense_output':True,'rtol':1e-10,'atol':1e-10}

def add_arrow_to_line2D(
        axes, line, arrow_locs=[0.2, 0.4, 0.6, 0.8],
        arrowstyle='-|>', arrowsize=1.2, transform=None,
        tail_space=50):

    if not isinstance(line, mlines.Line2D):
        raise ValueError("expected a matplotlib.lines.Line2D object")
    x, y = line.get_xdata(), line.get_ydata()

    arrow_kw = {
        "arrowstyle": arrowstyle,
        "mutation_scale": 10 * arrowsize,
    }

    color = line.get_color()
    use_multicolor_lines = isinstance(color, np.ndarray)
    #if use_multicolor_lines:
    #    raise NotImplementedError("multicolor lines not supported")
    #else:
    arrow_kw['color'] = color

    linewidth = line.get_linewidth()
    if isinstance(linewidth, np.ndarray):
        raise NotImplementedError("multiwidth lines not supported")
    else:
        arrow_kw['linewidth'] = linewidth

    if transform is None:
        transform = axes.transData

    arrows = []
    for loc in arrow_locs:
        s = np.nancumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        n = np.searchsorted(s, s[-1] * loc)
        arrow_tail = (x[n-tail_space], y[n-tail_space])
        arrow_head = (np.nanmean(x[n:n + 2]),
                      np.nanmean(y[n:n + 2]))
        p = mpatches.FancyArrowPatch(
            arrow_tail, arrow_head, transform=transform,
            **arrow_kw)
        axes.add_patch(p)
        arrows.append(p)
    return arrows



def get_phase(t,sol_arr,skipn,system1):
    """
    Get the brute-force phase estimate over time
    t: time array
    sol_arr: solution array (must include all dimensions of 
    one and only one oscillator)
    skipn: number of points to skip
    system1: object of response functions
    """

    phase1 = np.zeros(len(t[::skipn]))
    for i in range(len(t[::skipn])):
        d1 = np.linalg.norm(sol_arr[::skipn][i,:]-system1.lc['dat'],axis=1)
        idx1 = np.argmin(d1)
        
        phase1[i] = idx1/len(system1.lc['dat'])
        
    return t[::skipn],2*np.pi*phase1


def freq_est(t,y,transient=.5,width=10,prominence=.05,return_idxs=False):
    """ 
    Estimate the oscillator frequency
    For use only with the frequency ratio plots
    """
    
    peak_idxs = sp.signal.find_peaks(y,width=width,prominence=prominence)[0]
    peak_idxs = peak_idxs[int(len(peak_idxs)*transient):]
    freq = 2*np.pi/np.mean(np.diff(t[peak_idxs]))
    
    if return_idxs:
        return freq,peak_idxs
    else:
        return freq

def pl_exist_1d(eps,del1,a,th_init=0,return_data=False):
    """
    
    """
    sys1 = a.system1
    
    if eps == 0:
        return -1
    err = 1
    
    th_temp = np.linspace(0, 2*np.pi, 200)
    rhs = rhs_avg_ld(0,th_temp,a,eps,del1)

    if return_data:
        return th_temp,rhs
    
    if np.abs(np.sum(np.sign(rhs))) < len(rhs):
        return 1
    else:
        return -1
    
def get_tongue_1d(del_list,a,deps=.002,max_eps=.3,min_eps=0):
    """
    Get the Arnold tongue for the low-dim reduced model
    (where the isostable coordinate was eliminated)
    """
    ve_exist = np.zeros(len(del_list))
    
    for i in range(len(del_list)):
        print(np.round((i+1)/len(del_list),2),'    ',end='\r')

        if np.isnan(ve_exist[i-1]):
            eps = 0
        else:
            eps = max(ve_exist[i-1] - 2*deps,0)
        while not(pl_exist_ld(eps,del_list[i],a)+1)\
        and eps <= max_eps:
            eps += deps
            #print('existloop',eps)
        if eps >= max_eps:
            ve_exist[i] = np.nan
        else:
            deps2 = deps
            flag = False
            while not(flag) and deps2 < .2:
                #print('while loop',deps2)
                try:
                    out = bisect(pl_exist_ld,0,eps+deps2,args=(del_list[i],a))
                    flag = True
                except ValueError:
                    deps2 += .001
            if flag:
                ve_exist[i] = out
            else:
                ve_exist[i] = np.nan
    print('')
    return del_list,ve_exist

def is_stable(J):
    u,v = np.linalg.eig(J)
    print('eigs',u)
    print('')
    # if all real parts are negative, return true
    if np.sum(np.real(u)<0) == len(u):
        return True
    else:
        return False

def jac_2d(y,a,eps,del1,h=.01):
    """
    Jacobian for 2d averaged system (forcing only)
    """
    print('y',y,'eps',eps,'del1',del1)
    dx = np.array([h,0])
    dy = np.array([0,h])

    args = (a,eps,del1)
    
    c1 = (rhs.rhs_avg_2d(0,y+dx,*args)-rhs.rhs_avg_2d(0,y,*args))/h
    c2 = (rhs.rhs_avg_2d(0,y+dy,*args)-rhs.rhs_avg_2d(0,y,*args))/h

    return np.array([c1,c2])


import collections
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spatial
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hier


def intersection(points1, points2, eps):
    tree = spatial.KDTree(points1)
    distances, indices = tree.query(points2, k=1, distance_upper_bound=eps)
    intersection_points = tree.data[indices[np.isfinite(distances)]]
    return intersection_points


def cluster(points, cluster_size):
    dists = dist.pdist(points, metric='sqeuclidean')
    linkage_matrix = hier.linkage(dists, 'average')
    groups = hier.fcluster(linkage_matrix, cluster_size, criterion='distance')
    return np.array([points[cluster].mean(axis=0)
                     for cluster in clusterlists(groups)])


def contour_points(contour, steps=1):
    for linecol in contour.collections:
        
        if len(linecol.get_paths()) == 0:
            return 0
    
    return np.row_stack([path.interpolated(steps).vertices
                         for linecol in contour.collections
                         for path in linecol.get_paths()])

def clusterlists(T):
    '''
    http://stackoverflow.com/a/2913071/190597 (denis)
    T = [2, 1, 1, 1, 2, 2, 2, 2, 2, 1]
    Returns [[0, 4, 5, 6, 7, 8], [1, 2, 3, 9]]
    '''
    groups = collections.defaultdict(list)
    for i, elt in enumerate(T):
        groups[elt].append(i)
    return sorted(groups.values(), key=len, reverse=True)


def period_is_multi(sol,idx:int=0,prominence:float=.25):
    """
    check if period is uneven during n:m locking
    assumes transients have decayed
    """

    tn = len(sol.y.T)
    peak_idxs = sp.signal.find_peaks(sol.y.T[:,idx],
                                     prominence=prominence)[0]
    maxidx1 = peak_idxs[-2]    
    maxidx_prev1 = peak_idxs[-3]

    maxidx2 = peak_idxs[-3]    
    maxidx_prev2 = peak_idxs[-4]

    
    def sol_min(t):
        return -sol.sol(t)[idx]

    # get stronger estimates of max values
    # use optimization on ode solution
    pad1lo = (sol.t[maxidx1]-sol.t[maxidx1-1])/2
    pad1hi = (sol.t[maxidx1+1]-sol.t[maxidx1])/2
    bounds1 = [sol.t[maxidx1]-pad1lo,sol.t[maxidx1]+pad1hi]
    res1 = sp.optimize.minimize_scalar(sol_min,bounds=bounds1)

    pad2lo = (sol.t[maxidx_prev1]-sol.t[maxidx_prev1-1])/2
    pad2hi = (sol.t[maxidx_prev1+1]-sol.t[maxidx_prev1])/2
    bounds2 = [sol.t[maxidx_prev1]-pad2lo,sol.t[maxidx_prev1]+pad2hi]
    res2 = sp.optimize.minimize_scalar(sol_min,bounds=bounds2)

    T_init = res1.x - res2.x

    
    pass

def get_period(sol,idx:int=0,prominence:float=.25,idx_shift:int=0,sort_by='max'):
    """
    sol: solution object from solve_ivp
    idx: index of desired variable
    nm: n:m locking numbers    
    """
    
    tn = len(sol.y.T)
    peak_idxs = sp.signal.find_peaks(sol.y.T[:,idx],prominence=prominence)[0]
    n_half = int(2.5*len(peak_idxs)/5) # make this later

    # instead of always using the last points
    # start with the max idx after 50% of the simulation as a starting point.

    # sorted indices of peak values
    if sort_by == 'max':
        peak_idxs_sorted = np.argsort(sol.y.T[peak_idxs[n_half:],idx])[::-1]
    elif sort_by == 'min':
        peak_idxs_sorted = np.argsort(sol.y.T[peak_idxs[n_half:],idx])
    else:
        raise ValueError('Invalid option for peak sorting',sort_by)

    #print('peak_idxs_sorted',peak_idxs_sorted)
    if peak_idxs_sorted[0] == 0:
        idx_temp = peak_idxs_sorted[1]
    else:
        idx_temp = peak_idxs_sorted[0]
        
    maxidx = peak_idxs[n_half:][idx_temp]
    maxidx_prev = peak_idxs[n_half:][idx_temp-1]

    def sol_min(t):
        return -sol.sol(t)[idx]

    # get stronger estimates of max values
    # use optimization on ode solution
    pad1t = (sol.t[maxidx]-sol.t[maxidx-5])/2
    bounds1 = [sol.t[maxidx]-pad1t,sol.t[maxidx]+pad1t]
    res1 = sp.optimize.minimize_scalar(sol_min,bounds=bounds1)

    pad2t = (sol.t[maxidx_prev]-sol.t[maxidx_prev-5])/2
    bounds2 = [sol.t[maxidx_prev]-pad2t,sol.t[maxidx_prev]+pad2t]
    res2 = sp.optimize.minimize_scalar(sol_min,bounds=bounds2)

    T_init = res1.x - res2.x
    
    return T_init,res1.x


def get_period_v2(sol,a,idx:int=0,prominence:float=.25,idx_shift:int=0,sort_by='max',c_sign=None,use_point=None):
    """
    return all periods for first oscillator
    brute force it.
    
    sol: solution object from solve_ivp
    idx: index of desired variable
    nm: n:m locking numbers    
    """
    
    tn = len(sol.y.T)
    peak_idxs = sp.signal.find_peaks(sol.y.T[:,idx],prominence=prominence)[0]


    peak_idxs = peak_idxs[1:]
    peak_times = sol.t[peak_idxs]
    peak2peak_times = np.diff(peak_times)

    return np.sum(peak2peak_times[:a._n[1]])/a._n[1]


def get_period_v3(sol,a,idx:int=0,prominence:float=.25,idx_shift:int=0,sort_by='max'):
    """
    return all periods for first oscillator
    brute force it.
    
    sol: solution object from solve_ivp
    idx: index of desired variable
    nm: n:m locking numbers    
    """
    
    tn = len(sol.y.T)
    peak_idxs = sp.signal.find_peaks(sol.y.T[:,idx],prominence=prominence)[0]


    peak_idxs = peak_idxs[1:]
    peak_times = sol.t[peak_idxs]
    peak2peak_times = np.diff(peak_times)
    
    if idx == 0:
        return peak2peak_times[:a._n[1]]
    else:
        return peak2peak_times[:a._m[1]]

def get_period_v2b(sol,a,idx:int=0,prominence:float=.25,idx_shift:int=0,sort_by='max'):
    """
    Just making a separate function... i know it's not good coding but i'm tired.
    return all periods for first oscillator
    brute force it.
    
    sol: solution object from solve_ivp
    idx: index of desired variable
    nm: n:m locking numbers    
    """
    
    tn = len(sol.y.T)
    peak_idxs = sp.signal.find_peaks(sol.y.T[:,idx],prominence=prominence)[0]


    peak_idxs = peak_idxs[1:]
    peak_times = sol.t[peak_idxs]
    peak2peak_times = np.diff(peak_times)

    return np.sum(peak2peak_times[:a._n[1]]),peak_times[-2]


def get_init_point_force(fun,a,eps,del1,phi0,max_time=200,dt=.04,c_sign=1):
    """
    Get a first estimate of the initial condition given eps, delta, phase.

    It doesn't necessarily need to get to steady-state, since Newton's method will take care of that.

    If there is no convergence to steady-state, that will be detected when Newton's method fails to converge.
    """

    init_idx = int(-phi0*a.system1.TN/(2*np.pi))
    init = a.system1.lc['dat'][init_idx,:]
    
    t = np.arange(0,max_time,dt)
    sol = solve_ivp(fun,[0,max_time],init,args=(a,eps,del1,c_sign),t_eval=t,**kw_bif)
    
    T_est = 2*np.pi/(a._m[1]+del1) # just the forcing period
    
    # get index of n*T later in simulation, at 90% of the total time (90% is arbitrary)
    multiple = int(0.9*max_time/T_est)
    idx = np.argmin(np.abs(sol.t-multiple*T_est))
    Y_est = sol.y[:,idx]

    return Y_est,T_est

def get_dy_nm_force(Y,T_est,args_temp,eps_pert=1e-4,eps_pert_time=1e-4,return_sol=False):
    """
    Get dy for phase locking state in full model with forcing
    ignore the time variable. this must be chosen as the forcing function period.

    If there is no phase-locking, then Newton's method will not converge given a fixed period.

    Y: starting point from which to estimate the Jacobian
    T_est: estimated period, just the forcing period.
    """

    J = np.zeros([len(Y),len(Y)])

    for jj in range(len(Y)):
        pert = np.zeros(len(Y));pert[jj] = eps_pert

        t_span = [0,T_est]
        solp = solve_ivp(y0=Y+pert,t_span=t_span,**args_temp)
        solm = solve_ivp(y0=Y-pert,t_span=t_span,**args_temp)

        # Jacobian
        J[:,jj] = (solp.y.T[-1,:]-solm.y.T[-1,:])/(2*eps_pert)

    J = J - np.eye(len(Y))

    solb = solve_ivp(y0=Y,t_span=t_span,**args_temp)

    b = np.array(list(Y-solb.y.T[-1,:]))
    dy = np.linalg.solve(J,b)

    if return_sol:
        return dy,solb
    else:
        return dy


def get_period_all(sol,a,idx:int=0,prominence:float=.25,
                   idx_shift:int=0,sort_by='max',tol=8e-1)->float:
    """
    Gets each peak-to-peak period T_i of a given oscillator,
    for i = 1,...,n, where n is the number of peak to peak
    periods that add up to 2pi (the oscillator locking number).

    Assume that the input has already converged, and that the solution
    is already at steady-state in terms of phase differences.
    
    sol: solution object from solve_ivp
    a: coupling object
    idx: index of desired variable
    sort_by: sort peak by max of the peaks or the mins of the peaks
    tol: phase drift tolerance

    Returns n numbers. T1,...Tn

    It's just really rare for there to be nonequal peaks with different periods.
    Small eps means peak-to-peak periods and amplitudes are generally the same.
    Greater eps means peak-to-peak periods differ along with amplitudes.
    So picking the largest peak amplitude (after transients) is a reasonable
    starting point.
    """

    n = a._n[1]
    tn = len(sol.y.T)
    peak_idxs = sp.signal.find_peaks(sol.y.T[:,idx],prominence=prominence)[0]

    # start with the max idx after 50% of the simulation as a starting point.
    #n_half = int(len(peak_idxs)/2)
    # 
    
    # get sorted indices of peak values
    if sort_by == 'max':
        peak_idxs_sorted = np.argsort(sol.y.T[peak_idxs,idx])[::-1]
    elif sort_by == 'min':
        peak_idxs_sorted = np.argsort(sol.y.T[peak_idxs,idx])
    else:
        raise ValueError('Invalid option for peak sorting',sort_by)

    #print('peak_idxs_sorted',peak_idxs_sorted[:10])
    #print('values of peaks',sol.y.T[peak_idxs[n_half:],idx][peak_idxs_sorted[:10]])

    # avoids getting a peak at zero or last index and having issues with opt.
    
    choice = 0
    idx_temp = peak_idxs_sorted[choice]
    nn = len(peak_idxs)
    while idx_temp in [0,nn,nn-1,nn-2,nn-3,nn-4]:
        #print('idx_temp choice',choice,peak_idxs_sorted[choice])
        choice += 1
        idx_temp = peak_idxs_sorted[choice]

    #print('idx_temp',idx_temp)

    # the max idx is taken to be at the largest of all peaks
    # also get next n peaks
    maxidxs = peak_idxs[idx_temp:idx_temp+n]
    maxidxs_prev = peak_idxs[idx_temp-1:idx_temp-1+n]

    #print('maxidxs',maxidxs)
    #print('maxidxs_prev',maxidxs_prev)

    def sol_min(t):
        return -sol.sol(t)[idx]

    if False:
        fig,axs = plt.subplots(1,2)

        t1 = sol.t[maxidxs[0]]-50
        t2 = sol.t[maxidxs[0]]+50
        t1_idx = np.argmin(np.abs(sol.t-t1))
        t2_idx = np.argmin(np.abs(sol.t-t2))
        
        axs[0].scatter(sol.t[maxidxs],sol.y[0][maxidxs])
        axs[0].plot(sol.t[t1_idx:t2_idx],sol.y[0][t1_idx:t2_idx])
        axs[0].set_xlim(sol.t[maxidxs[0]]-50,sol.t[maxidxs[0]]+50)

        x = sol.t[peak_idxs]
        y = np.mod(sol.t[peak_idxs]*n,2*np.pi/(1+a.del1))
        axs[1].plot(x,y)

        d1 = np.abs(y[0]-y[-1])
        d2 = np.abs(y[int(len(y)/4)]-y[int(3*len(y)/4)])
        print('diff',d1+d2)
        plt.show()

    # peak solution indices are zero phase in the oscillator.
    # the modulus is the forcing function period.
    y = np.mod(sol.t[peak_idxs]*n,2*np.pi/(a._m[1]+a.del1)*a._m[1])

    Tlist = []

    for maxidx,maxidx_prev in zip(maxidxs,maxidxs_prev):
       # get stronger estimates of max values
       # use optimization on ode solution
       pad1t = (sol.t[maxidx]-sol.t[maxidx-5])/2
       bounds1 = [sol.t[maxidx]-pad1t,sol.t[maxidx]+pad1t]
       res1 = sp.optimize.minimize_scalar(sol_min,bounds=bounds1)

       pad2t = (sol.t[maxidx_prev]-sol.t[maxidx_prev-5])/2
       bounds2 = [sol.t[maxidx_prev]-pad2t,sol.t[maxidx_prev]+pad2t]
       res2 = sp.optimize.minimize_scalar(sol_min,bounds=bounds2)

       T_init = res1.x - res2.x

       Tlist.append((T_init,res1.x))

    #print('tlist,y[-1]',Tlist,y[-1])
    return (Tlist,y[-1])


def newton_iter_force(Y_est,T_est,fun,a,eps,del1,max_iter=50,tol=1e-6):
    print('Y_est',Y_est)
    dy = np.ones(len(Y_est))/2 # initial dy
    counter = 0
    
    while (np.linalg.norm(dy) > tol) and (counter <= max_iter):
        args_temp = {'fun':fun,'args':(a,eps,del1),**kw_bif}
        dy = get_dy_nm_force(Y_est,a._m[1]*T_est,args_temp)
    
        Y_est += dy
        counter += 1
        #print('counter,Y,dy',counter,Y_est,dy)

    if counter >= max_iter:
        return np.full((len(Y_est),),np.nan)
    else:
        return Y_est


    
def run_bif1d_f(rhs,a,del1,eps_init,eps_final,deps,phi0=0,max_iter:int=25,tol:float=1e-5,
period_multiple=None,c_sign=1):
    """
    get phase-locked initial conditions
    for forcing only
    """

    print('phi0 in run_bif1d_f',phi0)
    eps_range = np.arange(eps_init,eps_final,deps)

    inits = np.zeros([len(eps_range),len(a.system1.var_names)])

    Y_prev,T_est = get_init_point_force(rhs,a,eps_init,del1,phi0)
    
    print('Y_prev,T_est',Y_prev,T_est,del1,eps_init,eps_final,deps,phi0)
    
    for i,eps in enumerate(eps_range):

        if np.isnan(Y_prev).sum() > 0:
            Y_est,T_est = get_init_point_force(rhs,a,eps,del1,phi0,c_sign=c_sign)
        else:
            Y_est = Y_prev
            T_est = 2*np.pi/(a._m[1]+del1) # just the forcing period

        kws = {'Y_est':Y_est,'T_est':T_est,'fun':rhs,'a':a,'eps':eps,'del1':del1, 'max_iter':max_iter,'tol':tol}
        Y_est = newton_iter_force(**kws)

        Y_prev = deepcopy(Y_est)
        
        counter = 0
        print('{}, eps={:.5f}'.format(i,eps),Y_est)
        
        inits[i,:] = Y_est
        
    return eps_range,inits


def load_bif1d_f(rhs,a,del1,etup,recompute=False,data_dir='bifdat',phi0=0,c_sign=1,**kwargs):

    
    eps_init,eps_final,deps = etup

    data_dir = os.getcwd()+'/'+data_dir
    if not(os.path.isdir(data_dir)):
        os.mkdir(data_dir)

    model_name = a.system1.model_name
    ratio = str(a._n[1])+str(a._m[1])
    fpars = [data_dir,model_name,ratio,del1,eps_init,eps_final,deps,phi0]
    
    raw = ('{}/bif1_inits_{}_ratio={}_d={}_elo={}_ehi={}_de={}_phi0={}.txt')

    fname = raw.format(*fpars)
    file_dne  = not(os.path.isfile(fname))
    
    if file_dne or recompute:
        erange,inits = run_bif1d_f(rhs,a,del1,eps_init,eps_final,deps,phi0=phi0,c_sign=c_sign,**kwargs)

        data = np.zeros([len(erange),len(inits.T)+1])
        data[:,0] = erange
        data[:,1:] = inits
        np.savetxt(fname,data)
    else:
        data = np.loadtxt(fname)

    return data


def run_bif1d_f_u(rhs,a,del1,eps_init,eps_final,deps,phi0=0,max_iter:int=25,tol:float=1e-5,
                  use_point=None,return_point=False,period_multiple=None):
    """
    get phase-locked initial conditions
    for forcing only
    """

    print('phi0 in run_bif1d_f',phi0)
    eps_range = np.arange(eps_init,eps_final,deps)

    inits = np.zeros([len(eps_range),len(a.system1.var_names)])

    init_idx = int(-phi0*a.system1.TN/(2*np.pi))
    if use_point is None:
        Y_prev = a.system1.lc['dat'][init_idx,:]
    else:
        Y_prev = use_point

    _,T_est = get_init_point_force(rhs,a,eps_init,del1,phi0)    
    
    for i,eps in enumerate(eps_range):
        #print('Y_prev',Y_prev)

        if np.isnan(Y_prev).sum() > 0:
            init_idx = int(-phi0*a.system1.TN/(2*np.pi))
            if use_point is None:
                Y_prev = a.system1.lc['dat'][init_idx,:]
            else:
                Y_prev = use_point
            #print('phi0',phi0)
            _,T_est = get_init_point_force(rhs,a,eps,del1,phi0)
        else:
            Y_est = Y_prev
            T_est = 2*np.pi/(a._m[1]+del1) # just the forcing period
            
        
        kws = {'Y_est':Y_est,'T_est':T_est,'fun':rhs,'a':a,'eps':eps,'del1':del1, 'max_iter':max_iter,'tol':tol}

        Y_est = newton_iter_force(**kws)

        Y_prev = deepcopy(Y_est)
        
        counter = 0
        print('{}, eps={:.5f}'.format(i,eps),Y_est)

        if np.isnan(Y_est).sum() > 0:
            inits[i:,:] = np.nan
            break
        
        inits[i,:] = Y_est
        
    return eps_range,inits


def load_bif1d_f_u(rhs,a,del1,etup,recompute=False,data_dir='bifdat',phi0=0,**kwargs):

    
    eps_init,eps_final,deps = etup

    data_dir = os.getcwd()+'/'+data_dir
    if not(os.path.isdir(data_dir)):
        os.mkdir(data_dir)

    model_name = a.system1.model_name
    ratio = str(a._n[1])+str(a._m[1])
    fpars = [data_dir,model_name,ratio,del1,eps_init,eps_final,deps,phi0]
    
    raw = ('{}/bif1_u_inits_{}_ratio={}_d={}_elo={}_ehi={}_de={}_phi0={}.txt')

    fname = raw.format(*fpars)
    file_dne  = not(os.path.isfile(fname))

    #print('fname_inits in bifu',fname)
    
    if file_dne or recompute:
        erange,inits = run_bif1d_f_u(rhs,a,del1,eps_init,eps_final,deps,phi0=phi0,**kwargs)

        data = np.zeros([len(erange),len(inits.T)+1])
        data[:,0] = erange
        data[:,1:] = inits
        np.savetxt(fname,data)
    else:
        data = np.loadtxt(fname)

    return data


def compute_phis_force(fun,inits,etup,a,del1,period_multiple=10,use_point=None,c_sign=1):
    """
    initial conditions come from run_bif1d_f
    """

    erange = np.arange(*etup)
    philist = []

    for i,init in enumerate(inits):
        
        T = period_multiple*2*np.pi
        t = np.arange(0,T,.01)

        if np.isnan(init).sum() > 0:
            philist.append(np.nan)
        else:
            sol = solve_ivp(fun,[0,T],init,args=(a,erange[i],del1,c_sign),t_eval=t,**kw_bif)
        
            Tlist,phi = get_period_all(sol,a)
            philist.append(phi)

    return np.array(philist)


def load_phis_force(rhs,a,del1,etup,recompute=False,data_dir='bifdat',phi0=0,c_sign=1,**kwargs):
    """
    filename must match load_bif1d_f
    initial conditions (for phase-locked solutions) must exist
    before running this function.
    """
    eps_init,eps_final,deps = etup

    data_dir = os.getcwd()+'/'+data_dir
    if not(os.path.isdir(data_dir)):
        os.mkdir(data_dir)

    model_name = a.system1.model_name
    ratio = str(a._n[1])+str(a._m[1])
    raw_phis = ('{}/bif1_phis_{}_ratio={}_d={}_elo={}_ehi={}_de={}_phi0={}.txt')
    raw_inits = ('{}/bif1_inits_{}_ratio={}_d={}_elo={}_ehi={}_de={}_phi0={}.txt')
    fpars = [data_dir,model_name,ratio,del1,eps_init,eps_final,deps,phi0]

    fname_inits = raw_inits.format(*fpars)
    file_inits_dne = not(os.path.isfile(fname_inits))
    #print('fname_inits',fname_inits)
    if file_inits_dne:
        raise ValueError("Compute phase-locked initial conditions before running this file")

    fname_phis = raw_phis.format(*fpars)
    file_dne = not(os.path.isfile(fname_phis))

    if file_dne or recompute:
        out = load_bif1d_f(rhs,a,del1,etup,phi0=phi0,recompute=recompute,c_sign=c_sign,**kwargs)
        inits = out[:,1:]
        phis = compute_phis_force(rhs,inits,etup,a,del1,c_sign=c_sign,**kwargs)

        np.savetxt(fname_phis,phis)

    else:
        phis = np.loadtxt(fname_phis)
        
    return phis

def load_phis_force_u(rhs,a,del1,etup,recompute=False,data_dir='bifdat',phi0=0,use_point=None,**kwargs):
    """
    filename must match load_bif1d_f
    initial conditions (for phase-locked solutions) must exist
    before running this function.
    """
    eps_init,eps_final,deps = etup

    data_dir = os.getcwd()+'/'+data_dir
    if not(os.path.isdir(data_dir)):
        os.mkdir(data_dir)

    model_name = a.system1.model_name
    ratio = str(a._n[1])+str(a._m[1])
    raw_phis = ('{}/bif1_u_phis_{}_ratio={}_d={}_elo={}_ehi={}_de={}_phi0={}.txt')
    raw_inits = ('{}/bif1_u_inits_{}_ratio={}_d={}_elo={}_ehi={}_de={}_phi0={}.txt')
    fpars = [data_dir,model_name,ratio,del1,eps_init,eps_final,deps,phi0]

    fname_inits = raw_inits.format(*fpars)
    file_inits_dne = not(os.path.isfile(fname_inits))
    #print('fname_inits',fname_inits)
    if file_inits_dne:
        raise ValueError("Compute phase-locked initial conditions before running this file")

    fname_phis = raw_phis.format(*fpars)
    file_dne = not(os.path.isfile(fname_phis))

    if file_dne or recompute:
        out = load_bif1d_f_u(rhs,a,del1,etup,phi0=phi0,use_point=use_point,recompute=recompute,**kwargs)
        inits = out[:,1:]
        phis = compute_phis_force(rhs,inits,etup,a,del1,**kwargs)

        np.savetxt(fname_phis,phis)

    else:
        phis = np.loadtxt(fname_phis)
        
    return phis



def get_phase_diff_f_v2(rhs,phi0,a,eps,b,max_time=100,sort_by='max'):
    """
    get phase difference given phase-locking for a forced system.

    rhs: right-hand side of full system
    phi0: initial phase of full system
    
    """

    #print('phi0 in f_v2',phi0)
    init_idx = int(-phi0*a.system1.TN/(2*np.pi))
    init = a.system1.lc['dat'][init_idx,:]

    # run for a while and get period
    t = np.arange(0,max_time,.002)
    sol = solve_ivp(rhs,[0,max_time],init,args=(a,eps,b),t_eval=t,**kw_bif)
    
    # get period estimates
    Tlist,phi_last = get_period_all(sol,a,sort_by=sort_by)
    print('eps=',eps,'; Tlist=',Tlist)

    total_period = 0
    for i in range(len(Tlist)):
        total_period += Tlist[i][0]

    # since the forcing phase always starts with zero at time zero
    # and does not change over time (excxept for delta)
    # we can just take the times of the peaks of the full model
    # mod 2pi+delta.

    # phase diff for each peak-to-peak period

    phis = []
    for (period_est,time_est) in Tlist:

        # diff taken in time not phase, so order is reversed.
        phi = np.mod(a._n[1]*2*np.pi*(-time_est)/total_period,2*np.pi)

        phis.append(phi)
    
    return (np.asarray(phis),phi_last)


def bif1d(a,eps,b,domain,rhs=_redu_c,miter=None):
    """
    for coupling only
    gets stable and unstable points for a given epsilon and delta
    """

    if miter is None:
        y = rhs(0,domain,a,eps,b)
    else:
        y = rhs(0,domain,a,eps,b,miter)


    # get all zeros
    z1 = domain[1:][(y[1:]>0)*(y[:-1]<=0)]
    z2 = domain[1:][(y[1:]<0)*(y[:-1]>=0)]

    return z1,z2




def get_initial_phase_diff_c(phi0,a,eps,b,max_time=2000,_full_rhs=None,
                             prominence=0.1,c_sign=1,**kwargs):

    #print('get_init_phase_diff_c',phi0,a,eps,b,max_time,_full_rhs,prominence,kwargs)
    # get first phase estimate based on phi0
    init1 = list(a.system1.lc['dat'][int(phi0*a.system1.TN/(2*np.pi)),:])
    init2 = list(a.system2.lc['dat'][0,:])
    init = init1+init2

    t = np.linspace(0,max_time,int(max_time/.01))
    
    # run for a while and get period and steady-state phase difference
    sol = solve_ivp(_full_rhs,[0,max_time],init,args=(a,eps,b,'val',c_sign),
    t_eval=t,**kw_bif)

    # get period estimate and time at which period was estimated.
    period_est = 0

    # this is in case the ISIs are unequal.
    # adding up the ISIs between n+1 peaks makes the period more accurate.
    period_est,tt = get_period_v2b(sol,a,prominence=prominence,**kwargs)
    #pp,tt = get_period(sol,prominence=prominence,**kwargs)
    #period_est = pp*a._n[1]
    time_est = tt
    
    # get solution values at time_est.
    init = list(sol.sol(time_est))
    init = np.array(init+[period_est])
    
    print('period_est',period_est,'time_est',tt)
    
    if False:
        sol = solve_ivp(_full_rhs,[0,period_est],init[:-1],args=(a,eps,b),**kw_bif)
        fig,axs = plt.subplots()
        axs.plot(sol.t,sol.y[0])
        axs.plot(sol.t,sol.y[4])
        axs.set_title('2pi solution from steady-state')
        plt.show()
    
    # return solution values at steady-state phase difference.
    return init


def get_phase_lock_full(Y0,a,b,eps_init,eps_final,deps,max_iter:int=20,_full_rhs=None,
                        fudge=.01,prominence=.05,max_time=1000,phi0=0,tol=1e-7,
                        al_factor=1,return_point=False,shift=0,c_sign=1,**kwargs):
    
    """
    "fudge' extends the integration a tiny amount to ensure that the peak
    of the solution is not exactly at the edge (which makes find_peaks fail)
    """

    Y_init = deepcopy(Y0)
    dy_init = np.ones(len(Y0))/10
    eps_range = np.arange(eps_init,eps_final,deps)
    ts = np.zeros([len(eps_range),a._m[1]])
    
    phase_diffs = np.zeros([len(eps_range),a._m[1]])

    Y = deepcopy(Y0)
    
    for i,eps in enumerate(eps_range):
        
        dy = deepcopy(dy_init)
        
        counter = 0
        
        # search for periodic solution
        while np.linalg.norm(dy) > tol and counter <= max_iter and np.linalg.norm(dy)<10:
            
            args_temp = {'fun':_full_rhs,'args':(a,eps,b,'val',c_sign),'dense_output':True,**kw_bif}


            dy,solb = get_dy_nm_coupling(Y,args_temp,_full_rhs=_full_rhs,return_sol=True)
            

            Y += dy*al_factor
            
            str1 = 'iter={}, rel. err ={:.2e}'
            end = '                                              \r'
            printed = str1.format(counter,np.linalg.norm(dy))
            print(printed,end=end)
            counter += 1
    
        # if not converge after max_iter...
        if (counter >= max_iter) or (np.linalg.norm(dy) >= 1) or np.linalg.norm(dy) == 0:
            #print('counter max or norm dy large',counter,np.linalg.norm(dy))
            ts[i,:] = np.nan
            phase_diffs[i,:] = np.nan

            #init = get_initial_phase_diff_c(phi0,a,eps_init,b,max_time=max_time,
            #                                _full_rhs=_full_rhs,prominence=prominence)
            Y = Y#deepcopy(Y0)
            dy = np.ones(len(Y))/10
            
            # break out of loop
            ts[i:,:] = np.nan
            phase_diffs[i:,:] = np.nan
            break
        
        # if converged...
        else:
            # 5 multiples of the full period
            ff = 10*max(a._n[1],a._m[1])*2*np.pi
            
            ttemp = np.linspace(0,ff,int(ff/.002))
            
            solf = solve_ivp(y0=Y[:-1],t_span=[0,ff],
            t_eval=ttemp,**args_temp)
            
            idx1 = sp.signal.find_peaks(solf.y[0],prominence=prominence)[0]
            idx2 = sp.signal.find_peaks(solf.y[4],prominence=prominence)[0]
            tpeaks2_all = solf.t[idx2]

            if len(idx1) > len(idx2):
                len_diff = len(idx1) - len(idx2)
                idx1 = idx1[len_diff:]

            if len(idx2) > len(idx1):
                len_diff = len(idx2) - len(idx1)
                idx2 = idx2[len_diff:]

            tpeaks1 = solf.t[idx1]
            tpeaks2 = solf.t[idx2]
            
            # get max peak
            ypeaks1 = solf.y[0][idx1]
            mid_index = len(ypeaks1) // 2
            
            second_half_indices_sorted = np.argsort(ypeaks1[mid_index:])
            global_indices = mid_index + second_half_indices_sorted
            
            idx1_peaks = global_indices[::-1] # second half indices sorted.
            idx2_peaks = np.argsort(solf.y[4][idx2])
            
            #fig,axs = plt.subplots()
            #axs.plot(solf.t,solf.y[0])
            #axs.plot(solf.t,solf.y[4])
            #axs.scatter(tpeaks1[idx1_peaks],np.zeros(len(tpeaks1[idx1_peaks]))-.1)
            #axs.scatter(tpeaks2[idx2_peaks],np.zeros(len(tpeaks2[idx2_peaks]))-.1)
            #axs.scatter(tpeaks1[idx1_peaks[0]],[-.05])
            #plt.show()


            if len(idx2) == 0:
                Y = Y_init
                dy = dy_init
                ts[i,:] = np.nan
                phase_diffs[i,:] = np.nan
                
            else:
                # get periods and tdiffs for oscillator 1
                #per = get_period(solf,idx=0,prominence=prominence,**kwargs)[0]
                per = get_period_v2(solf,a,idx=0,prominence=prominence,**kwargs)
                
                #per1 = get_period(solf,0,prominence=prominence)
                #per2 = get_period(solf,len(a.system1.var_names),prominence=prominence)
                

                #per = per1[0]/a._n[1]
                #print('period',per)
                #per = per1[0]/a._n[1]
                for ll in range(a._m[1]):
                    # assume all periods are same for now.
                    ts[i,ll] = per # legacy. keep it simple
                    
                    # time diff, equivalent to phase diff.
                    # given a tpeak1, we want largest tpeaks2 s.t. tpeaks2<=tpeak1
                    #osc1_t_idx = idx1[idx1_peaks[peak_temp_idx]]
                    
                    #peak_temp_idx = 0
                    #osc1_t = solf.t[idx1[idx1_peaks[peak_temp_idx]]]
                    #print('osc1_t',osc1_t)
                    
                    #print('tpeaks1',tpeaks1)
                    #print('tpeaks2',tpeaks2)
                    
                    # get each previous phase for oscillator 2.
                    #while len(np.where((osc1_t - tpeaks2) >= 0)[0]) == 0:
                    #    peak_temp_idx += 1
                    #    osc1_t_idx = idx1[idx1_peaks[peak_temp_idx]]
                    #    osc1_t = solf.t[osc1_t_idx]
                    #    print('osc1_t',osc1_t,'peak_temp_idx',peak_temp_idx)
                    osc1_t_idx = idx1[idx1_peaks[0]]
                    osc1_t = solf.t[osc1_t_idx]
                    #print('(osc1_t - tpeaks2)',(osc1_t - tpeaks2),np.where((osc1_t - tpeaks2) >= 0)[0])
                    osc2_t_idx = np.where((osc1_t - tpeaks2_all) >= 0)[0][-(ll+1)]
                    osc2_t = tpeaks2_all[osc2_t_idx]                    

                    tdiff = osc1_t - osc2_t # rescale time to be consistent with paper
                    #tdiff = osc2_t - a._m[1]*osc1_t # rescale time to be consistent with paper

                    # normalize phase difference so that it's in [0,2*pi)
                    # the negative sign is to account for switching from time to phase.
                    #phase = 2*np.pi*(-tdiff)/(ts[i,ll])
                    phase = 2*np.pi*(-tdiff)/per
                    phase_diffs[i,ll] = np.mod(phase,2*np.pi)

        print('iter, eps phase_diffs',i,eps,phase_diffs[i,:],end='                \n')

    print('')
    if return_point:
        return eps_range, ts, phase_diffs, Y
    else:
        return eps_range, ts, phase_diffs

def get_phase_lock_full_v2(Y0,a,b,eps_init,eps_final,deps,max_iter:int=20,_full_rhs=None,
                        fudge=.01,prominence=.05,max_time=1000,phi0=0,tol=1e-7,
                        al_factor=1,return_point=False,shift=0,**kwargs):
    
    """
    "fudge' extends the integration a tiny amount to ensure that the peak
    of the solution is not exactly at the edge (which makes find_peaks fail)
    """

    Y_init = deepcopy(Y0)
    dy_init = np.ones(len(Y0))/10
    eps_range = np.arange(eps_init,eps_final,deps)
    tsx = np.zeros([len(eps_range),a._n[1]])
    tsy = np.zeros([len(eps_range),a._m[1]])
    tdiffs =  np.zeros([len(eps_range),a._n[1]*a._m[1]])

    Y = deepcopy(Y0)
    
    for i,eps in enumerate(eps_range):
        
        dy = deepcopy(dy_init)
        
        counter = 0
        
        # search for periodic solution
        while np.linalg.norm(dy) > tol and counter <= max_iter and np.linalg.norm(dy)<10:
            
            args_temp = {'fun':_full_rhs,'args':(a,eps,b),'dense_output':True,**kw_bif}

            dy,solb = get_dy_nm_coupling(Y,args_temp,_full_rhs=_full_rhs,return_sol=True)

            Y += dy*al_factor
            
            str1 = 'iter={}, rel. err ={:.2e}'
            end = '                                              \r'
            printed = str1.format(counter,np.linalg.norm(dy))
            print(printed,end=end)
            counter += 1
    
        # if not converge after max_iter...
        if (counter >= max_iter) or (np.linalg.norm(dy) >= 1):
            #print('counter max or norm dy large',counter,np.linalg.norm(dy))
            tsx[i,:] = np.nan
            tsy[i,:] = np.nan
            tdiffs[i,:] = np.nan

            #init = get_initial_phase_diff_c(phi0,a,eps_init,b,max_time=max_time,
            #                                _full_rhs=_full_rhs,prominence=prominence)
            #Y = deepcopy(init) # try saving new point
            #dy = np.ones(len(Y))/10
            
            # break out of loop
            break
        
        # if converged...
        else:
            # 3 multiples of the period
            ff = 2*max(a._n[1],a._m[1])*2*np.pi
            ttemp = np.linspace(0,ff,int(ff/.001))
            
            solf = solve_ivp(y0=Y[:-1],t_span=[0,ff],t_eval=ttemp,**args_temp)
            
            
            idx1 = sp.signal.find_peaks(solf.y[0],prominence=prominence)[0]
            idx2 = sp.signal.find_peaks(solf.y[4],prominence=prominence)[0]
            
            
            

            if len(idx1) > len(idx2):
                len_diff = len(idx1) - len(idx2)
                idx1 = idx1[len_diff:]

            if len(idx2) > len(idx1):
                len_diff = len(idx2) - len(idx1)
                idx2 = idx2[len_diff:]

            tpeaks1 = solf.t[idx1]
            tpeaks2 = solf.t[idx2]
            
            #fig,axs = plt.subplots(figsize=(8,2))
            #axs.plot(solf.t,solf.y[0])
            #axs.plot(solf.t,solf.y[4])
            #axs.scatter(solf.t[idx1],np.zeros(len(idx1))-.1)
            #axs.scatter(solf.t[idx2],np.zeros(len(idx2))-.1)
            #plt.show()
            
            
            # get periods and tdiffs for oscillator 1
            #per = get_period(solf,idx=0,prominence=prominence,**kwargs)[0]
            tsx[i,:] = get_period_v3(solf,a,idx=0,prominence=prominence,**kwargs)
            tsy[i,:] = get_period_v3(solf,a,idx=4,prominence=prominence,**kwargs)
            
            print('periods',tsx[i,:],tsy[i,:])
            #per = per1[0]/a._n[1]
            for kk in range(a._n[1]):
                for ll in range(a._m[1]):
                
                    # time diff, equivalent to phase diff.
                    # given a tpeak1, we want largest tpeaks2 s.t. tpeaks2<=tpeak1
                    osc1_t_idx = idx1[(-kk+2)]
                    osc1_t = solf.t[osc1_t_idx]

                    # get each previous phase for oscillator 2.
                    osc2_t_idx = np.where((osc1_t - tpeaks2) >= 0)[0][-(ll+2)]
                    osc2_t = tpeaks2[osc2_t_idx]                    

                    tdiff = osc1_t - osc2_t 
                    
                    tdiffs[i,kk*a._m[1]+ll] = tdiff

        print('iter, eps phase_diffs',i,eps,end='                \n')

    print('')
    if return_point:
        return eps_range, tsx, tsy, tdiffs, Y
    else:
        return eps_range, tsx, tsy, tdiffs


def get_phase_lock_full_del(Y0,a,eps,del_init,del_final,ddel,max_iter:int=20,_full_rhs=None,
                            fudge=.01,prominence=.05,max_time=1000,phi0=0,tol=1e-5,
                            al_factor=1,return_point=False,**kwargs):
    
    """
    "fudge' extends the integration a tiny amount to ensure that the peak
    of the solution is not exactly at the edge (which makes find_peaks fail)
    """

    Y_init = deepcopy(Y0)
    dy_init = np.ones(len(Y0))/10
    del_range = np.arange(del_init,del_final,ddel)
    ts = np.zeros([len(del_range),a._m[1]])
    
    phase_diffs = np.zeros([len(del_range),a._m[1]])

    Y = deepcopy(Y0)
    
    for i,b in enumerate(del_range):
        
        dy = deepcopy(dy_init)
        
        counter = 0
        print('del iter',i,b,phase_diffs[i-1,0],end='                \n')
        
        while np.linalg.norm(dy) > tol and counter <= max_iter and np.linalg.norm(dy)<10:
            

            args_temp = {'fun':_full_rhs,'args':(a,eps,b),'dense_output':True,**kw_bif}
            #print(a.system1.pardict['del0'])
            
            #a.system1.pardict['del0'] = del1
            
            #print('\tcounter, dy,Y next',counter,dy,Y)
            dy,solb = get_dy_nm_coupling(Y,args_temp,_full_rhs=_full_rhs,return_sol=True)

            if False:
                fig,axs = plt.subplots()
                axs.plot(solb.t,solb.y[0])
                axs.plot(solb.t,solb.y[4])
                plt.show()

            Y += dy
            
            str1 = 'del iter={}, rel. err ={:.2e}'
            end = '                                              \r'
            printed = str1.format(counter,np.linalg.norm(dy))
            print(printed,end=end)
            counter += 1

        print('')
    
        if (counter >= max_iter) or (np.linalg.norm(dy) >= 1):
            #print('counter max or norm dy large',counter,np.linalg.norm(dy))
            ts[i,:] = np.nan
            phase_diffs[i,:] = np.nan

            init = get_initial_phase_diff_c(phi0,a,eps,b,max_time=max_time,
                                            _full_rhs=_full_rhs,prominence=prominence)
            Y = Y#deepcopy(init)
            dy = np.ones(len(Y))/10
            break
        
        else:
            # 10 multiples of the period
            ff = 10*max(a._n[1],a._m[1])
            
            solf = solve_ivp(y0=Y[:-1],t_span=[0,ff*Y[-1]+Y[-1]/4],**args_temp)
            
            idx1 = sp.signal.find_peaks(solf.y[0],prominence=prominence)[0]
            idx2 = sp.signal.find_peaks(solf.y[4],prominence=prominence)[0]

            if len(idx1) > len(idx2):
                len_diff = len(idx1) - len(idx2)
                idx1 = idx1[len_diff:]

            if len(idx2) > len(idx1):
                len_diff = len(idx2) - len(idx1)
                idx2 = idx2[len_diff:]

            tpeaks1 = solf.t[idx1]
            tpeaks2 = solf.t[idx2]

            if len(idx2) == 0:
                Y = Y_init
                dy = dy_init
                ts[i,:] = np.nan
                phase_diffs[i,:] = np.nan
                
            else:
                # get periods and tdiffs for oscillator 1
                #per = get_period(solf,idx=0,prominence=prominence,**kwargs)[0]
                per = get_period_v2(solf,a,idx=0,prominence=prominence,**kwargs)

                for ll in range(a._m[1]):
                    #ts[i,ll] = get_period(solf,idx=0,idx_shift=-ll,prominence=prominence)[0]
                    ts[i,ll] = per 
                    
                    # time diff, equivalent to phase diff.
                    # given a tpeak1, we want largest tpeaks2 s.t. tpeaks2<=tpeak1
                    osc1_t_idx = idx1[-2]
                    osc1_t = solf.t[osc1_t_idx]

                    # get each previous phase for oscillator 2.
                    osc2_t_idx = np.where((osc1_t - tpeaks2) >= 0)[0][-(ll+1)]
                    osc2_t = tpeaks2[osc2_t_idx]                    

                    #tdiff = a._m[1]*osc1_t - osc2_t # rescale time to be consistent with paper
                    #tdiff = osc2_t - a._m[1]*osc1_t # rescale time to be consistent with paper
                    tdiff = osc1_t - osc2_t

                   # normalize phase difference so that it's in [0,2*pi)
                    # the negative sign is to account for switching from time to phase.
                    #phase = 2*np.pi*(-tdiff)/(a._m[1]*ts[i,ll])
                    phase = 2*np.pi*(-tdiff)/per
                    
                    phase_diffs[i,ll] = np.mod(phase,2*np.pi)
        
    if return_point:
        return del_range, ts, phase_diffs, Y
    else:
        return del_range, ts, phase_diffs


def follow_phase_diffs(phi0=0,a=None,b=0,eps_init=0.02,eps_final=0.0005,deps=-.0005,
                       recompute=False,bifdir='bif1d/',_full_rhs=None,max_iter=20,
                       max_time=1000,prominence=0.05,tol=1e-5,return_point=False,shift=0,c_sign=1,use_point=[],**kwargs):

    print('phi init',phi0)
    # branch init
    if c_sign == 1:
        fname_template1 = 'phase_diffs_init={}_nm={}{}_b={}_ei={}_ef={}_de={}.txt'
        fname_template2 = 'phase_diffs_s_pt_init={}_nm={}{}_b={}_ei={}_ef={}_de={}.txt' # for inits
        fname1 = fname_template1.format(phi0,a._n[1],a._m[1],b,eps_init,eps_final,deps)
        fname2 = fname_template2.format(phi0,a._n[1],a._m[1],b,eps_init,eps_final,deps)
    else:
        fname_template1 = 'phase_diffs_init={}_nm={}{}_b={}_ei={}_ef={}_de={}_cs={}.txt'
        fname_template2 = 'phase_diffs_s_pt_init={}_nm={}{}_b={}_ei={}_ef={}_de={}_cs={}.txt' # for inits
        fname1 = fname_template1.format(phi0,a._n[1],a._m[1],b,eps_init,eps_final,deps,c_sign)
        fname2 = fname_template2.format(phi0,a._n[1],a._m[1],b,eps_init,eps_final,deps,c_sign)
    
    
    
    
    # force save to curent working directory (where the script was called)
    # using absolute path.
    bifdir = os.getcwd()+'/'+bifdir
    if not(os.path.exists(bifdir)):
        os.makedirs(bifdir)
        
    fname1 = bifdir + fname1
    fname2 = bifdir + fname2
    
    kw1 = {'a':a,'b':b,'eps_init':eps_init,'eps_final':eps_final,'deps':deps,
           '_full_rhs':_full_rhs,'max_iter':max_iter,'prominence':prominence,
           'max_time':max_time,'phi0':phi0,'tol':tol,'shift':shift,'c_sign':c_sign,**kwargs}

    if return_point:
        if not(os.path.isfile(fname1)) or not(os.path.isfile(fname2)) or recompute:
            
            if list(use_point):
                init = use_point
            else:
                init = get_initial_phase_diff_c(phi0,a,eps_init,b,c_sign=c_sign,max_time=max_time,_full_rhs=_full_rhs,prominence=prominence)
            
            eps_range,ts,phase_diffs,Y = get_phase_lock_full(Y0=deepcopy(init),return_point=True,**kw1)
            np.savetxt(fname2,Y)

            data1 = np.zeros([len(eps_range),1+2*a._m[1]])
            data1[:,0] = eps_range
            data1[:,1:1+a._m[1]] = ts
            data1[:,1+a._m[1]:] = phase_diffs
            print('data1 shape',data1)

            np.savetxt(fname1,data1)

        else:
            data1 = np.loadtxt(fname1)
            Y =  np.loadtxt(fname2)

        return data1,Y

        
    if not(return_point):

        if not(os.path.isfile(fname1)) or recompute:
            if list(use_point):
                init = use_point
            else:
                init = get_initial_phase_diff_c(phi0,a,eps_init,b,c_sign=c_sign,max_time=max_time,_full_rhs=_full_rhs,prominence=prominence)
            
            
            eps_range,ts,phase_diffs = get_phase_lock_full(Y0=deepcopy(init),**kw1)

            data1 = np.zeros([len(eps_range),1+2*a._m[1]])
            data1[:,0] = eps_range
            data1[:,1:1+a._m[1]] = ts
            data1[:,1+a._m[1]:] = phase_diffs
            print('data1 shape',data1)

            np.savetxt(fname1,data1)

        else:
            data1 = np.loadtxt(fname1)

        return data1





def follow_phase_diffs_u(phi0=0,a=None,b=0,eps_init=0.02,eps_final=0.0005,deps=-.0005,
                         recompute=False,bifdir='bif1d/',_full_rhs=None,max_iter=20,
                         max_time=1000,prominence=0.05,tol=1e-7,use_point=[],
                         return_point=False,**kwargs):
    """
    follow unstable branches
    """

    print('phi init',phi0)
    # branch init
    fname_template1 = 'phase_diffs_u_init={}_nm={}{}_b={}_ei={}_ef={}_de={}.txt' # for phase diffs
    fname_template2 = 'phase_diffs_u_pt_init={}_nm={}{}_b={}_ei={}_ef={}_de={}.txt' # for inits
    
    fname1 = fname_template1.format(phi0,a._n[1],a._m[1],b,eps_init,eps_final,deps)
    fname2 = fname_template2.format(phi0,a._n[1],a._m[1],b,eps_init,eps_final,deps)
    
    print(fname1)
    print(fname2)
    
    # force save to curent working directory (where the script was called)
    # using absolute path.
    bifdir = os.getcwd()+'/'+bifdir
    if not(os.path.exists(bifdir)):
        os.makedirs(bifdir)
        
    fname1 = bifdir + fname1
    fname2 = bifdir + fname2

    kw1 = {'a':a,'b':b,'eps_init':eps_init,'eps_final':eps_final,'deps':deps,
           '_full_rhs':_full_rhs,'max_iter':max_iter,'prominence':prominence,
           'max_time':max_time,'phi0':phi0,'tol':tol,**kwargs}

    #print('isfile fname1,fname2')
    #print(os.path.isfile(fname1),fname1)
    #print(os.path.isfile(fname2),fname2)
    #print('return_point',return_point)
    #print(not(os.path.isfile(fname1)), not(os.path.isfile(fname2))*return_point, recompute)
    
    if not(os.path.isfile(fname1)) or not(os.path.isfile(fname2)) or recompute:

        if list(use_point):
            init = use_point
        else:
            init1 = list(a.system1.lc['dat'][int(phi0*a.system1.TN/(2*np.pi)),:])
            init2 = list(a.system2.lc['dat'][0,:])
            init = init1+init2+[2*np.pi]

        eps_range,ts,phase_diffs,Y = get_phase_lock_full(Y0=deepcopy(init),return_point=True,**kw1)
        np.savetxt(fname2,Y)

        # Y is a vector of initial conditions + time
        
        data1 = np.zeros([len(eps_range),1+2*a._m[1]])
        
        data1[:,0] = eps_range
        data1[:,1:1+a._m[1]] = ts
        data1[:,1+a._m[1]:] = phase_diffs

        print('phase diff',phase_diffs)

        np.savetxt(fname1,data1)
        

    else:
        data1 = np.loadtxt(fname1)
        if return_point:
            Y =  np.loadtxt(fname2)

    if return_point:
        return data1,Y
    else:
        return data1


def follow_phase_diffs_v2(phi0=0,a=None,b=0,eps_init=0.02,eps_final=0.0005,deps=-.0005,
                       recompute=False,bifdir='bif1d/',_full_rhs=None,max_iter=20,
                       max_time=1000,prominence=0.05,tol=1e-5,return_point=False,shift=0,use_point=[],c_sign=1,**kwargs):

    print('phi init',phi0)
    
    fname_template_eps = 'eps_init={}_nm={}{}_b={}_ei={}_ef={}_de={}.txt'
    fname_template_tsx = 'tsx_init={}_nm={}{}_b={}_ei={}_ef={}_de={}.txt'
    fname_template_tsy = 'tsy_init={}_nm={}{}_b={}_ei={}_ef={}_de={}.txt'
    fname_template_tdiffs = 'tdiffs_init={}_nm={}{}_b={}_ei={}_ef={}_de={}.txt'
    
    fname_template2 = 'phase_diffs_s_pt_init={}_nm={}{}_b={}_ei={}_ef={}_de={}.txt' # for inits
    
    fname_eps = fname_template_eps.format(phi0,a._n[1],a._m[1],b,eps_init,eps_final,deps)
    fname_tsx = fname_template_tsx.format(phi0,a._n[1],a._m[1],b,eps_init,eps_final,deps)
    fname_tsy = fname_template_tsy.format(phi0,a._n[1],a._m[1],b,eps_init,eps_final,deps)
    fname_tdiffs = fname_template_tdiffs.format(phi0,a._n[1],a._m[1],b,eps_init,eps_final,deps)
    
    fname2 = fname_template2.format(phi0,a._n[1],a._m[1],b,eps_init,eps_final,deps)
    
    # force save to curent working directory (where the script was called)
    # using absolute path.
    bifdir = os.getcwd()+'/'+bifdir
    if not(os.path.exists(bifdir)):
        os.makedirs(bifdir)
        
    fname_eps = bifdir + fname_eps
    fname_tsx = bifdir + fname_tsx
    fname_tsy = bifdir + fname_tsy
    fname_tdiffs = bifdir + fname_tdiffs
    
    fname2 = bifdir + fname2
    
    kw1 = {'a':a,'b':b,'eps_init':eps_init,'eps_final':eps_final,'deps':deps,
           '_full_rhs':_full_rhs,'max_iter':max_iter,'prominence':prominence,
           'max_time':max_time,'phi0':phi0,'tol':tol,'shift':shift,**kwargs}

    file_dne = not(os.path.isfile(fname_eps))
    file_dne += not(os.path.isfile(fname_tsx))
    file_dne += not(os.path.isfile(fname_tsy))
    file_dne += not(os.path.isfile(fname_tdiffs))

    if return_point:
        if file_dne or recompute:
            if list(use_point):
                init = use_point
            else:
                init = get_initial_phase_diff_c(phi0,a,eps_init,b,c_sign=c_sign,max_time=max_time,_full_rhs=_full_rhs,prominence=prominence)
            
            eps_range,tsx, tsy, tdiffs,Y = get_phase_lock_full_v2(Y0=deepcopy(init),return_point=True,**kw1)
            
            np.savetxt(fname_eps,eps_range)
            np.savetxt(fname_tsx,tsx)
            np.savetxt(fname_tsy,tsy)
            np.savetxt(fname_tdiffs,tdiffs)
            
            np.savetxt(fname2,Y)
            

        else:
            eps_range = np.loadtxt(fname_eps)
            tsx = np.loadtxt(fname_tsx)
            tsy = np.loadtxt(fname_tsy)
            tdiffs = np.loadtxt(fname_tdiffs)
            
            Y =  np.loadtxt(fname2)

        return data1,Y

    if not(return_point):

        if file_dne or recompute:
            if list(use_point):
                init = use_point
            else:
                init = get_initial_phase_diff_c(phi0,a,eps_init,b,c_sign=c_sign,max_time=max_time,_full_rhs=_full_rhs,prominence=prominence)
            
            
            eps_range,tsx, tsy, tdiffs, Y = get_phase_lock_full_v2(Y0=deepcopy(init),return_point=True,**kw1)

            np.savetxt(fname_eps,eps_range)
            np.savetxt(fname_tsx,tsx)
            np.savetxt(fname_tsy,tsy)
            np.savetxt(fname_tdiffs,tdiffs)
            

        else:
            eps_range = np.loadtxt(fname_eps)
            tsx = np.loadtxt(fname_tsx)
            tsy = np.loadtxt(fname_tsy)
            tdiffs = np.loadtxt(fname_tdiffs)

        return eps_range,tsx,tsy,tdiffs





def follow_phase_diffs_u_v2(phi0=0,a=None,b=0,eps_init=0.02,eps_final=0.0005,deps=-.0005,
                         recompute=False,bifdir='bif1d/',_full_rhs=None,max_iter=20,
                         max_time=1000,prominence=0.05,tol=1e-5,use_point=[],
                         return_point=False,**kwargs):
    """
    follow unstable branches
    """

    print('phi init',phi0)
    # branch init
    fname_template1 = 'phase_diffs_u_init={}_nm={}{}_b={}_ei={}_ef={}_de={}.txt' # for phase diffs
    fname_template2 = 'phase_diffs_u_pt_init={}_nm={}{}_b={}_ei={}_ef={}_de={}.txt' # for inits
    
    fname1 = fname_template1.format(phi0,a._n[1],a._m[1],b,eps_init,eps_final,deps)
    fname2 = fname_template2.format(phi0,a._n[1],a._m[1],b,eps_init,eps_final,deps)
    
    # force save to curent working directory (where the script was called)
    # using absolute path.
    bifdir = os.getcwd()+'/'+bifdir
    if not(os.path.exists(bifdir)):
        os.makedirs(bifdir)
        
    fname1 = bifdir + fname1
    fname2 = bifdir + fname2

    kw1 = {'a':a,'b':b,'eps_init':eps_init,'eps_final':eps_final,'deps':deps,
           '_full_rhs':_full_rhs,'max_iter':max_iter,'prominence':prominence,
           'max_time':max_time,'phi0':phi0,'tol':tol,**kwargs}

    #print('isfile fname1,fname2')
    #print(os.path.isfile(fname1),fname1)
    #print(os.path.isfile(fname2),fname2)
    #print('return_point',return_point)
    #print(not(os.path.isfile(fname1)), not(os.path.isfile(fname2))*return_point, recompute)
    
    if not(os.path.isfile(fname1)) or not(os.path.isfile(fname2)) or recompute:

        if list(use_point):
            init = use_point
        else:
            init1 = list(a.system1.lc['dat'][int(phi0*a.system1.TN/(2*np.pi)),:])
            init2 = list(a.system2.lc['dat'][0,:])
            init = init1+init2+[2*np.pi]

        eps_range,ts,phase_diffs,Y = get_phase_lock_full_v2(Y0=deepcopy(init),return_point=True,**kw1)
        np.savetxt(fname2,Y)

        # Y is a vector of initial conditions + time
        
        data1 = np.zeros([len(eps_range),1+2*a._m[1]])
        
        data1[:,0] = eps_range
        data1[:,1:1+a._m[1]] = ts
        data1[:,1+a._m[1]:] = phase_diffs

        print('phase diff',phase_diffs)

        np.savetxt(fname1,data1)
        

    else:
        data1 = np.loadtxt(fname1)
        if return_point:
            Y =  np.loadtxt(fname2)

    if return_point:
        return data1,Y
    else:
        return data1


def follow_phase_diffs_del(phi0=0,a=None,eps=0,del_init=0.02,del_final=0.0005,ddel=-.0005,
                           recompute=False,bifdir='bif1d/',_full_rhs=None,max_iter=20,
                           max_time=1000,prominence=0.05,tol=1e-5,**kwargs):
    
    # branch init
    fname_template1 = 'phase_diffs_init={}_nm={}{}_eps={}_di={}_df={}_dd={}.txt'
    fname1 = fname_template1.format(phi0,a._n[1],a._m[1],eps,del_init,del_final,ddel)
    
    # force save to curent working directory (where the script was called)
    # using absolute path.
    bifdir = os.getcwd()+'/'+bifdir
    if not(os.path.exists(bifdir)):
        os.makedirs(bifdir)
        
    fname1 = bifdir + fname1

    kw1 = {'a':a,'eps':eps,'del_init':del_init,'del_final':del_final,'ddel':ddel,
           '_full_rhs':_full_rhs,'max_iter':max_iter,'prominence':prominence,
           'max_time':max_time,'phi0':phi0,'tol':tol,**kwargs}

    if not(os.path.isfile(fname1)) or recompute:
        init = get_initial_phase_diff_c(phi0,a,eps,del_init,max_time=max_time,
                                        _full_rhs=_full_rhs,prominence=prominence)

        del_range,ts,phase_diffs = get_phase_lock_full_del(Y0=deepcopy(init),**kw1)
        
        data1 = np.zeros([len(del_range),1+2*a._m[1]])
        data1[:,0] = del_range
        data1[:,1:1+a._m[1]] = ts
        data1[:,1+a._m[1]:] = phase_diffs
        print('data1 shape',data1)
        
        np.savetxt(fname1,data1)
        
    else:
        data1 = np.loadtxt(fname1)

    return data1


def follow_phase_diffs_u_del(phi0=0,a=None,eps=0,del_init=0.02,del_final=0.0005,ddel=-.0005,
                           recompute=False,bifdir='bif1d/',_full_rhs=None,max_iter=20,
                           max_time=1000,prominence=0.05,tol=1e-5,use_point=[],
                           return_point=False,**kwargs):
    
    # branch init
    fname_template1 = 'phase_diffs_u_init={}_nm={}{}_eps={}_di={}_df={}_dd={}.txt'
    fname_template2 = 'phase_diffs_u_pt_init={}_nm={}{}_eps={}_di={}_df={}_dd={}.txt'
    
    fname1 = fname_template1.format(phi0,a._n[1],a._m[1],eps,del_init,del_final,ddel)
    fname2 = fname_template2.format(phi0,a._n[1],a._m[1],eps,del_init,del_final,ddel)
    
    # force save to curent working directory (where the script was called)
    # using absolute path.
    bifdir = os.getcwd()+'/'+bifdir
    if not(os.path.exists(bifdir)):
        os.makedirs(bifdir)
        
    fname1 = bifdir + fname1
    fname2 = bifdir + fname2

    kw1 = {'a':a,'eps':eps,'del_init':del_init,'del_final':del_final,'ddel':ddel,
           '_full_rhs':_full_rhs,'max_iter':max_iter,'prominence':prominence,
           'max_time':max_time,'phi0':phi0,'tol':tol,**kwargs}

    if not(os.path.isfile(fname1)) or not(os.path.isfile(fname2)) or recompute:

        if list(use_point):
            init = use_point
        else:
            init1 = list(a.system1.lc['dat'][int(phi0*a.system1.TN/(2*np.pi)),:])
            init2 = list(a.system2.lc['dat'][0,:])
            init = init1+init2+[2*np.pi]

        del_range,ts,phase_diffs,Y = get_phase_lock_full_del(Y0=deepcopy(init),return_point=True,**kw1)
        np.savetxt(fname2,Y)
        
        # Y is a vector of initial conditions + time
        
        data1 = np.zeros([len(del_range),1+2*a._m[1]])
        data1[:,0] = del_range
        data1[:,1:1+a._m[1]] = ts
        data1[:,1+a._m[1]:] = phase_diffs

        print('phase diff',phase_diffs)
        
        np.savetxt(fname1,data1)
        
    else:
        data1 = np.loadtxt(fname1)
        if return_point:
            Y =  np.loadtxt(fname2)

    if return_point:
        return data1,Y
    else:
        return data1
    

def get_dy_nm_coupling(Y,args_temp,eps_pert=1e-6,eps_pert_time=1e-6,_full_rhs=None,
                       return_sol=False):
    """
    get dy for Newton's method for estimating a phase locked state
    """
    
    J = np.zeros([len(Y),len(Y)])
    
    aa = args_temp['args'][0]
    #tt = np.linspace(0,Y[-1],int(Y[-1]/.0001/max(aa._n[1],aa._m[1])))

    for jj in range(len(Y)-1):
        pert = np.zeros(len(Y)-1);pert[jj] = eps_pert

        t_span = [0,Y[-1]]
        
        
        solp = solve_ivp(y0=Y[:-1]+pert,t_span=t_span,**args_temp)
        solm = solve_ivp(y0=Y[:-1]-pert,t_span=t_span,**args_temp)

        # Jacobian
        J[:-1,jj] = (solp.y.T[-1,:]-solm.y.T[-1,:])/(2*eps_pert)

    J[:-1,:-1] = J[:-1,:-1] - np.eye(len(Y)-1)

    #Tp = Y[-1]+eps_pert_time
    #Tm = Y[-1]-eps_pert_time
    t_spanp = [0,Y[-1]+eps_pert_time]
    t_spanm = [0,Y[-1]-eps_pert_time]
    
    #ttp = np.linspace(0,Tp,int(Tp/.0001/max(aa._n[1],aa._m[1])))
    #ttm = np.linspace(0,Tm,int(Tm/.0001/max(aa._n[1],aa._m[1])))
    
    solpt = solve_ivp(y0=Y[:-1],t_span=t_spanp,**args_temp)
    solmt = solve_ivp(y0=Y[:-1],t_span=t_spanm,**args_temp)

    J[:-1,-1] = (solpt.y.T[-1,:]-solmt.y.T[-1,:])/(2*eps_pert_time)
    J[-1,-1] = 0
    J[-1,:-1] = _full_rhs(0,Y[:-1],*args_temp['args'])

    solb = solve_ivp(y0=Y[:-1],t_span=t_span,**args_temp)
    
    b = np.array(list(Y[:-1]-solb.y.T[-1,:])+[0])
    dy = np.linalg.solve(J,b)

    if return_sol:
        return dy,solb
    else:
        return dy



def _get_sol(rhs,y0,t,args,recompute=False,data_dir='sols',idx='',
               kw_sim={'rtol':1e-7,'atol':1e-7,'method':'LSODA'}):
    """
    load or generate solution.
    args must be in the order obj,eps,b
    """
    a = args[0];eps = args[1];b = args[2]

    data_dir = os.getcwd()+'/'+data_dir
    if not(os.path.isdir(data_dir)):
        os.mkdir(data_dir)

    rhs_name = rhs.__name__

    y0 = np.round(y0,4)

    #tot = 0
    #for q in range(len(a.het_coeffs)):
    #    tot += eps**q*b**(q+1)*a.het_coeffs[q]
    #a.system1.pardict['del0'] = tot
    
    
    ratio = str(a._n[1])+str(a._m[1])
    raw = ('{}/sol_{}_y0={}_ratio={}_T={}_dt={}_e={}_b={}_idx={}.txt')
    fpars = [data_dir,rhs_name,y0,ratio,t[-1],t[1]-t[0],eps,b,idx]
    args1 = {'t_eval':t,'t_span':[0,t[-1]],'args':(*args,),**kw_sim}

    

    fname = raw.format(*fpars)
    file_dne  = not(os.path.isfile(fname))

    if file_dne or recompute:
        solf = solve_ivp(rhs,y0=y0,**args1)
        y = solf.y.T
        np.savetxt(fname,y)
    else:
        y = np.loadtxt(fname)

    return y


def _get_sol_3d(rhs,y0,t,args,recompute=True,data_dir='sols',idx='',
                kw_sim={'rtol':1e-7,'atol':1e-7,'method':'LSODA'}):
    """
    load or generate solution.
    args must be in the order obj,eps,b
    """
    a = args[0];eps = args[1];del1 = a.del1

    data_dir = os.getcwd()+'/'+data_dir
    if not(os.path.isdir(data_dir)):
        os.mkdir(data_dir)

    rhs_name = rhs.__name__

    y0 = np.round(y0,4)
    
    ratio = str(a._n[1])+str(a._m[1])
    raw = ('{}/sol_{}_y0={}_ratio={}_T={}_dt={}_e={}_d={}_idx={}.txt')
    fpars = [data_dir,rhs_name,y0,ratio,t[-1],t[1]-t[0],eps,del1,idx]
    args1 = {'t_eval':t,'t_span':[0,t[-1]],'args':(*args,),**kw_sim}

    fname = raw.format(*fpars)
    file_dne  = not(os.path.isfile(fname))

    if file_dne or recompute:
        solf = solve_ivp(rhs,y0=y0,**args1)
        y = solf.y.T
        np.savetxt(fname,y)
    else:
        y = np.loadtxt(fname)

    return y

def get_smallest_eps(data_list,return_idx=False):
    """get smallest non-nan epsilon value"""
    e_val = 100
    list_idx = np.nan
    e_idx = np.nan
    
    for i,data in enumerate(data_list):
        isnans = np.isnan(data[:,1])
        if np.sum(isnans) != len(data[:,1]):
            
            idx = np.argmax(isnans>0)-1 # get previous non-nan idx
            e_val_new = data[idx,0]
            if e_val > e_val_new:
                e_val = e_val_new
                list_idx = i
                e_idx = idx

    if return_idx:
        return e_val, list_idx, e_idx
    else:
        return e_val


def get_largest_eps(data_list,return_idx=False):
    """get largest non-nan epsilon value"""
    e_val = -1
    list_idx = np.nan
    e_idx = np.nan
    
    
    for i,data in enumerate(data_list):
        isnans = np.isnan(data[:,1])
        
        if np.sum(isnans) != len(data[:,1]):
            idx = np.argmax(isnans>0)-1 # get previous non-nan idx
            e_val_new = data[idx,0]

            # get smallest idx
            e_val_new2 = data[0,0] # assumes termination on nan

            if e_val_new > e_val_new2:
                pass
            else:
                e_val_new = e_val_new2
            
            if e_val < e_val_new:
                e_val = e_val_new
                list_idx = i
                e_idx = idx
        
    if return_idx:
        return e_val, list_idx, e_idx
    return e_val


def numerical_jac(fn,x,eps2=1e-7,**kwargs):
    """
    return numerical Jacobian function
    """
    n = len(x)
    J = np.zeros((n,n))
        
    PM = np.zeros_like(J)
    PP = np.zeros_like(J)
        
    for k in range(n):
        epsvec = np.zeros(n)
        epsvec[k] = eps2
        PP[:,k] = fn(0,x+epsvec,**kwargs)
        PM[:,k] = fn(0,x-epsvec,**kwargs)
            
    J = (PP-PM)/(2*eps2)
    
    return J

def get_es_min(data_bs_list,data_bu_list):
    """
    get es_min across either list + phi0
    """
    es_min,lists_idx,es_idx = get_smallest_eps(data_bs_list,return_idx=True) # get smallest eps value. bifurcation occurs here.
    eu_min,listu_idx,eu_idx = get_smallest_eps(data_bu_list,return_idx=True) # get smallest eps value. bifurcation occurs here.
    
    if es_min <= eu_min:
        list_idx = lists_idx
        e_idx = es_idx
        phi0 = data_bs_list[list_idx][e_idx] # get initial phi at this epsilon
    else:
        list_idx = listu_idx
        e_idx = eu_idx
        es_min = eu_min
        phi0 = data_bu_list[list_idx][e_idx] # get initial phi at this epsilon
    
    return es_min,phi0


def get_es_max(data_bs_list,data_bu_list):
    """
    get es_min across either list + phi0
    """
    es_max,lists_idx,es_idx = get_largest_eps(data_bs_list,return_idx=True) # get smallest eps value. bifurcation occurs here.
    eu_max,listu_idx,eu_idx = get_largest_eps(data_bu_list,return_idx=True) # get smallest eps value. bifurcation occurs here.
    
    if es_max >= eu_max:
        list_idx = lists_idx
        e_idx = es_idx
        phi0 = data_bs_list[list_idx][e_idx] # get initial phi at this epsilon
    else:
        list_idx = listu_idx
        e_idx = eu_idx
        es_max = eu_max
        phi0 = data_bu_list[list_idx][e_idx] # get initial phi at this epsilon
    
    return es_max,phi0
