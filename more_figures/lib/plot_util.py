from .rhs import _redu_c, _redu_c2
from . import util
from .util import (bif1d, _get_sol, get_phase, add_arrow_to_line2D)

from scipy.spatial.distance import cdist
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

import scipy as sp
import numpy as np

import os

def construct_branch(eps_list,zu,tol:float=.25):
    """
    tol: tolerance to cut discontinuous points.
    zu must be list for this code to work.
    """
    
    x = [];y = []
    idx2 = 0
    zu = list(zu)

    #print('*****zu',zu)
    for i in range(len(zu)):

        if len(zu[i]) > 0:
            
            # take first point
            if len(x) == 0:
                y.append(zu[i][0])
                x.append(eps_list[i])
                zu[i] = np.delete(zu[i],0)

            # save subsequent points
            else:
                min_idx = np.argmin(np.abs(y[idx2]-zu[i]))
                dx = np.abs(eps_list[idx2]-eps_list[min_idx])
                dy = np.abs(y[idx2]-zu[i][min_idx])
                if dx+dy < tol:
                #    print(dx,dy,'zu',zu[i][min_idx],'eps',eps_list[i])

                #if np.abs(y[idx2]-zu[i][min_idx]) < tol or\
                #   np.abs(np.tan(y[idx2])-np.tan(zu[i][min_idx])) < tol:
                    y.append(zu[i][min_idx])
                    x.append(eps_list[i])
                    zu[i] = np.delete(zu[i],min_idx)
                    idx2 += 1
        
    return zu, x, y


def add_diagram_1d(axs,a,del1,eps_tup=(0.001,.5,200),
                   rhs=_redu_c,domain=np.linspace(-.1,2*np.pi+.1,2000),
                   include_limits=True,label='',tol=.25,miter=None,
                   color='tab:red',lw=1.2,ls='-'):
    """
    for non-scatter diagrams (a little slower than scatter)
    """
    
    zu = [];zs = []

    eps_list = np.linspace(*eps_tup)
    for eps in eps_list:
        z1,z2 = bif1d(a,eps,del1,domain,rhs=rhs,miter=miter)

        zu.append(np.mod(z1,2*np.pi))
        zs.append(np.mod(z2,2*np.pi))

    # separate branches
    x = [1]
    while len(x)>0:
        zu,x,y = construct_branch(eps_list,zu,tol=tol)

        if np.sum((np.abs(np.diff(x))>tol) + (np.abs(np.diff(y))>tol)):
            pass
        else:
            axs.plot(x,y,color=color,ls='--',lw=lw)


    x = [1]
    count = 0
    while len(x)>0:
        zs,x,y = construct_branch(eps_list,zs,tol=tol)
        if np.sum((np.abs(np.diff(x))>tol) + (np.abs(np.diff(y))>tol)):
            pass
        else:
            if count == 0:
                label1 = label
            else:
                label1 = ''
            axs.plot(x,y,color=color,label=label1,lw=lw)
        count += 1

    if include_limits:
        axs.set_ylim(-.1,2*np.pi+.1)
        axs.set_xlim(eps_list[0],eps_list[-1])

    return axs



def add_diagram_1d_del(axs,a,eps,del_tup=(0.001,.1,200),
                       rhs=_redu_c,domain=np.linspace(-.1,2*np.pi+.1,2000),
                       include_limits=True,label='',tol=.25,miter=None,
                       color='tab:red',lw=1.2):
    """
    for non-scatter diagrams (a little slower than scatter)
    """
    
    zu = [];zs = []

    del_list = np.linspace(*del_tup)
    for del1 in del_list:
        z1,z2 = bif1d(a,eps,del1,domain,rhs=rhs,miter=miter)

        zu.append(np.mod(z1,2*np.pi))
        zs.append(np.mod(z2,2*np.pi))

    # separate branches
    x = [1]
    while len(x)>0:
        zu,x,y = construct_branch(del_list,zu,tol=tol)

        if np.sum((np.abs(np.diff(x))>tol) + (np.abs(np.diff(y))>tol)):
            pass
        else:
            axs.plot(x,y,color=color,ls='--',lw=lw)


    x = [1]
    count = 0
    while len(x)>0:
        zs,x,y = construct_branch(del_list,zs,tol=tol)
        if np.sum((np.abs(np.diff(x))>tol) + (np.abs(np.diff(y))>tol)):
            pass
        else:
            if count == 0:
                label1 = label
            else:
                label1 = ''
            axs.plot(x,y,color=color,label=label1,lw=lw)
        count += 1

    if include_limits:
        axs.set_ylim(-.1,2*np.pi+.1)
        axs.set_xlim(del_list[0],del_list[-1])

    return axs


def add_diagram_1d_scatter(axs,a,b,eps_tup=(0.001,.5,200),
                           rhs=_redu_c,
                           domain=np.linspace(-.1,2*np.pi+.1,2000),
                           include_limits=True,label='',**kwargs):

    eps_list = np.linspace(*eps_tup)
    zu = [];zs = []
    n = a._n[1]

    for eps in eps_list:
        z1,z2 = bif1d(a,eps,b,domain=domain,rhs=rhs,**kwargs)

        zu.append(np.mod(z1,2*np.pi))
        zs.append(np.mod(z2,2*np.pi))

    for xe, ye in zip(eps_list, zu):
        axs.scatter([xe] * len(ye), ye,s=2,c='red')

    for xe, ye in zip(eps_list, zs):
        axs.scatter([xe] * len(ye), ye,s=2,c='k')

    if include_limits:
        axs.set_ylim(-.1,2*np.pi+.1)
        axs.set_xlim(eps_list[0],eps_list[-1])

    return axs


def add_diagram_1d_del_scatter(axs,a,eps,del_tup=(0.001,.5,200),
                               rhs=_redu_c,
                               domain=np.linspace(-.1,2*np.pi+.1,2000),
                               include_limits=True,label='',**kwargs):

    del_list = np.linspace(*del_tup)
    zu = [];zs = []
    n = a._n[1]

    for del1 in del_list:
        z1,z2 = bif1d(a,eps,del1,domain=domain,rhs=rhs,**kwargs)

        zu.append(np.mod(z1,2*np.pi))
        zs.append(np.mod(z2,2*np.pi))

    for xe, ye in zip(del_list, zu):
        axs.scatter([xe] * len(ye), ye,s=2,c='red')

    for xe, ye in zip(del_list, zs):
        axs.scatter([xe] * len(ye), ye,s=2,c='k')

    if include_limits:
        axs.set_ylim(-.1,2*np.pi+.1)
        axs.set_xlim(del_list[0],del_list[-1])

    return axs



def load_diagram_full(a,del1,eps_tup=[0.01,0.011,2],dir1='bifdat',
                      rhs=None,phi0=np.pi,recompute=False,
                      maxt=100,scale_t_eps=True):
    
    """
    winow shift: needed to search for other phase-locked states
    besides the peaks that are closest to the peak of the oscillator.
    important for, e.g., 1:2 forcing.
    """
    
    model_name = a.system1.model_name
    ratio = str(a._n[1])+str(a._m[1])
    elo, ehi, N = eps_tup
    rawname_s = '{}/{}_full_s_del1={}_p0={}_ratio={}_elo={}_ehi={}_N={}.npy'
    aa = (dir1,model_name,del1,phi0,ratio,elo,ehi,N,)
    fname_s = rawname_s.format(*aa)
    
    if not(os.path.isdir(dir1)):
        os.mkdir(dir1)

    file_dne = not(os.path.isfile(fname_s))
    #file_dne += not(os.path.isfile(fname_u))

    eps_list = np.linspace(*eps_tup)

    if file_dne or recompute:
        eps_list = np.linspace(*eps_tup)
        phi_list = []
        for eps_init in eps_list:
            if scale_t_eps:
                tot = maxt/eps_init
            else:
                tot = maxt
            
            phi = util.get_phase_diff_f(rhs=rhs,
                                        phi0=phi0,
                                        a=a,
                                        eps=eps_init,
                                        del1=del1,
                                        max_time=tot)
            
            phi_list.append(phi)

        np.savetxt(fname_s,np.array(phi_list))
        #np.save(fname_u,np.array(zu,dtype=object))

        
    else:
         phi_list = list(np.loadtxt(fname_s))
         #zu = np.load(fname_u,allow_pickle=True)

    return phi_list#,zu

def add_diagram_full(axs,a,del1,eps_tup,rhs=None,phi0=np.pi,
                     recompute=False,maxt=100,scale_t_eps=True,
                     branch_tol=6e-1,similarity_tol=5e-1):

    zs = load_diagram_full(a,del1,eps_tup=eps_tup,dir1='bifdat',
                           rhs=rhs,phi0=phi0,recompute=recompute,
                           maxt=maxt,scale_t_eps=scale_t_eps)

    zs = np.mod(zs,2*np.pi)
    eps_list = np.linspace(*eps_tup)


    #print('zs original',zs)
    if a._m[1] > 1:
        for i in range(a._m[1]):
            zs,x,y = construct_branch(eps_list,zs,tol=branch_tol)
            axs.plot(x,y,color='k')
        #print('zs remaining',zs)
        # now that zs has a bunch of empty elements, remove them until
        # a nonempty element is found.
        zs_remainder = []
        eps_remainder = []
        for i in range(len(zs)):
            temp_list = []
            if len(zs[i]) != 0:
                
                # check if point is close to existing points on diagram
                for pt_y in zs[i]:
                    pt_x = eps_list[i]
                    #print('')
                    print('check point',pt_x,pt_y)
                    diff_prev = 10
                    for line in axs.lines:
                        pt = np.array([pt_x,pt_y])
                        diff = np.linalg.norm(pt-line.get_xydata(),axis=1)
                        
                        #print('diff min',np.amin(diff),diff_prev)
                        if np.amin(diff) < diff_prev:
                            diff_prev = np.amin(diff)
                    #print(diff_prev,similarity_tol)
                    if diff_prev > similarity_tol:
                        temp_list.append(pt_y)

                if len(temp_list) == 0:
                    pass
                else:
                    print('zs[i]',i,zs[i],len(zs[i]),temp_list)
                    zs_remainder.append(temp_list)
                    eps_remainder.append(eps_list[i])

        print('zs collected',zs_remainder)
        if len(zs_remainder) == 0:
            pass
        else:
                           
            for i in range(len(zs_remainder[0])):
                
                zs_remainder,x,y = construct_branch(eps_remainder,zs_remainder,
                                                    tol=branch_tol)
                axs.plot(x,y,color='k')
        print('zs_remainder',zs_remainder)
    else:
        axs.plot(eps_list,zs,color='k')
    

    return axs




def load_diagram_full_f(a,del1,eps_tup=[0.01,0.011,2],dir1='bifdat',
                      rhs=None,phi0=np.pi,recompute=False,
                      maxt=100,scale_t_eps=True,sort_by='max'):
    
    """
    load diagram for forced case
    
    winow shift: needed to search for other phase-locked states
    besides the peaks that are closest to the peak of the oscillator.
    important for, e.g., 1:2 forcing.
    """
    
    model_name = a.system1.model_name
    ratio = str(a._n[1])+str(a._m[1])
    elo, ehi, N = eps_tup
    rawname_s = '{}/{}_full_s_del1={}_p0={}_ratio={}_elo={}_ehi={}_N={}.npy'
    aa = (dir1,model_name,del1,phi0,ratio,elo,ehi,N,)
    fname_s = rawname_s.format(*aa)
    
    if not(os.path.isdir(dir1)):
        os.mkdir(dir1)

    file_dne = not(os.path.isfile(fname_s))
    #file_dne += not(os.path.isfile(fname_u))

    phi0_original = phi0

    eps_list = np.linspace(*eps_tup)

    if file_dne or recompute:
        eps_list = np.linspace(*eps_tup)
        phi_list = []
        for eps_init in eps_list:
            if scale_t_eps:
                tot = maxt/eps_init
            else:
                tot = maxt

            phis,phi_last = util.get_phase_diff_f_v2(rhs=rhs,phi0=phi0,a=a,eps=eps_init,
                                                     b=del1,max_time=tot,sort_by=sort_by)

            phi0 = phi_last

            phi_list.append(phis)

        np.savetxt(fname_s,np.array(phi_list))
        
    else:
         phi_list = list(np.loadtxt(fname_s))

    return phi_list#,zu


def draw_full_solutions(axs,a,T,eps,b,init,full_rhs,recompute=False,
                        label='Full',skipn=100):
    """
    draw full solution give axis object
    init is initial phase.
    """
    
    system1 = a.system1;system2 = a.system2
    kw1 = {'eps':eps,'a':a,'return_data':True}
    dt = .02;t = np.arange(0,T,dt)

    # estimate initial phase
    y0a = list(system1.lc['dat'][int((init/(2*np.pi))*system1.TN),:])
    y0b = list(system2.lc['dat'][int((0/(2*np.pi))*system2.TN),:])
    y0 = np.array(y0a+y0b)

    tot = 0
    for i in range(len(a.het_coeffs)):
        tot += eps**i*b**(i+1)*a.het_coeffs[i]
    a.system1.pardict['del0'] = tot
    print(a.system1.pardict['del0'],'eps',eps,'b',b)
    #print(pd1['del0'])

    solf = _get_sol(full_rhs,y0,t,args=(a,eps,b),recompute=recompute)

    tpa, phasea = get_phase(t,solf[:,:4],skipn=skipn,system1=system1)
    tpb, phaseb = get_phase(t,solf[:,4:],skipn=skipn,system1=system2)

    y = np.mod(phasea-a.om*phaseb,2*np.pi)
    axs.scatter(y,tpa,s=5,color='k',label=label)

    return axs


def draw_1d_rhs(axs,a,eps,b,miter=None,rhs=_redu_c,color='#57acdc',ls='-',label='',lw=1,n_factor=False):
    x = np.linspace(0,2*np.pi,200)
    y = rhs(0,x,a,eps,b,miter=miter,n_factor=n_factor)
    axs.plot(x,y,color=color,lw=lw,label=label,ls=ls)
    

    return axs

def draw_1d_solutions(axs,a,T,eps,b,init,rhs=_redu_c2,miter=None,ls='-',lw=1,
                      color='#57acdc',label='',zorder=5,arrow_locs=[0.75],n_factor=False):

    # trajectory
    dt = .02;t = np.arange(0,T,dt);th_init = init

    args1 = {'args':[a,eps,b,miter,n_factor],'rtol':1e-7,'atol':1e-7,'method':'LSODA',
             't_span':[0,t[-1]],'t_eval':t}

    print('nfactor in draw1d',n_factor)
    # solution on 1d phase plane
    solr1d = solve_ivp(rhs,y0=[th_init],**args1)
    

    # 1d solution over time
    xs = np.mod(solr1d.y.T[:,0],2*np.pi); ys = np.zeros(len(xs))
    discont_idxs = np.abs(np.gradient(xs,1)) > np.pi/2
    xs[discont_idxs] = np.nan; t[discont_idxs] = np.nan

    #axs.plot(np.mod(solr1d.y.T[:,0],2*np.pi),t,color=color,label=label,ls=ls,zorder=zorder)
    axs.plot(xs,t,color=color,alpha=.75,label=label,ls=ls)

    # add arrow
    line, = axs.plot(xs,t,color=color,alpha=.75,ls=ls)
    add_arrow_to_line2D(axs,line,arrow_locs=arrow_locs,arrowsize=2)

        
    return axs

def fix_flips(data_bu_list,a):
    """fix flips in bifurcation diagram"""
    tol1 = 1
    for i,data in enumerate(data_bu_list):
        phases = data[:,1+a._m[1]:]
        for j in range(1,len(phases)):
            phase0_prev = phases[j-1][0]
            phase0_curr = phases[j][0]
    
            d = circd(phase0_prev,phase0_curr)
            iter = 0
            while d > tol1 and iter < a._n[1]+a._m[1]:
                phases[j] = np.roll(phases[j],1)
                phase0_curr = phases[j][0]
                d = circd(phase0_prev,phase0_curr)
                #print('d',d,phase0_curr,phase0_prev)
                #print('phases edit',phases[j])
                iter += 1
            data[j,1+a._m[1]:] = phases[j]
    return data_bu_list

def circd(th1,th2):
    """circular distance"""
    return np.min([np.abs(th1-th2),2*np.pi-np.abs(th1-th2)])

def draw_quick_plot_f(axs,data_list,a,norm=None,**kwargs):
    data_list = fix_flips(data_list,a)
    for i in range(len(data_list)):
        data1 = data_list[i] # data1 is periods, data2 is t_diffs
        
        phase_diffs = data1[:,1+a._m[1]:] # just use first set of phase diffs.
        
        dtheta = np.diff(phase_diffs,axis=0)
        wraps = np.abs(dtheta) > np.pi   # threshold for wrap

        theta_plot = phase_diffs.astype(float).copy()
        theta_plot[1:][wraps] = np.nan
        
        #phase_diffs[np.abs(phase_diffs) > np.pi/2] = np.nan
        
        axs.plot(data1[:,0],theta_plot,lw=2,**kwargs)
        
    #axs.set_title('Full del={}'.format(a.del1))

    return axs

def quick_plot_f(data_list,a,ylim=[-.1,2*np.pi+.1]):
    kwargs = locals()
    
    fig,axs = plt.subplots(figsize=(3,2))

    axs = draw_quick_plot_f(axs,**kwargs)
    axs.set_ylim(*ylim)
    axs.set_xlabel('eps')
    axs.set_ylabel('phi')
    
    plt.tight_layout()

def quick_plot_r(a,etup):
    fig,axs = plt.subplots(figsize=(3,2))
    axs = add_diagram_1d_scatter(axs,a,a.del1,etup)
    axs.set_xlabel('eps')
    axs.set_ylabel('phi')
    axs.set_title('r1d delta={}, k={}'.format(a.del1,a.system1.trunc_order))
    plt.tight_layout()


def draw_isi(a,data_list):
    for i in range(len(data_list)):
        data1,data2 = data_list

    
    if i == 0:
        data_t = data1
        data_diff = data2
    else:
        data_t = np.append(data_t,data1,axis=0)
        data_diff = np.append(data_diff,data2,axis=0)
                
    axs[0].plot(data1[:,0],data1[:,1:1+a._n[1]],color='gray')
    axs[1].plot(data1[:,0],data1[:,1+a._n[1]:],color='k')


        
    
def quick_plot_combined(a,kw_f=None,kw_r=None,kw_r3d=None,
                        ylim=[-.1,2*np.pi+.1]):
    
    fig,axs = plt.subplots(figsize=(3,2))
    
    if not(kw_f) is None:
        axs = draw_quick_plot_f(axs,kw_f['data_list'],a)

    if not(kw_r) is None:
        axs = add_diagram_1d_scatter(axs,a,a.del1,kw_r['etup'])

    if not(kw_r3d) is None:
        axs = draw_quick_plot_r3d(axs,kw_r3d['data_list'],a)

    if not(kw_f) is None:
        fig,axs = plt.subplots(2,1,figsize=(3,4))

        max_diff = .02
        eps_min1 = 1
        eps_min2 = 1

        for i in range(len(kw_f['data_list'])):
            data1,data2 = kw_f['data_list'][i]

            if i == 0:
                data_t = data1
                data_diff = data2
            else:
                data_t = np.append(data_t,data1,axis=0)
                data_diff = np.append(data_diff,data2,axis=0)
                
            axs[0].plot(data1[:,0],data1[:,1:1+a._n[1]],color='gray')
            axs[1].plot(data1[:,0],data1[:,1+a._n[1]:],color='k')

        if a._n[1]>1:
            y1 = data_t[:,1]
            for j in range(1,a._n[1]):
                idxs = (np.abs(y1-data_t[:,1+j])>max_diff)
                
                if sum(idxs) == 0:
                    pass
                elif eps_min1 > np.amin(data_t[:,0][idxs]):
                    eps_min1 = np.amin(data_t[:,0][idxs])

        if a._m[1]>1:
            y2 = data_t[:,1+a._n[1]]
            for j in range(1,a._m[1]):
                idxs = (np.abs(y2-data_t[:,1+a._n[1]+j])>max_diff)
                
                if sum(idxs) == 0:
                    pass
                elif eps_min2 > np.amin(data_t[:,0][idxs]):
                    eps_min2 = np.amin(data_t[:,0][idxs])

        if eps_min1 != 1 or eps_min2 != 1:
            eps_min = min(eps_min1,eps_min2)
            print('eps_min',eps_min)

            for j in range(2):
                axs[j].axvline(eps_min,ls='--',color='gray')
                axs[j].axvspan(0,eps_min,color='tab:green',alpha=.15)
                axs[j].axvspan(kw_r['etup'][1],eps_min,color='tab:red',alpha=.15)
        

        axs[0].set_title('Oscillator 1')
        axs[1].set_title('Oscillator 2')

        plt.tight_layout()

