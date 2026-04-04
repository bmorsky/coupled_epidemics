"""
Figure generation code to present a couple bifurcation diagrams
"""


import os
#from matplotlib.gridspec import GridSpec
import numpy as np
import time

import matplotlib
import matplotlib.pyplot as plt
import string
import sympy as sym
import scipy as sp

from scipy.interpolate import interp1d



matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
#matplotlib.rcParams['font.size'] = 12

panels = [r'\textbf{(a)}',r'\textbf{(b)}',r'\textbf{(c)}',
          r'\textbf{(d)}',r'\textbf{(e)}',r'\textbf{(f)}']

preamble = (r'\usepackage{amsmath}'
            r'\usepackage{siunitx}'
            r'\usepackage{bm}'
            r'\newcommand{\ve}{\varepsilon}')

matplotlib.rcParams['text.latex.preamble'] = preamble
fontsize = 12

def remove_y_disc(y,threshold=0.1):
    discontinuities_idx = np.where(np.abs(np.diff(y)) > threshold)[0] + 1
    y_discontinuous = y.copy()
    y_discontinuous[discontinuities_idx] = np.nan
    return y_discontinuous

def remove_x_disc(data_x, data_y, threshold=10):
    delta_x = np.abs(np.diff(data_x))
    large_step_indices = np.where(delta_x > threshold)[0] + 1

    new_x = np.insert(data_x, large_step_indices, np.nan)
    new_y = np.insert(data_y, large_step_indices, np.nan)
    return new_x, new_y

def get_stable_unstable_bool(dat):
    pt_bool_s = dat[:,3] == 1
    pt_bool_u = dat[:,3] == 2
    per_bool_s = dat[:,3] == 3
    per_bool_u = dat[:,3] == 4

    return pt_bool_s, pt_bool_u, per_bool_s, per_bool_u

def seir1_onepars():
    """
    Figure for SEIR1 uncoupled model.
    See xpp file sirv1_social.ode
    # kappa
    """
    
    fig,axs = plt.subplots(1,2,figsize=(6,3))


    ########### delta = 0.01
    dat = np.loadtxt('dat/sirv1_kappa2.dat')
    pt_bool_s, pt_bool_u, per_bool_s, per_bool_u = get_stable_unstable_bool(dat)
    
    stable_pt_x = dat[:,0][pt_bool_s];stable_pt_y = dat[:,1][pt_bool_s]
    unstable_pt_x = dat[:,0][pt_bool_u];unstable_pt_y = dat[:,1][pt_bool_u]
    
    stable_per_x_max = dat[:,0][per_bool_s];stable_per_y_max = dat[:,1][per_bool_s]
    stable_per_x_min = dat[:,0][per_bool_s];stable_per_y_min = dat[:,2][per_bool_s]

    unstable_per_x_max = dat[:,0][per_bool_u];unstable_per_y_max = dat[:,1][per_bool_u]
    unstable_per_x_min = dat[:,0][per_bool_u];unstable_per_y_min = dat[:,2][per_bool_u]

    y = remove_y_disc(stable_pt_y)
    x,y = remove_x_disc(stable_pt_x,y)
    axs[0].plot(x,y,color='gray',label='Stable Point')

    y = remove_y_disc(unstable_pt_y)
    x,y = remove_x_disc(unstable_pt_x,y)
    axs[0].plot(x,y,color='gray',ls=':',label='Unstable Point')

    # periodic S
    y = remove_y_disc(stable_per_y_min,threshold=1)
    x,y = remove_x_disc(stable_per_x_min,y)
    axs[0].plot(x,y,color='k',label='Stable Orbit')

    y = remove_y_disc(stable_per_y_max,threshold=1)
    x,y = remove_x_disc(stable_per_x_max,y)
    axs[0].plot(x,y,color='k')
    
    ##### annotation
    idx = np.argmin(np.abs(stable_per_x_max))
    lp_x = stable_per_x_max[idx]
    lp_y = stable_per_y_max[idx]
    axs[0].annotate("LP",xy=(lp_x,lp_y), xycoords='data',
                 xytext=(lp_x-20,lp_y+0.13), textcoords='data',
                 arrowprops=dict(facecolor='k',arrowstyle="-|>", connectionstyle="arc3,rad=.5"))
    axs[0].scatter([lp_x],[lp_y],marker='D',facecolor='none',edgecolor='k',zorder=20)

    lp_y_min = stable_per_y_min[idx]
    axs[0].scatter([lp_x],[lp_y_min],marker='D',facecolor='none',edgecolor='k',zorder=20)
    print('LP at',lp_x,lp_y)

    # periodic U
    y = remove_y_disc(unstable_per_y_min,threshold=1)
    x,y = remove_x_disc(unstable_per_x_min,y)
    axs[0].plot(x,y,color='tab:red',ls='--',label='Unstable Orbit')

    y = remove_y_disc(unstable_per_y_max,threshold=1)
    x,y = remove_x_disc(unstable_per_x_max,y)
    axs[0].plot(x,y,color='tab:red',ls='--')


    ##### annotations
    # get index where y coordinate of unstable orbit is close to zero
    # this is the subcritical hopf
    idx = np.argmin(np.abs(unstable_per_y_max))
    hb_x = unstable_per_x_max[idx]
    hb_y = unstable_per_y_max[idx]
    axs[0].annotate("HB",xy=(hb_x,hb_y), xycoords='data',
                 xytext=(hb_x+50,hb_y-0.1), textcoords='data',
                 arrowprops=dict(facecolor='k',arrowstyle="-|>", connectionstyle="arc3,rad=-.5"))
    axs[0].scatter([hb_x],[hb_y],marker='o',facecolor='none',edgecolor='k',zorder=20)
    print('HB at',hb_x,hb_y)

    # get smallest x where y-coordinate of stable points is close to hb_y
    # this the transcritical point
    idxy = np.where(np.abs(stable_pt_y-hb_y)<1e-2)[0]
    idx = np.argmin(stable_pt_x[idxy])
    tc_x = stable_pt_x[idx]
    tc_y = stable_pt_y[idx]
    axs[0].annotate("TC",xy=(tc_x,tc_y), xycoords='data',
                 xytext=(tc_x+50,tc_y-0.1), textcoords='data',
                 arrowprops=dict(facecolor='k',arrowstyle="-|>", connectionstyle="arc3,rad=-.5"))
    axs[0].scatter([tc_x],[tc_y],marker='s',facecolor='none',edgecolor='k',zorder=20)
    print('TC at',tc_x,tc_y)

    # fill in remainder of upper unstable branch
    dat = np.loadtxt('dat/sirv1_kappa2b.dat')
    pt_bool_s, pt_bool_u, per_bool_s, per_bool_u = get_stable_unstable_bool(dat)

    x = dat[:,0][pt_bool_u]
    y = dat[:,1][pt_bool_u]
    
    x_bool = x > 195
    y_bool = y > .5
    bool1 = x_bool & y_bool
    axs[0].plot(x[bool1],y[bool1],color='gray',ls=':')
    


    
    ########### delta = 0.02
    # periodic S
    dat = np.loadtxt('dat/sirv1_kappa_theta=0.026_per_s_fixed.dat')
    axs[1].plot(dat[:,0],dat[:,1],color='k') # min
    axs[1].plot(dat[:,0],dat[:,2],color='k') # max

    idx = np.argmin(np.abs(dat[:,0]))
    lp_x = dat[:,0][idx]
    lp_y = dat[:,2][idx]
    axs[1].scatter([lp_x],[lp_y],marker='D',facecolor='none',edgecolor='k',zorder=20)

    lp_y_min = dat[:,1][idx]
    axs[1].scatter([lp_x],[lp_y_min],marker='D',facecolor='none',edgecolor='k',zorder=20)
    print('LP at',lp_x,lp_y)
    
    # periodic U
    dat = np.loadtxt('dat/sirv1_kappa_theta=0.026_per_u_fixed.dat')
    axs[1].plot(dat[:,0],dat[:,1],color='tab:red',ls='--')
    axs[1].plot(dat[:,0],dat[:,2],color='tab:red',ls='--')

    # points S
    dat = np.loadtxt('dat/sirv1_kappa_theta=0.026_pts_s_fixed.dat')
    axs[1].plot(dat[:,0],dat[:,1],color='gray')

    y_val = dat[-1,1] # get steady-state S-value
    # get smallest x for which we are within 1e-2 of y_val.

    idxy = np.where(np.abs(dat[:,1]-y_val)<1e-3)[0]
    print(idxy)
    idx = np.argmin(dat[:,0][idxy])
    print('idx min',idx)
    tc_x = dat[:,0][idxy][idx]
    tc_y = dat[:,1][idxy][idx]
    print('(b) TC',tc_x,tc_y)
    axs[1].scatter([tc_x],[tc_y],marker='s',facecolor='none',edgecolor='k',zorder=20)


    # points U
    dat = np.loadtxt('dat/sirv1_kappa_theta=0.026_pts_u_fixed.dat')
    axs[1].plot(dat[:,0],dat[:,1],color='gray',ls=':')


    # points S
    dat = np.loadtxt('dat/sirv1_kappa_theta=0.026_pts2_s_fixed.dat')
    axs[1].plot(dat[:,0],dat[:,1],color='gray')

    idx = np.argmin(dat[:,0])
    axs[1].axvline(dat[idx,0]-5,ls='-.',lw=1)
    axs[1].text(dat[idx,0],0.6,'SNIC',color='tab:blue')
    #axs[1].scatter([dat[idx,0]],[dat[idx,1]],marker='p',facecolor='none',edgecolor='k')

    # points U
    dat = np.loadtxt('dat/sirv1_kappa_theta=0.026_pts2_u_fixed.dat')
    axs[1].plot(dat[:,0],dat[:,1],color='gray',ls=':')

    
    ##### axis technicals
    
    axs[0].set_xlim(0,600)
    axs[0].set_ylim(0,1)

    axs[1].set_xlim(0,600)
    axs[1].set_ylim(0,1)

    axs[0].set_title(panels[0],loc='left')
    axs[1].set_title(panels[1],loc='left')
    
    axs[0].set_title(r'$\delta=0.01$')
    axs[1].set_title(r'$\delta=0.03$')

    axs[0].set_xlabel(r'$\kappa$')
    axs[1].set_xlabel(r'$\kappa$')
    
    axs[0].set_ylabel(r'$S$',labelpad=0)
    axs[1].set_ylabel(r'$S$',labelpad=0)

    axs[0].legend(fontsize=8,labelspacing=.25,bbox_to_anchor=(1,.9),loc='upper right')

    plt.tight_layout()
    return fig


def seir1_onepar():
    """
    Figure for SEIR1 uncoupled model.
    See xpp file sirv1_social.ode
    # kappa
    """

    fig,axs = plt.subplots(figsize=(3.5,3))

    dat = np.loadtxt('dat/sirv1_kappa.dat')
    pt_bool_s, pt_bool_u, per_bool_s, per_bool_u = get_stable_unstable_bool(dat)    

    stable_pt_x = dat[:,0][pt_bool_s]
    stable_pt_y = dat[:,1][pt_bool_s]
    
    unstable_pt_x = dat[:,0][pt_bool_u]
    unstable_pt_y = dat[:,1][pt_bool_u]
    
    stable_per_x_max = dat[:,0][per_bool_s]
    stable_per_y_max = dat[:,1][per_bool_s]

    stable_per_x_min = dat[:,0][per_bool_s]
    stable_per_y_min = dat[:,2][per_bool_s]

    unstable_per_x_max = dat[:,0][per_bool_u]
    unstable_per_y_max = dat[:,1][per_bool_u]

    unstable_per_x_min = dat[:,0][per_bool_u]
    unstable_per_y_min = dat[:,2][per_bool_u]


    y = remove_y_disc(stable_pt_y)
    x,y = remove_x_disc(stable_pt_x,y)
    axs.plot(x,y,color='gray',label='Stable Point')

    y = remove_y_disc(unstable_pt_y)
    x,y = remove_x_disc(unstable_pt_x,y)
    axs.plot(x,y,color='gray',ls=':',label='Unstable Point')

    
    y = remove_y_disc(stable_per_y_min,threshold=1)
    x,y = remove_x_disc(stable_per_x_min,y)
    axs.plot(x,y,color='k',label='Stable Orbit')

    y = remove_y_disc(stable_per_y_max,threshold=1)
    x,y = remove_x_disc(stable_per_x_max,y)
    axs.plot(x,y,color='k')
    
    ##### annotation
    idx = np.argmin(np.abs(stable_per_x_max))
    lp_x = stable_per_x_max[idx]
    lp_y = stable_per_y_max[idx]
    axs.annotate("LP",xy=(lp_x,lp_y), xycoords='data',
                 xytext=(lp_x-20,lp_y+0.05), textcoords='data',
                 arrowprops=dict(facecolor='k',arrowstyle="-|>", connectionstyle="arc3,rad=.5"))
    print('LP at',lp_x,lp_y)


    
    y = remove_y_disc(unstable_per_y_min,threshold=1)
    x,y = remove_x_disc(unstable_per_x_min,y)
    axs.plot(x,y,color='tab:red',ls='--',label='Unstable Orbit')

    y = remove_y_disc(unstable_per_y_max,threshold=1)
    x,y = remove_x_disc(unstable_per_x_max,y)
    axs.plot(x,y,color='tab:red',ls='--')


    ##### annotations
    # get index where y coordinate of unstable orbit is close to zero
    # this is the subcritical hopf
    idx = np.argmin(np.abs(unstable_per_y_max))
    hb_x = unstable_per_x_max[idx]
    hb_y = unstable_per_y_max[idx]
    axs.annotate("HB",xy=(hb_x,hb_y), xycoords='data',
                 xytext=(hb_x+50,hb_y-0.05), textcoords='data',
                 arrowprops=dict(facecolor='k',arrowstyle="-|>", connectionstyle="arc3,rad=-.5"))
    print('HB at',hb_x,hb_y)

    # get smallest x where y-coordinate of stable points is close to hb_y
    # this the transcritical point
    idxy = np.where(np.abs(stable_pt_y-hb_y)<1e-2)[0]
    idx = np.argmin(stable_pt_x[idxy])
    tc_x = stable_pt_x[idx]
    tc_y = stable_pt_y[idx]
    axs.annotate("TC",xy=(tc_x,tc_y), xycoords='data',
                 xytext=(tc_x+50,tc_y-0.05), textcoords='data',
                 arrowprops=dict(facecolor='k',arrowstyle="-|>", connectionstyle="arc3,rad=-.5"))
    print('TC at',tc_x,tc_y)

    ##### axis technicals
    
    axs.set_xlim(0,600)
    axs.set_ylim(0,0.4)

    axs.set_xlabel(r'$\kappa$')
    axs.set_ylabel(r'$S$')

    axs.legend()

    plt.tight_layout()
    return fig

def get_interp(x,y,n=100):
    
    fn = interp1d(x,y)
    x2 = np.linspace(min(x),max(x),n)
    
    return x2, fn(x2)

def seir1_twopars():
    """
    Figure for SEIR1 uncoupled model.
    See xpp file sirv1_social.ode
    kappa vs different parameters
    we may need to fix kappa large (500-1000) and vary other pairs of parameters
    """

    y_labels = [r'$\alpha$',r'$\beta$',r'$\eta$',r'$\gamma$',r'$\rho$',r'$\delta$']
    
    fig,axs = plt.subplots(2,3,figsize=(6,4))
    axs = axs.flatten()

    dat_k_al = np.loadtxt('dat/sirv1_kap_al_fixed.dat')
    dat_k_al_hb = np.loadtxt('dat/sirv1_kap_al_hb_fixed.dat')
    dat_k_al_tc = np.loadtxt('dat/sirv1_kap_al_tc_fixed.dat')
    
    
    dat_k_be = np.loadtxt('dat/sirv1_kap_be_fixed.dat')
    dat_k_be_hb = np.loadtxt('dat/sirv1_kap_be_hb_fixed.dat')
    dat_k_be_tc = np.loadtxt('dat/sirv1_kap_be_tc_fixed.dat')

    dat_k_eta = np.loadtxt('dat/sirv1_kap_eta_fixed.dat')
    dat_k_eta_hb = np.loadtxt('dat/sirv1_kap_eta_hb_fixed.dat')
    dat_k_eta_tc = np.loadtxt('dat/sirv1_kap_eta_tc_fixed.dat')

    dat_k_gm = np.loadtxt('dat/sirv1_kap_gm_fixed.dat')
    dat_k_gm_hb = np.loadtxt('dat/sirv1_kap_gm_hb_fixed.dat')
    dat_k_gm_tc = np.loadtxt('dat/sirv1_kap_gm_tc_fixed.dat')
    
    
    dat_k_rho = np.loadtxt('dat/sirv1_kap_rho_fixed.dat')
    dat_k_rho_hb = np.loadtxt('dat/sirv1_kap_rho_hb_fixed.dat')
    dat_k_rho_tc = np.loadtxt('dat/sirv1_kap_rho_tc_fixed.dat')
    
    
    
    
    dat_k_theta = np.loadtxt('dat/sirv1_kap_theta_fixed.dat')
    dat_hb_k_theta = np.loadtxt('dat/sirv1_hb_kap_theta_fixed.dat')
    dat_bp_k_theta = np.loadtxt('dat/sirv1_bp_kap_theta_fixed.dat')
    dat_bp_low_k_theta = np.loadtxt('dat/sirv1_bp_low_kap_theta_fixed.dat')

    ### kappa alpha LP
    x_ka,y_ka = dat_k_al[:,0],dat_k_al[:,1]
    axs[0].plot(x_ka,y_ka,color='k')
    axs[0].plot(x_ka[100:],y_ka[100:],color='none',marker='D',markevery=200,markerfacecolor='none',markeredgecolor='k')
    
    ### kappa alpha HB
    x_kbh,y_kbh = dat_k_al_hb[:,0],dat_k_al_hb[:,1]
    axs[0].plot(x_kbh,y_kbh,color='tab:purple',marker='o',markevery=300,markerfacecolor='none',markeredgecolor='k',alpha=.75,ls='-')
    
    # fill between for LP and HB. 
    axs[0].fill_betweenx(y_ka,600,x_ka,color='gray',alpha=.25,lw=0)

    # Get max y coordinate of LP and use it to define cutoff for fill-in in HB
    y_max_lp = np.max(y_ka-.0001)
    y_kah_subset = y_kbh[y_kbh>y_max_lp]
    x_kah_subset = x_kbh[y_kbh>y_max_lp]
    axs[0].fill_betweenx(y_kah_subset,600,x_kah_subset,color='gray',lw=0,alpha=.25)

    ### kappa alpha TC
    x1,y1 = dat_k_al_tc[:,0],dat_k_al_tc[:,1]
    axs[0].plot(x1,y1,color='gray',ls='-',marker='s',markevery=350,markerfacecolor='none',markeredgecolor='k')



    

    ### kappa beta
    # LP
    x_kb,y_kb = dat_k_be[:,0],dat_k_be[:,1]
    axs[1].plot(x_kb,y_kb,color='k')
    axs[1].plot(x_kb[100:],y_kb[100:],color='none',marker='D',markevery=300,markerfacecolor='none',markeredgecolor='k')
    
    ### kappa beta HB
    y_kbh,x_kbh = get_interp(dat_k_be_hb[:,1],dat_k_be_hb[:,0],n=500)
    axs[1].plot(x_kbh,y_kbh,color='tab:purple',alpha=.75)

    # upper half of HB
    idx_split = np.argmin(dat_k_be_hb[:,0])
    xhb_up = dat_k_be_hb[:,0][idx_split+50:];yhb_up = dat_k_be_hb[:,1][idx_split+50:]
    xhb_do = dat_k_be_hb[:,0][:idx_split-200];yhb_do = dat_k_be_hb[:,1][:idx_split-200]
    
    xhb_up,yhb_up = get_interp(xhb_up,yhb_up,n=100)
    axs[1].plot(xhb_up,yhb_up,color='none',marker='o',markevery=30,markerfacecolor='none',markeredgecolor='k',alpha=.75)
    xhb_do,yhb_do = get_interp(xhb_do,yhb_do,n=100)
    axs[1].plot(xhb_do,yhb_do,color='none',marker='o',markevery=40,markerfacecolor='none',markeredgecolor='k',alpha=.75)
    
    # fill between for LP and HB. 
    axs[1].fill_betweenx(y_kb,600,x_kb,color='gray',alpha=.25,lw=0)

    # Get max y coordinate of LP and use it to define cutoff for fill-in in HB
    y_max_lp = np.max(y_kb-.001)
    y_kbh_subset = y_kbh[y_kbh>y_max_lp]
    x_kbh_subset = x_kbh[y_kbh>y_max_lp]
    axs[1].fill_betweenx(y_kbh_subset,600,x_kbh_subset,color='gray',lw=0,alpha=.25)

    ### kappa beta TC
    x1,y1 = dat_k_be_tc[:,0],dat_k_be_tc[:,1]
    axs[1].plot(x1,y1,color='gray',ls='-',marker='s',markevery=350,markerfacecolor='none',markeredgecolor='k')



    ### kappa eta
    # LP
    x_kb,y_kb = dat_k_eta[:,0],dat_k_eta[:,1]
    axs[2].plot(x_kb,y_kb,color='k')
    axs[2].plot(x_kb[100:],y_kb[100:],color='none',marker='D',markevery=300,markerfacecolor='none',markeredgecolor='k')
    
    ### kappa eta HB
    y_kbh,x_kbh = get_interp(dat_k_eta_hb[:,1],dat_k_eta_hb[:,0],n=500)
    axs[2].plot(x_kbh,y_kbh,color='tab:purple',alpha=.75)

    # upper half of HB
    idx_split = np.argmin(dat_k_eta_hb[:,0])
    xhb_up = dat_k_eta_hb[:,0][idx_split+50:];yhb_up = dat_k_eta_hb[:,1][idx_split+50:]
    xhb_do = dat_k_eta_hb[:,0][:idx_split-200];yhb_do = dat_k_eta_hb[:,1][:idx_split-200]
    
    xhb_up,yhb_up = get_interp(xhb_up,yhb_up,n=100)
    axs[2].plot(xhb_up,yhb_up,color='none',marker='o',markevery=30,markerfacecolor='none',markeredgecolor='k',alpha=.75)
    xhb_do,yhb_do = get_interp(xhb_do,yhb_do,n=100)
    axs[2].plot(xhb_do,yhb_do,color='none',marker='o',markevery=40,markerfacecolor='none',markeredgecolor='k',alpha=.75)

    
    # fill between for LP and HB. 
    axs[2].fill_betweenx(y_kb,600,x_kb,color='gray',alpha=.25,lw=0)


    ### kappa eta TC
    x1,y1 = dat_k_eta_tc[:,0],dat_k_eta_tc[:,1]
    axs[2].plot(x1,y1,color='gray',ls='-',marker='s',markevery=350,markerfacecolor='none',markeredgecolor='k')
    axs[2].set_ylim(-.1,2)




    ### kappa gm
    # LP
    x_kb,y_kb = dat_k_gm[:,0],dat_k_gm[:,1]
    axs[3].plot(x_kb,y_kb,color='k')
    axs[3].plot(x_kb[100:],y_kb[100:],color='none',marker='D',markevery=300,markerfacecolor='none',markeredgecolor='k')
    
    ### kappa gm HB
    y_kbh,x_kbh = get_interp(dat_k_gm_hb[:,1],dat_k_gm_hb[:,0],n=500)
    axs[3].plot(x_kbh,y_kbh,color='tab:purple',alpha=.75,zorder=2)

    # upper half of HB
    idx_split = np.argmin(dat_k_gm_hb[:,0])
    xhb_up = dat_k_gm_hb[:,0][idx_split+50:];yhb_up = dat_k_gm_hb[:,1][idx_split+50:]
    xhb_do = dat_k_gm_hb[:,0][:idx_split-200];yhb_do = dat_k_gm_hb[:,1][:idx_split-200]
    
    xhb_up,yhb_up = get_interp(xhb_up,yhb_up,n=100)
    axs[3].plot(xhb_up,yhb_up,color='none',marker='o',markevery=30,markerfacecolor='none',markeredgecolor='k',alpha=.75)
    xhb_do,yhb_do = get_interp(xhb_do,yhb_do,n=100)
    axs[3].plot(xhb_do,yhb_do,color='none',marker='o',markevery=40,markerfacecolor='none',markeredgecolor='k',alpha=.75)

    
    # fill between for LP and HB. 
    axs[3].fill_betweenx(y_kb,600,x_kb,color='gray',alpha=.25,lw=0)
    
    # Get min y coordinate of LP and use it to define cutoff for fill-in in HB
    y_max_lp = np.min(y_kb+.0001)
    y_kah_subset = y_kbh[y_kbh<y_max_lp]
    x_kah_subset = x_kbh[y_kbh<y_max_lp]
    #axs[3].plot(x_kah_subset,y_kah_subset,lw=10)
    axs[3].fill_betweenx(y_kah_subset,600,x_kah_subset,color='gray',lw=0,alpha=.25)



    ### kappa gm TC
    x1,y1 = dat_k_gm_tc[:,0],dat_k_gm_tc[:,1]
    axs[3].plot(x1,y1,color='gray',ls='-',marker='s',markevery=350,markerfacecolor='none',markeredgecolor='k')
    axs[3].set_ylim(0,.6)





    ### kappa rho
    # LP
    x_kb,y_kb = dat_k_rho[:,0],dat_k_rho[:,1]
    axs[4].plot(x_kb,y_kb,color='k')
    axs[4].plot(x_kb[100:],y_kb[100:],color='none',marker='D',markevery=300,markerfacecolor='none',markeredgecolor='k')
    
    ### kappa rho HB
    y_kbh,x_kbh = get_interp(dat_k_rho_hb[:,1],dat_k_rho_hb[:,0],n=500)
    axs[4].plot(x_kbh,y_kbh,color='tab:purple',alpha=.75,zorder=2)

    # upper half of HB
    idx_split = np.argmin(dat_k_rho_hb[:,0])
    xhb_up = dat_k_rho_hb[:,0][idx_split+50:];yhb_up = dat_k_rho_hb[:,1][idx_split+50:]
    xhb_do = dat_k_rho_hb[:,0][:idx_split-200];yhb_do = dat_k_rho_hb[:,1][:idx_split-200]
    
    xhb_up,yhb_up = get_interp(xhb_up,yhb_up,n=100)
    axs[4].plot(xhb_up,yhb_up,color='none',marker='o',markevery=30,markerfacecolor='none',markeredgecolor='k',alpha=.75)
    xhb_do,yhb_do = get_interp(xhb_do,yhb_do,n=100)
    axs[4].plot(xhb_do,yhb_do,color='none',marker='o',markevery=40,markerfacecolor='none',markeredgecolor='k',alpha=.75)

    
    # fill between for LP and HB. 
    axs[4].fill_betweenx(y_kb,600,x_kb,color='gray',alpha=.25,lw=0)
    
    # Get min y coordinate of LP and use it to define cutoff for fill-in in HB
    y_max_lp = np.min(y_kb+.0001)
    y_kah_subset = y_kbh[y_kbh<y_max_lp]
    x_kah_subset = x_kbh[y_kbh<y_max_lp]
    #axs[4].plot(x_kah_subset,y_kah_subset,lw=10)
    axs[4].fill_betweenx(y_kah_subset,600,x_kah_subset,color='gray',lw=0,alpha=.25)



    ### kappa rho TC
    x1,y1 = dat_k_rho_tc[:,0],dat_k_rho_tc[:,1]
    axs[4].plot(x1,y1,color='gray',ls='-',marker='s',markevery=150,markerfacecolor='none',markeredgecolor='k')
    axs[4].set_ylim(0,.05)




    
    # kappa delta fold
    x1,y1 = get_interp(dat_k_theta[:,0],dat_k_theta[:,1])
    axs[5].plot(x1,y1,color='k',marker='D',markevery=20,markerfacecolor='none',markeredgecolor='k')
    
    # kappa delta TC
    x,y = get_interp(dat_bp_k_theta[:,0], dat_bp_k_theta[:,1])
    axs[5].plot(x,y,color='gray',ls='-',marker='s',markevery=20,markerfacecolor='none',markeredgecolor='k')
    
    # kappa delta SNIC
    x2,y2 = get_interp(dat_bp_low_k_theta[:,0], dat_bp_low_k_theta[:,1])
    axs[5].plot(x2,y2,color='tab:blue',ls='-.')

    
    # get y2 on same x coordinates
    fn1 = interp1d(x1,y1)
    fn2 = interp1d(x2,y2)
    
    # fill between
    axs[5].fill_between(x1,fn1(x1),fn2(x1),color='gray',alpha=.25)

    # kappa delta HB
    x1,y1 = dat_hb_k_theta[:,0],dat_hb_k_theta[:,1]
    axs[5].plot(x1,y1,color='tab:purple',alpha=.75,marker='o',markevery=900,markeredgecolor='k',markerfacecolor='none')


    
    axs[5].set_ylim(0,.05)



    

    
    ##### annotations
    for k,ax in enumerate(axs):

        
        if k in [5]:
            x = .4; y = .3
        elif k in [0,1]:
            x = .6; y=.5
        else:
            x = .7;y = .35

        ax.text(x,y,'LC',transform=ax.transAxes,fontsize=8)

        if k in [2]:
            x = .02
            y = .2
        elif k in [3,4]:
            x = 0.02
            y = .9
        else:
            x = 0.02
            y = .05
        #ax.text(x,y,'No LC',transform=ax.transAxes,fontsize=8)

    ##### baseline parameter values
    

    #axs[0].scatter(0.01,ls='--',color='gray',lw=1) # alpha
    #axs[1].axhline(0.5,ls='--',color='gray',lw=1) # beta
    #axs[2].axhline(1/7,ls='--',color='gray',lw=1) # eta
    #axs[3].axhline(1/7,ls='--',color='gray',lw=1) # gamma
    #axs[4].axhline(0.01,ls='--',color='gray',lw=1) # rho
    #axs[5].axhline(0.01,ls='--',color='gray',lw=1) # delta
    default_values = [0.01,.5,1/7,1/7,.01,.01]
    
    
    ##### axis technicals
    for k,ax in enumerate(axs):
        ax.set_xlabel(r'$\kappa$',labelpad=0)
        ax.set_ylabel(y_labels[k])
        ax.set_title(panels[k],loc='left')
        ax.set_xlim(0,600)

        ax.scatter([0],[default_values[k]],marker='*',s=100,clip_on=False,zorder=5)
        

    #plt.tight_layout()
    #plt.subplots_adjust(left=.075,right=.97,bottom=.18,top=.89,wspace=.3
    plt.subplots_adjust(left=.1,wspace=.45,hspace=.5,top=.94,right=.96)
    
    return fig

    
def generate_figure(function, args, filenames, dpi=200):
    # workaround for python bug where forked processes use the same random 
    # filename.
    #tempfile._name_sequence = None;

    fig = function(*args)

    if type(filenames) is list:
        for name in filenames:
            fig.savefig(name,dpi=dpi)
    else:
        fig.savefig(filenames,dpi=dpi)

def main():

    #quick_plots_thalamic()

    # create figs directory if it doesn't exist
    if not(os.path.isdir('figs')):
        os.mkdir('figs')
    
    # listed in order of Figures in paper
    figures = [
        #(seir1_onepars,[],['figs/f_seir1_onepars_kap.pdf']),
        #(seir1_onepar,[],['figs/f_seir1_onepar_kap.pdf']),
        (seir1_twopars,[],['figs/f_seir1_twopars.pdf']),
        
    ]
    
    for fig in figures:
        generate_figure(*fig)



if __name__ == "__main__":
    main()
