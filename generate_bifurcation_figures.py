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



matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
#matplotlib.rcParams['font.size'] = 12


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

def seir1_onepar():
    """
    Figure for SEIR1 uncoupled model.
    See xpp file sirv1_social.ode
    # kappa
    """

    fig,axs = plt.subplots(figsize=(3.5,3))

    dat = np.loadtxt('dat/sirv1_kappa.dat')

    stable_pt_bool = dat[:,3] == 1
    unstable_pt_bool = dat[:,3] == 2
    stable_per_bool = dat[:,3] == 3
    unstable_per_bool = dat[:,3] == 4
    

    stable_pt_x = dat[:,0][stable_pt_bool]
    stable_pt_y = dat[:,1][stable_pt_bool]
    
    unstable_pt_x = dat[:,0][unstable_pt_bool]
    unstable_pt_y = dat[:,1][unstable_pt_bool]
    
    stable_per_x_max = dat[:,0][stable_per_bool]
    stable_per_y_max = dat[:,1][stable_per_bool]

    stable_per_x_min = dat[:,0][stable_per_bool]
    stable_per_y_min = dat[:,2][stable_per_bool]

    unstable_per_x_max = dat[:,0][unstable_per_bool]
    unstable_per_y_max = dat[:,1][unstable_per_bool]

    unstable_per_x_min = dat[:,0][unstable_per_bool]
    unstable_per_y_min = dat[:,2][unstable_per_bool]


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

    dat_k_al = np.loadtxt('dat/sirv1_kap_al.dat')
    dat_k_be = np.loadtxt('dat/sirv1_kap_be.dat')
    dat_k_eta = np.loadtxt('dat/sirv1_kap_eta.dat')
    dat_k_gm = np.loadtxt('dat/sirv1_kap_gm.dat')
    dat_k_rho = np.loadtxt('dat/sirv1_kap_rho.dat')
    dat_k_theta = np.loadtxt('dat/sirv1_kap_theta.dat')



    y = remove_y_disc(dat_k_al[:,1],threshold=.005)
    x,y = remove_x_disc(dat_k_al[:,0],y)
    axs[0].plot(x,y,color='k')

    x,y = dat_k_be[:,0], dat_k_be[:,1]
    y = remove_y_disc(y,threshold=.1)
    x,y = remove_x_disc(x,y)
    axs[1].plot(x,y,color='k')

    x,y = dat_k_eta[:,0], dat_k_eta[:,1]
    y = remove_y_disc(y,threshold=.1)
    x,y = remove_x_disc(x,y)
    axs[2].plot(x,y,color='k')

    x,y = dat_k_gm[:,0], dat_k_gm[:,1]
    y = remove_y_disc(y,threshold=.05)
    x,y = remove_x_disc(x,y)
    axs[3].plot(x,y,color='k')

    x,y = dat_k_rho[:,0], dat_k_rho[:,1]
    y = remove_y_disc(y,threshold=.005)
    x,y = remove_x_disc(x,y)
    axs[4].plot(x,y,color='k')

    x,y = dat_k_theta[:,0], dat_k_theta[:,1]
    y = remove_y_disc(y,threshold=.005)
    x,y = remove_x_disc(x,y)
    axs[5].plot(x,y,color='k')

    ##### annotations
    for k,ax in enumerate(axs):

        ax.text(.7,.35,'LC',transform=ax.transAxes,fontsize=8)

        if k in [2]:
            x = .02
            y = .2
        elif k in [3,4]:
            x = 0.02
            y = .9
        else:
            x = 0.02
            y = .05
        ax.text(x,y,'No LC',transform=ax.transAxes,fontsize=8)

    ##### baseline parameter values
    #axs[0].scatter(0.01,ls='--',color='gray',lw=1) # alpha
    #axs[1].axhline(0.5,ls='--',color='gray',lw=1) # beta
    #axs[2].axhline(1/7,ls='--',color='gray',lw=1) # eta
    #axs[3].axhline(1/7,ls='--',color='gray',lw=1) # gamma
    #axs[4].axhline(0.01,ls='--',color='gray',lw=1) # rho
    #axs[5].axhline(0.01,ls='--',color='gray',lw=1) # delta
    
    ##### axis technicals
    labels = ['A','B','C','D','E','F']
    for k,ax in enumerate(axs):
        ax.set_xlabel(r'$\kappa$',labelpad=0)
        ax.set_ylabel(y_labels[k])
        ax.set_title(labels[k],loc='left')

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
        #(seir1_onepar,[],['figs/f_seir1_onepar_kap.pdf']),
        (seir1_twopars,[],['figs/f_seir1_twopars.pdf']),
        
    ]
    
    for fig in figures:
        generate_figure(*fig)



if __name__ == "__main__":
    main()
