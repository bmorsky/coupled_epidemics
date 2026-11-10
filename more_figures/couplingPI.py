r"""
$\newcommand{\ve}{\varepsilon}$

`hetCouplingPI.py` computes the (Taylor) higher-order interaction functions for $N=2$ heterogeneous models. It does NOT reduce to a single scalar equation like in Park and Wilson 2021. It keeps the isostable equations explicit so that a system of $N=2$ heterogeous models reduces to 3 equations (one phase difference and one isostable for each oscillator).

This code is essentially Nicks et al  "Insights into oscillator network dynamics using a phase-isostable framework" Chaos 2024.

`response.py` must be run prior to using this code.

Assumptions:
* $N=2$, i.e., only two models.
* At most one nontrivial Floquet multiplier for each model.
* No heterogeneous terms per oscillator.

To generate docs, run one of the following in the project root:
1. `./build_docs.sh`
2. `pdoc --math --footer-text "Generated with: pdoc --math *.py lib -o ./docs" *.py lib -o ./docs`
"""



# Tells pdoc to parse the NumPy-style sections below.
__docformat__ = "numpy"

# _n and _m have leading underscores, so pdoc treats them as private and
# hides them by default; whitelist them so they appear in the docs.
__pdoc__ = {"hetCoupling._n": True, "hetCoupling._m": True}


import copy
import lib.lib_sym as slib

#import lib.lib as lib
from lib import lib
from lib.fast_interp import interp1d,interp2d
from lib.interp_basic import interp_basic as interpb
from lib import fnames
#from lam_vec import lam_vec

#import inspect
import time
import os

import math
import sys
#import multiprocessing as multip
import tqdm
from pathos.pools import ProcessPool
from pathos.pools import _ProcessPool
import multiprocessing as mp
from multiprocessing import shared_memory

import scipy.interpolate as si
import numpy as np
#from numpy.fft import fft,ifft,fftfreq,fft2
from scipy.fft import fft,ifft,fft2,ifft2

#import scipy as sp
import sympy as sym
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import dill
dill.settings['recurse'] = True

from sympy import Matrix, symbols, Sum, Indexed, collect, expand
from sympy import sympify
from sympy.physics.quantum import TensorProduct as kp
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy.matrices.dense import matrix_multiply_elementwise as me
from sympy.tensor.array import derive_by_array

from scipy.integrate import quad
from scipy.signal import fftconvolve
#import pdoc

from scipy.integrate import solve_ivp

#import logging # for debugging
import warnings

#logging.getLogger('matplotlib.font_manager').disabled = True

#np.set_printoptions(linewidth=4)

warnings.filterwarnings('ignore',category=dill.PicklingWarning)
warnings.filterwarnings('ignore',category=dill.UnpicklingWarning)


imp_fn = implemented_function
imp_fn.__doc__ = "@private" # exclude from pdoc


class couplingPI(object):
    system1 : object
    """Ojbect of ODE system 1. Its index is system1.idx"""

    system2 : object
    """Ojbect of ODE system 2. Its index is system2.idx"""

    save_fig : bool = False
    """Save a figure of each response function for all variables"""

    pfactor : int = None
    r"""A numerical parameter. Determine how many periods to integrate $p_i^{(\ell)}$."""

    recompute_list : list = []
    """Force recomputation of all functions or specific functions. Enter ['all'] to recompute all cached functions and data files. This is unreliable if there are edits to the right-hand side or coupling function. For the most robust way to recompute everything, delete the `./data/model_name` directory."""

    NH : int = 100
    """Number of discretization points for the p (`self.NP`) and H (`self.NH`) functions. Higher values give finer phase resolution at greater cost."""

    trunc_deriv : int = 3
    """Truncation order for the Taylor expansion of the coupling function's derivatives in `generate_k_sym`. Important to define in case there higher-order derivatives are zero."""

    _n : tuple = (None, 1)
    """`(parameter_name, n)` pair for oscillator 1 in the n:m coupling. Only the integer `_n[1]` is used numerically; `_n[0]` names the associated parameter."""

    _m : tuple = (None, 1)
    """`(parameter_name, m)` pair for oscillator 2 in the n:m coupling, analogous to `_n`."""

    method : str = 'LSODA'
    """ODE solver name. Reserved for selecting the integrator; currently accepted but not used internally."""

    #log_level : str = 'CRITICAL'
    #"""Logging verbosity: 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'. Mapped to the matching `logging` level."""

    #log_file : str = 'log_nm.log'
    #"""Path to the file logging output is written to."""

    def __init__(self,system1,system2,
                 NH=100,
                 trunc_deriv=None,

                 _n=(None,1),_m=(None,1),
                 
                 method='LSODA',
                 
                 log_level='CRITICAL',log_file='log_nm.log',
                 save_fig=None,
                 pfactor=None,
                 recompute_list=None,
                 load_data_only=False,
                 
                 ):

        """Initialize the two-oscillator heterogeneous coupling problem.

        * Only the `N=2` case is supported. All constructor inputs are
        documented as class-level attributes above.

        Notes
        -----
        # het_fns: heterogeneous functions; index 0 is order eps_i, index 1
        # is order eps_i^2, and so on.
        """

        #print('Initializing {}{} Coupling...'.format(_n[1],_m[1]))

        if save_fig is not None:
            self.save_fig = save_fig

        if pfactor is not None:
            self.pfactor = pfactor

        self._expand_kws = {'basic':True,'deep':True,'power_base':False,
                            'power_exp':False,'mul':True,'log':False,
                            'multinomial':True}

        system1 = copy.deepcopy(system1)
        system2 = copy.deepcopy(system2)

        # ensures het. term carries through isostable calc.
        # constant het. only.
        #self.del1 = del1
        #system1.pardict['del'+str(system1.idx)] = del1
        #system1.rule_par['del'+str(system1.idx)] = del1

        #system1.pardict['del1'] = 0
        #system1.rule_par['del1'] = 0
        #system1.pardict_sym['del1'] = 0
        
        assert(type(_n) is tuple);assert(type(_m) is tuple)
        #assert(_n[0] in system1.pardict);assert(_m[0] in system2.pardict)
        
        self.system1 = system1;self.system2 = system2

        if recompute_list is not None:
            self.recompute_list = recompute_list

        if trunc_deriv is not None:
            self.trunc_deriv = trunc_deriv
        
        self._n = _n
        """n for the first oscillator."""
        
        self._m = _m
        """m for the second oscillator."""
        
        self.om = self._n[1]/self._m[1]
        """Frequency ratio $\omega=n/m$."""
        
        self.T = self.system1.T
        assert(self.system1.T == self.system2.T)
        """Period $T=2\pi$."""

        self.system1.pardict['om'+str(system1.idx)] = self._n[1]
        #self.system1.kappa_val *= self._n[1]
        
        if not(system1.forcing):
            self.system2.pardict['om'+str(system2.idx)] = self._m[1]
            #self.system2.kappa_val *= self._m[1]
        
        ths_str = ' '.join(['th'+str(i) for i in range(2)])
        #psis_str = ' '.join(['ps'+str(i) for i in range(2)])
        
        self.ths = symbols(ths_str,real=True)
        """All symbolic phase variables (th0,th1)"""
        
        self.psis = [self.system1.psi,self.system2.psi]#symbols(psis_str,real=True)
        """All symbolic isostable variables (ps0,ps1)"""

        # discretization for p_X,p_Y
        self.NP = NH
        """Legacy. NP used to be the discretization for p. Now it uses the same as H."""
        
        self.x,self.dx = np.linspace(0,self.T,self.NP,retstep=True,
                                     endpoint=False)
        
        self.an,self.dan = np.linspace(0,self.T*self._n[1],self.NP*self._n[1],
                                       retstep=True,endpoint=False)

        # discretization for H_X,H_Y
        self.NH = NH
        self.bn,self.dbn = np.linspace(0,self.T*self._m[1],NH,
                                       retstep=True,endpoint=False)

        self.be,self.dbe = np.linspace(0,self.T*self._m[1],NH*self._m[1],
                                       retstep=True,endpoint=True)

        self.bne,self.dbne = np.linspace(0,self.T*self._n[1],NH*self._n[1],
                                         retstep=True,endpoint=True)

        self.load_data_only = load_data_only


        system1.h['dat']={};system1.h['imp']={};system1.h['lam']={}
        system1.h['sym']={}

        system1.h['dat_hom']={};system1.h['dat_het']={}
        system1.h['sym_hom']={};system1.h['sym_het']={}
        system1.h['lam_hom']={};system1.h['lam_het']={}
        
        system2.h['dat_hom']={};system2.h['dat_het']={}
        system2.h['sym_hom']={};system2.h['sym_het']={}
        system2.h['lam_hom']={};system2.h['lam_het']={}

        # global replacement rules based on system index
        system1.rule_lc={};system1.rule_g={};system1.rule_z={}
        system1.rule_i={};

        system2.rule_lc={};system2.rule_g={};system2.rule_z={}
        system2.rule_i={}

        for j,key in enumerate(system1.var_names):
            in1 = self.ths[system1.idx]
            system1.rule_lc[system1.syms[j]] = system1.lc['imp_'+key](in1)
            
        for key in system1.var_names:
            for k in range(system1.miter):
                in1 = self.ths[system1.idx]

                kg = sym.Indexed('g'+key,k)
                system1.rule_g[kg] = system1.g['imp_'+key][k](in1)

                kz = sym.Indexed('z'+key,k)
                system1.rule_z[kz] = system1.z['imp_'+key][k](in1)
                
                ki = sym.Indexed('i'+key,k)
                system1.rule_i[ki] = system1.i['imp_'+key][k](in1)
                
        self._init_noforce(system1,system2)
        """legacy/redundant. was used when there was an option to include a forced system. hetCoupling only considers a pair of coupled oscillators."""

        #self.load_4d(system1,system2)
        #self.load_4d(system2,system1)


    def load_4d(self,system1,system2):
        """
        get functions for use in 3d reduction
        """
        
        s1 = system1;s2 = system2
        rule_temp1 = {**s1.rule_g_local,**s1.rule_lc_local,
                      **s1.rule_z_local,**s1.rule_i_local,
                      **s1.rule_par}
        rule_temp2 = {**s2.rule_g_local,**s2.rule_lc_local,
                      **s2.rule_z_local,**s2.rule_i_local,
                      **s2.rule_par}

        d1 = s1.gz.subs(rule_temp1)
        d1 = d1.subs({s1.t:self.ths[system1.idx]})
        d1 = d1.subs(rule_temp2)
        d1 = d1.subs({s2.t:self.ths[system2.idx]})
        

        d2 = s1.gi.subs(rule_temp1)
        d2 = d2.subs({s1.t:self.ths[system1.idx]})
        d2 = d2.subs(rule_temp2)
        d2 = d2.subs({s2.t:self.ths[system2.idx]})

        # system1.het_sym
        e0 = self.system1.het_sym['eps0']
        e1 = self.system2.het_sym['eps1']
        
        hs = [e0,e1]

        #print('*******d1,d2')
        
        #print([system1.het_sym['eps'+str(system1.idx)],*self.psis])
        #poly1 = sym.Poly(d1,[system1.het_sym['eps'+str(system1.idx)],*self.psis]).as_dict()
        #print('collect',sym.Poly(d1,[e0,*self.psis]).as_dict())
        #d1 = poly1[(1,0,0)]
        #print('d1:')
        #print(d1)
        #print('d2:')
        #print(d2)
        #print('')

        s1.gz_lam = lambdify([*self.ths,*self.psis,*hs],d1)
        s1.gi_lam = lambdify([*self.ths,*self.psis,*hs],d2)

        
    def _init_noforce(self,system1,system2):
        """Set up the problem when both oscillators are dynamical systems.

        Called by `__init__` when neither system is a forcing function.
        Builds the time grids, chooses a default `pfactor` if none was
        given, constructs the symbolic substitution rules for `system2`,
        lambdifies the Z and I expansions in psi, and then loads (or
        computes) the K terms, coupling response, and the symbolic and
        numerical p and H functions for both oscillators at every order.

        Parameters
        ----------
        system1 : object
            First oscillator system object.
        system2 : object
            Second oscillator system object.
        """
        self.t,self.dt = np.linspace(0,self.T,100,retstep=True)

        self.forcing = False # legacy

        # keeping pfactor as an explicit variable may be a bit antiquated. It may be that this heuristic is sufficient.
        if self.pfactor is None:
            pfactor1 = int((np.log(.01)/system1.kappa_val)/self.T)
            pfactor2 = int((np.log(.01)/system2.kappa_val)/self.T)

            self.pfactor = max([pfactor1,pfactor2])
            self.pfactor = max([2,self.pfactor])

            #print('pfactor default',self.pfactor)
            
        else:
            pass
            #print('pfactor user value',self.pfactor)
        
        # just keep these here.
        system2.h['dat']={};system2.h['imp']={};system2.h['lam']={}
        system2.h['sym']={}
        
        for j,key in enumerate(system2.var_names):
            in1 = self.ths[system2.idx]
            system2.rule_lc[system2.syms[j]] = system2.lc['imp_'+key](in1)
        
        for key in system2.var_names:
            for k in range(system2.miter):
                in1 = self.ths[system2.idx]
                
                kg = sym.Indexed('g'+key,k)
                system2.rule_g[kg] = system2.g['imp_'+key][k](in1)

                kz = sym.Indexed('z'+key,k)
                system2.rule_z[kz] = system2.z['imp_'+key][k](in1)
                
                ki = sym.Indexed('i'+key,k)
                system2.rule_i[ki] = system2.i['imp_'+key][k](in1)
        
        fnames.load_fnames_nm(system1,self)
        fnames.load_fnames_nm(system2,self)

        if not(self.load_data_only):
            #print('hello1')
            slib.load_coupling_expansions(system1,True)
            slib.load_coupling_expansions(system2,True)

            #print('hello1b')
        
            # load G and K
            self.load_k_sym(system1,system2)
            self.load_k_sym(system2,system1)
            #print('hello1c')
            
            self.load_coupling_response(system1)
            self.load_coupling_response(system2)
            #print('hello2')

        # convert from polynomial exponents to indices
        #self.exponent2idx = {(1,0,0):0,(1,0,1):1,(1,1,0):2}
        #het_term_idx_dict

        # [eps0,psi0,psi1,del0] for oscillator 1 and 2
        self.exponent2idx = {(1,0,0,0):0,(1,1,0,0):1,(1,0,1,0):2,
                             (0,0,0,1):6,(0,0,0,1):7}
        
        #self.exponent2idx = {(1,0,0,0):0,(0,1,0,0):1,
        #                     (0,0,1,0):2,(2,0,0,0):3,
        #                     (1,1,0,0):4,(0,2,0,0):5,
        #                     (1,0,1,0):6,(0,1,1,0):7,
        #                     (0,0,2,0):8,(0,0,1,1):9}

        #self.eps = sym.symbols('eps')
        r"""global epsilon. where $\varepsilon_i\rightarrow\varepsilon b_i$ and $\delta_i\rightarrow\varepsilon q_i$"""

        #print('hello3')
        # h functions
        recompute = ('h_sym' in self.recompute_list) or ('all' in self.recompute_list)
        self.load_h_sym(system1,system2,recompute=recompute)
        self.load_h_sym(system2,system1,recompute=recompute)
        

        #print('hello4')
        #self.load_h_sym(system1,system2,recompute=recompute)
        #self.load_h_sym(system2,system1,recompute=recompute)
        recompute = ('h' in self.recompute_list) or ('all' in self.recompute_list)
        # 0-5: H1..H6, 6: het term (theta eq.), 7: het term (psi eq.)
        for k in range(8):
            self.load_h(system1,system2,k,recompute=recompute)
            self.load_h(system2,system1,k,recompute=recompute)

        #print('hello5')
        #recompute = ('h' in self.recompute_list) or ('all' in self.recompute_list)
        #for k in range(6):
        #    self.load_h(system1,system2,k,recompute=recompute)
        #    self.load_h(system2,system1,k,recompute=recompute)

        # compute het. term
        #self.load_h(system1,system2,6,recompute=recompute)

        
        

            
    def load_k_sym(self,system1,system2):
        
        """Load or compute the symbolic coupling terms G and K.

        G is the full expanded coupling function; K is the same expansion
        indexed by powers of eps. Results are cached to disk and reloaded
        unless recomputation is requested.

        Parameters
        ----------
        system1 : object
            Oscillator whose G and K terms are computed.
        system2 : object
            The other oscillator, whose variables enter the coupling.
        """
        #print('load_k_sym')
            
        # check that files exist
        val = not(os.path.isfile(system1.G['fname_psi']))
        val += not(os.path.isfile(system1.K['fname_psi']))

        files_dne = val

        system1.G['sym_psi'] = {}
        system1.K['sym_psi'] = {}
        
        if 'k_'+system1.model_name in self.recompute_list or files_dne or ('all' in self.recompute_list):
            #print('* Computing G symbolic...')
            #print('trunc update?')
            rule_trunc = {}
            for k in range(system1.miter,system1.miter+50):
                rule_trunc.update({system1.eps**(k+1):0})
                rule_trunc.update({system1.psi**(k+1):0})

            #print('trunc update end')

            #print('ehllo 1b1')
            self.generate_k_sym(system1,system2)
            #print('ehllo 1b2')

            # now collect terms
            # G contains full summation
            # K contains terms of summation indexed by power
            for key in system1.var_names:
                tmp_p = expand(system1.G['sym_psi'][key],**self._expand_kws)
                system1.G['sym_psi'][key] = tmp_p.subs(rule_trunc)
                
                system1.K['sym_psi'][key] = []
                for k in range(system1.miter):
                    e_term_p = tmp_p.coeff(system1.psi,k)
                    system1.K['sym_psi'][key].append(e_term_p)

            dill.dump(system1.G['sym_psi'],open(system1.G['fname_psi'],'wb'),
                      recurse=True)
            dill.dump(system1.K['sym_psi'],open(system1.K['fname_psi'],'wb'),
                      recurse=True)
                
        else:
            #print('load dill?')
            system1.G['sym_psi'] = dill.load(open(system1.G['fname_psi'],'rb'))
            system1.K['sym_psi'] = dill.load(open(system1.K['fname_psi'],'rb'))

        # create symbolic vector of psi for convenience
        # epsilon is assumed to be included in the coupling function.

        #print('finish load?')
        system1.G['vec_psi'] = sym.zeros(system1.dim,1)
        for i,key in enumerate(system1.var_names):
            
            out = system1.G['sym_psi'][key]
            system1.G['vec_psi'][i] = [out]
            #print(i,key,system1.var_names)

        #print('finish vec_psi?')
        


    def generate_k_sym(self,system1,system2):
        """Generate the symbolic coupling terms (K in the paper).

        Implements equations (8) and (9) of Park & Wilson 2021: expands the
        coupling function to `trunc_deriv` order, substitutes the g
        expansions in eps and in psi, and stores the results in
        `system1.G['sym']` and `system1.G['sym_psi']`.

        Parameters
        ----------
        system1 : object
            Oscillator whose coupling terms are generated.
        system2 : object
            The other oscillator.
        """

        psym = system1.pardict_sym # shorten for readability
        hsym = {**system1.het_sym,**system2.het_sym}
        fn = system1.coupling # coupling function

        # all variables and d_variables
        var_names_all = system1.var_names + system2.var_names
        z = sym.Matrix(system1.syms + system2.syms)
        dz = sym.Matrix(system1.dsyms + system2.dsyms)

        # get full coupling term for given oscillator
        system1.G['sym_fn'] = sym.flatten(fn(z,psym,hsym,option='sym',idx=system1.idx))
        
        # 0 and 1st derivative
        c_temp = {}
        for key_idx,key in enumerate(system1.var_names):
            c_temp[key] = system1.G['sym_fn'][key_idx]
            d = lib.df(system1.G['sym_fn'][key_idx],z,1).dot(dz)
            c_temp[key] += d
            
        # 2nd + derivative        
        for key_idx,key in enumerate(system1.var_names):
            for k in range(2,self.trunc_deriv+1):
                kp = lib.kProd(k,dz)
                da = lib.vec(lib.df(system1.G['sym_fn'][key_idx],z,k))
                c_temp[key] += (1/math.factorial(k))*kp.dot(da)

        # save 
        for key in system1.var_names:
            system1.G['sym_psi'][key] = c_temp[key]
                        
        # replace dx with (g expansion in eps).
        rule = {}

        for key_idx,key in enumerate(system1.var_names):
            rule.update({system1.dsyms[key_idx]:system1.g[key+'_eps']})
        for key_idx,key in enumerate(system2.var_names):
            rule.update({system2.dsyms[key_idx]:system2.g[key+'_eps']})

        # do the same for psi expansion
        rule_psi = {}
        
        for key_idx,key in enumerate(system1.var_names):
            rule_psi.update({system1.dsyms[key_idx]:system1.g['expand_'+key]})
            
        # replace psi with psi_0        
        for key in system1.var_names:
            d1 = system1.G['sym_psi'][key].subs(rule_psi)
            d1 = d1.subs({system1.psi:self.psis[system1.idx]})
            system1.G['sym_psi'][key] = d1
            
        for key_idx,key in enumerate(system2.var_names):
            rule_psi.update({system2.dsyms[key_idx]:system2.g['expand_'+key]})

        # replace psi with psi_i  
        for key in system1.var_names:
            d1 = system1.G['sym_psi'][key].subs(rule_psi)
            d1 = d1.subs({system2.psi:self.psis[system2.idx]})            
            system1.G['sym_psi'][key] = d1

        
        

    def load_coupling_response(self,system):
        """Construct the truncated phase and isostable response functions.

        Builds the gz and gi products (the coupling term times the Z and I
        psi expansions) needed in the averaging step, truncates them, and
        caches them to disk. Assumes the coupling function is nonzero only
        in the first coordinate.

        Parameters
        ----------
        system : object
            Oscillator whose response functions are constructed.
        """

        file_dne = not(os.path.isfile(system.G['fname_gz']))
        file_dne += not(os.path.isfile(system.G['fname_gi']))

        if file_dne or 'gz_'+system.model_name in self.recompute_list or ('all' in self.recompute_list):

            # multiply G['sym_psi'] with the z and i psi expansions
            z = system.z['vec_psi']
            i = system.i['vec_psi']
            g = system.G['vec_psi']

            # specify psi_index
            z = z.subs({system.psi:self.psis[system.idx]})
            i = i.subs({system.psi:self.psis[system.idx]})

            gz = sym.expand(z.dot(g))
            gi = sym.expand(i.dot(g))

            # truncate
            rule_trunc = {}
            for k in range(system.miter,system.miter**2):
                for ll in range(k+1):
                    rule_trunc.update({self.psis[0]**ll*self.psis[1]**(k-ll):0})

            #remove_het = {'del0':0,'del1':0,'sig0':0,'sig1':0}
            gz = gz.subs(rule_trunc)#.subs(remove_het)
            gi = gi.subs(rule_trunc)#.subs(remove_het)

            system.gz = gz
            system.gi = gi

            dill.dump(system.gz,open(system.G['fname_gz'],'wb'),recurse=True)
            dill.dump(system.gi,open(system.G['fname_gi'],'wb'),recurse=True)

        else:
            system.gz = dill.load(open(system.G['fname_gz'],'rb'))
            system.gi = dill.load(open(system.G['fname_gi'],'rb'))

    def load_h_sym(self, system1, system2, recompute=False):
        """Generate or load the symbolic H functions for an oscillator."""
        fname = system1.h['fname']
        
        if recompute or not os.path.isfile(fname):
            #print('* Computing H symbolic...')
            system1.h['sym'] = {k: sym.S.Zero for k in range(8)}
            v1 = system1.z['vec_psi']
            v2 = system1.i['vec_psi']
            v3 = system1.G['vec_psi']
            
            tmp1 = sym.expand(v1.dot(v3), **self._expand_kws)
            tmp2 = sym.expand(v2.dot(v3), **self._expand_kws)

            # [own eps, own psi (psi_i), other psi (psi_j), own het] -- uses the
            # method arguments so h2/h3, h5/h6 are correct for both oscillators.
            sym_values = [system1.eps, system1.psi, system2.psi,
                          list(system1.het_sym.values())[0]]
            theta_idx = {(1,0,0,0):0, (1,1,0,0):1, (1,0,1,0):2, (0,0,0,1):6}
            psi_idx   = {(1,0,0,0):3, (1,1,0,0):4, (1,0,1,0):5, (0,0,0,1):7}

            # monomials absent from the tables are higher order and dropped
            for key, val in sym.Poly(tmp1, sym_values).as_dict().items():
                if key in theta_idx:
                    #print('key exists', key)
                    system1.h['sym'][theta_idx[key]] = val
            for key, val in sym.Poly(tmp2, sym_values).as_dict().items():
                if key in psi_idx:
                    #print('key exists', key)
                    system1.h['sym'][psi_idx[key]] = val
                    
            dill.dump(system1.h['sym'], open(fname, 'wb'))

        else:
            system1.h['sym'] = dill.load(open(fname,'rb'))

        


    def load_h(self,system1,system2,k,recompute=False):
        """Split the order-k H function into homogeneous and heterogeneous parts.


        Explicitly computes (or loads) and saves the homogeneous part and the
        heterogeneous parts of the H function separately, building a periodic
        interpolant for each.


        Parameters
        ----------
        system1 : object
            Oscillator whose H function is split.
        system2 : object
            The other oscillator.
        k : int
            Order in epsilon.
        recompute : bool, optional
            If True, recompute instead of loading the cache.
        """

        fname = system1.h['fnames_data'][k]
        file_dne = not(os.path.isfile(fname))

        if recompute or file_dne:
            str1 = '* Computing H {}, idx={}...'
            #print(str1.format(system1.model_name,k))
            expr = system1.h['sym'][k]
            dat = self.generate_h_old(system1,system2,expr,k)
            np.savetxt(system1.h['fnames_data'][k],dat)

        else:
            str1 = '* Loading H {}, idx={}...'
            #print(str1.format(system1.model_name,k))

            dat = np.loadtxt(fname)

        system1.h['dat'][k] = dat

        n=self._n[1];m=self._m[1]

        system1.h['lam'][k] = interp1d(0,self.T,self.dx,dat,p=True,k=3)

            
    def check_h_dne(self,system1,system2,k):
        """Check whether the order-k H data file is missing.

        Parameters
        ----------
        system1 : object
            Oscillator whose H data file is checked.
        system2 : object
            The other oscillator (unused; kept for signature consistency).
        k : int
            Order in epsilon.

        Returns
        -------
        bool
            True if the H data file does not exist, otherwise False.
        """
        fname = system1.h['fnames_data'][k]
        file_dne = not(os.path.isfile(fname))

        return file_dne
        



    def generate_h_new(self, system1, system2, expr, k):
        """Compute the order-k averaged interaction function H(phi) via 2D FFT.

        Newer implementation. Samples `expr` on a grid sized to avoid
        aliasing on the n:m resonant line, takes a 2D FFT, extracts the
        Fourier modes lying on the resonant line (modes n*i, -m*i), and
        inverse-transforms to recover H over a single slow phase.

        Parameters
        ----------
        system1 : object
            Oscillator the H function is being computed for.
        system2 : object
            The other oscillator.
        expr : sympy.Expr
            Symbolic expression for the order-k H term.
        k : int
            Order in epsilon.

        Returns
        -------
        numpy.ndarray
            A 1D real array of length `self.NH` giving H(phi).
        """
        rule = {**system1.rule_par,**system2.rule_par, 
                **system1.rule_lc, **system2.rule_lc,
                **system1.rule_g, **system2.rule_g,
                **system1.rule_z, **system2.rule_z,
                **system1.rule_i, **system2.rule_i}

        if expr == 0:
            return np.zeros(len(self.x))

        h_lam = lambdify(self.ths, expr.subs(rule))

        n = self._n[1]
        m = self._m[1]

        # Use enough samples to avoid aliasing on the resonant line.
        # Minimum: n*NH by m*NH (often you may want a safety factor 2–4).
        N1 = n * self.NH
        N2 = m * self.NH

        th1 = np.linspace(0, self.T, N1, endpoint=False)
        th2 = np.linspace(0, self.T, N2, endpoint=False)
        TH1, TH2 = np.meshgrid(th1, th2, indexing='ij')

        h_grid = h_lam(TH1, TH2)
        F = fft2(h_grid)  # unnormalized

        # Build 1D Fourier coefficients for H(phi) on modes i=0..NH-1:
        # coeff[i] = c_{k1=n*i, k2=-m*i} with modular wrap for negative index.
        coeff = np.zeros(self.NH, dtype=complex)
        for i in range(self.NH):
            k1 = (n * i) % N1
            k2 = (-m * i) % N2
            coeff[i] = F[k1, k2]

        # Normalize: for numpy/scipy FFT conventions, ifft divides by len.
        # We want coefficients consistent with ifft over NH points:
        # scaling here should include the 2D grid size.
        coeff = coeff / (N1 * N2)

        out = ifft(coeff).real
        return out


    def generate_h_old(self,system1,system2,expr,k,integration='riemann'):

        
        rule = {**system1.rule_par,**system2.rule_par,
                **system1.rule_lc,**system2.rule_lc,
                **system1.rule_g,**system2.rule_g,
                **system1.rule_z,**system2.rule_z,
                **system1.rule_i,**system2.rule_i}
                

        if expr == 0:
            return np.zeros(len(self.x))

        h_lam = lambdify(self.ths,expr.subs(rule))
        
        m = self._m[1]
        # quadrature over one common period [0, T*m); endpoint=False makes the
        # rectangle rule spectrally accurate for a periodic integrand
        s, ds = np.linspace(0, self.T, self.NH, endpoint=False, retstep=True)
        # output grid must match the interp1d in load_h: chi_j = j*T/NH
        chi = np.arange(self.NH)*self.dx
        # argument order is (th0, th1) = (s, chi + om*s) for BOTH systems,
        # so both arrays are functions of chi = th1 - th0
        h = np.array([np.sum(h_lam(s, c + self.om*s))*ds for c in chi])
        return h/(self.T)

    
    def generate_h(self,system1,system2,expr,k):
        """Compute the order-k interaction function H over phase.

        Original implementation. Evaluates `expr` on the n:m phase
        meshgrid, takes a 2D FFT, keeps the n:m resonant (diagonal) modes,
        and inverse-transforms to obtain H over a single slow phase.

        Parameters
        ----------
        system1 : object
            Oscillator the H function is being computed for.
        system2 : object
            The other oscillator.
        expr : sympy.Expr
            Symbolic expression for the order-k H term.
        k : int
            Order in epsilon.

        Returns
        -------
        numpy.ndarray
            A 1D real array giving H over phase. Returns zeros when `expr`
            is 0.
        """

        rule = {**system1.rule_par,**system2.rule_par,
                **system1.rule_lc,**system2.rule_lc,
                **system1.rule_g,**system2.rule_g,
                **system1.rule_z,**system2.rule_z,
                **system1.rule_i,**system2.rule_i}
                
        #print('rulez',system1.rule_z,system2.rule_z)

        if expr == 0:
            return np.zeros(len(self.x))


        h_lam = lambdify(self.ths,expr.subs(rule))

        x=self.x;dx=self.dx
        n=self._n[1];m=self._m[1]

        #print('idx, expr in h:',system1.idx,expr)
        #print('idx, imp in h:',system1.idx,expr.subs(rule))
        
        X,Y = np.meshgrid(x*n,x*m,indexing='ij')
        h_integrand = h_lam(X,Y)
        ft2 = fft2(h_integrand)
        # get only those in n:m indices
        # n*i,-m*i
        ft2_new = list([ft2[0,0]])
        ft2_new += list(np.diag(np.fliplr(ft2[1:,1:])))
        ft2_new = np.asarray(ft2_new)
        ft2_new *= self.T/self.NH

        out = ifft(ft2_new).real

        return out

        """
        bn=self.be;dbn=self.dbe
        #bn=self.be;dbn=self.dbe
        n=self._n[1];m=self._m[1]

        # calculate mean
        #X,Y = np.meshgrid(self.an,self.an*self._m[1],indexing='ij')
        #system1._H0[k] = np.sum(h_lam(X,Y))*self.dan**2/(2*np.pi*self._m[1])**2

        def integrand(x,y):
            return h_lam(y+self.om*x,x)
                
        # calculate coupling function
        h = np.zeros(self.NH)
        for j in range(self.NH):
            #val = np.sum(h_lam(bn[j] + self.om*bn,bn))*dbn
            val = quad(integrand,bn[0],bn[-1],args=(bn[j],),limit=10000)[0]

            h[j] = val

        out = h/(2*np.pi*self._m[1])

        return out
        """
        
    def fast_interp_lam(self,fn):
        """Wrap a fast_interp interp2d object as a phase-indexed callable.

        Needed so interp2d objects can be used inside sympy implemented
        functions.

        Parameters
        ----------
        fn : object
            An interp2d object (from `fast_interp`).

        Returns
        -------
        callable
            A function of the two phase variables that evaluates `fn`.
        """
        return lambda t0=self.ths[0],t1=self.ths[1]: fn(t0,t1)

    
    def save_temp_figure(self,data,k,fn='plot',path_loc='figs_temp/'):
        """Save a quick diagnostic figure of a trajectory to disk.

        Parameters
        ----------
        data : numpy.ndarray
            Array of shape (TN, dim) to plot, one subplot per state variable.
        k : int
            Index used in the figure title and file name.
        fn : str, optional
            Base file name / label for the plot.
        path_loc : str, optional
            Directory to save the figure in; created if it does not exist.
        """

        if (not os.path.exists(path_loc)):
            os.makedirs(path_loc)
        
        fig, axs = plt.subplots(nrows=self.dim,ncols=1)
        
        for j,ax in enumerate(axs):
            key = self.var_names[j]
            ax.plot(self.tLC,data[:,j],label=key)
            ax.legend()
            
        #logging.info(fn+str(k)+' ini'+str(data[0,:]))
        #logging.info(fn+str(k)+' fin'+str(data[-1,:]))
        axs[0].set_title(fn+str(k))
        plt.tight_layout()
        plt.savefig(path_loc+fn+str(k)+'.png')
        plt.close()


    def _collect_impls(self, *objs):
        """name -> implemented_function class (carrying its numba _imp_)."""
        reg = {}
        for o in objs:
            for a in sym.sympify(o).atoms(AppliedUndef):
                if hasattr(a.func, '_imp_'):
                    reg[a.func.__name__] = a.func
        return reg

    
    @staticmethod
    def _reattach(expr, reg):
        """Swap plain undefined functions back to their numba-backed versions."""
        reps = {a: reg[a.func.__name__](*a.args)
                for a in expr.atoms(AppliedUndef) if a.func.__name__ in reg}
        return expr.xreplace(reps)

    
