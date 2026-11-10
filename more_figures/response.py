"""
Calculate limit cycle, monodromy, and response functions.

---
To generate docs, run the following in the project root:
```pdoc --math --footer-text "Generated with: pdoc --math *.py lib -o ./docs" *.py lib -o ./docs```

"""

import lib.lib_sym as slib
from lib import lib
from lib import fnames
from lib.fast_interp import interp1d
from lib.util import get_period

import os
import logging
import numpy as np

import time
import dill
import copy

import scipy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import sympy as sym
from sympy import Matrix, symbols, Sum, Indexed, collect, expand
from sympy import sympify
from sympy.utilities.lambdify import lambdify, implemented_function

from sympy.physics.quantum import TensorProduct as kp
import math

imp_fn = implemented_function
imp_fn.__doc__ = "@private" # exclude from pdoc

class Response(object):
    var_names : list
    """Names of variables in the model. Must match the number of variables in the model. These variable names are used by sympy. Variable names will have oscillator `idx` appended to the end of the name."""

    par_dict : dict
    """Dictionary of parameter names and values."""

    rhs : callable
    """Right-hand side of a given model. Input must be of the form (t,z,par_dict,option,idx). The right-hand side must be compatible with both sympy and numpy (so `option` is either `sym` or `val`). The input `idx` specifies the oscillator index and must be an integer."""

    init : list | np.typing.NDArray
    """Initial condition vector + period estimate. This script will NOT find this initial vector in a vacuum!! You must find a point and period that's close to the limit cycle and its period."""

    TN : int = 5000
    """total number of time points to use in a single period. `2000` is a decent starting point, but you will need to increase this if response functions vary rapidly as a function of time. It would be nice to make this discretization non-uniform one day, but this is good enough for now."""

    idx : int | str
    """Index of the oscillator. This is mostly used for bookkeeping, so it just needs to be distinct between the two oscillators. Each variable in `var_names` will have `str(idx)` appended to them under the hood."""
    
    model_name : str
    """Name of the model. must be unique between the two oscillators. A directory will be created using this name. `model_name` is also used internally for debugging."""
    
    trunc_order : int = 1
    """The number of higher-order terms to calculate. 0 calculates the classic infinitesimal phase response curve and infinitesimal isostable response curve. 1 is the next-order correction, 2 is the next-order correct, and so on."""
    
    recompute_list : list = []
    """Force recomputation of all functions or specific functions. Enter ['all'] to recompute all response functions. If you wish to recompute symbolic calculations, delete the ./data/`model_name` directory."""

    g_forward : bool | list = True
    """Determine if the Floquet eigenfunction calculation is performed forwards or backwards in time."""
    
    z_forward : bool | list = False
    """Determine if the phase response function calculation is performed forwards or backwards in time."""
    
    i_forward : bool | list = False
    """Determine if the isostable response function calculation is performed forwards or backwards in time."""
    
    i_bad_dx : bool | list
    """This is manually set to `True` when the inverse matrix of Newton's method is effectively singular. The inverse is replaced with a least squares approximation."""
    
    coupling : callable
    """the coupling function from above. This example happens to be symmetric, so we can use the same coupling function as the input to both oscillators."""

    het_pars : list | dict
    r"""Parameters for heterogeneity including $\varepsilon_i$ and $\delta_{ij}$ for oscillator $i$ and parameter $j$."""

    lc_prominence : float = 0.25
    r"""Parameter `prominence` in `scipy.signal.find_peaks` for the limit cycle. Needed to determine the time/index where a variable of the limit cycle reaches its peak."""

    max_time_lc : float | int = 5000
    """Max time to integrate the RHS to converge to a limit cycle solution"""

    save_fig : bool = False
    """Save a figure of each response function g, Z, I, including all variables."""

    tol_root_lc : float = 1e-13
    """Tolerance for Newton's method to find LC"""

    init_is_precise : bool = False
    """If False, this class will integrate the underlying RHS for `max_t_lc` time units to hopefully converge onto the limit cycle. If the limit cycle does not converge, you will get an error saying that no peaks were detected. Set the `save_fig` flag to True to see if plots show a limit cycle solution. If yes, then reduce the `lc_prominence` parameter (it is the same as the `prominence` parameter in `scipy.signal.find_peaks`). """

    dir_root : str = './data'
    """Directory for outputs of response.py. Mostly numpy arrays and pickled bojects."""
    
    #rhs_kwargs is for when using a more general right-hand side.
    
    def __init__(self,var_names,
                 rhs,
                 init,                 
                 idx,
                 
                 init_is_precise=False, # force use input init in lc calculation
                 
                 TN=None,

                 coupling=None,
                 het_pars=None,
                 trunc_order=None,
                 recompute_list=None,

                 pardict=None,
                 forcing_fn=None,
                 
                 dir_root=None,
                 model_name=None,
                 pars_for_fname='',

                 method='LSODA',
                 log_level='CRITICAL',
                 log_file='log_response.log',
                 
                 g_forward=None,
                 z_forward=None,
                 i_forward=None,
                 
                 g_jac_eps=1e-3,
                 z_jac_eps=1e-3,
                 i_jac_eps=1e-3,
                 i_bad_dx=False,

                 rtol=1e-10,
                 atol=1e-10,
                 max_iter=30,
                 rel_tol=1e-10,
                 rtol_lc=1e-12,
                 atol_lc=1e-12,

                 mode='nm',
                 lc_prominence=None,
                 max_time_lc=None,
                 tol_root_lc=None,
                 save_fig=None,

                 **rhs_kwargs):
            
        var_names = copy.deepcopy(var_names)
        pardict = copy.deepcopy(pardict)

        if lc_prominence is not None:
            self.lc_prominence = lc_prominence

        self.mode = mode
        
        if save_fig is not None:
            self.save_fig = save_fig
        
        # if forcing, period is assumed known.
        self.forcing = False
        if not(forcing_fn is None):
            self.forcing_fn = forcing_fn
            self.forcing = True

            self.syms = [symbols('f'+str(idx))]
            self.dsyms = [symbols('df'+str(idx))]

        self.idx = idx

        self.pars_for_fname = pars_for_fname

        self.init_is_precise = init_is_precise

        if max_time_lc is not None:
            self.max_time_lc = max_time_lc

        if tol_root_lc is not None:
            self.tol_root_lc = tol_root_lc

        self.max_iter = max_iter
        self.rtol = rtol
        self.atol = atol
        self.rel_tol = rel_tol

        self.rtol_lc = rtol_lc
        self.atol_lc = atol_lc
        
        self.g_jac_eps = g_jac_eps
        self.z_jac_eps = z_jac_eps
        self.i_jac_eps = i_jac_eps

        if g_forward is not None:
            self.g_forward = g_forward

        if z_forward is not None:
            self.z_forward = z_forward

        if i_forward is not None:
            self.i_forward = i_forward

        self.i_bad_dx = i_bad_dx

        if trunc_order is not None:
            self.trunc_order = trunc_order
        
        self.miter = self.trunc_order+1

        self.log_level = log_level
        self.log_file = log_file

        if self.log_level == 'DEBUG':
            self.log_level = logging.DEBUG
        elif self.log_level == 'INFO':
            self.log_level = logging.INFO
        elif self.log_level == 'WARNING':
            self.log_level = logging.WARNING
        elif self.log_level == 'ERROR':
            self.log_level = logging.ERROR
        elif self.log_level == 'CRITICAL':
            self.log_level = logging.CRITICAL

        FORMAT = '%(asctime)s %(message)s'
        logging.basicConfig(filename=self.log_file,level=self.log_level,
                            format=FORMAT)

        for i in range(len(var_names)):
            var_names[i] += str(self.idx)
        self.var_names = var_names
        self.dim = len(self.var_names)
        
        keys = list(pardict.keys())
        for key in keys:
            pardict[key+str(idx)] = pardict.pop(key)
        
        self.pardict = pardict
        self.rhs = rhs
        self.coupling = coupling

        self.het_pars = het_pars
        self.het_sym = {}

        het_pars_new = []

        for key in het_pars:
            self.het_sym[key] = symbols(key)
            # needed to make all names consistent after appending idx

        # create eps_i symbolic
        self.het_sym['eps'+str(idx)] = symbols('eps'+str(idx))
        self.init = init

        if TN is not None:
            self.TN = TN

        if dir_root is not None:
            self.dir_root = dir_root

        if recompute_list is not None:
            self.recompute_list = recompute_list
        
        self.method = method
        self.rule_par = {}
        self.pardict_sym = {}
        for (prop, value) in pardict.items():

            parname = prop
            symvar = symbols(parname)

            # define replacement rule for parameters
            # i.e. parname (sympy) to parname_val (float/int)
            self.rule_par.update({prop:value})
            self.pardict_sym.update({prop:symvar})

        if self.forcing:
            self.pardict_sym.update({'omf':symbols('omf')})
            
        assert(not(model_name is None))
        self.om_fix_key = 'om_fix'+str(idx)
        #print('omfixkey',self.om_fix_key,'pardict',pardict)
        #assert(self.om_fix_key in pardict)
        assert(type(self.om_fix_key) is str)


        self.model_name = model_name

        ########### create directories
        if not(os.path.isdir(self.dir_root)):
            os.mkdir(self.dir_root)

        self.dir1 = self.dir_root+'/'+self.model_name+'/'

        if not(os.path.isdir(self.dir1)):
            os.mkdir(self.dir1)

        #print('mkdir',self.dir1)

        ########## define dicts
        self.lc = {};self.g = {}
        self.z = {};self.i = {}
        self.G = {};self.K = {}
        self.p = {};self.h = {}

        self.t = symbols('t',real=True)

        self.eye = np.identity(self.dim)
        #self.psi, self.eps, self.kappa = sym.symbols('psi eps kappa')

        self.eps = self.het_sym['eps'+str(idx)]#sym.symbols('eps'+str(idx))
        self.psi = sym.symbols('psi'+str(idx))
        self.kappa = sym.symbols('kappa'+str(idx))

        self.syms = []; self.dsyms = []
        for j,name in enumerate(var_names):
            self.syms.append(symbols(name))
            self.dsyms.append(symbols('d'+name))

        #self.x_vec = sym.Matrix(self.syms)
        #self.dx_vec = sym.Matrix(self.dsyms)
        #### load fnames
        fnames.load_fnames_response(self,model_pars=self.pars_for_fname)

        ########## calculate stuff
        recompute = ('lc' in self.recompute_list) or ('all' in self.recompute_list)
        self.load_lc(recompute=recompute)

        slib.generate_expansions(self)
        slib.load_coupling_expansions(self)
        #print('loaded coupling expansions')
        
        # make rhs callable
        self.rhs_sym = rhs(0,self.syms,self.pardict_sym,option='sym',idx=self.idx,**rhs_kwargs)
        slib.load_jac_sym(self) # callable jac
        slib.load_jac_sym_general_fast(self)
        
        # get monodromy matrix
        recompute = ('m' in self.recompute_list) or ('all' in self.recompute_list)
        self.load_monodromy(recompute=recompute)

        # get heterogeneous terms for g, floquet e. fun.
        recompute = ('g_sym' in self.recompute_list) or ('all' in self.recompute_list)
        self.load_g_sym(recompute=recompute)

        # get g
        self.load_g()

        # get het. terms for z and i
        self.load_het_sym()

        # get iPRC, iIRC.
        self.load_z()
        self.load_i()
            
    def load_lc(self,recompute = False):
        """
        LC = limit cycle. This function either loads or computes the LC solution. On first run, it will compute the limit cycle using Newton's method (see `generate_lc()`). It uses the peak of the first variable of the limit cycle as zero phase and the initial condition of the limit cycle. In generate_lc(), the limit cycle is saved as a data array of shape `(self.TN+1,len(self.var_names))`, i.e., the final time point at $T=2\pi$ is included. So care must be taken to ensure that the final point is excluded in Fourier transforms.

        The period (`T`) is normalized so that $T=2\pi$.
        
        Force recompute if 'lc' is in `self.recompute_list`.

        Parameters
        ----------
        recompute : bool
            If True, recompute LC from scratch. Otherwise load from storage.

        Returns
        -------
        None. However, several variables are created:

        * `self.lc`: dict with the following keys (let `key` a the variable name  from `self.var_names`):
          * `'imp_'+key`: the sympy implemented interpolation of the LC.
          * `'lam_'+key`: the lambdified interpolation of the LC.
          * `'t_old'`: non-normalized time array for limit cycle.
          * `'dat'`: limit cycle data points. Shape `(self.TN,len(self.var_names))` Excluding $T=2\pi$. 
        * `self.tlc`: Array of limit cycle time points (length of `self.TN`).
        * `self.T`: normalized period
        * `self.T_old`: non-normalized period
        * `self.tlc,self.dtlc`: LC time array and time step WITHOUT the endpoint
        * `self.tlc_endpt,self.dtlc_endpt`: LC time array and time step WITH the endpoint
        * `self.init`: initial condition of limit cycle
        * `self.lc_vec`: lambdified limit cycle vector. if called, e.g., `self.lc_vec(10)`, a numpy array will be returned of the LC values at time 10.
        * `self.rule_lc_local`: a replacement rule dictionary. replace each sympy variable with an LC implemeneted function. In the code, the replacement dictionary is constructed as follows:
        
        ```python
        for j,key in enumerate(self.var_names):
            self.rule_lc_local[self.syms[j]] = self.lc['imp_'+key](self.t)
        ```
        
        where `self.syms` is an array of sympy versions of the model variables, `self.t` is the symbolic time variable.
        
        """

        self.lc['dat'] = []
                
        for key in self.var_names:
            self.lc['imp_'+key] = []
            self.lc['lam_'+key] = []

        file_dne = not(os.path.isfile(self.lc_fname))

        if recompute or file_dne:
        
            #print('* Computing LC data...')
            y,t = self.generate_lc(max_time_lc=self.max_time_lc,tol_root_lc=self.tol_root_lc)

            nr,nc = np.shape(y)
            z = np.zeros([nr,nc+1])
            z[:,0] = t
            z[:,1:] = y

            np.savetxt(self.lc_fname,z)

        else:
            #print('* Loading LC data...')
            z = np.loadtxt(self.lc_fname)

        self.T = z[-1,0]

        #print('self.T',self.T)

        self.tlc,self.dtlc = np.linspace(0,self.T,self.TN,retstep=True,
                                         endpoint=False)
        self.tlc_endpt,self.dtlc_endpt = np.linspace(0,self.T,self.TN+1,
                                                     retstep=True,
                                                     endpoint=True)

        # tlc_enpt has the exact same points as tlc but with the
        # final point being the endpoint.
        # use tlc_endpt for integratin, tlc for interpolation.
        
        self.lc['t'] = self.tlc
        
        self.lc['dat'] = z[:,1:]
        self.init = z[0,1:]

        #logging.debug('* LC period = '+str(self.T))
                    
        # Make LC data callable from inside sympy
        imp_lc = sym.zeros(self.dim)
        for i,key in enumerate(self.var_names):

            fn = interp1d(0,self.T,self.dtlc,self.lc['dat'][:-1,i],p=True,k=5)
            
            #imp = imp_fn('lc'+key+'_'+str(i),self.fmod(fn))
            imp = imp_fn('lc'+key+'_'+str(i),fn)
            
            self.lc['imp_'+key] = imp #imp_fn(key,fn)
            self.lc['lam_'+key] = fn
            
            imp_lc[i] = self.lc['imp_'+key](self.t)
            #lam_list.append(self.lc['lam_'+key])
            
        self.lc_vec = lambdify(self.t,imp_lc,modules='numpy')
        #self.lc_vec = lam_vec(lam_list)

        if self.save_fig:
            self.save_temp_figure(z[:,1:],0,'LC')

        self.rule_lc_local = {}
        for j,key in enumerate(self.var_names):
            self.rule_lc_local[self.syms[j]] = self.lc['imp_'+key](self.t)

    def generate_lc(self,max_time_lc=5000,method='LSODA',tol_root_lc=1e-13):
        """
        generate limit cycle data for system
        system: dict. 
        """

        eps = np.zeros(self.dim) + 1e-4
        epstime = 1e-4
        dy = np.zeros(self.dim+1) + 10

        pardict = self.pardict

        if not(self.init_is_precise):
            #print("Assuming imprecise limit cycle initial condition. Integrating for {} time units to converge near the limit cycle.".format(max_time_lc))
            #print("If you know that the initial condition is sufficiently close to the limit cycle such that you know the solution will converge to the desired limit cycle, then set keyword arg init_is_precise to True to skip this integration.")
            #print("Note that no matter the choice of imprecise and precise LC initial conditions are used in Newton's method")
            init = self.init
            T_init = self.init[-1]

            sol = solve_ivp(self.rhs,[0,max_time_lc],
                            self.init[:-1],
                            args=(pardict,'value',self.idx),
                            method=method,
                            dense_output=True,
                            rtol=self.rtol_lc,atol=self.atol_lc)
            
            if self.save_fig:
                if not(os.path.exists('figs_temp')):
                    os.makedirs('figs_temp')
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(sol.t,sol.y.T[:,0])
                ax.set_xlim(sol.t[-1]-(sol.t[-1]/10),sol.t[-1])
                plt.savefig('figs_temp/plot_limit_cycle_long.png')
                #plt.show()
            
            T_init,res1 = get_period(sol,prominence=self.lc_prominence)
            yinit = sol.sol(res1)
            init = np.append(yinit,T_init)
            

        else:
            init = self.init

        counter = 0
        while np.linalg.norm(dy) > tol_root_lc and\
              counter < self.max_iter:
        
            J = np.zeros((self.dim+1,self.dim+1))
            t = np.linspace(0,init[-1],self.TN)

            for p in range(self.dim):
                pertp = np.zeros(self.dim)
                pertm = np.zeros(self.dim)

                pertp[p] = eps[p]
                pertm[p] = -eps[p]

                initp = init[:-1] + pertp
                initm = init[:-1] + pertm

                # get error in position estimate
                solp = solve_ivp(self.rhs,[0,t[-1]],initp,
                                 method=self.method,
                                 rtol=1e-13,atol=1e-13,
                                 args=(pardict,'value',self.idx))

                solm = solve_ivp(self.rhs,[0,t[-1]],initm,
                                 method=self.method,
                                 rtol=1e-13,atol=1e-13,
                                 args=(pardict,'value',self.idx))

                yp = solp.y.T
                ym = solm.y.T

                J[:-1,p] = (yp[-1,:]-ym[-1,:])/(2*eps[p])

            J[:-1,:-1] = J[:-1,:-1] - np.eye(self.dim)

            tp = np.linspace(0,init[-1]+epstime,self.TN)
            tm = np.linspace(0,init[-1]-epstime,self.TN)

            # get error in time estimate
            solp = solve_ivp(self.rhs,[0,tp[-1]],initp,
                             method=self.method,
                             rtol=1e-13,atol=1e-13,
                             args=(pardict,'value',self.idx),
                             t_eval=tp)

            solm = solve_ivp(self.rhs,[0,tm[-1]],initm,
                             method=self.method,
                             rtol=1e-13,atol=1e-13,
                             args=(pardict,'value',self.idx),
                             t_eval=tm)

            yp = solp.y.T
            ym = solm.y.T

            J[:-1,-1] = (yp[-1,:]-ym[-1,:])/(2*epstime)

            J[-1,:] = np.append(self.rhs(0,init[:-1],pardict,
                                         'value',self.idx),0)

            sol = solve_ivp(self.rhs,[0,init[-1]],init[:-1],
                            method=self.method,
                            rtol=1e-13,atol=1e-13,
                            args=(pardict,'value',self.idx),
                            t_eval=t)

            y_final = sol.y.T[-1,:]
            b = np.append(init[:-1]-y_final,0)

            dy = np.linalg.solve(J,b)
            init += dy

            if False:
                print('dy=',dy)
                fig,axs = plt.subplots(len(self.var_names),1)
                for k in range(len(self.var_names)):
                    axs[k].plot(sol.t,sol.y[k],label='unp')
                    axs[k].plot(solm.t,solm.y[k],label='pm')
                    axs[k].plot(solp.t,solp.y[k],label='pp')
                    axs[k].legend()
                
                if not(os.path.isdir('figs_temp/iters')):
                    os.mkdir('figs_temp/iters')
                
                plt.savefig('figs_temp/iters/lc_'+str(counter)+'.png')


            to_disp = (counter,np.linalg.norm(dy))
            

            str1 = 'iter={}, LC rel. err ={:.2e}     '
            end = '            \r'
            printed = str1.format(counter,np.linalg.norm(dy))
            print(printed,end=end)
            counter += 1

        print("") # force newline
        # find index of peak voltage and initialize.
        peak_idx = np.argmax(sol.y.T[:,0])
        init = sol.y.T[peak_idx,:]

        # run finalized limit cycle solution
        sol = solve_ivp(self.rhs,[0,sol.t[-1]],sol.y.T[peak_idx,:],
                        method=self.method,
                        t_eval=np.linspace(0,sol.t[-1],self.TN+1),
                        rtol=1e-13,atol=1e-13,
                        args=(pardict,'value',self.idx))

        return sol.y.T,sol.t
    
    
    def load_monodromy(self,recompute=False):
        
        file_dne = not os.path.isfile(self.m_fname)
        
        
        if recompute or file_dne:
            #print('* Computing monodromy (augmented)...')
            #logging.info('* Computing monodromy (augmented)...')
            
            solx = solve_ivp(lambda t,x: self.rhs(t,x,self.pardict,'value',self.idx),
                     [0, self.T], self.lc['dat'][0],
                     method=self.method, rtol=1e-10, atol=1e-10)

            
            x0 = self.lc['dat'][0]
            dx = solx.y[:,-1] - x0
            v0 = self.rhs(0.0, x0, self.pardict, 'value', self.idx)
            dT = np.dot(dx, v0) / np.dot(v0, v0)
            #print("dT:", dT)

            y0 = np.asarray(self.lc['dat'][0], dtype=float)  # shape (n,)
            n = y0.size
            T = float(self.T) - dT

            Phi0 = np.eye(n)
            Phi0_flat = Phi0.reshape(n * n, order='F')

            Y0 = np.concatenate([y0, Phi0_flat])

            # tolerances
            scale_x = np.maximum(1.0, np.abs(y0))
            atol_x = 1e-12 * scale_x
            atol_phi = 1e-12 * np.ones(n * n)   
            atol = np.concatenate([atol_x, atol_phi])
            rtol = 1e-12

            # augmented RHS
            def aug_rhs(t, Y):
                x = Y[:n]
                Phi = Y[n:].reshape((n, n), order='F')

                fx = self.rhs(t, x, self.pardict, 'value', self.idx)
                J = self.jac(t, x) 

                dPhi = (J @ Phi).reshape(n * n, order='F')
                return np.concatenate([fx, dPhi])

            start = time.time()
            sol = solve_ivp(
                aug_rhs,
                [0.0, T],
                Y0,
                method=self.method,
                rtol=rtol,
                atol=atol,
                dense_output=False,   # OK; solver returns state at final time
                max_step = (T/2)/50
            )
            end = time.time()
            #print('mon eval time', end - start)

            if not sol.success:
                raise RuntimeError(f"Monodromy integration failed: {sol.message}")

            YT = sol.y[:, -1]
            xT = YT[:n]
            M = YT[n:].reshape((n, n), order='F')

            self.sol = sol.y.T
            self.M = M
            np.savetxt(self.m_fname, self.M)

        else:
            #print('* Loading monodromy...')
            #logging.info('* Loading monodromy...')
            self.M = np.loadtxt(self.m_fname)

        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.M)

        self.min_lam_idx = np.argsort(self.eigenvalues)[-2]
        self.lam = self.eigenvalues[self.min_lam_idx]

        self.kappa_val = np.log(self.lam) / self.T
        if np.abs(self.kappa_val.imag) <= 1e-10:
            self.kappa_val = np.real(self.kappa_val)

        einv = np.linalg.inv(self.eigenvectors)
        idx = np.argsort(np.abs(self.eigenvalues - 1))[0]

        self.g1_init = self.eigenvectors[:, self.min_lam_idx]
        self.z0_init = einv[idx, :]
        self.i0_init = einv[self.min_lam_idx, :]

        if np.abs(self.g1_init.imag).sum() < 1e-10:
            self.g1_init = self.g1_init.real

        if np.abs(self.z0_init.imag).sum() < 1e-10:
            self.z0_init = self.z0_init.real

        if np.abs(self.i0_init.imag).sum() < 1e-10:
            self.i0_init = self.i0_init.real

        #print('* Floquet Exponent kappa =', self.kappa_val)


    def monodromy(self,t,z):
        """
        calculate right-hand side of system
        
        $\dot \Phi = J\Phi, \Phi(0)=I$,        
        where $\Phi$ is a matrix solution
        
        jaclc is the jacobian evaluated along the limit cycle
        """
        
        jac = self.jaclc(t)
        
        n = int(np.sqrt(len(z)))
        z = np.reshape(z,(n,n), order='F')        
        dy = np.dot(jac,z)
        
        return np.reshape(dy,n*n, order='F')
    
    def load_g_sym(self,recompute=False):
        # load het. functions h if they exist. otherwise generate.
        #self.rule_g0 = {sym.Indexed('gx',0):s(0),sym.Indexed('gy',0):s(0)}
        
        # create dict of gv0=0,gh0=0,etc for substitution later.
        self.rule_g0 = {sym.Indexed('g'+name,0):
                        sympify(0) for name in self.var_names}

        for key in self.var_names:
            self.g['sym_'+key] = []
        
        # check that files exist
        val = 0
        for key in self.var_names:
            val += not(lib.files_exist(self.g['sym_fnames_'+key]))
        
        if val != 0:
            files_dne = True
        else:
            files_dne = False        
        
        
        if recompute or files_dne:
            #print('* Computing g symbolic...')
            #logging.info('* Computing g symbolic...')
            
            # create symbolic derivative
            sym_collected = slib.generate_g_sym(self)
            
            for i in range(self.miter):
                for key in self.var_names:
                    expr = sym_collected[key].coeff(self.psi,i)
        
                    self.g['sym_'+key].append(expr)
                    dill.dump(self.g['sym_'+key][i],
                              open(self.g['sym_fnames_'+key][i],'wb'),
                              recurse=True)
                    
        else:
            #print('* Loading g symbolic...')
            #logging.info('* Loading g symbolic...')
            for key in self.var_names:
                self.g['sym_'+key] = lib.load_dill(self.g['sym_fnames_'+key])


    def load_g(self):
        """
        load all Floquet eigenfunctions g or recompute
        """
        
        self.g['dat'] = []
        
        for key in self.var_names:
            self.g['imp_'+key] = []
            self.g['lam_'+key] = []

        #print('* Computing g...')        
        #logging.info('* Computing g...')
        for i in range(self.miter):

            fname = self.g['dat_fnames'][i]
            
            file_dne = not(os.path.isfile(fname))
            if 'g'+str(i) in self.recompute_list or file_dne or\
               'g' in self.recompute_list:
                
                het_vec = self.interp_lam(i,self.g,fn_type='g')
                
                data = self.generate_g(i,het_vec)
                np.savetxt(self.g['dat_fnames'][i],data)
                
            else:
                data = np.loadtxt(fname)

            self.g['dat'].append(data)
            if self.save_fig:
                self.save_temp_figure(data,i,'g')
            
            for j,key in enumerate(self.var_names):
                
                fn = interp1d(self.tlc[0],self.T,
                              self.dtlc,data[:-1,j],p=True,k=3)
                imp = imp_fn('g'+key+'_'+str(i),fn)
                
                self.g['imp_'+key].append(imp)
                self.g['lam_'+key].append(fn)                
        
        # replacement rules.        
        self.rule_g_local = {}  # g function

        t = self.t
        for key in self.var_names:
            for k in range(self.miter):
                
                fn_loc = sym.Indexed('g'+key,k)
                d_loc = {fn_loc:self.g['imp_'+key][k](t)}
                self.rule_g_local.update(d_loc) # local

    
    def generate_g(self,k,het_vec):
        """
        generate Floquet eigenfunctions g
        
        uses Newtons method
        """
        
        if type(self.g_forward) is bool:
            backwards = not(self.g_forward)
        elif type(self.g_forward) is list:
            backwards = not(self.g_forward[k])
        else:
            raise ValueError('g_forward must be bool or list, not',
                             type(self.g_forward))
            
        
        if type(self.g_jac_eps) is float:
            eps = self.g_jac_eps
        elif type(self.g_jac_eps) is list:
            eps= self.g_jac_eps[k]
        else:
            raise ValueError('g_jac_eps must be float or list or floats, not',
                             type(self.g_jac_eps))
        
        # load kth expansion of g for k >= 0
        if k == 0:
            # g0 is 0. do this to keep indexing simple.
            return np.zeros((self.TN+1,len(self.var_names)))
        
        if k == 1:
            # pick correct normalization
            init = copy.deepcopy(self.g1_init)
            eps = 1e-7
        else:
            init = np.zeros(self.dim)
            eps = 1e-7
            init = lib.run_newton2(self,self._dg,init,k,het_vec,
                                  max_iter=self.max_iter,eps=eps,
                                  rel_tol=self.rel_tol,rel_err=10,
                                  alpha=1,backwards=backwards)
        
        # get full solution
        if backwards:
            tlc = -self.tlc_endpt
            
        else:
            tlc = self.tlc_endpt

        sol = solve_ivp(self._dg,[0,tlc[-1]],
                        init,args=(k,het_vec),
                        t_eval=tlc,
                        method=self.method,
                        rtol=self.rtol,atol=self.atol,
                        dense_output=True)

        if backwards:
            gu = sol.y.T[::-1,:]
            
        else:
            gu = sol.y.T
        return gu

    def load_z(self):
        """
        load all PRCs z or recompute
        """
        
        self.z['dat'] = []
        
        for key in self.var_names:
            self.z['imp_'+key] = []
            self.z['lam_'+key] = []
            self.z['avg_'+key] = []

        #print('* Computing z...')
        #logging.info('* Computing z...')
        for i in range(self.miter):
            
            fname = self.z['dat_fnames'][i]
            file_dne = not(os.path.isfile(fname))
            if 'z'+str(i) in self.recompute_list or file_dne or\
               'z' in self.recompute_list:
                
                het_vec = self.interp_lam(i,self.z,fn_type='z')
                data = self.generate_z(i,het_vec)
                np.savetxt(self.z['dat_fnames'][i],data)
                
            else:
                data = np.loadtxt(fname)

            self.z['dat'].append(data)
            if self.save_fig:
                self.save_temp_figure(data,i,'z')
            
            for j,key in enumerate(self.var_names):

                fn = interp1d(self.tlc[0],self.T,
                              self.dtlc,data[:-1,j],p=True,k=5)
                imp = imp_fn('z'+key+'_'+str(i),fn)
                
                self.z['imp_'+key].append(imp)
                self.z['lam_'+key].append(fn)
                self.z['avg_'+key].append(np.mean(data[:,j]))
        
        self.rule_z_local = {}

        # messy but keeps global and local indices clear
        t = self.t
        for key in self.var_names:
            for k in range(self.miter):
                fn = sym.Indexed('z'+key,k)
                d = {fn:self.z['imp_'+key][k](t)}
                self.rule_z_local.update(d) # global

        
    def generate_z(self,k,het_vec):
        if type(self.z_forward) is bool:
            backwards = not(self.z_forward)
        elif type(self.z_forward) is list:
            backwards = not(self.z_forward[k])
        else:
            raise ValueError('z_forward must be bool or list, not',
                             type(self.z_forward))
            
        if type(self.z_jac_eps) is float:
            eps = self.z_jac_eps
        elif type(self.z_jac_eps) is list:
            eps= self.z_jac_eps[k]
        else:
            raise ValueError('z_jac_eps must be bool or list, not',
                             type(self.z_jac_eps))
        
        if k == 0:
            init = copy.deepcopy(self.z0_init)
            #init = [-1.389, -1.077, 9.645, 0]
        else:
            
            init = np.zeros(self.dim)
            
            init = lib.run_newton2(self,self._dz,init,k,het_vec,
                                  max_iter=self.max_iter,eps=eps,alpha=1,
                                  rel_tol=self.rel_tol,rel_err=10,
                                  backwards=backwards)
        
        if backwards:
            tlc = -self.tlc_endpt
            
        else:
            tlc = self.tlc_endpt
            
        sol = solve_ivp(self._dz,[0,tlc[-1]],
                        init,args=(k,het_vec),
                        method=self.method,
                        t_eval=tlc,
                        rtol=self.rtol,atol=self.atol,
                        dense_output=True)
        
        if backwards:
            zu = sol.y.T[::-1,:]
            
        else:
            zu = sol.y.T
        
        if k == 0:
            # normalize
            dlc = self.rhs(0,self.lc_vec(0)[0],self.pardict,
                           'value',self.idx)
            zu = zu/(np.dot(dlc,zu[0,:]))
            
        
        return zu

    def load_i(self):
        """
        load all IRCs i or recomptue
        """
        
        self.i['dat'] = []
        
        for key in self.var_names:
            self.i['imp_'+key] = []
            self.i['lam_'+key] = []

        #print('* Computing i...')
        #logging.info('* Computing i...')
        for i in range(self.miter):

            fname = self.i['dat_fnames'][i]
            file_dne = not(os.path.isfile(fname))
            
            if 'i'+str(i) in self.recompute_list or file_dne or\
               'i' in self.recompute_list:

                het_vec = self.interp_lam(i,self.i,fn_type='i')
                data = self.generate_i(i,het_vec)
                np.savetxt(fname,data)
                
            else:
                data = np.loadtxt(fname)

            self.i['dat'].append(data)

            if self.save_fig:
                self.save_temp_figure(data,i,'i')
            
            for j,key in enumerate(self.var_names):

                fn = interp1d(self.tlc[0],self.T,
                              self.dtlc,data[:-1,j],p=True,k=5)
                imp = imp_fn('i'+key+'_'+str(i),fn)
                
                self.i['imp_'+key].append(imp)
                self.i['lam_'+key].append(fn)
        
        # coupling
        # messy but keeps global and local indices clear

        self.rule_i_local = {}
        t = self.t
        
        for key in self.var_names:
            for k in range(self.miter):                
                fn = sym.Indexed('i'+key,k)
                d = {fn:self.i['imp_'+key][k](t)}
                self.rule_i_local.update(d) # global
        
    
    def generate_i(self,k,het_vec):
        """
        i0 equation is stable in forwards time
        i1, i2, etc equations are stable in backwards time.

        """
        
        if type(self.i_forward) is bool:
            backwards = not(self.i_forward)
        elif type(self.i_forward) is list:
            backwards = not(self.i_forward[k])
        else:
            raise ValueError('i_forward must be bool or list, not',
                             type(self.i_forward))
        
        if type(self.i_bad_dx) is bool:
            exception = self.i_bad_dx
        elif type(self.i_bad_dx) is list:
            exception = self.i_bad_dx[k]
        else:
            raise ValueError('i_bad_dx must be bool or list, not',
                             type(self.i_bad_dx))
            
        if type(self.i_jac_eps) is float:
            eps = self.i_jac_eps
        elif type(self.i_jac_eps) is list:
            eps= self.i_jac_eps[k]
        else:
            raise ValueError('i_jac_eps must be bool or list, not',
                             type(self.i_jac_eps))
        
        if k == 0:
            init = copy.deepcopy(self.i0_init)
        else:
            
            init = np.zeros(self.dim)
            init = lib.run_newton2(self,self._di,init,k,het_vec,
                                   max_iter=self.max_iter,rel_tol=self.rel_tol,
                                   eps=eps,alpha=1,backwards=backwards,
                                   exception=exception)

        
        if backwards:
            tlc = -self.tlc_endpt
            
        else:
            tlc = self.tlc_endpt
        
        sol = solve_ivp(self._di,[0,tlc[-1]],init,
                        args=(k,het_vec),
                        t_eval=tlc,method=self.method,
                        rtol=self.rtol,atol=self.atol,
                        dense_output=True)
    
        if backwards:
            iu = sol.y.T[::-1,:]
            
        else:
            iu = sol.y.T
                
        if k == 0:
            # normalize. classic weak coupling theory normalization
            c = np.dot(self.g1_init,iu[0,:])
            iu /= c
            
            logging.debug('norm const i0'+str(c))
    
        if k == 1:  # normalize
        
            # kill off nonzero v
            #if np.sum(self.g['dat'][1][:,-1]) < 1e-20:
            #    iu[:,-1] = 0
            
            # see Wilson 2020 PRE for normalization formula.
            lc0 = []
            g10 = []
            z00 = []
            i00 = []
        
            for varname in self.var_names:
                key = 'lam_'+varname
                lc0.append(self.lc[key](0))
                g10.append(self.g[key][1](0))
                z00.append(self.z[key][0](0))
                i00.append(self.i[key][0](0))
                
            F = self.rhs(0,lc0,self.pardict,'value',self.idx)
            g1 = np.array(g10)
            z0 = np.array(z00)
            i0 = np.array(i00)
            
            J = self.jaclc(0)
            i1 = iu[0,:]
            
            ijg = np.dot(i0,np.dot(J,g1))
            be = (self.kappa_val - ijg - np.dot(i1,F))/(np.dot(z0,F))
            
            init = iu[0,:] + be*z0

            sol = solve_ivp(self._di,[0,tlc[-1]],init,
                            args=(k,het_vec),
                            t_eval=tlc,
                            method=self.method,
                            rtol=self.rtol,atol=self.atol,
                            dense_output=True)


            #iu = sol.y.T[::-1,:]
            if backwards:
                iu = sol.y.T[::-1,:]
                
            else:
                iu = sol.y.T

            #logging.debug('norm const i1='+str(be))
            #print('norm const i1='+str(be),'iu0',iu[0,:],'z0',z0)
            
        return iu

    def load_het_sym(self):
        # load het. for z and i if they exist. otherwise generate.
        for key in self.var_names:
            self.z['sym_'+key] = []
            self.i['sym_'+key] = []
        
        # check that files exist
        val = 0
        for key in self.var_names:
            val += not(lib.files_exist(self.z['sym_fnames_'+key]))
            val += not(lib.files_exist(self.i['sym_fnames_'+key]))
        
        val += not(lib.files_exist([self.A_fname]))
        
        if val != 0:
            files_dne = True
        else:
            files_dne = False
            
        
        if 'het' in self.recompute_list or files_dne:
            #print('* Computing heterogeneous terms...')
            #logging.info('* Computing heterogeneous terms...')
            sym_collected = self.generate_het_sym()
            
            for i in range(self.miter):
                for key in self.var_names:
                    
                    expr = sym_collected[key].coeff(self.psi,i)
                    
                    self.z['sym_'+key].append(expr)
                    self.i['sym_'+key].append(expr)
                    
                    dill.dump(self.z['sym_'+key][i],
                              open(self.z['sym_fnames_'+key][i],'wb'),
                              recurse=True)
                    dill.dump(self.i['sym_'+key][i],
                              open(self.i['sym_fnames_'+key][i],'wb'),
                              recurse=True)
                
            # save matrix of a_i
            dill.dump(self.A,open(self.A_fname,'wb'),recurse=True)
            
        else:
            #print('* Loading heterogeneous terms...')
            #logging.info('* Loading heterogeneous terms...')
            self.A, = lib.load_dill([self.A_fname])
            for key in self.var_names:
                self.z['sym_'+key] = lib.load_dill(self.z['sym_fnames_'+key])
                self.i['sym_'+key] = lib.load_dill(self.i['sym_fnames_'+key])

                
    def generate_het_sym(self):
        """
        Generate heterogeneous terms for integrating the Z_i and I_i terms.

        Returns
        -------
        None.

        """
        
        # get the general expression for h in z before plugging in g,z.        
        # column vectors ax ay for use in matrix A = [ax ay]
        self.a = {k: sym.zeros(self.dim,1) for k in self.var_names}
     
        for i in range(1,self.miter):
            logging.debug('z,i het sym deriv order='+str(i))
            p1 = lib.kProd(i,self.dx_vec)
            p2 = kp(p1,sym.eye(self.dim))

            #print('iter',i)
            #print('p1,p2',p1,p2)

            for j,key in enumerate(self.var_names):
                logging.debug('\t var='+str(key))
                d1 = lib.vec(lib.df(self.rhs_sym[j],self.x_vec,i+1))
                d1 = sym.powsimp(d1)
                self.a[key] += (1/math.factorial(i))*(p2*d1)
                #print('given i, var name',key,', term is',self.a[key])
                
        self.A = sym.zeros(self.dim,self.dim)
        
        for i,key in enumerate(self.var_names):            
            self.A[:,i] = self.a[key]

        #print('A is',self.A)
        
        het = self.A*self.z['vec_psi']

        # expand all terms
        out = {}        
        rule = {**self.rule_g0,**self.rule_d2g}
        
        rule_trunc = {}
        for k in range(self.miter,self.miter+200):
            rule_trunc.update({self.psi**k:0})
            #print('truncating order',k)
            
        for i,key in enumerate(self.var_names):
            logging.info('z,i het sym subs key='+str(key))

            #print('subsituting key',key)
            tmp = het[i]

            tmp = sym.expand(tmp)
            #print('after zero expand')
            
            tmp = tmp.subs(rule)

            #print('after first sub')
            
            tmp = sym.expand(tmp,basic=True,deep=True,
                             power_base=False,power_exp=False,
                             mul=True,log=False,
                             multinomial=True)

            #print('after first expand')
            
            tmp = tmp.subs(rule_trunc)
            #print('after first subs')
            
            tmp = sym.collect(tmp,self.psi).subs(rule_trunc)

            #print('after first collect')
            tmp = sym.expand(tmp).subs(rule_trunc)

            #print('after second expand')
            tmp = sym.collect(tmp,self.psi).subs(rule_trunc)

            #print('after final power simplification')
            tmp = sym.powsimp(tmp)
            out[key] = tmp
            
        return out

    
    def _dg(self,t,z,order,het_vec):
        """
        g functon rhs with ith het. term
        
        z: position
        t: time
        jaclc: jacobian on lc
        het: heterogeneous terms
        
        order determines the Taylor expansion term
        """

        jac = self.jaclc(t)*(order > 0)
        hom = np.dot(jac-order*self.kappa_val*self.eye,z.T)
        out = hom + het_vec(t).T
    
        return out

    def _dz(self,t,z,order,het_vec):
        """
        g functon rhs with ith het. term
        
        z: position
        t: time
        jaclc: jacobian on lc
        het: heterogeneous terms
        
        order determines the Taylor expansion term
        """
        
        hom = np.dot(self.jaclc(t).T+order*self.kappa_val*self.eye,z.T)
        out = -hom - het_vec(t).T
        
        return out

    def _di(self,t,z,order,het_vec):
        """
        g functon rhs with ith het. term
        
        z: position
        t: time
        jacLC: jacobian on LC
        het: heterogeneous terms
        
        order determines the Taylor expansion term
        """

        hom = np.dot(self.jaclc(t).T+(order-1)*self.kappa_val*self.eye,z)
        out = -hom - het_vec(t).T
        
        return out
    
    def interp_lam(self,k,fn_dict,fn_type='z'):
        """
        it is too slow to call individual interpolated functions
        in the symbolic heterogeneous terms.
        since the heterogeneous terms only depend on t, just make
        and interpolated version and use that instead so only 1 function
        is called for the het. terms per iteration in numerical iteration.
        """
        
        # lambdify heterogeneous terms for use in integration
        # create lambdified heterogeneous term and interpolate
        # load kth expansion of g for k >= 1
        
        # z and i use the same heterogeneous terms
        # so they are indistinguishable in this function.
        if fn_type == 'z' or fn_type == 'i':
            fn_type = 'z'
        
        rule = {}

        for key in self.var_names:
            tmp = {sym.Indexed(fn_type+key,i):fn_dict['imp_'+key][i](self.t)
                   for i in range(k)}
            
            rule.update(tmp)

        
        rule = {**rule,**self.rule_lc_local,**self.rule_par}
        if fn_type == 'z':
            rule.update({**self.rule_g_local})            
            
        het_imp = sym.zeros(self.dim,1)
        for i,key in enumerate(self.var_names):
            sym_fn = fn_dict['sym_'+key][k].subs(rule)

            lam = lambdify(self.t,sym_fn,modules='numpy')

            # evaluate
            if fn_type == 'g' and (k == 0 or k == 1):
                y = np.zeros(self.TN+1)
            elif fn_type == 'z' and k == 0:
                y = np.zeros(self.TN+1)
            elif fn_type == 'i' and k == 0:
                y = np.zeros(self.TN+1)
            elif sym_fn == 0:
                y= np.zeros(self.TN+1)
            else:
                y = lam(self.tlc_endpt)
            
            #fn = interp1d(0,self.T,self.dtlc,y[:-1],p=True,k=5)
            fn = interp1d(self.tlc[0],self.T,self.dtlc,y[:-1],p=True,k=5)
            imp = imp_fn(key,fn)
            
            # save as implemented fn
            het_imp[i] = imp(self.t)
            
        het_vec = lambdify(self.t,het_imp,modules='numpy')
            
        return het_vec
    
    def fmod(self,fn):
        """
        fn has mod built-in
        
        input function-like. usually interp1d object
        
        needed to keep lambda input variable unique to fn.
        
        otherwise lambda will use the same input variable for 
        all lambda functions.
        """
        return lambda x=self.t: fn(x)
    
    def save_temp_figure(self,data,k,fn='plot',path_loc='figs_temp/'):
        """
        data should be (TN,dim)
        """

        if (not os.path.exists(path_loc)):
            os.makedirs(path_loc)
        
        fig, axs = plt.subplots(nrows=self.dim,ncols=1,figsize=(8,self.dim*2))
        
        for j,ax in enumerate(axs):
            key = self.var_names[j]
            ax.plot(self.tlc_endpt,data[:,j],label=key)
            ax.legend()

        print(fn+str(k)+' ini'+str(data[0,:]))
        print(fn+str(k)+' fin'+str(data[-1,:]))
              
        logging.info(fn+str(k)+' ini'+str(data[0,:]))
        logging.info(fn+str(k)+' fin'+str(data[-1,:]))
        axs[0].set_title(fn+str(k))
        plt.tight_layout()
        plt.savefig(path_loc+fn+str(k)+'_'+self.model_name+'.png')
        plt.close()
        

    def __call__(self,th,option='add'):
        if option == 'add':
            #om = self.pardict['om'+str(self.idx)]
            #a = self.pardict['amp'+str(self.idx)]
            return self.coupling(self.forcing_fn(th),self.pardict)
        
