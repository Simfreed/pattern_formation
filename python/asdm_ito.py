#!/usr/bin/env python

import numpy as np
import scipy.integrate
import sdeint as sdeint
import argparse 
import pickle as pkl
import os

parser = argparse.ArgumentParser()

parser.add_argument("--nsamp",         type=int, help="# of samps", default=20)
parser.add_argument("--diff_rats",     type=float, help="ratio of diffusion coefficients (da/ds)", nargs = '+', default=[1])
parser.add_argument("--seed",          type=int, help="random number seed", default=None)
parser.add_argument("--save_start",    type=int, help="first timestep to save", default = -1)
parser.add_argument("--save_stop",     type=int, help="last timestep to save", default = None)
parser.add_argument("--save_incr",     type=int, help="save increment", default = 1)
parser.add_argument("--dt",            type=float, help="simulation timestep", default=5e-4)
parser.add_argument("--tmax",          type=float, help="simulation endtime",  default=50)
parser.add_argument("--noise",            type=float, help="noise scale",      default=10)
parser.add_argument("--outdir",        type=str,   help = "directory for output",   default='/Users/simonfreedman/data/droso/turing/noise_scales') 

args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)
np.random.seed(args.seed)

with open('{0}/args.pkl'.format(args.outdir), 'wb') as f:
    pkl.dump(args, f)

def dc_dt(
    c,
    t,
    x,
    derivs_0,
    derivs_L,
    diff_coeff_fun,
    diff_coeff_params,
    rxn_fun,
    rxn_params,
    n_species,
    h,
):
    """
    Time derivative of concentrations in an R-D system 
    for constant flux BCs.
    
    Parameters
    ----------
    c : ndarray, shape (n_species * n_gridpoints)
        The concentration of the chemical species interleaved in a
        a NumPy array.  The interleaving allows us to take advantage 
        of the banded structure of the Jacobian when using the 
        Hindmarsh algorithm for integrating in time.
    t : float
        Time.
    derivs_0 : ndarray, shape (n_species)
        derivs_0[i] is the value of the diffusive flux,
        D dc_i/dx, at x = 0, the leftmost boundary of the domain of x.
    derivs_L : ndarray, shape (n_species)
        derivs_0[i] is the value of the diffusive flux,
        D dc_i/dx, at x = L, the rightmost boundary of the domain of x.
    diff_coeff_fun : function
        Function of the form diff_coeff_fun(c_tuple, t, x, *diff_coeff_params).
        Returns an tuple where entry i is a NumPy array containing
        the diffusion coefficient of species i at the grid points.
        c_tuple[i] is a NumPy array containing the concentrations of
        species i at the grid poitns.
    diff_coeff_params : arbitrary
        Tuple of parameters to be passed into diff_coeff_fun.
    rxn_fun : function
        Function of the form rxn_fun(c_tuple, t, *rxn_params).
        Returns an tuple where entry i is a NumPy array containing
        the net rate of production of species i by chemical reaction
        at the grid points.  c_tuple[i] is a NumPy array containing 
        the concentrations of species i at the grid poitns.
    rxn_params : arbitrary
        Tuple of parameters to be passed into rxn_fun.
    n_species : int
        Number of chemical species.
    h : float
        Grid spacing (assumed to be constant)
        
    Returns
    -------
    dc_dt : ndarray, shape (n_species * n_gridpoints)
        The time derivatives of the concentrations of the chemical
        species at the grid points interleaved in a NumPy array.
    """
    # Tuple of concentrations
    c_tuple = tuple([c[i::n_species] for i in range(n_species)])

    # Compute diffusion coefficients
    D_tuple = diff_coeff_fun(c_tuple, t, x, *diff_coeff_params)

    # Compute reaction terms
    rxn_tuple = rxn_fun(c_tuple, t, *rxn_params)

    # Return array
    conc_deriv = np.empty_like(c)

    # Convenient array for storing concentrations
    da_dt = np.empty(len(c_tuple[0]))

    # Useful to have square of grid spacing around
    h2 = h ** 2

    # Compute diffusion terms (central differencing w/ Neumann BCs)
    for i in range(n_species):
        # View of concentrations and diffusion coeff. for convenience
        a = np.copy(c_tuple[i])
        D = np.copy(D_tuple[i])

        # Time derivative at left boundary
        da_dt[0] = D[0] / h2 * 2 * (a[1] - a[0] - h * derivs_0[i])

        # First derivatives of D and a
        dD_dx = (D[2:] - D[:-2]) / (2 * h)
        da_dx = (a[2:] - a[:-2]) / (2 * h)

        # Time derivative for middle grid points
        da_dt[1:-1] = D[1:-1] * np.diff(a, 2) / h2 + dD_dx * da_dx

        # Time derivative at left boundary
        da_dt[-1] = D[-1] / h2 * 2 * (a[-2] - a[-1] + h * derivs_L[i])

        # Store in output array with reaction terms
        conc_deriv[i::n_species] = da_dt + rxn_tuple[i]

    return conc_deriv



def constant_diff_coeffs(c_tuple, t, x, diff_coeffs):
    n = len(c_tuple[0])
    return tuple([diff_coeffs[i] * np.ones(n) for i in range(len(c_tuple))])


def asdm_rxn(as_tuple, t, mu):
    """
    Reaction expression for activator-substrate depletion model.

    Returns the rate of production of activator and substrate, respectively.

    r_a = a**2 * s - a
    r_s = mu * (1 - a**2 * s)
    """
    # Unpack concentrations
    a, s = as_tuple

    # Compute and return reaction rates
    a2s = a ** 2 * s
    return (a2s - a, mu * (1.0 - a2s))

def dispersion_relation(k_vals, d, mu):
    lam = np.empty_like(k_vals)
    for i, k in enumerate(k_vals):
        A = np.array([[1-d*k**2,          1],
                      [-2*mu,    -mu - k**2]])
        lam[i] = np.linalg.eigvals(A).real.max()
        
    return lam

def dispersion_eig(k_vals, d, mu):
    lam = np.zeros((k_vals.shape[0],2))
    evec = np.zeros((k_vals.shape[0],2,2))
    for i, k in enumerate(k_vals):
        A = np.array([[1-d*k**2,          1],
                      [-2*mu,    -mu - k**2]])
        w,v = np.linalg.eig(A)
        lam[i] = w.real
        evec[i] = np.abs(v)
        
    return lam, evec


def rd_solve_ito(
    c_0_tuple,
    dt,
    tmax,
    noise_scale = 1,
    L=1,
    derivs_0=0,
    derivs_L=0,
    diff_coeff_fun=None,
    diff_coeff_params=(),
    rxn_fun=None,
    rxn_params=(),
    rtol=1.49012e-8,
    atol=1.49012e-8,
):
    """
    Parameters
    ----------
    c_0_tuple : tuple
        c_0_tuple[i] is a NumPy array of length n_gridpoints with the 
        initial concentrations of chemical species i at the grid points.
    t : ndarray
        An array of time points for which the solution is desired.
    L : float
        Total length of the x-domain.
    derivs_0 : ndarray, shape (n_species)
        derivs_0[i] is the value of dc_i/dx at x = 0.
    derivs_L : ndarray, shape (n_species)
        derivs_L[i] is the value of dc_i/dx at x = L, the rightmost
        boundary of the domain of x.
    diff_coeff_fun : function
        Function of the form diff_coeff_fun(c_tuple, x, t, *diff_coeff_params).
        Returns an tuple where entry i is a NumPy array containing
        the diffusion coefficient of species i at the grid points.
        c_tuple[i] is a NumPy array containing the concentrations of
        species i at the grid poitns.
    diff_coeff_params : arbitrary
        Tuple of parameters to be passed into diff_coeff_fun.
    rxn_fun : function
        Function of the form rxn_fun(c_tuple, t, *rxn_params).
        Returns an tuple where entry i is a NumPy array containing
        the net rate of production of species i by chemical reaction
        at the grid points.  c_tuple[i] is a NumPy array containing 
        the concentrations of species i at the grid poitns.
    rxn_params : arbitrary
        Tuple of parameters to be passed into rxn_fun.
    rtol : float
        Relative tolerance for solver.  Default os odeint's default.
    atol : float
        Absolute tolerance for solver.  Default os odeint's default.
        
    Returns
    -------
    c_tuple : tuple
        c_tuple[i] is a NumPy array of shape (len(t), n_gridpoints)
        with the initial concentrations of chemical species i at 
        the grid points over time.
        
    Notes
    -----
    .. When intergrating for long times near a steady state, you
       may need to lower the absolute tolerance (atol) because the
       solution does not change much over time and it may be difficult
       for the solver to maintain tight tolerances.
    """
    # Number of grid points
    n_gridpoints = len(c_0_tuple[0])

    # Number of chemical species
    n_species = len(c_0_tuple)

    # Grid spacing
    h = L / (n_gridpoints - 1)

    # Grid points
    x = np.linspace(0, L, n_gridpoints)

    # Set up boundary conditions
    if np.isscalar(derivs_0):
        derivs_0 = np.array(n_species * [derivs_0])
    if np.isscalar(derivs_L):
        derivs_L = np.array(n_species * [derivs_L])

    # Set up initial condition
    c0 = np.empty(n_species * n_gridpoints)
    for i in range(n_species):
        c0[i::n_species] = c_0_tuple[i]

    # Solve using odeint, taking advantage of banded structure
    dc_dt_simple = lambda c, t: dc_dt(c, t, x, derivs_0, derivs_L, diff_coeff_fun, diff_coeff_params,
                                      rxn_fun,rxn_params,n_species,h)
    noise_func   = lambda c,t: noise_scale*np.sqrt(h)*np.sqrt(dt)*np.ones((n_species * n_gridpoints, 1))
    c = sdeint.itoEuler(
        dc_dt_simple,
        noise_func,
        c0,
        np.arange(0,tmax+dt, dt),
    )

    return np.array([c[:, i::n_species] for i in range(n_species)])


# Set up steady state (using 500 grid points)
a_0 = np.ones(500)
s_0 = np.ones(500)


# Physical length of system
L = 20.0

# x-coordinates for plotting
x = np.linspace(0, L, len(a_0))

# Reaction parameter (must be a tuple of params, even though only 1 for ASDM)
rxn_params = (1.5,)

for d in args.diff_rats:
    print('diff coeff ratio = {0}'.format(d))
    for i in range(args.nsamp):
        print('\tsample = {0}'.format(i))
        
        # Make a small perturbation to a_0
        a_0 += 0.01 * np.random.rand(len(a_0))

        conc = np.array(rd_solve_ito(
            (a_0, s_0,),
            args.dt,
            args.tmax,
            args.noise,
            L=L,
            derivs_0=0,
            derivs_L=0,
            diff_coeff_fun=constant_diff_coeffs,
            diff_coeff_params=((d, 1),),
            rxn_fun=asdm_rxn,
            rxn_params=rxn_params,
            ))[:,args.save_start:args.save_stop:args.save_incr]
        
        np.save('{0}/diff{1:.3f}_samp{2:d}.npy'.format(args.outdir, d, i), conc)
