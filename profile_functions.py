#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 13:44:40 2019

@author: drsmith
"""

import sympy as sp
from scipy.constants import physical_constants as _pc

_x = sp.symbols('x')
_c = [sp.symbols('c{}'.format(i)) for i in range(6)]
_str_expr = "c0*(c1-c4+(1-c1+c4)*(1-x**c2)**c3 + c4*(1-exp(-x**2/c5**2)))"
_expr = sp.sympify(_str_expr)

_ne_coef = [10.22178262, 0.0267643, 1.84138753, 1.62809221, 0.3, 0.31130153]
_ne = _expr.subs(_x, _x/1.14491)  # dens [10**19/m**3]
for i in range(6):
    _ne = _ne.subs(_c[i], _ne_coef[i])
_dne = _ne.diff(_x)
_islne = _dne / _ne

ne = sp.lambdify(_x, _ne)
dne = sp.lambdify(_x, _dne)
islne = sp.lambdify(_x, _islne)

_te_coef = [ 3.8076127, 0.00651745, 2.22383115, 2.85146903, -0.20842746, 0.20145577]
_te = _expr.subs(_x, _x/1.09491) # Te [keV]
for i in range(6):
    _te = _te.subs(_c[i], _te_coef[i])
_dte = _te.diff(_x)
_islte = _dte / _te

te = sp.lambdify(_x, _te)
dte = sp.lambdify(_x, _dte)
islte = sp.lambdify(_x, _islte)

_bmag = 2 # mag. field strength [T]

_rho_d = 6.461e-3/sp.sqrt(2) * sp.sqrt(_te)/_bmag  # deut. gyroradius [m] (no sqrt(2))
_omega_d = 4.791e7 * _bmag  # deut. cyclotron freq. [rad/s]

_md = _pc['deuteron mass'][0]  # deut. mass [kg]
_j_per_ev = _pc['electron volt-joule relationship'][0] # J/eV ratio
_c_s = sp.sqrt(_te*1e3*_j_per_ev/_md)  # sound speed [m/s]
_rho_s = _c_s / _omega_d  # sound gyroradius [m]

c_s = sp.lambdify(_x, _c_s)
rho_d = sp.lambdify(_x, _rho_d)
rho_s = sp.lambdify(_x, _rho_s)

if __name__=='__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    plt.close('all')
    x = np.linspace(0,1.2,20)
    for f,df,isl,fname in zip([ne,te],[dne,dte],[islne,islte],['ne','Te']):
        plt.figure(figsize=(10.5,3.25))
        plt.subplot(1,3,1)
        plt.plot(x, f(x), '-o')
        plt.xlabel('r/a')
        plt.ylabel(fname)
        plt.subplot(1,3,2)
        plt.plot(x, df(x), '-o')
        plt.xlabel('r/a')
        plt.ylabel('d('+fname+')/dx')
        plt.subplot(1,3,3)
        plt.plot(x, isl(x), '-o')
        plt.xlabel('r/a')
        plt.ylabel('1/'+fname+' * d('+fname+')/dx')
        plt.tight_layout()
    plt.figure(figsize=(10.5,3.25))
    plt.subplot(1,2,1)
    plt.plot(x, c_s(x)/1e3)
    plt.xlabel('r/a')
    plt.ylabel('trans. time = c_s/(1 m) (kHz)')
    plt.subplot(1,2,2)
    plt.plot(x, rho_d(x), label='rho_d')
    plt.plot(x, rho_s(x), label='rho_s')
    plt.xlabel('r/a')
    plt.ylabel('rho_d,s (m)')
    plt.legend()
    plt.tight_layout()
