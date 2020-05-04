#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:28:30 2019

@author: drsmith
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

plt.close('all')

#c[0] * (c[1] - c[4] + (1-c[1]+c[4])*(1-np.abs(x)**c[2])**c[3] + \
#                     c[4]*(1-np.exp(-(x**2)/(c[5]**2))))
# [10.22178262  0.0267643   1.84138753  1.62809221  0.3         0.31130153]

x = sp.symbols('x')
c = [sp.symbols('c{}'.format(i)) for i in range(6)]
str_expr = "c0*(c1-c4+(1-c1+c4)*(1-x**c2)**c3 + c4*(1-exp(-x**2/c5**2)))"
expr = sp.sympify(str_expr)

coef = np.array([10.22178262,  0.0267643,   1.84138753,  1.62809221,  0.3,0.31130153])

expr2 = expr
for i in range(6):
    expr2 = expr2.subs(c[i],coef[i])

xval = np.linspace(0,1.1,10)
xscaled = xval/1.14491

f = sp.lambdify(x, expr2)
fval = f(xscaled)
plt.figure()
plt.subplot(1,2,1)
plt.plot(xval, fval, '-o')

dexpr = sp.diff(expr, x)
dexpr2 = dexpr
for i in range(6):
    dexpr2 = dexpr2.subs(c[i],coef[i])

df = sp.lambdify(x, dexpr2)
dfval = df(xscaled)
plt.subplot(1,2,2)
plt.plot(xval, dfval, '-o')