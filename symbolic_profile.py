#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 08:13:22 2020

@author: drsmith
"""

import sympy as sp

"""
Y = a[0] * [ aa[1]-aa[4]+(1-aa[1]+aa[4])*(1-XX^aa[2])^aa[3] + \
    aa[4]*(1-exp(-XX^2/aa[5]^2)) ]
    XX - r/a
"""

sp.init_printing()

x = sp.symbols('x', nonnegative=True, finite=True)
a0,a1,a2,a3,a4,a5 = sp.symbols('a0:6', nonnegative=True, finite=True)

y = a0 * ( a1 - a4 + (1-a1-a4)*(1-x**a2)**a3 + \
           a4*(1-sp.exp(-x**2/(a5**2))) )

dydx = sp.diff(y, x)