#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 10:22:18 2020

@author: drsmith
"""


import numpy as np
import matplotlib.pyplot as plt

def make_filters():
    f1 = {'name':'OD6 Ultra Narrow',
          'wl':np.arange(648,652.1,0.2)}
    f1['od'] = np.empty(f1['wl'].shape)*np.nan
    f1['t'] = np.empty(f1['wl'].shape)*np.nan
    f1['od'][:14] = [7.09, 6.814, 6.52, 6.21, 5.88, 5.52, 5.134, 4.715, 
                    4.256, 3.746, 3.171, 2.51, 1.73, 0.7902]
    f1['t'][14:] = [0.829, 0.937, 0.949, 0.9565, 0.956, 0.9605, 0.964]
    
    f2 = {'name':'OD7 Ultra',
          'wl':np.arange(591,595.1,0.2)}
    f2['od'] = np.empty(f2['wl'].shape)*np.nan
    f2['t'] = np.empty(f2['wl'].shape)*np.nan
    f2['od'][:16] = [5.96, 5.72, 5.47, 5.21, 4.94, 4.65, 4.36, 4.05, 3.72, 3.37, 
                     3.01, 2.61, 2.18, 1.7, 1.18, 0.627]
    f2['t'][16:] = [0.669, 0.958, 0.966, 0.958, 0.97]
    
    
    f3 = {'name':'OD6 Ultra',
          'wl':np.arange(642,646.1,0.2)}
    f3['od'] = np.empty(f3['wl'].shape)*np.nan
    f3['t'] = np.empty(f3['wl'].shape)*np.nan
    f3['od'][:16] = [4.45, 4.25, 4.04, 3.82, 3.59, 3.35, 3.09, 2.83, 2.54, 2.24, 
                     1.92, 1.57, 1.2, 0.802, 0.424, 0.1475]
    f3['t'][16:] = [0.93, 0.972, 0.966, 0.964, 0.97]
    
    filters = [f1, f2, f3]
    
    for f in filters:
        for i in range(f['t'].size):
            if np.isnan(f['t'][i]):
                f['t'][i] = 10**(-f['od'][i])
        assert(np.all(f['t']))
        
        f['edgeind']= np.argwhere(f['t']>=0.9)[0][0] # index of smallest wl > 90% transmission
        f['edgewl'] = f['wl'][f['edgeind']]
        
    return filters


def plot_filters():
    filters = make_filters()
    plt.figure(figsize=[8.75,3.75])
    plt.subplot(1,2,1)
    for f in filters:
        plt.plot(f['wl']-f['edgewl'], f['t'], '-x', label=f['name'])
    plt.subplot(1,2,2)
    for f in filters:
        plt.semilogy(f['wl']-f['edgewl'], f['t'], '-x', label=f['name'])
    
    for ax in plt.gcf().axes:
        plt.sca(ax)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Transmission')
        plt.legend()
    plt.tight_layout()
    
    
if __name__=='__main__':
    plt.close('all')
    plot_filters()