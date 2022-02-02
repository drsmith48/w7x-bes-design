#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:29:56 2020

@author: drsmith
"""

import numpy as np
import matplotlib.pyplot as plt
import profiles
import beams
import optics_calculations
import plotfida


c2cs = np.array([1.6,1.6])  # center-to-center spot distances (radial,binorm) (cm)
grid = np.array([8,8])

rhoi = np.array([0.4,0.1])

assert(c2cs.size==grid.size)

for i,c2c in enumerate(c2cs):
    if i==0:
        print('Radial')
    else:
        print('Binormal')
    
    # k_max = 1/2 *  2pi / center-to-center
    lambda_min = 2*c2c
    k_max = 2*np.pi / lambda_min
    
    # k_min = 1/2 * 2pi / domain size
    grid_length = c2c * grid[i]
    lambda_max = 2*grid_length
    k_min = 2*np.pi / lambda_max
    
    print('  center-to-center spacing: {:.2f} cm'.format(c2c))
    print('  lambda min/max: {:.2f} {:.2f} cm'.format(lambda_min, lambda_max))
    print('  k min/max: {:.2f} {:.2f} 1/cm'.format(k_min, k_max))
    print('  k*rhoi (core) min/max: {:.2f} {:.2f}'.format(
        k_min*rhoi[0], k_max*rhoi[0]))
    print('  k*rhoi (edge) min/max: {:.2f} {:.2f}'.format(
        k_min*rhoi[1], k_max*rhoi[1]))
    


if __name__=='__main__':
    plt.close('all')
    # layout=beams.SightlineGrid(beam=beams.HeatingBeam(pini=2),
    #                             port='A21-lolo',
    #                             r_obs=5.86, 
    #                             z_obs=-0.46,
    #                             grid_shape=grid, 
    #                             c2c_normal=c2cs[0],
    #                             c2c_binormal=c2cs[1])
    # layout.plot(save=True)
    # profiles.profile_calculations(ifit=0)
    optics_calculations.plot()
    # plotfida.plotfida()
