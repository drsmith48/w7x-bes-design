#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 07:56:39 2020

@author: drsmith
"""

import numpy as np
import matplotlib.pyplot as plt


def spotsize(fiber_radius=0.2,  # fiber radius in mm
             fiber_na=0.333,  # fiber numerical aperture
             lens_radius = 3.0,  # lens radius in cm
             ):
    fiber_radius = np.squeeze(np.array(fiber_radius))
    spot_distance = 200.0  # cm

    print('Inputs')
    print('  Lens radius = {:.2f} cm'.format(lens_radius))
    print('  Spot distance = {:.1f} cm'.format(spot_distance))
    print('  Fiber NA = {:.3f}'.format(fiber_na))
    print('  Fiber radius in mm:')
    print(fiber_radius)
    
    t = spot_distance
    a = lens_radius
    assert(a>0 and t>0)
    
    theta = np.arctan2(a,t)
    print('  Theta = {:.3g} rad  (axial marginal ray)'.format(theta))
    assert(theta>0)
    
    hp = -fiber_radius/10
    # assert(hp<0)
    
    thetap = -np.arcsin(fiber_na)
    print('  Thetap = {:.4g} rad  (set by fiber NA)'.format(thetap))
    assert(thetap<0)
    
    mag = theta / thetap
    print('  Magnification = {:.4f}  (axial marginal ray)'.format(mag))
    print('  1/Mag = {:.4f}'.format(1/mag))
    assert(np.abs(mag)<1 and mag<0)
    
    spot_radius = h = hp / mag
    # print('  Spot radius in cm (set by eq 1 of edge chief ray):')
    # print(spot_radius)

    beta = -np.arctan2(h,t)
    # print('  Beta in rad  (object edge chief ray):')
    # print(beta)
    # assert(beta<0)
    
    tps = np.array([4,8])
    
    for tp in tps:
        print('Image distance = {:.1f} cm'.format(tp))
        ap = -tp * np.tan(thetap)
        print('  Exit pupil radius = {:.3f} cm'.format(ap))
        betap = np.arctan2(hp, tp)
        # print('  Betap in rad:')
        # print(betap)
        # assert(betap<0)
        k = (beta/mag-betap) / h
        print('  EFL in cm  (eq 2 for object edge chief ray):')
        # print(k)
        print(1/k)

    return spot_radius
    
def plot():
    plt.close('all')
    fiber_diameters = np.array([0.05,0.1,0.2,0.3,0.4,0.6,0.8,1.0])
    
    # NA scan
    na_values = [0.5, 0.333, 0.2]
    plt.figure(figsize=[9,3.5])
    plt.subplot(1,2,1)
    for na in na_values:
        lens_radius = 3
        spot_diameters = 2*spotsize(fiber_diameters/2, fiber_na=na, lens_radius=lens_radius)
        plt.plot(fiber_diameters, 
                 spot_diameters, 
                 marker='x', 
                 label='Fiber NA = {:.2f}'.format(na))
        plt.annotate('lens radius = {:.1f} cm'.format(lens_radius), 
                     [fiber_diameters.max()/2,0])
    
    # lens scan
    lens_values = [2,3,5]
    plt.subplot(1,2,2)
    for lens_radius in lens_values:
        na = 0.333
        spot_diameters = 2*spotsize(fiber_diameters/2, lens_radius=lens_radius, fiber_na=na)
        plt.plot(fiber_diameters, 
                 spot_diameters, 
                 marker='x', 
                 label='Lens radius = {:.1f} cm'.format(lens_radius))
        plt.annotate('fiber NA = {:.2f}'.format(na), 
                     [fiber_diameters.max()/2,0])
    
    for ax in plt.gcf().axes:
        plt.sca(ax)
        plt.legend()
        plt.xlabel('Fiber diameter (mm)')
        plt.ylabel('Spot diameter (cm)')
        for mag in [10,20,30]:
            plt.plot([0,fiber_diameters.max()], 
                     [0,mag*fiber_diameters.max()/10], 
                     linestyle='--', 
                     color='k', 
                     linewidth=0.4)
            plt.annotate('1/M={:d}'.format(mag), 
                         [fiber_diameters.max(),mag*fiber_diameters.max()/10], 
                         ha='right')
        plt.tight_layout()
    

if __name__=='__main__':
    # spotsize(fiber_radius=[0.05,0.1,0.2,0.5,1.0])
    plot()