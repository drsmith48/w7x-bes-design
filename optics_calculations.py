#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 07:56:39 2020

@author: drsmith
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


class FiberImage(object):
    
    def __init__(self,
                 fiber_diameter=0.2,  # fiber diameter in mm
                 fiber_na=0.333,  # fiber numerical aperture
                 lens_diameter=6.0,  # lens diameter in cm
                 fiber_distance = 3.0,  # image distance in cm
                 spot_distance = 200.0,  # object distance in cm
                 ):
        assert(fiber_na>0 and fiber_diameter>0 and lens_diameter>0 and fiber_distance>0)

        t = spot_distance
        a = lens_diameter/2  # entrance pupil radius
        assert(a>0 and t>0)
            
        theta = np.arctan2(a,t)
        # print('  Theta = {:.3g} rad  (axial marginal ray)'.format(theta))
        assert(theta>0)
        
        hp = -(fiber_diameter/2)/10
        assert(hp<0)
        
        thetap = -np.arcsin(fiber_na)
        # print('  Thetap = {:.3g} rad  (set by fiber NA)'.format(thetap))
        assert(thetap<0)
        
        mag = theta / thetap
        # print('  Magnification = {:.3g}  (axial marginal ray)'.format(mag))
        # print('  1/Mag = {:.3g}'.format(1/mag))
        assert(np.abs(mag)<1 and mag<0)
        
        h = hp / mag
        # print('  Spot radius = {:.3g} cm  (set by eq 1 of edge chief ray)'.format(h))
        assert(h>0)
        
        beta = -np.arctan2(h,t)
        # print('  Beta = {:.3g} rad  (object edge chief ray)'.format(beta))
        assert(beta<0)
        
        tp = fiber_distance
        assert(tp>0)
        
        ap = -tp * np.tan(thetap)  # exit pupil radius
        # print('  Exit pupil radius = {:.3g} cm'.format(ap))
        assert(ap>0)
        
        betap = np.arctan2(hp, tp)
        # print('  Betap = {:.3g} rad'.format(betap))
        assert(betap<0)
    
        K = (beta/mag-betap) / h
        assert(K>0)
        # print('  Refractive power = {:.4g} 1/cm'.format(K))
        
        efl = 1/K
        # print('  Effective focal length = {:.4g} cm'.format(efl))
        
        C = -K
        A = mag - C*tp
        D = 1/mag - C*t
        B = (A*D-1)/C
        
        
        self.spot_diameter = 2*h
        self.h = h  # spot radius
        self.mag = mag
        self.ap = ap  # exit pupil radius
        self.K = K
        self.efl = efl
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        
        
def plot(save=False):
    plt.close('all')
    fiber_diameters = np.array([0.05,0.1,0.2,0.3,0.4,0.6,0.8,1.0])

    na_values = [0.5, 0.333, 0.2]
    lens_values = [4,6,8]
    
    plt.figure(figsize=[11,3.25])

    # NA scan
    for iplot, lens_diameter in enumerate(lens_values):
        plt.subplot(1,len(lens_values),iplot+1)
        for na in na_values:
            images = [FiberImage(fiber_diameter=fiber_diameter,
                                 fiber_na=na,
                                 lens_diameter=lens_diameter)
                      for fiber_diameter in fiber_diameters]
            spot_diameters = [image.spot_diameter for image in images]
            efl = images[0].efl
            plt.plot(fiber_diameters, 
                     spot_diameters, 
                     marker='x', 
                     label='Fiber NA = {:.2g} (EFL={:.2g} cm)'.format(na,efl))
            plt.title('Lens diameter = {:.2g} cm'.format(lens_diameter))
            plt.legend(fontsize='small')
            plt.xlabel('Fiber diameter (mm)')
            plt.ylabel('Spot diameter (cm)')
            for mag in [10,20,30,40]:
                plt.plot([0,fiber_diameters.max()], 
                         [0,mag*fiber_diameters.max()/10], 
                         linestyle='--', 
                         color='k', 
                         linewidth=0.4)
                plt.annotate('1/M={:d}'.format(mag), 
                             [fiber_diameters.max(),mag*fiber_diameters.max()/10], 
                             ha='right')
    plt.tight_layout()
    if save:
        fname = Path('optics-calculations.pdf')
        plt.savefig(fname.as_posix(), format='pdf', transparent=True)
    

if __name__=='__main__':
    # spotsize()
    # plot2()
    # a=FiberImage()
    plot()