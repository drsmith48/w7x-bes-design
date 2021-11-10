#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 09:45:07 2020

@author: drsmith
"""

from pathlib import Path
import numpy as np
import scipy.constants as pc


class _Diode(object):
    
    bw_ref = 1e6  # signal bandwidth [Hz]
    p_ref = 10  # reference power [nW]
    t_ref = 20  # reference temp [C]
    vb_ref = 10  # reference reverse bias [V]
    
    def __init__(self, 
                 name='', 
                 responsivity=0, 
                 qe=0, 
                 darkcurrent=0, 
                 darkcurrent_surface=0,
                 junction_cap=0,
                 ):
        self.name = name
        self.gain = None
        self.noise_factor = None
        if responsivity:
            # responsivity at 650 nm [A/W]
            self.responsivity = responsivity
            self.qe = self.responsivity * 1240/656
        elif qe:
            self.qe = qe
            self.responsivity = self.qe * 656/1240
        assert(self.qe>0 or self.responsivity>0)
        # dark current at V_b=10V, T=20C [nA]
        # drops by half for 20 C decrease
        # slight increase with V_b
        assert(darkcurrent>0)
        self.darkcurrent_ref = darkcurrent  # total or volumetric, by context
        self.darkcurrent_surface_ref = darkcurrent_surface  # 0 if n/a
        # junction cap at V_b=10V, T=20C [pF]
        # sharp decrease with V_b; C_j ~ 1/sqrt(V_b)
        # C_j(10V) ~ C_j(0V)/5
        assert(junction_cap>0)
        self.junction_cap_ref = junction_cap
        print(self.name)
        print(f'  QE = {self.qe*1e2:.1f}')
        print(f'  Resp. = {self.responsivity:.2f} A/W')
        
    def photocurrent(self, p_inc=p_ref):
        return self.gain * self.responsivity * p_inc * 1e-9
    
    def darkcurrent_temperature_factor(self, t=t_ref):
        return 2**((t-self.t_ref)/10)
    
    def junction_cap_with_vb(self, vb=vb_ref):
        return self.junction_cap_ref * np.sqrt(self.vb_ref)/np.sqrt(vb)
    
    def sigmasq_shot(self, p_inc=p_ref, t=t_ref):
        photocurrent = self.photocurrent(p_inc=p_inc)
        darkcurrent_factor = self.darkcurrent_temperature_factor(t=t)
        volume_current =  + photocurrent + \
            self.darkcurrent_ref * darkcurrent_factor
        darkcurrent_surface = self.darkcurrent_surface_ref * darkcurrent_factor
        return 2*pc.e * (volume_current * self.gain**2 * self.noise_factor + \
            darkcurrent_surface)
    
    def SNR(self, p_inc=p_ref, t=t_ref):
        photocurrent = self.photocurrent(p_inc=p_inc)
        sigmasq = self.sigmasq_shot(p_inc=p_inc, t=t) * self.bw_ref
        return photocurrent / np.sqrt(sigmasq)
    

class PinDiode(_Diode):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gain = 1
        self.noise_factor = 1
    

class ApdDiode(_Diode):
    
    def __init__(self, 
                 gain=0, 
                 noise_factor=0,
                 noise_figure=0,
                 noise_current=0, 
                 **kwargs):
        super().__init__(**kwargs)
        self.gain = gain
        if noise_factor:
            # noise factor specified
            self.noise_factor = noise_factor
        elif noise_figure:
            print(f'Noise figure: {noise_figure:.3g}')
            # calc noise factor from noise figure
            self.noise_factor = 10**(noise_figure/10)
        elif noise_current:
            print(f'Noise current: {noise_current:.3g}')
            # calc noise factor from noise current
            sigma_sq_ideal = self.sigmasq_shot(p_inc=0)/self.bw_ref
            self.noise_factor = noise_current**2 / sigma_sq_ideal
        else:
            raise ValueError
        assert(self.noise_factor>1)
    
    
if __name__=='__main__':
    plt.close('all')
    diodes = plot_diodes(save=False)
