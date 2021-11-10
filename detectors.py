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
    
    bw_ref = 1e6  # reference bandwidth [Hz]
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
        print(f'  QE = {self.qe*1e2:.1f} %')
        print(f'  Resp. = {self.responsivity:.2f} A/W')
        print(f'  Dark current (total or vol.) = {self.darkcurrent_ref*1e9:.3f} nA')
        if self.darkcurrent_surface_ref:
            print(f'  Dark current (surf.) = {self.darkcurrent_surface_ref*1e9:.3f} nA')
        
    def photocurrent(self, p_inc=p_ref):
        # p_inc in nW
        return self.gain * self.responsivity * p_inc * 1e-9
    
    def junction_cap_with_vb(self, vb=vb_ref):
        # junction cap scales with sqrt(V_b,ref/V_b)
        return self.junction_cap_ref * np.sqrt(self.vb_ref/vb)
    
    def shot_noise_power_density(self, p_inc=p_ref, t=t_ref):
        photocurrent = self.photocurrent(p_inc=p_inc)
        # decrease by half for every 10C below ref. temp.
        darkcurrent_temp_factor = 2**((t-self.t_ref)/10)
        # dark current or volumetric dark current depending on context
        darkcurrent = self.darkcurrent_ref * darkcurrent_temp_factor
        # surface dark current, if applicable
        darkcurrent_surface = self.darkcurrent_surface_ref * darkcurrent_temp_factor
        return 2*pc.e * ((photocurrent + darkcurrent) * self.gain**2 * self.noise_factor + \
            darkcurrent_surface)

    def shot_noise_current(self, p_inc=p_ref, t=t_ref, bw=bw_ref):
        return np.sqrt(self.shot_noise_power_density(p_inc=p_inc, t=t) * bw)
    
    def SNR(self, p_inc=p_ref, t=t_ref, bw=bw_ref):
        photocurrent = self.photocurrent(p_inc=p_inc)
        shot_noise_current = self.shot_noise_current(p_inc=p_inc, t=t, bw=bw)
        return photocurrent / shot_noise_current
    

class PinDiode(_Diode):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gain = 1
        self.noise_factor = 1
    

class ApdDiode(_Diode):
    
    def __init__(self, 
                 gain=0, 
                 noise_factor=0,
                 noise_figure=0,
                 noise_current=0, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert(gain>0)
        assert(noise_factor>0 or noise_figure>0 or noise_current>0)
        self.gain = gain
        if noise_factor:
            # noise factor specified
            self.noise_factor = noise_factor
        elif noise_figure:
            print(f'  Noise figure = {noise_figure:.3f}')
            # calc noise factor from noise figure
            self.noise_factor = 10**(noise_figure/10)
        elif noise_current:
            print(f'  Noise current = {noise_current:.3g}')
            # calc noise factor from noise current
            sigma_sq_ideal = self.shot_noise_power_density(p_inc=0)/self.bw_ref
            self.noise_factor = noise_current**2 / sigma_sq_ideal
        assert(self.noise_factor>1)
        print(f'  Noise factor = {self.noise_factor:.3f}')


diodes = {
    'PDF_C164' : PinDiode(
        name='API PDB-C164 disc. PIN',
        responsivity=0.43,  # A/W @ 650 nm
        darkcurrent=1e-9,  # typ 1 nA, max 10 nA @ T=20C
        junction_cap=7e-12,  # typ 7 pF @ Vr=10V
    ),
    'S13620' : PinDiode(
        name='Hamm S13620 8x8 PIN',
        responsivity=0.47,  # A/W @ 650 nm
        darkcurrent=0.3e-9,  # typ 10 pA, max 300 pA @ Vr=10mV, T=20C
        junction_cap=60e-12,  # typ 60 pF @ Vr=0V
    ),
    'S8550' : ApdDiode(
        name='Hamm S8550 4x8 APD',
        qe=0.85,  # QE @ 650 nm
        gain=50,  # gain @ Vr=330 V
        darkcurrent=1e-9,  # typ 1 nA, max 10 nA @ Vr=330V, M=50
        junction_cap=9e-12,  # typ 9 pF @ Vr=330V, M=50
        noise_figure=0.2,
    ),
}