#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 09:45:07 2020

@author: drsmith
"""

from pathlib import Path
import numpy as np
import scipy.constants as pc

#  CPD = current power density [A^2/Hz]
#  VPD = voltage power density [V^2/Hz]

class _Diode(object):

    #  Diode noise sources
    #  CPD: Shot noise from diode photocurrent and dark currents
    #  CPD: Thermal (Johnson) noise from diode shunt resistance
    
    p_ref = 10  # reference incident power [nW]
    t_ref = 20  # reference temp [C]
    
    def __init__(self, 
                 name='', 
                 responsivity=0,  # responsivity [V/A]
                 qe=0,  # quantum efficiancy [0-1]
                 darkcurrent=10e-9,  # dark (leakage) current (or volumetric if APD) [A]
                 darkcurrent_surface=0,  # dark (leakage) surface current for APD [A]
                 junction_cap=10e-15,  # diode junction cap. [F]
                 r_shunt=300e6,  # shunt resistance [Ohms]
                 vb_ref=10,  # reference reverse bias [V]
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
        self.darkcurrent_ref = darkcurrent  # total or volumetric, by PIN/APD context
        self.darkcurrent_surface_ref = darkcurrent_surface  # 0 if n/a
        # junction cap at V_b=10V, T=20C [pF]
        # sharp decrease with V_b; C_j ~ 1/sqrt(V_b)
        # C_j(10V) ~ C_j(0V)/5
        assert(junction_cap>0)
        self.junction_cap_ref = junction_cap
        self.r_shunt = r_shunt
        self.vb_ref = vb_ref
        print(self.name)
        print(f'  QE = {self.qe*1e2:.1f} %')
        print(f'  Resp. = {self.responsivity:.2f} A/W')
        print(f'  Dark current (total or vol.) = {self.darkcurrent_ref*1e9:.3f} nA')
        if self.darkcurrent_surface_ref:
            print(f'  Dark current (surf.) = {self.darkcurrent_surface_ref*1e9:.3f} nA')
        
    def photocurrent(self, p_inc=p_ref):
        # p_inc in nW
        photocurrent = self.gain * self.responsivity * p_inc * 1e-9
        return photocurrent
    
    def junction_cap(self, vb=None, ideal=False):
        if vb is None:
            vb = self.vb_ref
        # junction cap scales with sqrt(V_b,ref/V_b)
        junction_cap = self.junction_cap_ref * np.sqrt(self.vb_ref/vb)
        if ideal:
            junction_cap *= 0
        return junction_cap
    
    def shot_noise_CPD(self, p_inc=p_ref, t=t_ref, ideal=False):
        #  Shot current noise power density [A^2/Hz]
        photocurrent = self.photocurrent(p_inc=p_inc)
        # decrease by half for every 10C below ref. temp.
        darkcurrent_temp_factor = 2**((t-self.t_ref)/10)
        # dark current or volumetric dark current depending on PIN/APD context
        darkcurrent = self.darkcurrent_ref * darkcurrent_temp_factor
        # surface dark current, if applicable
        darkcurrent_surface = self.darkcurrent_surface_ref * darkcurrent_temp_factor
        noise_factor = self.noise_factor
        if ideal:
            darkcurrent *= 0
            darkcurrent_surface *= 0
            noise_factor = 1
        shot_noise = 2*pc.e * ( \
            (photocurrent + darkcurrent) * self.gain**2 * noise_factor + \
            darkcurrent_surface)
        return shot_noise

    def shunt_noise_CPD(self, t=t_ref, ideal=False):
        #  Shunt thermal current noise power density [A^2/Hz]
        shunt_noise = 4*pc.k*(273+t) / self.r_shunt
        if ideal:
            shunt_noise *= 0
        return shunt_noise

    # def shot_noise_current(self, p_inc=p_ref, t=t_ref, bw=bw_ref):
    #     shot_noise_current = np.sqrt(self.shot_noise_power_density(p_inc=p_inc, t=t) * bw)
    #     return shot_noise_current

    # def shunt_noise_current(self, r_shunt=r_shunt_ref, t=t_ref, bw=bw_ref):
    #     shunt_noise_current = np.sqrt(self.shunt_noise_power_density(r_shunt=r_shunt, t=t) * bw)
    #     return shunt_noise_current

    # def total_noise_current(self, p_inc=p_ref, t=t_ref, r_shunt=r_shunt_ref, bw=bw_ref):
    #     shot_noise_power_density = self.shot_noise_power_density(p_inc=p_inc, t=t)
    #     shunt_noise_power_density = self.shunt_noise_power_density(r_shunt=r_shunt, t=t)
    #     total_noise_power_density = shot_noise_power_density + shunt_noise_power_density
    #     total_noise_current = np.sqrt(total_noise_power_density * bw)
    #     return total_noise_current
    
    # def SNR(self, p_inc=p_ref, t=t_ref, bw=bw_ref, r_shunt=r_shunt_ref):
    #     photocurrent = self.photocurrent(p_inc=p_inc)
    #     total_noise_current = self.total_noise_current(p_inc=p_inc, t=t, bw=bw, r_shunt=r_shunt)
    #     return photocurrent / total_noise_current
    

class PinDiode(_Diode):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gain = 1
        self.noise_factor = 1
    

class ApdDiode(_Diode):
    
    def __init__(self, 
                 gain=50, 
                 noise_factor=1.05,
                 noise_figure=0,
                 noise_current=0, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert(gain>=1)
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
            sigma_sq_ideal = self.shot_noise_CPD(p_inc=0)/self.bw_ref
            self.noise_factor = noise_current**2 / sigma_sq_ideal
        assert(self.noise_factor>1)
        print(f'  Noise factor = {self.noise_factor:.3f}')


diodes = {
    'API_C164_PIN' : PinDiode(
        name='API PDB-C164 disc. PIN',
        responsivity=0.43,  # A/W @ 650 nm
        darkcurrent=1e-9,  # typ 1 nA, max 10 nA @ T=20C
        junction_cap=7e-12,  # typ 7 pF @ Vr=10V
    ),
    'Hama_S13620_PIN' : PinDiode(
        name='Hama S13620 8x8 PIN',
        responsivity=0.47,  # A/W @ 650 nm
        darkcurrent=0.3e-9,  # typ 10 pA, max 300 pA @ Vr=10mV, T=20C
        junction_cap=60e-12,  # typ 60 pF @ Vr=0V
    ),
    'Hama_S8550_APD' : ApdDiode(
        name='Hama S8550 4x8 APD',
        qe=0.85,  # QE @ 650 nm
        gain=50,  # gain @ Vr=330 V
        darkcurrent=1e-9,  # typ 1 nA, max 10 nA @ Vr=330V, M=50
        junction_cap=9e-12,  # typ 9 pF @ Vr=330V, M=50
        noise_figure=0.2,
    ),
}


class TIA(object):

    #  Noise sources:
    #  Shot noise from JFET gate current
    #  Thermal (Johnson) noise from TIA feedback resistor
    #  JFET channel voltage noise and "ENC" current noise
    
    t_ref = 20  # reference temp [C]
    p_ref = 10  # reference incident power [nW]

    # bw_ref = 1e6  # reference bandwidth [Hz]

    def __init__(self,
            diode=None,
            r_feedback=100e6,  # reference feedback resistor [Ohms]
            jfet_enoise=0.8e-9,  # reference JFET e-nosie [V/root(Hz)], typ. 1 nV/rt(Hz)
            jfet_gatecurrent=1e-9,  # reference JFET gate current [A], typ. 10 mA
            jfet_input_cap=10e-15,  # reference JFET input cap. [F]
            stray_cap=10e-15,   # PCB stray capacitance [F]
            ):
        assert(diode and isinstance(diode, _Diode))
        self.diode = diode
        self.r_feedback = r_feedback
        self.jfet_enoise = jfet_enoise
        self.jfet_gatecurrent = jfet_gatecurrent
        self.jfet_input_cap = jfet_input_cap
        self.stray_cap = stray_cap

    def feedback_noise_CPD(self, t=t_ref, r_feedback=None, ideal=False):
        if not r_feedback:
            r_feedback = self.r_feedback
        #  current noise power density [A^2/Hz]
        feedback_noise = 4*pc.k*(273+t) / r_feedback
        if ideal:
            feedback_noise *= 0
        return feedback_noise

    def gate_noise_CPD(self, jfet_gatecurrent=None, ideal=False):
        if not jfet_gatecurrent:
            jfet_gatecurrent = self.jfet_gatecurrent
        #  current noise power density [A^2/Hz]
        gate_noise = 2*pc.e * jfet_gatecurrent
        if ideal:
            gate_noise *= 0
        return gate_noise

    def enc_noise_CPD(self, vb=None, ideal=False):
        #  current noise power density [A^2/Hz]
        diode_junction_cap = self.diode.junction_cap(vb=vb, ideal=ideal)
        total_cap = diode_junction_cap + self.jfet_input_cap + self.stray_cap
        enc_noise = 0
        if ideal:
            enc_noise *= 0
        return enc_noise

