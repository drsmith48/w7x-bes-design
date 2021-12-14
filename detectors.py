#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 09:45:07 2020

@author: drsmith
"""

from pathlib import Path
import numpy as np
import scipy.constants as pc


#  Terminology
#  CNPD = current noise power density [A^2/Hz]


class _Diode(object):

    #  Diode current noise sources:
    #  Shot noise from diode photocurrent and dark currents
    #  Thermal (Johnson) noise from diode shunt resistance
    
    p_ref = 10  # reference incident power [nW]
    t_ref = 20  # reference temp [C]
    
    def __init__(self, 
                 name='', 
                 responsivity=0,  # responsivity [V/A]
                 qe=0,  # quantum efficiancy [0-1]
                 darkcurrent_ref=10e-9,  # dark (leakage) current (or volumetric if APD) [A]
                 darkcurrent_surface_ref=0,  # dark (leakage) surface current for APD [A]
                 junction_cap_ref=10e-15,  # diode junction cap. [F]
                 vb_ref=10,  # reference reverse bias [V]
                 r_shunt=250e6,  # shunt resistance [Ohms]
                 ):
        assert(qe>0 or responsivity>0)
        self.name = name
        self.gain = None
        self.noise_factor = None
        if responsivity:
            # responsivity at 650 nm [A/W]
            if qe:
                print('Warning: Using input responsivity, ignoring input QE')
            self.responsivity = responsivity
            self.qe = self.responsivity * 1240/656
        elif qe:
            self.qe = qe
            self.responsivity = self.qe * 656/1240
        self.darkcurrent_ref = darkcurrent_ref  # total or volumetric, by PIN/APD context
        self.darkcurrent_surface_ref = darkcurrent_surface_ref  # 0 if n/a
        self.junction_cap_ref = junction_cap_ref
        self.vb_ref = vb_ref
        self.r_shunt = r_shunt
        print(self.name)
        print(f'  QE = {self.qe*1e2:.1f} %')
        print(f'  Resp. = {self.responsivity:.2f} A/W')
        print(f'  Dark current (total or vol.) = {self.darkcurrent_ref*1e9:.1f} nA')
        if self.darkcurrent_surface_ref:
            print(f'  Dark current (surf.) = {self.darkcurrent_surface_ref*1e9:.1f} nA')
        print(f'  Junction cap = {self.junction_cap_ref*1e12:.1f} pF')
        print(f'  Shunt res. = {self.r_shunt/1e6:.0f} MOhms')
        
    def photocurrent(self, p_inc=p_ref):
        # p_inc in nW
        photocurrent = self.gain * self.responsivity * p_inc * 1e-9  # A
        return photocurrent
    
    def junction_cap(self, vb=None, ideal=False):
        if vb is None:
            vb = self.vb_ref
        # junction cap scales with sqrt(V_b,ref/V_b)
        junction_cap = self.junction_cap_ref * np.sqrt(self.vb_ref/vb)
        if ideal:
            junction_cap *= 0
        return junction_cap
    
    def photocurrent_shot_noise_CNPD(self, p_inc=p_ref, ideal=False):
        photocurrent = self.photocurrent(p_inc=p_inc)/self.gain
        noise_factor = self.noise_factor
        if ideal:
            noise_factor = 1
        shot_noise = 2*pc.e * photocurrent * self.gain**2 * noise_factor  # A^2/Hz
        return shot_noise

    def dark_current_shot_noise_CNPD(self, t=t_ref, ideal=False):
        # decrease by half for every 10C below ref. temp.
        darkcurrent_temp_factor = 2**((t-self.t_ref)/10)
        # dark current or volumetric dark current depending on PIN/APD context
        darkcurrent = self.darkcurrent_ref * darkcurrent_temp_factor
        darkcurrent_surface = self.darkcurrent_surface_ref * darkcurrent_temp_factor
        noise_factor = self.noise_factor
        if ideal:
            darkcurrent *= 0
            darkcurrent_surface *= 0
            noise_factor = 1
        shot_noise = 2*pc.e * ( darkcurrent * self.gain**2 * noise_factor + \
            darkcurrent_surface )  # A^2/Hz
        return shot_noise

    def shunt_noise_CNPD(self, t=t_ref, ideal=False):
        shunt_noise = 4*pc.k*(273+t) / self.r_shunt
        if ideal:
            shunt_noise *= 0
        return shunt_noise


class PinDiode(_Diode):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gain = 1
        self.noise_factor = 1
    

class ApdDiode(_Diode):
    
    def __init__(self, 
                 gain=50, 
                 noise_factor=0,
                 noise_figure=0,
                 noise_current=0, 
                 noise_index=0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert(gain>=1)
        assert(noise_factor>0 or noise_figure>0 or noise_current>0 or noise_index>0)
        self.gain = gain
        print(f'  Gain = {self.gain:.0f}')
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
            sigma_sq_ideal = self.shot_noise_CNPD(p_inc=0)/self.bw_ref
            self.noise_factor = noise_current**2 / sigma_sq_ideal
        elif noise_index:
            # noise factor from noise index
            print(f'  Noise index = {noise_index:.3g}')
            self.noise_factor = self.gain ** noise_index
        assert(self.noise_factor>1)
        print(f'  Noise factor = {self.noise_factor:.3f}')


class TIA(object):

    #  TIA current noise sources:
    #  Shot noise from JFET gate current
    #  Thermal (Johnson) noise from TIA feedback resistor
    #  JFET channel voltage noise and "ENC" current noise
    
    t_ref = _Diode.t_ref  # reference temp [C]
    f_ref = np.logspace(1e3, 1e6)

    def __init__(self,
            r_feedback=100e6,  # reference feedback resistor [Ohms]
            jfet_enoise=0.8e-9,  # reference JFET e-nosie [V/root(Hz)], typ. 1 nV/rt(Hz)
            jfet_gatecurrent=1e-9,  # reference JFET gate current [A], typ. 10 mA
            jfet_input_cap=10e-12,  # reference JFET input cap. [F]
            stray_cap=4e-12,   # PCB stray capacitance [F]
            ):
        self.r_feedback = r_feedback
        self.jfet_enoise = jfet_enoise
        self.jfet_gatecurrent = jfet_gatecurrent
        self.jfet_input_cap = jfet_input_cap
        self.stray_cap = stray_cap

    def feedback_noise_CNPD(self, t=t_ref, r_feedback=None, ideal=False):
        if not r_feedback:
            r_feedback = self.r_feedback
        feedback_noise = 4*pc.k*(273+t) / r_feedback
        if ideal:
            feedback_noise *= 0
        return feedback_noise

    def gate_noise_CNPD(self, ideal=False):
        gate_noise = 2*pc.e * self.jfet_gatecurrent
        if ideal:
            gate_noise *= 0
        return gate_noise

    def enc_noise_CNPD(self, f=f_ref, diode_junction_cap=0, ideal=False):
        total_cap = diode_junction_cap + self.jfet_input_cap + self.stray_cap
        enc_noise = ( 2*np.pi * self.jfet_enoise * total_cap * f ) ** 2.0
        if ideal:
            enc_noise *= 0
        return enc_noise


# diodes = {
#     'API_C164_PIN' : PinDiode(
#         name='API PDB-C164 disc. PIN',
#         responsivity=0.43,  # A/W @ 650 nm
#         darkcurrent_ref=1e-9,  # typ 1 nA, max 10 nA @ T=20C
#         junction_cap_ref=7e-12,  # typ 7 pF @ Vr=10V
#         vb_ref=10,  # reference bias voltage for junction cap.
#         r_shunt=500e6,  # shunt resistance
#     ),
#     'Hama_S13620_PIN' : PinDiode(
#         name='Hama. S13620 8x8 PIN',
#         responsivity=0.49,  # A/W @ 650 nm
#         darkcurrent_ref=0.3e-9,  # typ 10 pA, max 300 pA @ Vr=10mV, T=20C
#         junction_cap_ref=15e-12,  # typ 15 pF @ Vr=8.6V
#         vb_ref=8.6,
#     ),
#     'Hama_S8550_APD' : ApdDiode(
#         name='Hama. S8550 4x8 APD',
#         qe=0.85,  # QE @ 650 nm
#         gain=50,  # gain @ Vr=330 V
#         darkcurrent_ref=1e-9,  # typ 1 nA, max 10 nA @ Vr=330V, M=50
#         junction_cap_ref=9e-12,  # typ 9 pF @ Vr=330V, M=50
#         noise_figure=0.2,
#         vb_ref=330,  # reference bias for gain and junction cap.
#     ),
# }