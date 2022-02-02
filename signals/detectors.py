#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import scipy.constants as pc


# All inputs and outputs are in SI units:  W, A, V, F, Ohms, Hz, A^2/Hz, etc.
# All noise terms are current noise power density (CNPD) with units A^2/Hz
# For diodes, noise terms are output CNPD
# For TIA, noise terms are input-referenced CNPD


class PinDiode(object):

    def __init__(self, 
                 name='', 
                 responsivity=0,  # responsivity [V/A]
                 qe=0,  # quantum efficiancy [0-1]
                 darkcurrent_ref=10e-9,  # dark (leakage) current (or volumetric if APD) [A]
                 darkcurrent_surface_ref=0,  # dark (leakage) surface current for APD [A]
                 junction_cap_ref=10e-15,  # diode junction cap. [F]
                 vb_ref=10,  # reference reverse bias [V]
                 r_shunt=250e6,  # shunt resistance [Ohms]
                 t_ref=20,  # reference temp [C]
                 ):
        assert(qe>0 or responsivity>0)
        self.name = name
        self.gain = 1
        self.noise_factor = 1
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
        self.r_shunt = r_shunt
        self.junction_cap_ref = junction_cap_ref
        self.vb_ref = vb_ref
        self.t_ref = t_ref
        print(self.name)
        print(f'  QE = {self.qe*1e2:.1f} %')
        print(f'  Resp. = {self.responsivity:.2f} A/W')
        print(f'  Dark current (total or vol.) = {self.darkcurrent_ref*1e9:.1f} nA')
        if self.darkcurrent_surface_ref:
            print(f'  Dark current (surf.) = {self.darkcurrent_surface_ref*1e9:.1f} nA')
        print(f'  Junction cap = {self.junction_cap_ref*1e12:.1f} pF')
        print(f'  Shunt res. = {self.r_shunt/1e6:.0f} MOhms')
        print(f'  Ref. bias voltage = {self.vb_ref:.0f} V')
        print(f'  Ref. temp. = {self.t_ref:.0f} C')
        
    def photocurrent(self, p_inc=None):
        if p_inc is None:
            p_inc = 100e-9  # W
        photocurrent = self.gain * self.responsivity * p_inc  # A
        return photocurrent
    
    def junction_cap(self, vb=None, ideal=False):
        if vb is None:
            vb = self.vb_ref  # V
        # junction cap scales with sqrt(V_b,ref/V_b)
        junction_cap = self.junction_cap_ref * np.sqrt(self.vb_ref/vb)  # F
        if ideal:
            junction_cap *= 0
        return junction_cap
    
    def photocurrent_shot_noise_CNPD(self, p_inc=None, ideal=False):
        if p_inc is None:
            p_inc = 100
        photocurrent = self.photocurrent(p_inc=p_inc)/self.gain  # G=1 photocurrent, A
        noise_factor = self.noise_factor
        if ideal:
            noise_factor = 1
        shot_noise = 2*pc.e * photocurrent * self.gain**2 * noise_factor  # A^2/Hz
        return shot_noise

    def dark_current_shot_noise_CNPD(self, t=None, ideal=False):
        if t is None:
            t = self.t_ref  # C
        # dark current temperature factor
        # decrease by half for every 10C below ref. temp.
        darkcurrent_temp_factor = 2**((t-self.t_ref)/10)
        # dark current or volumetric dark current depending on PIN/APD context
        darkcurrent = self.darkcurrent_ref * darkcurrent_temp_factor  # A
        darkcurrent_surface = self.darkcurrent_surface_ref * darkcurrent_temp_factor  # A
        noise_factor = self.noise_factor
        if ideal:
            noise_factor = 1
        shot_noise = 2*pc.e * ( darkcurrent * self.gain**2 * noise_factor + \
            darkcurrent_surface )  # A^2/Hz
        return shot_noise

    def shunt_noise_CNPD(self, t=None, ideal=False):
        if t is None:
            t = self.t_ref  # C
        shunt_noise = 4*pc.k*(273+t) / self.r_shunt  # A^2/Hz
        if ideal:
            shunt_noise *= 0
        return shunt_noise


class ApdDiode(PinDiode):
    
    def __init__(self, 
                 gain=50, 
                 noise_factor=0,
                 noise_figure=0,
                 noise_current=0,  #  A/rt(Hz)
                 noise_index=0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert(gain>1)
        assert(noise_factor>0 or noise_figure>0 or noise_current>0 or noise_index>0)
        self.gain = gain
        print(f'  Gain = {self.gain:.0f}')
        if noise_factor:
            self.noise_factor = noise_factor
        elif noise_figure:
            print(f'  Noise figure = {noise_figure:.1f}')
            self.noise_factor = 10**(noise_figure/10)
        elif noise_current:
            print(f'  Noise current = {noise_current*1e12:.1f} pA/rt(Hz)')
            sigma_sq_ideal = self.dark_current_shot_noise_CNPD(ideal=True)
            print((noise_current*self.gain) ** 2, sigma_sq_ideal)
            self.noise_factor = (noise_current*self.gain) ** 2 / sigma_sq_ideal
        elif noise_index:
            print(f'  Noise index = {noise_index:.2f}')
            self.noise_factor = self.gain ** noise_index
        print(f'  Noise factor = {self.noise_factor:.1f}')
        assert(self.noise_factor>1)


class TIA(object):

    def __init__(self,
            r_feedback=100e6,  # reference feedback resistor [Ohms]
            jfet_enoise=0.8e-9,  # reference JFET e-nosie [V/rt(Hz)], typ. 1 nV/rt(Hz)
            jfet_gatecurrent=1e-9,  # reference JFET gate current [A], typ. 10 mA
            jfet_input_cap=10e-12,  # reference JFET input cap. [F]
            stray_cap=4e-12,   # PCB stray cap. [F]
            t_ref=20,  # ref. temp [C]
            ):
        self.r_feedback = r_feedback
        self.jfet_enoise = jfet_enoise
        self.jfet_gatecurrent = jfet_gatecurrent
        self.jfet_input_cap = jfet_input_cap
        self.stray_cap = stray_cap
        self.t_ref = t_ref

    def feedback_noise_CNPD(self, t=None, r_feedback=None, ideal=False):
        if t is None:
            t = self.t_ref  # C
        if r_feedback is None:
            r_feedback = self.r_feedback  # Ohms
        feedback_noise = 4*pc.k*(273+t) / r_feedback  # A^2/Hz
        if ideal:
            feedback_noise *= 0
        return feedback_noise

    def gate_noise_CNPD(self, ideal=False):
        gate_noise = 2*pc.e * self.jfet_gatecurrent  # A^2/Hz
        if ideal:
            gate_noise *= 0
        return gate_noise

    def enc_noise_CNPD(self, f=None, diode_junction_cap=None, ideal=False):
        assert(diode_junction_cap)
        if f is None:
            f = np.geomspace(1e3, 3e6)  # Hz
        total_cap = diode_junction_cap + self.jfet_input_cap + self.stray_cap  # F
        enc_noise = ( 2*np.pi * self.jfet_enoise * total_cap * f ) ** 2.0  # A^2/Hz
        if ideal:
            enc_noise *= 0
        return enc_noise
