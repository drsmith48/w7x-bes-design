#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:37:49 2020

@author: drsmith
"""

import numpy as np
import scipy.constants as pc


def calc_signal(emission_radiance=1e17,  # photons/m^2/ster
                fiber_NA=0.333,
                fiber_diameter=1.33,  # mm
                spot_diameter=2,  # cm
                distance=180,  # cm
                aperture=8,  # cm
                ):
    print(f'Fiber NA = {fiber_NA:.3f}')
    print(f'Fiber diameter = {fiber_diameter:.2f} mm')
    print(f'Spot diameter = {spot_diameter:.2f} cm')
    print(f'Lens-spot distance = {distance:.1f} cm')
    print(f'Lens diameter = {aperture:.1f} cm')
    print(f'Emission radiance = {emission_radiance:.2g} ph/m^2/ster')
    
    fiber_area = np.pi * (fiber_diameter/2)**2  # mm^2
    fiber_etendue = np.pi * fiber_area * fiber_NA**2  # mm^2-ster
    print(f'Fiber etendue = {fiber_etendue:.4f} mm**2-ster')
    
    # check emitter etuendue
    NA = (aperture/2) / distance
    emitter_area = np.pi * (spot_diameter/2)**2 * 1e2  # mm^2
    emitter_etendue = np.pi * emitter_area * NA**2  # mm^2-ster
    print(f'Emitter etendue = {emitter_etendue:.4f} mm**2-ster')
    
    # emitted flux
    photon_wavelength = 656e-9  # wavelength (m)
    photon_energy = pc.h * pc.c / photon_wavelength  # energy per photon (J)
    emission_power = emission_radiance * photon_energy * (emitter_etendue / 1e6) * 1e9  # nW
    print(f'Emission power = {emission_power:.1f} nW')
    
    # assume transmission loss
    transmission_factor = 0.5
    transmitted_power = transmission_factor * emission_power
    print(f'Assume {transmission_factor*1e2:.1f}% transmission loss')
    print(f'Transmitted power = {transmitted_power:.1f} nW')
    
    # preamp
    preamp_response = 4.5  # mV/nW ; with 10x preamp gain
    print(f'Assume {preamp_response:.1f} mV/nW preamp response (with x10 preamp gain)')
    preamp_signal = preamp_response * transmitted_power  # mV
    print(f'Preamp signal = {preamp_signal:.1f} mV')
    
    # amplifiers
    # gain = 50
    # output_signal = preamp_signal * gain / 1e3  # V
    # print('Output with additional x{} voltage gain = {:.3g} V'.format(gain, output_signal))
    

if __name__=='__main__':
    calc_signal(emission_radiance=0.5e18,  # photons/m^2/ster
                fiber_diameter=0.51,  # mm
                spot_diameter=1.0,  # cm
                distance=230,  # cm
                aperture=8,  # cm
                )