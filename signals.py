#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:37:49 2020

@author: drsmith
"""

import numpy as np
import scipy.constants as pc


emission_radiance = 2.5e17  # photons/m**2/ster

photon_wavelength = 656e-9  # wavelength (m)
photon_energy = pc.h * pc.c / photon_wavelength  # energy per photon (J)

etendue = 0.16 # throughput (mm**2-ster)

emission_power = emission_radiance * photon_energy * etendue / 1e6  # W

print('Emission power = {:.2f} nW'.format(emission_power*1e9))

preamp_response = 4.5  # mV/nW

preamp_signal = preamp_response * emission_power*1e9  #  mV

print('Preamp signal = {:.2f} mV'.format(preamp_signal))

gain = 50

output_signal = preamp_signal * gain

print('Output with x{} gain = {:.2f} V'.format(gain, output_signal/1e3))