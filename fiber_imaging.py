#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:39:58 2019

@author: drsmith
"""

import numpy as np

# fiber specs
fiber_core_diameter = 0.05   # in mm
# fiber NA is size of light cone ; NA = sin(acceptance angle), so NA = [0,1]
fiber_na = 0.2
# lens f/# is ratio of focal length to diameter ; by geometry, NA * f/# = 1/2
fiber_fn = 1/(2*fiber_na)

area = np.pi * (fiber_core_diameter/2)**2   # in mm**2
etendue = np.pi * area * fiber_na**2    # in mm**2-ster

# for fiber receiver, optics can under-fill fiber:
# NA(optics) <= NA(fiber)

# for fiber transmitter, fiber should exactly fill optics:
# NA(optics) = NA(fiber)

krhoi_max = 0.8
rhoi = np.linspace(1.5,3.5,10) # rho_i values in mm

k_max = krhoi_max / (rhoi/10) # in 1/cm
lambda_min = 2*np.pi / k_max # in cm
spotsize = lambda_min/2 # in cm
print('spotsize', spotsize)

spotsize = 0.75 # spotsize in cm
print('Using spotsize:', spotsize, 'cm')

dist_spot2lens = 200.0 # distance in cm
lens_diameter = 6 # in cm

fiber_core_diameter = 3 # in mm
fiber_na = 0.25

# fill fiber with spot image
image_size = fiber_core_diameter / 10  # in cm

# Magnification = sqrt(image size / object size) = image dist / object dist
mag = np.sqrt(image_size / spotsize)
dist_lens2fiber = mag * dist_spot2lens

focal_length = dist_lens2fiber * dist_spot2lens / (dist_lens2fiber + dist_spot2lens)

lens_fnum = 1/(2*fiber_na)
lens_diameter = focal_length / lens_fnum