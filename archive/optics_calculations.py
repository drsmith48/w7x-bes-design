#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 07:56:39 2020

@author: drsmith
"""

import numpy as np


# bundle_halfsize = 0.139 * 2.54 / 2  # cm
fiber_na = 0.4  # fiber numerical aperture, sin(theta) light cone
fiber_radius = 0.1  # fiber radius, cm
spot_radius = 1.5 / 2  # cm
spot_distance = 200.0  # cm
lens_radius = 3.0 # cm
print('Inputs')
print('  Spot radius = {:.2f} cm'.format(spot_radius))
print('  Spot distance = {:.1f} cm'.format(spot_distance))
print('  Lens radius = {:.2f} cm'.format(lens_radius))
print('  Fiber NA = {:.2f}'.format(fiber_na))

t = spot_distance
h = spot_radius
a = lens_radius

theta = np.arctan2(a,t)
print('  Theta = {:.3g} rad  (axial marginal ray)'.format(theta))
assert(theta>0)

beta = -np.arctan2(h,t)
print('  Beta = {:.3g} rad  (object edge chief ray)'.format(beta))
assert(beta<0)

# gamma = -np.arctan2(h+a, t)
# print('  Gamma = {:.4f} rad  (object edge marginal ray)'.format(gamma))
# assert(gamma<0)
# assert(np.abs(gamma)>np.abs(beta))

thetap = -np.arcsin(fiber_na)
print('  Thetap = {:.4g} rad'.format(thetap))
assert(thetap<0)

mag = theta / thetap
print('  Magnification = {:.4f}  (axial marginal ray)'.format(mag))
print('  1/Mag = {:.4f}'.format(1/mag))
assert(np.abs(mag)<1 and mag<0)

hp = image_halfsize = mag * h
print('  Image (fiber illum.) radius = {:.3f} mm  (eq 1 for object edge chief ray)'.format(hp*10))
assert(hp<0)

tps = np.array([1.5,2,4,6,8])

for tp in tps:
    print('Image distance = {:.1f} cm'.format(tp))
    ap = -tp * np.tan(thetap)
    print('  Exit pupil radius = {:.3f} cm'.format(ap))
    betap = np.arctan2(hp, tp)
    print('  Betap = {:.4g} rad'.format(betap))
    assert(betap<0)
    k = (beta/mag-betap) / h
    print('  K = {:.3g} 1/cm  (eq 2 for object edge chief ray)'.format(k))
