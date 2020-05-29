#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 07:56:39 2020

@author: drsmith
"""

import numpy as np


spot_distance = 200.0  # cm

fiber_na = 0.5  # bundle numerical aperture, sin(theta) light cone
fiber_radius = 0.02  # fiber radius, cm

lens_radius = 3.0 # cm

print('Inputs')
print('  Fiber NA = {:.3f}'.format(fiber_na))
print('  Fiber radius = {:.3f} mm'.format(fiber_radius*10))
print('  Spot distance = {:.1f} cm'.format(spot_distance))
print('  Lens radius = {:.2f} cm'.format(lens_radius))

t = spot_distance
a = lens_radius
assert (a>0 and t>0)

theta = np.arctan2(a,t)
print('  Theta = {:.3g} rad  (axial marginal ray)'.format(theta))
assert(theta>0)

hp = -fiber_radius
assert(hp<0)

thetap = -np.arcsin(fiber_na)
print('  Thetap = {:.4g} rad  (set by fiber NA)'.format(thetap))
assert(thetap<0)

mag = theta / thetap
print('  Magnification = {:.4f}  (axial marginal ray)'.format(mag))
print('  1/Mag = {:.4f}'.format(1/mag))
assert(np.abs(mag)<1 and mag<0)

spot_radius = h = hp / mag
print('  Spot radius = {:.3f} cm  (set by eq 1 of edge chief ray)'.format(h))
assert(h>0)

beta = -np.arctan2(h,t)
print('  Beta = {:.3g} rad  (object edge chief ray)'.format(beta))
assert(beta<0)

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
