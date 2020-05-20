#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:40:12 2019

@author: drsmith
"""

import matplotlib.pyplot as plt
import beams

rudix = beams.RudixBeam(species='protium', bvoltage=65e3)

def beam_axis_calculations(noplot=False, save=False):
    print('original candidate ports:')
    print(list(beams.rudix_ports.keys()))
    validports = rudix.plot_all_ports(save=save, noplot=noplot)
    print('ports with field alignment:')
    print(validports)
    return validports

def beam_plane_calculations(save=False):
    validports = beam_axis_calculations(noplot=True)
    for port in validports:
        rudix.plot_port(port, save=save)

def sightline_calculations(save=False):
    r_obs, z_obs = 5.96, -0.08
    for port in ['T50','F41']:
        rudix.plot_sightline(port, r_obs=r_obs, z_obs=z_obs, save=save)
    r_obs, z_obs = 5.8, 0.02
    rudix.plot_sightline('E41-mid', r_obs=r_obs, z_obs=z_obs, save=save)


if __name__=='__main__':
    plt.close('all')
#    beam_axis_calculations()
#    beam_plane_calculations()
    sightline_calculations()