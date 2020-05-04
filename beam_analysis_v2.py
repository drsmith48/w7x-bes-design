#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:05:03 2019

@author: drsmith
"""

import matplotlib.pyplot as plt
import beams

# reference equilibria
#eqs = {'A':[range(2,15,2), 'EIM standard'],
#       'C':[range(15,18), 'FTM high iota'],
#       'B':[range(18,21), 'DBM low iota'],
#       'D':[range(21,26,2), 'AIM low mirror'],
#       'E':[range(27,36,2), 'KJM high mirror']}
#eqs = {'A':[[9], 'EIM standard'],
#       'C':[[17], 'FTM high iota'],
#       'B':[[20], 'DBM low iota'],
#       'D':[[23], 'AIM low mirror'],
#       'E':[[29], 'KJM high mirror']}
eqs = [17, 20, 23, 29, 9]

if __name__=='__main__':
    plt.close('all')
    p1 = beams.HeatingBeam(pini=1)
    p2 = beams.HeatingBeam(pini=2)
    p4 = beams.HeatingBeam(pini=4)
    p1.plot_all_ports()
#    for port in ['W11','B21','Q20','A21-lo','A21-lolo','V11']:
#        p1.plot_port(port)
#    p1.plot_sightline('B21', r_obs=5)
#    for eq_int in eqs:
#        eq_str = 'w7x_ref_{}'.format(eq_int)
#        p1.plot_sightline('B21', r_obs=5, eq_tag=eq_str)
#        p4.plot_sightline('W11', r_obs=5.92, z_obs=-0.2, eq_tag=eq_str)
    ### B21 views
    r_obs = 5.0
    p1.plot_sightline('B21', r_obs=r_obs)
#    p1.plot_sightlines_eq('B21', r_obs=r_obs)
#    p1.plot_sightlines_eq('B21', r_obs=r_obs, betascan=True)
    ### A21-lolo views
    r_obs, z_obs = 5.8, -0.4
    p1.plot_sightline('A21-lolo', r_obs=r_obs, z_obs=z_obs)
    r_obs, z_obs = 5.8, -0.4
    p2.plot_sightline('A21-lolo', r_obs=r_obs, z_obs=z_obs)
    r_obs, z_obs = 5.88, -0.42
    p2.plot_sightline('A21-lolo', r_obs=r_obs, z_obs=z_obs)
    ### W11 views
    r_obs, z_obs = 5.97, -0.2
    p4.plot_sightline('W11', r_obs=r_obs, z_obs=z_obs)
#    p4.plot_sightlines_eq('W11', r_obs=r_obs, z_obs=z_obs)
#    p4.plot_sightlines_eq('W11', r_obs=r_obs, z_obs=z_obs, betascan=True)