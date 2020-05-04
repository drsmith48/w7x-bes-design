#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#%% startup
import matplotlib.pyplot as plt
import beams

plt.close('all')


#%% PINIs 1-4
pinis = [beams.HeatingBeam(pini=pini, 
                           species='deuterium', 
                           bvoltage=60e3) 
         for pini in [1,2,3,4]]


#%% beam axis calculations
plt.close('all')
print('original candidate ports:')
print(list(beams.k20_ports.keys()))
validports = set()
for pini in pinis:
    ports = pini.plot_all_ports(savefig=True)
    validports.update(ports)
validports = sorted(list(validports))
print('ports with field alignment:')
print(validports)


#%% remove bad ports
bad_ports = ['F21','U20','Y21']
for port in bad_ports:
    validports.remove(port)
print('ports with better field alignment:')
print(validports)


#%% beam verticle plane calculations
plt.close('all')
for port in validports:
    for pini in pinis:
        pini.plot_port(port, save=True)
    plt.close('all')


#%% A21-lolo beam-intensity-weighted sightlines
plt.close('all')

port = 'A21-lolo'
r_obs, z_obs = 5.8, -0.4
pinis[0].plot_sightline(port, r_obs=r_obs, z_obs=z_obs, save=True)
pinis[1].plot_sightline(port, r_obs=r_obs, z_obs=z_obs, save=True)
r_obs, z_obs = 5.88, -0.42
pinis[1].plot_sightline(port, r_obs=r_obs, z_obs=z_obs, save=True)


#%% A21-lo beam-intensity-weighted sightlines
plt.close('all')

port = 'A21-lo'
r_obs, z_obs = 5.8, -0.35
pinis[0].plot_sightline(port, r_obs=r_obs, z_obs=z_obs, save=True)
pinis[1].plot_sightline(port, r_obs=r_obs, z_obs=z_obs, save=True)


#%% W11 beam-intensity-weighted sightlines
plt.close('all')

port = 'W11'
r_obs, z_obs = 5.97, -0.2
pinis[3].plot_sightline(port, r_obs=r_obs, z_obs=z_obs, save=True)


#%% PINIs 5-8
pinis = [beams.HeatingBeam(pini=pini, 
                           species='deuterium', 
                           bvoltage=60e3) 
         for pini in [5,6,7,8]]


#%% beam axis calculations
plt.close('all')
print('original candidate ports:')
print(list(beams.k21_ports.keys()))
validports = set()
for pini in pinis:
    ports = pini.plot_all_ports(savefig=True)
    validports.update(ports)
validports = sorted(list(validports))
print('ports with field alignment:')
print(validports)


#%% RUDIX
rudix = beams.RudixBeam(species='protium', bvoltage=65e3)


#%% beam axis calculations
plt.close('all')
print('original candidate ports:')
print(list(beams.rudix_ports.keys()))
validports = rudix.plot_all_ports(savefig=True)
print('ports with field alignment:')
print(validports)


#%% beam verticle plane calculations
plt.close('all')
for port in validports:
    rudix.plot_port(port, save=True)


#%% A21-lo beam-intensity-weighted sightlines
plt.close('all')
r_obs, z_obs = 5.96, -0.08
#for port in ['T50','F41']:
#    rudix.plot_sightline(port, r_obs=r_obs, z_obs=z_obs, save=True)
r_obs, z_obs = 5.8, 0.02
rudix.plot_sightline('E41-mid', r_obs=r_obs, z_obs=z_obs, save=True)


#%%

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
#eqs = [17, 20, 23, 29, 9]
#
#for port in ['W11','B21','Q20','A21-lo','A21-lolo','V11']:
#    pinis[0].plot_port(port)
#pinis[0].plot_sightline('B21', r_obs=5)
#for eq_int in eqs:
#    eq_str = 'w7x_ref_{}'.format(eq_int)
#    pinis[0].plot_sightline('B21', r_obs=5, eq_tag=eq_str)
#    pinis[3].plot_sightline('W11', r_obs=5.92, z_obs=-0.2, eq_tag=eq_str)
#### B21 views
#r_obs = 5.0
#pinis[0].plot_sightline('B21', r_obs=r_obs)
##    p1.plot_sightlines_eq('B21', r_obs=r_obs)
##    p1.plot_sightlines_eq('B21', r_obs=r_obs, betascan=True)
