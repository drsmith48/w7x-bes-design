#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:04:46 2019

@author: drsmith
"""

import numpy as np
import matplotlib.pyplot as plt
#from cherab.core import atomic
import vmec_connection
import beam_class

plt.close('all')


## beam, plasma, and impurity species
#beam_species = atomic.lookup_isotope('hydrogen', number=2)
#target_species = atomic.lookup_isotope('hydrogen', number=2)
#impurity_species = atomic.lookup_isotope('carbon12')
#
#print('Beam species:     {}'.format(beam_species.symbol))
#print('Target species:   {}'.format(target_species.symbol))
#print('Impurity species: {}'.format(impurity_species.symbol))
#
## temperature, density, beam voltage
#beam_voltage_ev = 60e3
#beam_ev_amu = beam_voltage_ev / beam_species.mass_number
#plasma_density = 5e19
#impurity_density = plasma_density / 100
#ti_ev = 2e3
#
#print('Beam voltage:      {:.1f} keV'.format(beam_voltage_ev/1e3))
#print('Beam voltage/amu:  {:.1f} keV/amu'.format(beam_ev_amu/1e3))
#print('Target density:    {:.1e} 1/m**3'.format(plasma_density))
#print('Impurity density:  {:.1e} 1/m**3'.format(impurity_density))
#print('Ion temperature:   {:.1f} keV'.format(ti_ev/1e3))

# injector geometry
injector=0
beams= []
for ibeam in range(4):
    beams.append(beam_class.Beam(injector=injector, 
                                 beam=ibeam, 
                                 axis_spacing=0.01))
phi_fs = beams[0].injector_origin_rpz[1]
# plot flux surfaces and beam axis
plt.figure()
plt.scatter(beams[0].injector_origin_rpz[0], 
            beams[0].injector_origin_rpz[2], marker='x')
for beam in beams:
    plt.plot(beam.axis_rpz[0,:], beam.axis_rpz[2,:])
#    plt.annotate('NI2{} src {}'.format(beam.injector, beam.beam+1),
#                 (beam.axis_rpz[0,0]+0.05, beam.axis_rpz[2,0]+0.05))


# VMEC equilibrium
vmec = vmec_connection.connection()
eq_tag = 'w7x_ref_9'
# get flux surfaces
s = [0.001,0.2,0.4,0.6,0.8,1.0]
numPoints = 100
fs = vmec.service.getFluxSurfaces(eq_tag, phi_fs, s, numPoints)
for fss in fs:
    plt.plot(fss.x1, fss.x3, 'b')

plt.title('flux surfaces phi = {:.2f}'.format(phi_fs*180/np.pi))
plt.xlabel('R')
plt.ylabel('Z')
plt.axis([4.5, 7, -1.0, 1.0])
plt.gca().set_aspect('equal')
plt.tight_layout()


beam = beams[3]
type_factory = vmec.type_factory('ns1')
Points3D = type_factory.Points3D
vmec_rpz = Points3D(*beam.axis_rpz.tolist())
vmec_stp = vmec.service.toVMECCoordinates(eq_tag, vmec_rpz, 1E-3)
# delete leading inf's
while True:
    if np.isinf(vmec_stp.x1[0]) or \
        vmec_stp.x1[0] > 1 or \
        vmec_rpz.x1[0] > 6.4:
        for attrname in dir(vmec_stp):
            for obj in [vmec_stp, vmec_rpz]:
                attr = getattr(obj, attrname)
                del attr[0]
    else:
        break
# delete trailing inf's
i=0
while True:
    if np.isfinite(vmec_stp.x1[i]) and \
        vmec_stp.x1[i] <= 1 and \
        vmec_rpz.x1[i] <= 6.4:
        i += 1
    else:
        for attrname in dir(vmec_stp):
            for obj in [vmec_stp, vmec_rpz]:
                attr = getattr(obj, attrname)
                del attr[i:]
        break
plt.plot(vmec_rpz.x1, vmec_rpz.x3, 'k')
#for i in np.arange(len(vmec_rpz.x1)):
#    print('s,r,phi,z:  {:.3f}  {:.3f}  {:.3f}  {:.3f}'.
#          format(vmec_stp.x1[i], vmec_rpz.x1[i],
#                 vmec_rpz.x2[i], vmec_rpz.x3[i]))


