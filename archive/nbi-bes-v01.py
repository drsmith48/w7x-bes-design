#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:04:46 2019

@author: drsmith
"""

import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from cherab.core import atomic
import cherab.openadas as oa
import vmec_connection
import beams
import nbi_ports

plt.close('all')

adas = oa.OpenADAS(permit_extrapolation=True)

# beam, plasma, and impurity species
beam_species = atomic.lookup_isotope('protium')
#target_species = atomic.lookup_isotope('hydrogen', number=2)
#impurity_species = atomic.lookup_isotope('carbon12')

print('Beam species:     {}'.format(beam_species.symbol))
#print('Target species:   {}'.format(target_species.symbol))
#print('Impurity species: {}'.format(impurity_species.symbol))

# temperature, density, beam voltage
beam_voltage = 60e3
#plasma_density = 5e19
#impurity_density = plasma_density / 100
#ti_ev = 2e3

print('Beam voltage:      {:.1f} kV'.format(beam_voltage/1e3))
#print('Target density:    {:.1e} 1/m**3'.format(plasma_density))
#print('Impurity density:  {:.1e} 1/m**3'.format(impurity_density))
#print('Ion temperature:   {:.1f} keV'.format(ti_ev/1e3))

wavelength = adas.wavelength(beam_species, 0, (3, 2))
print('{} 3->2 wavelength: {:.2f} nm'.format(beam_species.symbol, wavelength))

#vbeam = sqrt(2qV/m)
beam_species_mass = beam_species.atomic_weight * constants.m_u
vbeam = np.sqrt(2 * constants.e * beam_voltage / beam_species_mass)


def trim_beamaxis(xyz, rpz, stp):
    # delete leading inf's
    while True:
        if np.isinf(stp.x1[0]) or \
            stp.x1[0] > 1 or \
            rpz.x1[0] > 6.4:
            for obj in [xyz, stp, rpz]:
                for attrname in dir(obj):
                    attr = getattr(obj, attrname)
                    del attr[0]
        else:
            break
    # delete trailing inf's
    i=0
    while True:
        if np.isfinite(stp.x1[i]) and \
            stp.x1[i] <= 1 and \
            rpz.x1[i] <= 6.4:
            i += 1
        else:
            for obj in [xyz, stp, rpz]:
                for attrname in dir(obj):
                    attr = getattr(obj, attrname)
                    del attr[i:]
            break
    return xyz, rpz, stp


# vmec
vmec = vmec_connection.connection()
type_factory = vmec.type_factory('ns1')
Points3D = type_factory.Points3D
eq_tag = 'w7x_ref_9'
numPoints = 80

# flux surfaces for module 2
phi = (np.linspace(0,72,num=7) + 36) * np.pi/180
mod2_fs = vmec.service.getFluxSurfaces(eq_tag, phi.tolist(), 1.0, numPoints)
    
# injector geometry
injector=0
axis_spacing = 0.06
beam = beams.Beam(injector=injector, 
                  source=0, 
                  axis_spacing=axis_spacing)
phi_fs = beam.injector_origin_rpz[1]
# get VMEC flux surfaces
s = [0.005,0.333,0.667,1.0]
numPoints = 100
fs = vmec.service.getFluxSurfaces(eq_tag, phi_fs, s, numPoints)
for ibeam in [0,3]:
    beam = beams.Beam(injector=injector, 
                      source=ibeam, 
                      axis_spacing=axis_spacing)
    # vmec coordinates
    vmec_xyz = Points3D(*beam.axis.tolist())
    vmec_rpz = Points3D(*beam.axis_rpz.tolist())
    vmec_stp = vmec.service.toVMECCoordinates(eq_tag, vmec_rpz, 1E-3)
    vmec_xyz, vmec_rpz, vmec_stp = trim_beamaxis(vmec_xyz, vmec_rpz, vmec_stp)
    xyz = np.array([vmec_xyz.x1, vmec_xyz.x2, vmec_xyz.x3])
    rpz = np.array([vmec_rpz.x1, vmec_rpz.x2, vmec_rpz.x3])
    stp = np.array([vmec_stp.x1, vmec_stp.x2, vmec_stp.x3])
    daxis = np.linalg.norm(xyz - beam.source.copy().reshape((3,1)), axis=0)
    naxis = daxis.size
    vmec_bvec_xyz = vmec.service.magneticField(eq_tag, vmec_xyz)
    bvec = np.array([vmec_bvec_xyz.x1, vmec_bvec_xyz.x2, vmec_bvec_xyz.x3])
    bmod = np.linalg.norm(bvec, axis=0)
    bpar = np.squeeze(np.matmul(bvec.T, beam.r_hat.reshape(3,1)))
    angle = np.arccos(bpar/bmod)
    bunit = bvec/bmod
    # calc dist from beam axis to LCFS along B
    distances = np.array([0.3,0.4,0.5,0.6,0.8,0.9,1.0,
                          1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,
                          2.2,2.4,2.6,2.8,3.0,3.2,3.5])
    bquiver = np.empty(xyz.shape+(2,))
    dist_lcfs = np.empty((naxis,2))
    for isign, sign in enumerate([1,-1]):
        for i in range(naxis):
            dist_par_b = bunit[:,i:i+1] * distances[np.newaxis,:]
            xyz_test = xyz[:,i:i+1] + sign * dist_par_b
            psi = np.array(vmec.service.getReff(eq_tag, Points3D(*xyz_test.tolist())))
            try:
                max_ind = psi[np.isfinite(psi)].argmax()
            except ValueError:
                max_ind=0
            if max_ind==psi.size: max_ind=psi.size-1
            bquiver[:,i,isign] = xyz_test[:,max_ind+1]
            dist_lcfs[i,isign] = distances[max_ind+1]
    # port sightline calculations
    views = nbi_ports.ports.copy()
    views.pop('A21')
    views['A21_1'] = nbi_ports.A21_lower1
    views['A21_2'] = nbi_ports.A21_lower2
    port_angles = {}
    port_distances = {}
    port_dshift = {}
    for portname,port in views.items():
        port = port[:,0:-1]
        center = np.mean(port, axis=1, keepdims=True)
        segments = xyz - np.tile(center, naxis)
        dist = np.linalg.norm(segments, axis=0)
        set_hat = segments / dist
        dotprod = np.sum(set_hat*bunit, axis=0)
        ang = 180/np.pi * np.arccos(dotprod)
        ang[ang>=90] -= 180
        port_angles[portname] = np.abs(ang)
        dist[port_angles[portname]>20] = np.nan
        port_distances[portname] = dist
        vpar = vbeam * np.squeeze(np.matmul(set_hat.T,
                                            beam.r_hat.reshape((3,1))))
        vpar[port_angles[portname]>20] = np.nan
        port_dshift[portname] = wavelength * vpar / constants.c
        
    

    #### make plots
    # plot flux surfaces and beam axis
    f1 = plt.figure(figsize=(9.7,5.3))
    plt.subplot(2,3,1)
    for fss in fs:
        plt.plot(fss.x1, fss.x3, 'b')
    plt.plot(rpz[0,:], rpz[2,:], 'k')
    plt.title('{} | Phi = {:.1f}'.
              format(beam.name, phi_fs*180/np.pi))
    plt.xlabel('R')
    plt.ylabel('Z')
    plt.axis([4.5, 7, -1.0, 1.0])
    plt.gca().set_aspect('equal')
    # plot psi norm along beam axis
    plt.subplot(2,3,2)
    plt.plot(daxis, stp[0,:], '-x')
    plt.xlabel('Dist. on axis from source [m]')
    plt.ylabel('psi norm')
    # plot B angle along beam axis
    plt.subplot(2,3,3)
    plt.plot(daxis, angle*180/np.pi)
    plt.xlabel('Dist. on axis from source [m]')
    plt.ylabel('B angle wrt axis (deg)')
    plt.ylim(105,130)
    # plot angles w/ ports along beam axis
    plt.subplot(2,3,4)
    for portname,angles in port_angles.items():
        if np.all(angles>15):
            continue
        plt.plot(daxis, angles, label=portname)
    plt.xlabel('Dist. on axis from source [m]')
    plt.ylabel('Port sightline ang. wrt B (deg)')
    plt.ylim(0,20)
    plt.legend()
    # plot distance from port to obs. volume on axis
    plt.subplot(2,3,5)
    for portname,distances in port_distances.items():
        if np.all(port_angles[portname]>15):
            continue
        plt.plot(daxis, distances, label=portname)
    plt.xlabel('Dist. on axis from source [m]')
    plt.ylabel('Dist. to axis [m]')
    plt.ylim(1,4)
    # plot doppler shift
    plt.subplot(2,3,6)
    for portname,dshift in port_dshift.items():
        if np.all(port_angles[portname]>15):
            continue
        plt.plot(daxis, dshift, label=portname)
    plt.xlabel('Dist. on axis from source [m]')
    plt.ylabel('Doppler shift (nm)')
    plt.ylim(-6,6)
    plt.tight_layout()
    #### plot 3D view of beam axis and flux surfaces
    f2 = plt.figure(figsize=(10.5,7.5))
    mngr = plt.get_current_fig_manager()
    rect = mngr.window.geometry().getRect()
    mngr.window.setGeometry(30,30,rect[2],rect[3])
    ax = plt.axes(projection='3d')
    ax.scatter(*beam.source, color='g')
    rpt1 = beam.source+6*beam.r_hat
    rpt2 = beam.source+9*beam.r_hat
    l = list(zip(rpt1.tolist(), rpt2.tolist()))
    ax.plot(*l, color='g', linewidth=0.5)
    ax.plot(*xyz, color='r', linewidth=2)
    # Bpar lines of sight along beam axis
    for i in range(naxis):
        for ii in [0,1]:
            bline = xyz + (1-2*ii)*4*bunit
            l = list(zip(xyz[:,i].tolist(), bline[:,i].tolist()))
            ax.plot(*l, color='k', linewidth=0.5)
            ax.scatter(*bquiver[:,i,ii], color='k')
    # LCFS's
    for i,fs3d in enumerate(mod2_fs):
        fs_x = np.array(fs3d.x1) * np.cos(phi[i])
        fs_y = np.array(fs3d.x1) * np.sin(phi[i])
        fs_z = np.array(fs3d.x3)
        ax.plot(fs_x, fs_y, fs_z, color='b')
    # ports
    for portname,port in nbi_ports.ports.items():
        ax.plot(*port, color='m')
        ax.text(port[0,0], port[1,0], port[2,0]+0.04, portname)
    # machine axes
    ax.plot([0,5],[0,0],[0,0], color='k')
    ax.plot([0,0],[0,7],[0,0], color='k')
    ax.plot([0,0],[0,0],[-1,1], color='k')
    plt.xlim(-2,5)
    plt.ylim(0,7)
    plt.title(beam.name)
    ax.set_xlabel('Machine X (m)')
    ax.set_ylabel('Machine Y (m)')
    ax.set_zlabel('Machine Z (m)')
    plt.tight_layout()
    break


