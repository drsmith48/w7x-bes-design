#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:39:54 2019

@author: drsmith
"""

import pathlib
import numpy as np
from scipy import constants, optimize
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
import mpl_toolkits.mplot3d
import vmec_connection
from cherab.core import atomic
import cherab.openadas as oa
import beams2

analysisdir = pathlib.Path(pathlib.Path.home()/'Documents/analysis/w7x')

# beam atomic data
adas = oa.OpenADAS(permit_extrapolation=True)
beam_species = atomic.lookup_isotope('protium')
beam_voltage = 60e3
beam_species_mass = beam_species.atomic_weight * constants.m_u
vbeam = np.sqrt(2 * constants.e * beam_voltage / beam_species_mass)
transition = (3,2)
ionization_state = 0
wavelength = adas.wavelength(beam_species, ionization_state, transition)

# vmec connection
vmec = vmec_connection.connection()
type_factory = vmec.type_factory('ns1')
Points3D = type_factory.Points3D
eq_tag = 'w7x_ref_9'

def port_to_beamaxis(beam=None):

    b_angle_limit = 12
    numPoints = 80
    phi = (np.linspace(0,72,num=7) + 36) * np.pi/180
    fs = vmec.service.getFluxSurfaces(eq_tag, phi.tolist(), 1.0, numPoints)
    # determine inside/outside of plasma along beamline
    axis_points3d = Points3D(*beam.axis.tolist())
    reff = np.array(vmec.service.getReff(eq_tag, axis_points3d))
    inplasma = np.isfinite(reff)
    # B vector along beamline in plasma
    axis2 = beam.axis[:,inplasma]
    axis_rpz2 = beam.axis_rpz[:,inplasma]
    naxis2 = axis2.shape[1]
    daxis2 = np.linalg.norm(axis2 - beam.source.reshape((3,1)), axis=0)
    vmec_stp = vmec.service.toVMECCoordinates(eq_tag, 
                                              Points3D(*axis_rpz2.tolist()),
                                              1e-3)
    stp = np.array([vmec_stp.x1, vmec_stp.x2, vmec_stp.x3])
    vmec_bvec = vmec.service.magneticField(eq_tag, Points3D(*axis2.tolist()))
    bvec = np.array([vmec_bvec.x1, vmec_bvec.x2, vmec_bvec.x3])
    bmod = np.linalg.norm(bvec, axis=0)
    bunit = bvec/bmod
    # port sightline calculations
    port_angles = {}
    port_distances = {}
    port_dshift = {}
    # loop over all ports for beam
    for portname,portcoord in beam.ports.items():
        segments = axis2 - np.tile(portcoord.reshape(3,1), naxis2)
        dist = np.linalg.norm(segments, axis=0)
        set_hat = segments / dist
        dotprod = np.sum(set_hat*bunit, axis=0)
        ang = 180/np.pi * np.arccos(dotprod)
        ang[ang>=90] -= 180
        port_angles[portname] = np.abs(ang)
        port_distances[portname] = dist
        vpar = vbeam * np.squeeze(np.matmul(set_hat.T,
                                            beam.r_hat.reshape((3,1))))
        port_dshift[portname] = wavelength * vpar / constants.c
    #### plot quanitities along beam axis
    plt.figure(figsize=(7,5.25))
    plt.subplot(2,2,1)
    plt.plot(daxis2, stp[0,:], '-x')
    plt.xlabel('Dist. on axis from source [m]')
    plt.ylabel('psi norm')
    plt.title(beam.name)
    # plot angles w/ ports along beam axis
    plt.subplot(2,2,2)
    for portname,angles in port_angles.items():
        if np.all(angles>b_angle_limit):
            continue
        plt.plot(daxis2, angles, label=portname)
    plt.xlabel('Dist. on axis from source [m]')
    plt.ylabel('Sightline ang. w/ B (deg)')
    plt.ylim(0,b_angle_limit)
    plt.legend()
    # plot distance from port to obs. volume on axis
    plt.subplot(2,2,3)
    for portname,distances in port_distances.items():
        if np.all(port_angles[portname]>b_angle_limit):
            continue
        plt.plot(daxis2, distances, label=portname)
    plt.xlabel('Dist. on axis from source [m]')
    plt.ylabel('Dist. to beam axis [m]')
    plt.ylim(0,4)
    plt.legend()
    # plot doppler shift
    plt.subplot(2,2,4)
    for portname,dshift in port_dshift.items():
        if np.all(port_angles[portname]>b_angle_limit):
            continue
        valid = np.logical_or(dshift>=3,dshift<=-1.5)
        plt.plot(daxis2[valid], dshift[valid], label=portname)
    plt.xlabel('Dist. on axis from source [m]')
    plt.ylabel('Doppler shift (nm)')
    plt.ylim(-6,6)
    plt.legend()
    plt.tight_layout()
    #### 3D plot
    plt.figure(figsize=(10.5,7.5))
    mngr = plt.get_current_fig_manager()
    rect = mngr.window.geometry().getRect()
    mngr.window.setGeometry(30,30,rect[2],rect[3])
    ax = plt.axes(projection='3d')
    # plot beam axis
    ax.plot(*beam.axis, color='b')
    ax.plot(*beam.axis[:,inplasma], color='r', linewidth=2)
    # ports
    for portname,portcoords in beam.ports.items():
        ax.plot(*portcoords.reshape(3,1), '*', color='m')
        ax.text(portcoords[0], portcoords[1], portcoords[2]+0.04, portname)
    # Bpar lines of sight along beam axis
    for i in range(naxis2):
        for ii in [0,1]:
            bline = axis2 + (1-2*ii)*2.5*bunit
            l = list(zip(axis2[:,i].tolist(), bline[:,i].tolist()))
            ax.plot(*l, color='k', linewidth=0.5)
    # plot LCFS's
    for i,fs3d in enumerate(fs):
        fs_x = np.array(fs3d.x1) * np.cos(phi[i])
        fs_y = np.array(fs3d.x1) * np.sin(phi[i])
        fs_z = np.array(fs3d.x3)
        ax.plot(fs_x, fs_y, fs_z, color='b')
    # machine axes
    ax.plot([-1,6],[0,0],[0,0], color='k')
    ax.plot([0,0],[-1,6],[0,0], color='k')
    ax.plot([0,0],[0,0],[-2,2], color='k')
    plt.title(beam.name)
    ax.set_xlabel('Machine X (m)')
    ax.set_ylabel('Machine Y (m)')
    ax.set_zlabel('Machine Z (m)')
    plt.tight_layout()


def beam_vert_plane(beam=None, portnames=None, save=False):
    
    # determine inside/outside of plasma along beamline
    axis_points3d = Points3D(*beam.axis.tolist())
    reff = np.array(vmec.service.getReff(eq_tag, axis_points3d))
    inplasma = np.isfinite(reff)
    # B vector along beamline in plasma
    axis = beam.axis[:,inplasma]
    mrad_axis = np.linalg.norm(axis - beam.source.reshape((3,1)), axis=0)
    rlim = [mrad_axis.min()-0.2, mrad_axis.max()+0.05]
    tlim = [-0.40,0.40]
    # plot beam intensity in r,t plane at s=0
    ngrid = 50
    rgrid = np.linspace(rlim[0],rlim[1],num=ngrid)
    tgrid = np.linspace(tlim[0],tlim[1],num=ngrid)
    int_values = np.empty([ngrid,ngrid])
    xyz_values = np.empty([3,ngrid,ngrid])
    for ir,r in enumerate(rgrid):
        for it,t in enumerate(tgrid):
            xyz_values[:,ir,it] = \
                beam.source + r*beam.r_hat + t*beam.t_hat
            int_values[ir,it] = beam.point_intensity(t=t,r=r)
    intlevel = 0.8*int_values.max()
    z_values = xyz_values[2,:,:]
    rmaj_values = np.sqrt(xyz_values[0,:,:]**2 + xyz_values[1,:,:]**2)
    rpz_values = beam.xyz_to_rpz(xyz_values).reshape(3,-1)
    vmec_stp = vmec.service.toVMECCoordinates(eq_tag, 
                                              Points3D(*rpz_values.tolist()),
                                              1e-3)
    psi_values = np.array(vmec_stp.x1).reshape(ngrid,ngrid)
    if not isinstance(portnames, (list,tuple)):
        portnames = [portnames]
    for portname in portnames:
        # plot dist. from port to r,t plane
        portcoord = beam.ports[portname]
        sightline = xyz_values - np.tile(portcoord.reshape(3,1,1), (1,ngrid,ngrid))
        d_values = np.linalg.norm(sightline, axis=0)
        sl_hat = sightline / np.tile(d_values, (3,1,1))
        # plot port sightline angle wrt B in r,t plane
        xyz_p3d = Points3D(*xyz_values.reshape(3,-1).tolist())
        vmec_bvec = vmec.service.magneticField(eq_tag, xyz_p3d)
        bvec = np.array([vmec_bvec.x1, vmec_bvec.x2, vmec_bvec.x3]).reshape(3,ngrid,ngrid)
        bnorm = np.linalg.norm(bvec, axis=0)
        b_hat = bvec / np.tile(bnorm, (3,1,1))
        b_angle = np.arccos(np.sum(sl_hat*b_hat, axis=0)) * 180/np.pi
        b_angle[b_angle>=90] -= 180
        b_angle = np.abs(b_angle)
        fig = plt.figure(figsize=[10.2,3.25])
        plt.subplot(1,2,1)
        plt.contourf(rmaj_values, z_values, b_angle,
                     levels=np.linspace(0,15,6))
        plt.clim(0,15)
        plt.xlabel('R [m]')
        plt.ylabel('z [m]')
        plt.title(beam.name+' | {} angle to B'.format(portname))
        plt.gca().set_aspect('equal')
        plt.colorbar()
        plt.contour(rmaj_values, z_values, psi_values, colors='k')
        plt.contour(rmaj_values, z_values, int_values, 
                    colors='k', levels=[intlevel])
        # plot doppler shift in r,t plane
        vpar = vbeam * np.sum(sl_hat*np.tile(beam.r_hat.reshape(3,1,1), 
                                             (1,ngrid,ngrid)),
                              axis=0)
        dshift_values = wavelength * vpar / constants.c
        plt.subplot(1,2,2)
        plt.contourf(rmaj_values, z_values, dshift_values,
                     levels=np.linspace(-5,5,11),
                     cmap=cm.RdYlBu_r)
        plt.clim(-4,4)
        plt.xlabel('R [m]')
        plt.ylabel('z [m]')
        plt.title(beam.name+' | {} doppler shift'.format(portname))
        plt.gca().set_aspect('equal')
        plt.colorbar()
        plt.contour(rmaj_values, z_values, psi_values, colors='k')
        plt.contour(rmaj_values, z_values, int_values, 
                    colors='k', levels=[intlevel])
        plt.tight_layout()
        if save:
            fname = analysisdir / 'pini_{:d}_view_from_{}.eps'.format(beam.pini, portname)
            plt.savefig(fname.as_posix(), transparent=True)
        return fig


def port_sightline(beam=None, portname=None, r_obs=None, z_obs=None, save=False):
    # calc dist from source to axis point at R=r_obs
    rxy = beam.r_hat[0:2]
    srcxy = beam.source[0:2]
    quad_a = np.linalg.norm(rxy)**2
    quad_b = 2*np.inner(rxy,srcxy)
    quad_c = np.linalg.norm(srcxy)**2-r_obs**2
    dist = (-quad_b - np.sqrt(quad_b**2-4*quad_a*quad_c)) / (2*quad_a)
    if z_obs is None:
        target = beam.source + dist*beam.r_hat
        z_obs = target[2]
    else:
        def fun(x):
            dr = x[0]
            dt = x[1]
            f1 = (beam.source[0]+dr*beam.r_hat[0]+dt*beam.t_hat[0])**2 \
                 + (beam.source[1]+dr*beam.r_hat[1]+dt*beam.t_hat[1])**2 \
                 - r_obs**2
            f2 = beam.source[2] + dr*beam.r_hat[2] + dt*beam.t_hat[2] - z_obs
            return [f1, f2]
        sol = optimize.root(fun, [dist,0], jac=False, options={})
        if not sol.success:
            print(sol.message)
            raise RuntimeError(sol.message)
        dr,dt = sol.x
        target = beam.source + dr*beam.r_hat + dt*beam.t_hat
    # sightline from port to target
    portcoord = beam.ports[portname]
    sightline = target - portcoord
    obs_distance = np.linalg.norm(sightline)
    ngrid = 121
    half_grid = 0.6
    sightline_uv = sightline / obs_distance
    sightline_grid = np.linspace(-half_grid,half_grid,ngrid) + obs_distance
    # x,y,z coords of sightline near beam axis
    sightline = portcoord.reshape(3,1) + np.outer(sightline_uv, sightline_grid)
    # check for inside LCFS
    reff = np.array(vmec.service.getReff(eq_tag, 
                                         Points3D(*sightline.tolist())))
    inplasma = np.isfinite(reff)
    sightline = sightline[:,inplasma]
    sightline_grid = sightline_grid[inplasma]
    ngrid = sightline_grid.size
    vmec_bvector = vmec.service.magneticField(eq_tag, 
                                              Points3D(*sightline.tolist()))
    bvector = np.array([vmec_bvector.x1, vmec_bvector.x2, vmec_bvector.x3])
    bnorm = np.linalg.norm(bvector,axis=0)
    bunit = bvector / np.tile(bnorm.reshape(1,ngrid), (3,1))
    assert(np.allclose(np.linalg.norm(bunit, axis=0), 1))
    # calc angle wrt B vector along sightline
    dots = np.sum(np.tile(sightline_uv.reshape(3,1), (1,ngrid)) * bunit, axis=0)
    bangle = np.arccos(dots) * 180 / np.pi
    bangle[bangle>=90] -= 180
    # psi
    sightline_rpz = beam.xyz_to_rpz(sightline)
    rvalues = sightline_rpz[0,:]
    zvalues = sightline[2,:]
    vmec_stp = vmec.service.toVMECCoordinates(eq_tag, 
                                              Points3D(*sightline_rpz.tolist()),
                                              1e-3)
    psi_values = np.array(vmec_stp.x1)
    theta_values = np.array(vmec_stp.x2) * 180 / np.pi
    # calc beam intensity along sightline
    intensity = np.empty(ngrid)
    for i in np.arange(ngrid):
        intensity[i] = beam.point_intensity(x=sightline[0,i],
                                            y=sightline[1,i],
                                            z=sightline[2,i])
    int_max = intensity.max()
    int_norm = intensity / int_max
    int_sum = np.sum(intensity)
    
    def weighted_avg(values):
        weighted_average = np.sum(values*intensity) / int_sum
        weighted_variance = np.sum(intensity*(values-weighted_average)**2) / int_sum
        weighted_sd = np.sqrt(weighted_variance)
        return weighted_average, weighted_sd
    
    def make_lc(xdata, ydata):
        plt.xlim(xdata.min(), xdata.max())
        plt.ylim(ydata.min()-np.abs(ydata.min())*0.1, 
                 ydata.max()+np.abs(ydata.max())*0.1)
        points = np.array([xdata, ydata]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, 
                            norm=plt.Normalize(0,1),
                            cmap='viridis_r')
        lc.set_array(int_norm)
        lc.set_linewidth(3)
        line = plt.gca().add_collection(lc)
        w_avg, w_sd = weighted_avg(ydata)
        plt.annotate('wt avg = {:.2f}'.format(w_avg), 
                     (0.05,0.9), 
                     xycoords='axes fraction')
        plt.annotate('wt sd = {:.3g}'.format(w_sd), 
                     (0.05,0.8), 
                     xycoords='axes fraction')
        return line, w_avg, w_sd
        
    
    plt.figure(figsize=(12,5.5))
    ### plot R along sightline
    plt.subplot(2,3,1)
    line, r_avg, r_sd = make_lc(sightline_grid, rvalues)
    plt.colorbar(line, ax=plt.gca())
    plt.xlabel('Distance from {} (m)'.format(portname))
    plt.ylabel('R-major (m)')
    plt.title('{} | {}'.format(beam.name, portname))
    ### plot B angle along sightline
    plt.subplot(2,3,2)
    line,_,_ = make_lc(sightline_grid, bangle)
    plt.colorbar(line, ax=plt.gca())
    plt.xlabel('Distance from {} (m)'.format(portname))
    plt.ylabel('Angle wrt B (deg)');
    plt.title('{} | {}'.format(beam.name, portname))
    ### plot psi along sightline
    plt.subplot(2,3,3)
    line,_,_ = make_lc(sightline_grid, psi_values)
    plt.colorbar(line, ax=plt.gca())
    plt.xlabel('Distance from {} (m)'.format(portname))
    plt.ylabel('Psi norm')
    plt.title('{} | {}'.format(beam.name, portname))
    ### plot Z along sightline
    plt.subplot(2,3,4)
    line,z_avg,z_sd = make_lc(sightline_grid, zvalues)
    plt.colorbar(line, ax=plt.gca())
    plt.xlabel('Distance from {} (m)'.format(portname))
    plt.ylabel('Z (m)')
    plt.title('{} | {}'.format(beam.name, portname))
    ### plot theta along sightline
    plt.subplot(2,3,5)
    line,_,_ = make_lc(sightline_grid, theta_values)
    plt.colorbar(line, ax=plt.gca())
    plt.xlabel('Distance from {} (m)'.format(portname))
    plt.ylabel('Theta (deg)')
    plt.title('{} | {}'.format(beam.name, portname))
    ### plot beam intensity along sightline
    plt.subplot(2,3,6)
#    line,_,_ = make_lc(psi_values, theta_values)
#    plt.colorbar(line, ax=plt.gca())
    line,_,_ = make_lc(sightline_grid, sightline_grid)
    plt.colorbar(line, ax=plt.gca())
    plt.xlabel('Distance from {} (m)'.format(portname))
    plt.ylabel('Distance from {} (m)'.format(portname))
    plt.title('{} | {}'.format(beam.name, portname))
    plt.tight_layout()
    if save:
        fname = analysisdir / 'pini_{:d}_view_from_{}_R{:.0f}_Z{:.0f}.eps'.format(
                beam.pini, portname, np.round(r_obs*1e2), np.round(z_obs*1e2))
        plt.savefig(fname.as_posix(), transparent=True)
    f2 = beam_vert_plane(beam=beam, portnames=portname)
    axes = f2.get_axes()
    for ax in axes:
        ax.plot(r_avg, z_avg, color='r', marker='s')
        ax.plot([r_avg-r_sd,r_avg+r_sd], [z_avg,z_avg],
                color='r', marker='|', linewidth=2)
        ax.plot([r_avg,r_avg], [z_avg-z_sd,z_avg+z_sd],
                color='r', marker='_', linewidth=2)


if __name__=='__main__':
    plt.close('all')
    k20_beams = [beams2.HeatingBeam(pini=pini) for pini in range(1,5)]
    pini1 = k20_beams[0]
    pini4 = k20_beams[3]
#    for beam in k20_beams:
#        port_to_beamaxis(beam=beam)
#    k20_ports = ['W11','B21','Q20','A21-lo','A21-lolo','V11']
#    for beam in [pini1,pini4]:
#        beam_vert_plane(beam=beam, portnames=k20_ports, save=True)
    port_sightline(beam=pini1, portname='B21', 
                   r_obs=5.0, z_obs=-0.42, save=False)
#    port_sightline(beam=pini1, portname='B21', 
#                   r_obs=5.0, z_obs=-0.4, save=False)
    port_sightline(beam=pini4, portname='W11', 
                   r_obs=5.98, z_obs=-0.2, save=False)
