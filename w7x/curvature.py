#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:09:47 2020

@author: drsmith
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from vtkTools import vtk_grids
from vmecTools.wout_files import wout_read
from vmecTools.wout_files import coord_convert
from vmecTools.wout_files import curveB_tools
import beams


keys = ['R', 'Z', 'Jacobian', 'dR_ds', 'dR_du', 'dR_dv', 'dZ_ds', 'dZ_du', 'dZ_dv', 
        'Bmod', 'dBmod_ds', 'dBmod_du', 'dBmod_dv', 'Bs', 'Bu', 'Bv', 
        'dBs_du', 'dBs_dv', 'dBu_ds', 'dBu_dv', 'dBv_ds', 'dBv_du']    

vmec_file = Path('data') / 'wout.nc'

wout = wout_read.readWout('', 
                          name=vmec_file.as_posix(), 
                          diffAmps=True, 
                          curvAmps=True)


def calc_vtk():
    sym = 5
    uNpts = 161
    u_dom = np.linspace(0, 2*np.pi, uNpts)
    v_dom = np.linspace(0, 2*np.pi, sym*uNpts)
    
    rEff = 0.8
    
    wout.transForm_2D_sSec(rEff, u_dom, v_dom, keys)
    
    cartCoord = coord_convert.cartCoord(wout)
    vtk_grids.scalar_mesh('', 'core_rEff_{}.vtk'.format(int(1e2*rEff)), cartCoord, wout.invFourAmps['Bmod'])
    
    curB = curveB_tools.BcurveTools(wout)
    curB.calc_curvature()
    
    curve = 'norm'
    
    if curve == 'vec':
        name = 'vecCurve_rEff_{}.vtk'.format(int(1e2*rEff))
        curB.make_vecVTK('', name)
    elif curve == 'norm':
        name = 'normCurve_rEff_{}.vtk'.format(int(1e2*rEff))
        curB.make_normVTK('', name)
    elif curve == 'geod':
        name = 'geodCurve_rEff_{}.vtk'.format(int(1e2*rEff))
        curB.make_geodVTK('', name)
    else:
        print(curve+' is not an option.')


def calc_curve():
    f,axes = plt.subplots(2,3, figsize=[14.25,6.75])
    u_dom = np.linspace(0, 2*np.pi, 161)
    tor_angles = np.array([0,7,14,21,28,36])+36
    for it,tor_angle in enumerate(tor_angles):
        wout.transForm_2D_vSec(u_dom, tor_angle*np.pi/180, keys)
        curB = curveB_tools.BcurveTools(wout)
        curB.calc_curvature()
        ax = axes.flat[it]
        curB.plot_K(f, ax)
        plt.sca(ax)
        plt.title(f'phi = {tor_angle}')
        plt.xlabel('R (m)')
        plt.ylabel('Z (m)')
        plt.ylim(-1,1)
        plt.xlim(4.5,6.3)
        ax.set_aspect('equal')
    plt.tight_layout()
    
    reffs = np.array([0.6,0.8,0.95])
    u_dom = np.linspace(-40, 10, 89)
    v_dom = np.linspace(30, 80, 99)
    f,axes = plt.subplots(1, reffs.size, figsize=[13,3.15])
    for ir,reff in enumerate(reffs):
        wout.transForm_2D_sSec(reff, u_dom*np.pi/180, v_dom*np.pi/180, keys)
        curB = curveB_tools.BcurveTools(wout)
        curB.calc_curvature()
        ax=axes.flat[ir]
        plt.sca(ax)
        plt.imshow(curB.k_norm.T,
                   origin='lower',
                   interpolation='bilinear',
                   aspect='auto',
                   extent=[v_dom[0], v_dom[-1], u_dom[0], u_dom[-1]],
                   cmap=plt.get_cmap('RdBu'))
        plt.clim(-0.15,0.15)
        plt.title(f'reff = {reff:.2}')
        plt.xlabel('Torodial angle (deg)')
        plt.ylabel('Poloidal angle (deg)')
        cb = plt.colorbar()
        cb.ax.set_ylabel(r'$\kappa_{norm}$')
    plt.tight_layout()


def plot_curvature(sl=None, save=False):
    if not isinstance(sl, beams.Sightline):
        raise ValueError
    beam = sl.beam
    imax=sl.imaxbeam
    s=sl.psinorm[imax]
    theta=sl.theta[imax]
    phi=sl.phi[imax]
    f, axes = plt.subplots(1,2,figsize=[9.5,3.6])
    # plot sightline in R,Z plane
    vint = beam.calc_vertical_plane_intensity()
    z_values = vint['z_values']
    rmaj_values = vint['rmaj_values']
    int_values = vint['int_values']
    intlevels = np.array([0.7,0.8,0.9]) * int_values.max()
    u_dom = np.linspace(0, 2*np.pi, 161)
    wout.transForm_2D_vSec(u_dom, phi, keys)
    curB = curveB_tools.BcurveTools(wout)
    curB.calc_curvature()
    ax = axes.flat[0]
    curB.plot_K(f, ax)
    plt.sca(ax)
    plt.title(f'phi = {phi*180/np.pi:.3} deg')
    plt.xlabel('R (m)')
    plt.ylabel('Z (m)')
    plt.ylim(-1,0.75)
    plt.xlim(4.75,6.3)
    ax.set_aspect('equal')
    plt.contour(rmaj_values, z_values, int_values, 
                colors='k', levels=intlevels)
    plt.plot(sl.r[imax], sl.z[imax], 'mx', mew=2)
    # plot in phi, theta plane
    u_dom = np.linspace(theta*180/np.pi-40, theta*180/np.pi+40, 89)
    v_dom = np.linspace(phi*180/np.pi-30, phi*180/np.pi+30, 99)
    wout.transForm_2D_sSec(s, u_dom*np.pi/180, v_dom*np.pi/180, keys)
    curB = curveB_tools.BcurveTools(wout)
    curB.calc_curvature()
    ax=axes.flat[1]
    plt.sca(ax)
    plt.imshow(curB.k_norm.T,
               origin='lower',
               interpolation='bilinear',
               aspect='auto',
               extent=[v_dom[0], v_dom[-1], u_dom[0], u_dom[-1]],
               cmap=plt.get_cmap('RdBu_r'))
    clim = np.max(np.abs(curB.k_norm))
    plt.clim(-clim,clim)
    cb = plt.colorbar()
    cb.ax.set_ylabel(r'$\kappa_{norm}$')
    plt.sca(ax)
    plt.title(f'psinorm = {s:.2}')
    plt.xlabel('Torodial angle, phi (deg)')
    plt.ylabel('Poloidal angle, theta (deg)')
    rgrid = curB.R_coord
    zgrid = curB.Z_coord
    xgrid = rgrid * np.cos(v_dom.reshape(-1,1)*np.pi/180)
    ygrid = rgrid * np.sin(v_dom.reshape(-1,1)*np.pi/180)
    intvalues = np.empty(rgrid.shape)
    for iv in np.arange(v_dom.size):
        for iu in np.arange(u_dom.size):
            intvalues[iv,iu] = beam.point_intensity(x=xgrid[iv,iu],
                                                  y=ygrid[iv,iu],
                                                  z=zgrid[iv,iu])
    # intlevels = np.array([0.7,0.8,0.9]) * intvalues.max()
    plt.contour(v_dom, u_dom, intvalues.T, 
                colors='k', levels=[0.7*beam.point_intensity()])
    plt.plot(phi*180/np.pi, theta*180/np.pi, 'mx', mew=2)
    plt.tight_layout()
    if save:
        fname = Path('plots') / 'curvature.pdf'
        plt.savefig(fname.as_posix(), transparent=True)


if __name__=='__main__':
    plt.close('all')
    # calc_vtk()
    # calc_curve()
    p2 = beams.HeatingBeam(pini=2)
    sl = beams.Sightline(p2, port='A21-lolo', r_obs=5.83, z_obs=-0.47)
    plot_curvature(sl=sl, save=True)
