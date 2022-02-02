#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:47:25 2019

@author: drsmith
"""

import numpy as np
import matplotlib.pyplot as plt
import vmec_connection
import beams

# vmec connection

def plot_heating_beams():
    plt.figure()
    mngr = plt.get_current_fig_manager()
    rect = mngr.window.geometry().getRect()
    mngr.window.setGeometry(30,30,rect[2],rect[3])
    ax = plt.axes(projection='3d')
    ax.plot([-2,12],[0,0],[0,0], color='k')
    ax.plot([0,0],[0,14],[0,0], color='k')
    ax.plot([0,0],[0,0],[-2,2], color='k')
    ax.set_xlabel('Machine X (m)')
    ax.set_ylabel('Machine Y (m)')
    ax.set_zlabel('Machine Z (m)')
    for inj,col in zip([0,1], ['g','r']):
        for pini in range(1,5):
            beam=beams.HeatingBeam(pini=inj*4+pini)
            source = beam.source
            # print('PINI {:d} r-hat: {:.6f} {:.6f} {:.6f}'.
                  # format(beam.injector, *beam.r_hat))
            if pini==0:
                ax.scatter(*beam.injector_origin, color='k')
                for vec in [beam.u_hat, beam.v_hat, beam.w_hat/8]:
                    axpt = beam.injector_origin+2*vec
                    l = list(zip(beam.injector_origin.tolist(), axpt.tolist()))
                    ax.plot(*l, color='k')
            ax.scatter(*source, color=col)
            ax.text(source[0],source[1],source[2]+0.08, beam.injector)
            rpt = source+10*beam.r_hat
            l = list(zip(source.tolist(), rpt.tolist()))
            ax.plot(*l, color=col)
    vmec = vmec_connection.connection()
    if vmec is not None:
        # get VMEC flux surfaces
        eq_tag = 'w7x_ref_9'
        numPoints = 80
        phi = (np.linspace(0,72,num=7) + 36) * np.pi/180
        fs = vmec.service.getFluxSurfaces(eq_tag, phi.tolist(), 1.0, numPoints)
        # plot LCFS's
        for i,fs3d in enumerate(fs):
            fs_x = np.array(fs3d.x1) * np.cos(phi[i])
            fs_y = np.array(fs3d.x1) * np.sin(phi[i])
            fs_z = np.array(fs3d.x3)
            ax.plot(fs_x, fs_y, fs_z, color='b')


def plot_beam_distributions():
    hb = beams.HeatingBeam()
    rb = beams.RudixBeam()
    for beam,gridsize,res in zip([hb,rb], [0.8,0.5], [0.02,0.01]):
        sg,tg,pro2d = beam.profile(axial_dist=2, 
                                   gridsize=gridsize, 
                                   resolution=res)
        plt.figure()
        plt.contourf(sg,tg,pro2d, levels=12)
        plt.clim(0,0.003)
        plt.gca().set_aspect('equal')
        plt.title(beam.name+' | profile 2 m from source')
        plt.xlabel('source horizontal direction (m)')
        plt.ylabel('source verticle direction (m)')
        plt.colorbar()
        plt.tight_layout()
    for ax_dist in [5.0,8.0]:
        sg,tg,pro2d = hb.profile(axial_dist=ax_dist, gridsize=0.5)
        plt.figure()
        plt.contourf(sg,tg,pro2d, levels=12)
        plt.clim(0,0.003)
        plt.gca().set_aspect('equal')
        plt.title('{} | ax dist. = {:.1f} m'.format(hb.name, ax_dist))
        plt.xlabel('source horizontal direction (m)')
        plt.ylabel('source verticle direction (m)')
        plt.colorbar()
        plt.tight_layout()
    

pinis = [beams.HeatingBeam(pini=pini) for pini in [1,2,3,4]]


def beam_axis_calculations(pini=2, noplot=False, save=False):
    print('original candidate ports:')
    print(list(beams.k20_ports.keys()))
    validports = set()
    for p in pinis:
        ports = p.plot_onaxis(save=save, noplot=noplot)
        validports.update(ports)
    validports = sorted(list(validports))
    print('ports with field alignment:')
    print(validports)
    # remove bad ports
    for port in ['F21','U20','Y21']:
        validports.remove(port)
    print('ports with better field alignment:')
    print(validports)
    # return validports
    

def beam_plane_calculations(save=False):
    for ipini in [0,1]:
        pinis[ipini].plot_vertical_plane('A21-lolo', save=save)
    for port in ['W11']:
        pinis[3].plot_vertical_plane(port, save=save)
    
    
def sightline_calculations(save=False):
    r_obs, z_obs = 5.8, -0.4
    for ipini in [0,1]:
        pinis[ipini].plot_sightline('A21-lolo', r_obs=r_obs, z_obs=z_obs, save=save)


if __name__=='__main__':
    plt.close('all')
    # plot_heating_beams()
    # plot_beam_distributions()
    # beam_axis_calculations()
    beam_plane_calculations()
    # sightline_calculations()