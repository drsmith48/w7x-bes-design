#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:38:30 2019

@author: drsmith
"""

import pathlib
import numpy as np
from scipy import integrate, interpolate, constants, optimize
import scipy.special as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from cherab.core import atomic
import cherab.openadas as oa
import vmec_connection


# vmec connection
vmec = vmec_connection.connection()
Points3D = vmec.type_factory('ns1').Points3D

# atomic data
adas = oa.OpenADAS(permit_extrapolation=True)

# candidate viewing ports for injectors
# input dimensions are mm
k20_ports = {
             'A21-mid':np.array([1981.8, 6099.4, 0]),
             'A21-lo':np.array([1981.8, 6099.4, -250]),
             'A21-lolo':np.array([1981.8, 6099.4, -375]),
             'B21':np.array([1718.3, 5960.8, -906.1]),
             'F21':np.array([1075.6, 6306.9, -449.3]),
             'Q20':np.array([2225, 5955.9, -650]),
             'T21':np.array([-930.7, 6276.9, -152.5]),
             'Y21':np.array([442.7, 6312.2, -736.7]),
             'W20':np.array([4773.9, 3671.2, 346.7]),
             'V11':np.array([4216.9, 2710.2, -642.1]),
             'W11':np.array([4966.7, 3405.8, -346.7]),
             'U20':np.array([5063.6, 3707.1, 0]),
             }
k21_ports = {
             'A21-mid':np.array([1981.8, 6099.4, 0]),
             'A21-hi':np.array([1981.8, 6099.4, 250]),
             'A21-hihi':np.array([1981.8, 6099.4, 375]),
             'B20':np.array([2113.5, 5832.3, 906.1]),
             'F20':np.array([2836.9, 5734.6, 449.3]),
             'Q21':np.array([1703.2, 6127.2, 650.4]),
             'T20':np.array([4457.1, 4548.3, 159.9]),
             'Y20':np.array([3352.1, 5366.8, 736.7]),
             'W21':np.array([-1704.3, 5776, -346.7]),
             'V30':np.array([-1818.2, 4671.7, 641.1]),
             'W30':np.array([-2016.1, 5674.7, 346.2]),
             'U30':np.array([-1939.3, 5968.7, 0]),
             }
rudix_ports = {
               'U50':np.array([-1939.3, -5968.7, 0]),
               'T50':np.array([-932.4, -6299.2, 159.8]),
               'W50':np.array([-1704.5, -5776, 346.2]),
               'V50':np.array([-1275, -4848.2, 641.1]),
               'Unk50':np.array([-867.5, -4729.5, -925.7]),
               'G41':np.array([-3974.2, -4029.3, -1037.3]),
               'Y41':np.array([-4068.3, -4846.4, -736.7]),
               'O41':np.array([-3214.7, -5091.1, 607.7]),
               'L41':np.array([-3222.2, -3968.1, 124.3]),
               'F41':np.array([-4586.4, -4479.6, -467.7]),
               'V41':np.array([-1818.6, -4671.3, -642.1]),
               'K41-lo':np.array([-3949.8, -4922.6, 194.2]),
               'K41-mid':np.array([-3906.4, -4870.4, 369.2]),
               'K41-up':np.array([-3863.1, -4818.1, 544.2]),
               'E41-lo':np.array([-4688.1, -4432.2, -150]),
               'E41-mid':np.array([-4650.9, -4397, 100]),
               'E41-up':np.array([-4613.7, -4361.8, 350]),
               }

# convert dimensions from mm to m
for ports in [k20_ports, k21_ports, rudix_ports]:
    for value in ports.values():
        value /= 1e3
del ports
del value


class _Beam(object):
    
    # subclasses must define these attributes
    _required_attr = ['source', 'source_rpz', 'ports',
                      'r_hat', 's_hat', 't_hat', 'name',
                      'src_width_1', 'src_width_2', 'divergence',
                      'torus_period']
    _analysisdir = pathlib.Path().absolute().parent
    _ref_eq = 'w7x_ref_9'

    def __init__(self, axis_spacing=0.04, species='protium', bvoltage=60e3,
                 eq_tag=None):
        # check for required attributes
        for attrname in self._required_attr:
            if getattr(self, attrname, None) is None:
                raise AttributeError('"{}" is not implemented'.format(attrname))
        self.eq_tag = None
        self.axis = None
        self.axis_rpz = None
        self.tantheta = np.tan(self.divergence * np.pi/180)

        self.species = atomic.lookup_isotope(species)
        self.voltage = bvoltage
        beam_species_mass = self.species.atomic_weight * constants.m_u
        self.vbeam = np.sqrt(2 * constants.e * self.voltage / beam_species_mass)
        transition = (3,2)
        ionization_state = 0
        self.wavelength = adas.wavelength(self.species, ionization_state, transition)
        self.set_eq(eq_tag=eq_tag, axis_spacing=axis_spacing)
        vmec_bvec = vmec.service.magneticField(self.eq_tag, 
                                               Points3D(*self.axis.tolist()))
        self.bvec = np.array([vmec_bvec.x1, vmec_bvec.x2, vmec_bvec.x3])
    
    @staticmethod
    def weighted_avg(values, weights):
        if values is None or values.size==0:
            return np.NaN, np.NaN
        weighted_average = np.average(values, weights=weights)
        weighted_std = np.sqrt(np.average((values-weighted_average)**2,
                                          weights=weights))
        return weighted_average, weighted_std
    
    def set_eq(self, eq_tag=None, axis_spacing=0.04):
        if eq_tag is None:
            self.eq_tag = self._ref_eq
        else:
            self.eq_tag = eq_tag
        self.make_axis(axis_spacing=axis_spacing)
        
    def make_axis(self, axis_spacing=0.04):
        beamaxis = np.empty((3,0))
        xyz = self.source.copy()
        while True:
            xyz += axis_spacing * self.r_hat
            rmajor = np.linalg.norm(xyz[0:2])
            if rmajor > 6.75:
                continue
            elif rmajor > 4.5:
                beamaxis = np.append(beamaxis, xyz.reshape((3,1)), axis=1)
            else:
                break
        # determine inside/outside of plasma along beamline
        axis_points3d = Points3D(*beamaxis.tolist())
        reff = np.array(vmec.service.getReff(self.eq_tag, axis_points3d))
        inplasma = np.isfinite(reff)
        self.axis = beamaxis[:,inplasma]
        self.naxis = self.axis.shape[1]
        self.daxis = np.linalg.norm(self.axis - self.source.reshape((3,1)), axis=0)
        self.axis_rpz = self.xyz_to_rpz(self.axis)

    def plot_onaxis(self, b_angle_limit=15, save=False, noplot=False):
        # get B vector along beam axis
        vmec_stp = vmec.service.toVMECCoordinates(self.eq_tag, 
                                                  Points3D(*self.axis_rpz.tolist()),
                                                  1e-3)
        stp = np.array([vmec_stp.x1, vmec_stp.x2, vmec_stp.x3])
        bmod = np.linalg.norm(self.bvec, axis=0)
        bunit = self.bvec/bmod
        # port sightline calculations
        port_angles = {}
        port_distances = {}
        port_dshift = {}
        # loop over all ports for beam
        for portname,portcoord in self.ports.items():
            segments = self.axis - np.tile(portcoord.reshape(3,1), self.naxis)
            dist = np.linalg.norm(segments, axis=0)
            set_hat = segments / dist
            dotprod = np.sum(set_hat*bunit, axis=0)
            ang = 180/np.pi * np.arccos(dotprod)
            ang[ang>=90] -= 180
            port_angles[portname] = np.abs(ang)
            port_distances[portname] = dist
            vpar = self.vbeam * np.squeeze(np.matmul(set_hat.T,
                                                self.r_hat.reshape((3,1))))
            port_dshift[portname] = self.wavelength * vpar / constants.c
        validports = []
        for portname,angles in port_angles.items():
            if np.any(angles<=b_angle_limit):
                validports.append(portname)
        if noplot:
            return validports
        #### plot quanitities along beam axis
        plt.figure(figsize=(7.55,5.6))
        plt.subplot(2,2,1)
        plt.plot(self.daxis, stp[0,:], '-x')
        plt.xlabel('Dist. along beam axis [m]')
        plt.ylabel('psi norm')
        plot_title = '{} axis'.format(self.name)
        plt.title(plot_title)
        # plot angles w/ ports along beam axis
        legend_kwargs = {'loc':'upper right',
                        'ncol':3, 
                        'columnspacing':0.5, 
                        'handletextpad':0.5,
                        'handlelength':1,
                        'labelspacing':0.25,
                        'fontsize':'small'}
        plt.subplot(2,2,2)
        for portname in validports:
            plt.plot(self.daxis, port_angles[portname], label=portname)
        plt.xlabel('Dist. along beam axis [m]')
        plt.ylabel('Sightline ang. w/ B (deg)')
        plt.ylim(0,b_angle_limit)
        plt.title(plot_title)
        plt.legend(**legend_kwargs)
        # plot distance from port to obs. volume on axis
        plt.subplot(2,2,3)
        for portname in validports:
            plt.plot(self.daxis, port_distances[portname], label=portname)
        plt.xlabel('Dist. along beam axis [m]')
        plt.ylabel('Dist. to beam axis [m]')
        plt.ylim(0,4)
        plt.title(plot_title)
        plt.legend(**legend_kwargs)
        # plot doppler shift
        plt.subplot(2,2,4)
        for portname in validports:
            plt.plot(self.daxis, port_dshift[portname], label=portname)
        plt.xlabel('Dist. along beam axis [m]')
        plt.ylabel('Doppler shift (nm)')
        plt.ylim(-6,6)
        plt.title(plot_title)
        plt.annotate('{} @ {:.1f} keV'.format(self.species.symbol, self.voltage/1e3),
                     [0.03,0.03], xycoords='axes fraction')
        plt.legend(**legend_kwargs)
        plt.tight_layout()
        if save:
            fname = 'pini_{:d}_axis.pdf'.format(self.injector)
            fname = self._analysisdir / fname
            plt.savefig(fname.as_posix(), format='pdf', transparent=True)
        # return validports

    def plot_3d(self, save=False):
        numPoints = 80
        phi = (np.linspace(0,72,num=7) - 36 + 72*(self.torus_period-1)) * np.pi/180
        fs = vmec.service.getFluxSurfaces(self.eq_tag, phi.tolist(), 1.0, numPoints)
        #### 3D plot
        plt.figure(figsize=(7.5,5.5))
        mngr = plt.get_current_fig_manager()
        rect = mngr.window.geometry().getRect()
        mngr.window.setGeometry(30,30,rect[2],rect[3])
        ax = plt.axes(projection='3d')
        # plot beam axis
        ax.plot(*self.axis, color='b')
        # ports
        for portname,portcoords in self.ports.items():
            ax.plot(*portcoords.reshape(3,1), '*', color='m')
            ax.text(portcoords[0], portcoords[1], portcoords[2]+0.04, portname)
        # Bpar lines of sight along beam axis
        bmod = np.linalg.norm(self.bvec, axis=0)
        bunit = self.bvec/bmod
        for i in range(self.naxis):
            for ii in [0,1]:
                bline = self.axis + (1-2*ii)*2.5*bunit
                l = list(zip(self.axis[:,i].tolist(), bline[:,i].tolist()))
                ax.plot(*l, color='k', linewidth=0.5)
        # plot LCFS's
        for i,fs3d in enumerate(fs):
            fs_x = np.array(fs3d.x1) * np.cos(phi[i])
            fs_y = np.array(fs3d.x1) * np.sin(phi[i])
            fs_z = np.array(fs3d.x3)
            ax.plot(fs_x, fs_y, fs_z, color='b')
        # machine axes
        ax.plot([-1,1],[0,0],[0,0], color='k')
        ax.plot([0,0],[-1,1],[0,0], color='k')
        ax.plot([0,0],[0,0],[-2,2], color='k')
        plt.title(self.name)
        ax.set_xlabel('Machine X (m)')
        ax.set_ylabel('Machine Y (m)')
        ax.set_zlabel('Machine Z (m)')
        plt.tight_layout()
        # return validports
        
    def plot_vertical_plane(self, 
                            port='A21-lolo', 
                            eq_tag=None, 
                            save=False, 
                            sp1=None, 
                            sp2=None):
        if eq_tag is not None:
            self.set_eq(eq_tag=eq_tag)
        # B vector along beamline in plasma
        mrad_axis = np.linalg.norm(self.axis - self.source.reshape((3,1)), axis=0)
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
                xyz_values[:,ir,it] = self.source + r*self.r_hat + t*self.t_hat
                int_values[ir,it] = self.point_intensity(t=t,r=r)
        intlevels = np.array([0.8,0.9]) * int_values.max()
        z_values = xyz_values[2,:,:]
        rmaj_values = np.sqrt(xyz_values[0,:,:]**2 + xyz_values[1,:,:]**2)
        rpz_values = self.xyz_to_rpz(xyz_values).reshape(3,-1)
        vmec_stp = vmec.service.toVMECCoordinates(self.eq_tag, 
                                                  Points3D(*rpz_values.tolist()),
                                                  1e-3)
        psi_values = np.array(vmec_stp.x1).reshape(ngrid,ngrid)
        # plot dist. from port to r,t plane
        portcoord = self.ports[port]
        sightline = xyz_values - np.tile(portcoord.reshape(3,1,1), (1,ngrid,ngrid))
        d_values = np.linalg.norm(sightline, axis=0)
        sl_hat = sightline / np.tile(d_values, (3,1,1))
        # plot port sightline angle wrt B in r,t plane
        xyz_p3d = Points3D(*xyz_values.reshape(3,-1).tolist())
        vmec_bvec = vmec.service.magneticField(self.eq_tag, xyz_p3d)
        bvec = np.array([vmec_bvec.x1, vmec_bvec.x2, vmec_bvec.x3]).reshape(3,ngrid,ngrid)
        bnorm = np.linalg.norm(bvec, axis=0)
        print(np.nanmin(bnorm), np.nanmax(bnorm))
        b_hat = bvec / np.tile(bnorm, (3,1,1))
        b_angle = np.arccos(np.sum(sl_hat*b_hat, axis=0)) * 180/np.pi
        b_angle[np.isnan(b_angle)] = 90
        b_angle = 90-np.abs(b_angle-90)
        if sp1 and sp2:
            plt.subplot(sp1)
        else:
            plt.figure(figsize=[10.2,3.25])
            plt.subplot(1,2,1)
        plt.contourf(rmaj_values, z_values, b_angle,
                     levels=np.linspace(0,15,6),
                     cmap=cm.viridis)
        plt.clim(0,15)
        plt.xlabel('R [m]')
        plt.ylabel('z [m]')
        plt.title('{} | {} | {} B-angle'.format(self.name, self.eq_tag, port))
        plt.gca().set_aspect('equal')
        plt.colorbar()
        plt.contour(rmaj_values, z_values, psi_values, colors='k')
        plt.contour(rmaj_values, z_values, int_values, 
                    colors='k', levels=intlevels)
        # plot doppler shift in r,t plane
        tiled_rhat = np.tile(self.r_hat.reshape(3,1,1), (1,ngrid,ngrid))
        vpar = self.vbeam * np.sum(sl_hat*tiled_rhat, axis=0)
        dshift_values = self.wavelength * vpar / constants.c
        if sp1 and sp2:
            plt.subplot(sp2)
        else:
            plt.subplot(1,2,2)
        plt.contourf(rmaj_values, z_values, dshift_values,
                     levels=np.linspace(-5,5,11),
                     cmap=cm.RdYlBu_r)
        plt.clim(-4,4)
        plt.xlabel('R [m]')
        plt.ylabel('z [m]')
        plt.title('{} | {} | {} doppler'.format(self.name, self.eq_tag, port))
        plt.gca().set_aspect('equal')
        plt.colorbar()
        plt.contour(rmaj_values, z_values, psi_values, colors='k')
        plt.contour(rmaj_values, z_values, int_values, 
                    colors='k', levels=intlevels)
        plt.annotate('{} @ {:.1f} keV'.format(self.species.symbol, self.voltage/1e3),
                     [0.03,0.92], xycoords='axes fraction')
        plt.tight_layout()
        if save:
            fname = 'port_{}_viewing_pini_{:d}.pdf'.format(port, self.injector)
            fname = self._analysisdir / fname
            plt.savefig(fname.as_posix(), format='pdf', transparent=True)
            
    def plot_sightline(self, port=None, r_obs=None, z_obs=None, 
                       eq_tag=None, save=False):
        sl = Sightline(self, port, 
                              r_obs=r_obs, 
                              z_obs=z_obs, 
                              eq_tag=eq_tag)
        print(sl.bnorm)
        ### plot quantities along sl
        plt.figure(figsize=(12,8.25))
        plotdata = [[sl.r, 'R-major (m)'],
                    [sl.z, 'Z (m)'],
                    [sl.bangle, 'Angle wrt B (deg)'],
                    [sl.psinorm, 'Psi-norm'], 
                    [sl.cosn, 'sl*nhat'],
                    [sl.cosbi, 'sl*bihat'],
                    ]
        imax = np.argmax(sl.intensity)
        for i,pdata in enumerate(plotdata):
            plt.subplot(3,3,i+1)
            xdata = sl.distance
            ydata = pdata[0]
            plt.xlim(xdata.min(), xdata.max())
            plt.ylim(ydata.min()-np.abs(ydata.min())*0.1, 
                      ydata.max()+np.abs(ydata.max())*0.1)
            points = np.array([xdata, ydata]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, norm=plt.Normalize(0,1),
                                cmap='viridis_r')
            lc.set_array(sl.intensity)
            lc.set_linewidth(3)
            line = plt.gca().add_collection(lc)
            w_avg, w_sd = self.weighted_avg(ydata, sl.intensity)
            plt.errorbar(xdata[imax], w_avg, yerr=w_sd, 
                         color='r', capsize=3)
            plt.colorbar(line, ax=plt.gca())
            plt.xlabel('Dist. on {} sightline (m)'.format(port))
            plt.ylabel(pdata[1])
            plt.title('{} | {} | {}'.format(self.eq_tag, port, self.name))
            if np.array_equal(ydata, sl.bangle):
                plt.ylim(0,10)
            plt.annotate('beam-wtd {}'.format(pdata[1]), (0.05,0.9), 
                          xycoords='axes fraction')
            plt.annotate('{:.3g} +/- {:.2g}'.format(w_avg, w_sd), (0.05,0.8), 
                          xycoords='axes fraction')
            if np.array_equal(ydata, sl.cosn) or np.array_equal(ydata, sl.cosbi):
                w_int = np.sum(ydata*sl.step*sl.intensity)
                plt.annotate('beam-wtd-sum: {:.3g} cm'.format(w_int*1e2), (0.05,0.7), 
                          xycoords='axes fraction')
            # if i<=1:
            #     plt.annotate('k_max*rho_i = {:.3g}'.format(np.pi*2.5e-3/(2*w_sd)), 
            #                   (0.05,0.7), 
            #                   xycoords='axes fraction')
        #### add measurement markers to beam profiles
        self.plot_vertical_plane(port, sp1=338, sp2=339)
        # r_avg, r_sd = self.weighted_avg(sl.r, sl.intensity)
        # z_avg, z_sd = self.weighted_avg(sl.z, sl.intensity)
        r_avg = np.average(sl.r, weights=sl.intensity)
        z_avg = np.average(sl.z, weights=sl.intensity)
        bi_disp = np.sum(sl.cosbi*sl.step*sl.intensity)/2
        n_disp = np.sum(sl.cosn*sl.step*sl.intensity)/2
        nhat = sl.nhat[:,imax]
        bihat = sl.bihat[:,imax]
        ax = plt.gcf().axes
        for iax,col in zip([12,14],['r','b']):
            # binormal displacement
            nhat_r = np.linalg.norm(nhat[0:2])
            nhat_z = nhat[2]
            bihat_r = np.linalg.norm(bihat[0:2])
            bihat_z = bihat[2]
            rseq = [r_avg + bi_disp*bihat_r,
                    r_avg + n_disp*nhat_r,
                    r_avg - bi_disp*bihat_r,
                    r_avg - n_disp*nhat_r,
                    r_avg + bi_disp*bihat_r]
            zseq = [z_avg + bi_disp*bihat_z,
                    z_avg + n_disp*nhat_z,
                    z_avg - bi_disp*bihat_z,
                    z_avg - n_disp*nhat_z,
                    z_avg + bi_disp*bihat_z]
            ax[iax].plot(rseq, zseq, color=col)
            # ax[iax].plot(r_avg, z_avg, color=col, marker='s', markersize=1)
            # ax[iax].plot([r_avg-r_sd,r_avg+r_sd], [z_avg,z_avg],
            #         color=col, marker='|', linewidth=2)
            # ax[iax].plot([r_avg,r_avg], [z_avg-z_sd,z_avg+z_sd],
            #         color=col, marker='_', linewidth=2)
        plt.tight_layout()
        if save:
            fname = self._analysisdir / 'pini_{:d}_view_from_{}_R{:.0f}_Z{:.0f}.pdf'.format(
                    self.injector, port, np.round(r_obs*1e2), np.round(z_obs*1e2))
            plt.savefig(fname.as_posix(), transparent=True)
            
    def plot_sightlines_eqscan(self, port=None, r_obs=None, z_obs=None, 
                           betascan=False, save=False):
        eq_tag_original = self.eq_tag
        if betascan:
            eqs = [[range(2,15,2), 'A: EIM standard'],
                   [range(18,21), 'B: DBM low iota'],
                   [range(15,18), 'C: FTM high iota'],
                   [range(21,26,2), 'D: AIM low mirror'],
                   [range(27,36,2), 'E: KJM high mirror']]
        else:
            eqs = [[[9], 'A: EIM standard'],
                   [[20], 'B: DBM low iota'],
                   [[17], 'C: FTM high iota'],
                   [[23], 'D: AIM low mirror'],
                   [[29], 'E: KJM high mirror'],
                   [[39], 'F: JLF low shear'],
                   [[45], 'G: FIS inward shift'],
                   [[48], 'H: DKH outward shift']]
        plt.figure()
        for eq in eqs:
            ind, desc = eq
            psi = np.empty([2,0])
            theta = np.empty([2,0])
            for i in ind:
                eq_tag = 'w7x_ref_{}'.format(i)
                sl = Sightline(self, port, 
                                      r_obs=r_obs, 
                                      z_obs=z_obs, 
                                      eq_tag=eq_tag)
                avg,std = self.weighted_avg(sl.psi, sl.intensity)
                psi = np.append(psi, 
                                np.array([[avg,std]]).T,
                                axis=1)
                avg,std = self.weighted_avg(sl.theta, sl.intensity)
                theta = np.append(theta, 
                                  np.array([[avg,std]]).T,
                                  axis=1)
            plt.errorbar(psi[0,:], theta[0,:], 
                         xerr=psi[1,:], 
                         yerr=theta[1,:],
                         fmt='s-',
                         label=desc,
                         capsize=4)
            if len(ind)>1:
                for i,xy in enumerate(zip(psi[0,:], theta[0,:])):
                    plt.annotate(str(i+1), (xy[0]+0.005, xy[1]+0.2), color='k')
        plt.legend()
        plt.xlabel('Psi norm')
        plt.ylabel('Theta (deg)')
        plt.title('{} | {} | R={:.2f} m | Z={:.2f} m'.
                  format(self.name, port, sl.r_obs, sl.z_obs))
        plt.tight_layout
        self.set_eq(eq_tag_original)

    def point_intensity(self, s=0, t=0, r=7.5, x=0, y=0, z=0):
        """
        Return beam intensity for either s,t,r coords or x,y,z coords.
        """
        # convert x,y,z to s,t,r
        if x or y or z:
            xyz = np.array([x,y,z])
            xyz_source = xyz - self.source
            s = np.inner(xyz_source, self.s_hat)
            t = np.inner(xyz_source, self.t_hat)
            r = np.inner(xyz_source, self.r_hat)
        # calculate intensity using s,t,r coords.
        if self.src_width_2:
            # rectangular source with analytic formula
            a = r * self.tantheta
            s_sq = (2*self.src_width_1) * (2*self.src_width_2)
            intensity = (sp.erf((self.src_width_1-s)/a) + 
                         sp.erf((self.src_width_1+s)/a)) * \
                        (sp.erf((self.src_width_2-t)/a) + 
                         sp.erf((self.src_width_2+t)/a)) / (4*s_sq)
        else:
            # circular source
            r_sq = s**2 + t**2
            a_sq = (r * self.tantheta)**2
            R_sq = self.src_width_1**2
            metric = np.sqrt(4*r_sq*R_sq/(a_sq**2))
            if metric < 0.25:
                # near-axis approximation
                intensity = np.exp(-r_sq/a_sq) / (np.pi*R_sq) * \
                            (1 - np.exp(-R_sq/a_sq) + r_sq/a_sq * 
                            (1-(1+R_sq/a_sq)*np.exp(-R_sq/a_sq)))
            else:
                # precise numerical integration
                def integrand(phip, rp, rr):
                    ret = rp*np.exp(-(rr**2+rp**2-2*rr*rp*np.cos(phip))/a_sq)
                    ret *= 1/(np.pi**2 * a_sq * R_sq)
                    return ret
                intensity,_ = integrate.dblquad(integrand, 0, self.src_width_1, 
                                                0, 2*np.pi, [np.sqrt(r_sq)])
        return intensity
    
    def profile(self, axial_dist=7.5, gridsize=0.8, resolution=0.02):
        sgrid = np.arange(-gridsize, gridsize+1e-4, resolution)
        tgrid = sgrid.copy()
        svalues, tvalues = np.meshgrid(sgrid, tgrid)
        profile2d = np.empty(svalues.shape)
        if self.src_width_2:
            # rectangular source
            for i in np.arange(svalues.size):
                profile2d.flat[i] = self.point_intensity(s=svalues.flat[i], 
                                                         t=tvalues.flat[i],
                                                         r=axial_dist)
        else:
            # circular source
            ssgrid = np.arange(0, gridsize+1e-5, resolution)
            profile_s = np.empty(ssgrid.shape)
            for i,s in enumerate(ssgrid):
                profile_s[i] = self.point_intensity(s=s, r=axial_dist)
            f_interp = interpolate.interp1d(ssgrid, profile_s,
                                            kind='cubic', 
                                            fill_value='extrapolate',
                                            assume_sorted=True)
            profile2d = f_interp(np.sqrt(svalues**2 + tvalues**2))
        # profile should sum to 1
        profile2d *= resolution**2
        assert(np.allclose(profile2d.sum(),1,rtol=1e-3))
        return sgrid, tgrid, profile2d
            
    @staticmethod
    def xyz_to_rpz(xyz):
        if xyz.shape[0] != 3: raise ValueError('{}'.format(xyz.shape))
        rpz = np.empty(xyz.shape)
        rpz[0,...] = np.linalg.norm(xyz[0:2,...], axis=0)
        rpz[1,...] = np.arctan2(xyz[1,...], xyz[0,...])
        rpz[2,...] = xyz[2,...]
        return rpz
    
    @staticmethod
    def rpz_to_xyz(rpz):
        if rpz.shape[0] != 3: raise ValueError('{}'.format(rpz.shape))
        xyz = np.empty(rpz.shape)
        xyz[0,...] = rpz[0,...] * np.cos(rpz[1,...])
        xyz[1,...] = rpz[0,...] * np.sin(rpz[1,...])
        xyz[2,...] = rpz[2,...]
        return xyz
    

class HeatingBeam(_Beam):
    
    def __init__(self, pini=1, *args, **kwargs):
        if pini<1 or pini>8:
            raise ValueError('invalid PINI: {}'.format(pini))
        self.injector = pini
        self.src_width_1 = 0.228/2  # half-width [m]
        self.src_width_2 = 0.506/2  # half-height [m]
        self.divergence = 1.0  # half-angle divergence [deg]
        self.torus_period = 2
        self.set_injector_geometry()
        self.set_pini_geometry()
        super().__init__(*args, **kwargs)

    def set_injector_geometry(self):
        if self.injector <= 4:
            # K20, PINI 1-4
            self.ports = k20_ports
            inj_phi = 56.9059 * np.pi/180
            inj_z = -0.305
            uv_rot = (90 - 36 + 2.9059 + 7.4441) * np.pi/180
        else:
            # K21, PINI 5-8
            self.ports = k21_ports
            inj_phi = 87.0941*np.pi/180
            inj_z = 0.305
            uv_rot = (90 - 36 + 2.9059 + 2*15.0941 - 7.4441) * np.pi/180
        # injector origin
        injector_origin_rpz = np.array([6.75, inj_phi, inj_z])
        self.injector_origin = self.rpz_to_xyz(injector_origin_rpz)
        # u/v unit vectors
        rot_mat = np.array([[np.cos(uv_rot), -np.sin(uv_rot), 0],
                            [np.sin(uv_rot), np.cos(uv_rot), 0],
                            [0, 0, 1]])
        self.u_hat = np.matmul(rot_mat, np.array([1,0,0]))
        self.v_hat = np.matmul(rot_mat, np.array([0,1,0]))
        if self.injector > 4:
            self.v_hat *= -1
        self.w_hat = np.cross(self.u_hat, self.v_hat)
    
    def set_pini_geometry(self):
        self.name = 'PINI {}'.format(self.injector)
        # source coord.
        v = 0.47
        if self.injector%4 in [2,3]:
            v *= -1
        w = 0.6
        if self.injector%4 in [0,3]:
            w *= -1
        self.source = self.injector_origin + 6.5 * self.u_hat + \
                      v * self.v_hat + w * self.w_hat
        self.source_rpz = self.xyz_to_rpz(self.source)
        # source crossing at u=v=0
        w_crossing = 0.0429
        if self.injector%4 in [0,3]:
            w_crossing *= -1
        crossing = self.injector_origin + w_crossing * self.w_hat
        # r-hat is the beam axis, i.e. direction perpendicular to source
        beam_vec = crossing - self.source
        self.r_hat = beam_vec / np.linalg.norm(beam_vec)
        # s-hat = unit vector(z-hat cross r-hat)
        # s-hat is horizontal direction of source extent
        s_direction = np.cross(np.array([0,0,1]), self.r_hat)
        self.s_hat = s_direction / np.linalg.norm(s_direction)
        # t-hat = r-hat cross s-hat
        # t-hat is near-vertical direction of source extent
        t_direction = np.cross(self.r_hat, self.s_hat)
        self.t_hat = t_direction / np.linalg.norm(t_direction)


class RudixBeam(_Beam):
    
    def __init__(self, *args, **kwargs):
        self.name = 'Rudix'
        self.source = np.array([-2.963, -5.648, -0.165])
        self.source_rpz = self.xyz_to_rpz(self.source)
        # r-hat is beam axis, i.e. direction perpendicular to source
        self.r_hat = np.array([0.50139767, 0.80628892, 0.31384478])
        # s-hat = unit vector(z-hat cross r-hat)
        # s-hat is horizontal direction of source extent
        s_direction = np.cross(np.array([0,0,1]), self.r_hat)
        self.s_hat = s_direction / np.linalg.norm(s_direction)
        # t-hat = r-hat cross s-hat
        # t-hat is near-vertical direction of source extent
        t_direction = np.cross(self.r_hat, self.s_hat)
        self.t_hat = t_direction / np.linalg.norm(t_direction)
        self.injector = 99
        self.ports = rudix_ports
        self.src_width_1 = 0.08
        self.src_width_2 = 0
        self.divergence = 0.7
        self.torus_period = 4
        super().__init__(*args, **kwargs)


def test_heating_beams():
    test_origins = np.array([[3.685606, 5.654981, -0.305000],
                             [0.342197, 6.741320, 0.305000]]).transpose()
    test_sources = np.array([[6.075594, 11.717889, 0.295000],
                             [6.922962, 11.310989, 0.295000],
                             [6.922962, 11.310989, -0.905000],
                             [6.075594, 11.717889, -0.905000],
                             [1.972344, 13.051116, -0.295000],
                             [1.047639, 13.219997, -0.295000],
                             [1.047639, 13.219997, 0.905000],
                             [1.972344, 13.051116, 0.905000],
                             ]).transpose()
    test_rhat =    np.array([[-0.365400, -0.926946, -0.085174],
                             [-0.494953, -0.864735, -0.085174],
                             [-0.494953, -0.864735, 0.085174],
                             [-0.365400, -0.926946, 0.085174],
                             [-0.249230, -0.964692, 0.085174],
                             [-0.107854, -0.990511, 0.085174],
                             [-0.107854, -0.990511, -0.085174],
                             [-0.249230, -0.964692, -0.085174],
                             ]).transpose()
    beam = []
    for pini in range(1,9):
        beam.append(HeatingBeam(pini=pini))
    # verify injector coordinate origin (u,v,w)
    for i in [0,4]:
        assert(np.allclose(beam[i].injector_origin, test_origins[:,i//4]))
    for i,b in enumerate(beam):
        # verify source coordinates
        assert(np.allclose(b.source, test_sources[:,i]))
        # verify r unit vector (central beam axis)
        assert(np.allclose(b.r_hat, test_rhat[:,i]))
        # verify r,s,t unit vectors are orthogonal
        assert(np.isclose(np.sum(b.r_hat*b.s_hat)+1,1))
        assert(np.isclose(np.sum(b.t_hat*b.s_hat)+1,1))
        # verify axis point is on r axis
        x2source = b.axis[:,5] - b.source
        x2scomp = np.sum(x2source * b.s_hat)
        x2tcomp = np.sum(x2source * b.t_hat)
        assert(np.isclose(x2scomp+1,1))
        assert(np.isclose(x2tcomp+1,1))
        assert(np.isclose(np.linalg.norm(x2source+b.r_hat),
                          np.linalg.norm(x2source) + np.linalg.norm(b.r_hat)))


class Sightline(object):
    
    def __init__(self, beam, port, r_obs=None, z_obs=None, eq_tag=None):
        if eq_tag is not None:
            beam.set_eq(eq_tag=eq_tag)
        eq_tag = beam.eq_tag
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
                dr,dt = x
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
        port_xyz = beam.ports[port]
        sightline = target - port_xyz
        obs_distance = np.linalg.norm(sightline)
        unitvec_sl = sightline / obs_distance
        step_sl= 0.02
        dist_sl = np.arange(-0.5, 0.5, step_sl) + obs_distance
        # x,y,z coords of sightline near beam axis
        xyz_sl = port_xyz.reshape(3,1) + np.outer(unitvec_sl, dist_sl)
        # check for inside LCFS
        vmec_reff = vmec.service.getReff(eq_tag, 
                                         Points3D(*xyz_sl.tolist()))
        reff = np.array(vmec_reff)
        inplasma = np.isfinite(reff)
        if np.count_nonzero(inplasma) == 0:
            return
        # remove sightline points outside LCFS
        xyz_sl = xyz_sl[:,inplasma]
        dist_sl = dist_sl[inplasma]
        ngrid = dist_sl.size
        vmec_bvector = vmec.service.magneticField(eq_tag, 
                                                  Points3D(*xyz_sl.tolist()))
        bvec_xyz_sl = np.array([vmec_bvector.x1, vmec_bvector.x2, vmec_bvector.x3])
        bnorm_sl = np.linalg.norm(bvec_xyz_sl,axis=0)
        print(bnorm_sl)
        bunit_sl = bvec_xyz_sl / np.tile(bnorm_sl.reshape(1,ngrid), (3,1))
        assert(np.allclose(np.linalg.norm(bunit_sl, axis=0), 1))
        # calc angle wrt B vector along sightline
        dots = np.sum(np.tile(unitvec_sl.reshape(3,1), (1,ngrid)) * bunit_sl, axis=0)
        bangle_sl = np.arccos(dots)
        bangle_sl[np.isnan(bangle_sl)] = np.pi/2
        bangle_sl = np.pi/2-np.abs(bangle_sl-np.pi/2)
        # psi
        rpz_sl = beam.xyz_to_rpz(xyz_sl)
        r_sl = rpz_sl[0,:]
        z_sl = xyz_sl[2,:]
        vmec_stp = vmec.service.toVMECCoordinates(eq_tag, 
                                                  Points3D(*rpz_sl.tolist()),
                                                  1e-4)
        s_sl = np.array(vmec_stp.x1) # norm. flux surface
        theta_sl = np.array(vmec_stp.x2) # poloidal angle [rad]
        phi_sl = np.array(vmec_stp.x3) # toroidal angle [rad]
        # calc beam intensity along sightline
        beam_intensity_sl = np.empty(ngrid)
        for i in np.arange(ngrid):
            beam_intensity_sl[i] = beam.point_intensity(x=xyz_sl[0,i],
                                                   y=xyz_sl[1,i],
                                                   z=xyz_sl[2,i])
        # normalize intensity along sightline
        beam_intensity_sl = beam_intensity_sl / beam_intensity_sl.max()
        # # set quantities along sightline
        # r_weighted_avg = np.average(r_sl, weights=beam_intensity_sl)
        # z_weighted_avg = np.average(z_sl, weights=beam_intensity_sl)
        # fourier components
        vmec_rcoscoeff = vmec.service.getFourierCoefficients(eq_tag, 'RCos')
        vmec_zsincoeff = vmec.service.getFourierCoefficients(eq_tag, 'ZSin')
        nrad = vmec_rcoscoeff.numRadialPoints
        rarray = np.linspace(0,1,nrad) # uniform psinorm grid
        pmodes = np.array(vmec_rcoscoeff.poloidalModeNumbers, dtype=np.int).reshape([-1,1])
        npol = pmodes.size
        tmodes = np.array(vmec_rcoscoeff.toroidalModeNumbers, dtype=np.int).reshape([1,-1])
        ntor = tmodes.size
        rcoeff = np.empty([nrad,npol,ntor])
        zcoeff = np.empty([nrad,npol,ntor])
        for k in np.arange(nrad):
            for m in np.arange(npol):
                for n in np.arange(ntor):
                    icoeff = (m * ntor + n) * nrad + k
                    rcoeff[k,m,n] = vmec_rcoscoeff.coefficients[icoeff]
                    zcoeff[k,m,n] = vmec_zsincoeff.coefficients[icoeff]
        imin = np.argmin(np.abs(rarray-s_sl.min()))-3
        imax = np.argmin(np.abs(rarray-s_sl.max()))+3
        rcoeff_interp = interpolate.interp1d(rarray[imin:imax], 
                                             rcoeff[imin:imax,...], 
                                             axis=0,
                                             kind='cubic',
                                             assume_sorted=True)
        zcoeff_interp = interpolate.interp1d(rarray[imin:imax], 
                                             zcoeff[imin:imax,...], 
                                             axis=0,
                                             kind='cubic',
                                             assume_sorted=True)
        u_sl = theta_sl
        dphidv = 1/5
        v_sl = phi_sl / dphidv
        def eval_vmec(s,u,v, rin, zin):
            rcoeff_local = rcoeff_interp(s)
            zcoeff_local = zcoeff_interp(s)
            phi = v * dphidv
            parray = np.broadcast_to(pmodes, rcoeff_local.shape)
            tarray = np.broadcast_to(tmodes, rcoeff_local.shape)
            operand = parray*u - tarray*v
            cosmn = np.cos(operand)
            sinmn = np.sin(operand)
            r = np.sum(rcoeff_local * cosmn)
            z = np.sum(zcoeff_local * sinmn)
            assert(np.allclose(r, rin))
            assert(np.allclose(z, zin, rtol=5e-5))
            dcosdu = -parray * sinmn
            dcosdv = tarray * sinmn
            dsindu = parray * cosmn
            dsindv = -tarray * cosmn
            eu = np.array([np.sum(rcoeff_local * dcosdu) * np.cos(phi),
                           np.sum(rcoeff_local * dcosdu) * np.sin(phi),
                           np.sum(zcoeff_local * dsindu)])
            evx = np.sum(rcoeff_local*dcosdv)*np.cos(phi) - r*dphidv*np.sin(phi)
            evy = np.sum(rcoeff_local*dcosdv)*np.sin(phi) + r*dphidv*np.cos(phi)
            evz = np.sum(zcoeff_local*dsindv)
            ev = np.array([evx, evy, evz])
            eu = eu / np.linalg.norm(eu)
            ev = ev / np.linalg.norm(ev)
            nvec = -np.cross(eu,ev)
            nvec = nvec / np.linalg.norm(nvec)
            nx = (-(np.sum(rcoeff_local*dcosdu)*np.sum(zcoeff_local*dsindv) -
                  np.sum(rcoeff_local*dcosdv)*np.sum(zcoeff_local*dsindu))*np.sin(phi) + 
                  r*dphidv*np.sum(zcoeff_local * dsindu)*np.cos(phi))
            ny = ((np.sum(rcoeff_local*dcosdu)*np.sum(zcoeff_local*dsindv) -
                  np.sum(rcoeff_local*dcosdv)*np.sum(zcoeff_local*dsindu))*np.cos(phi) +
                  r*dphidv*np.sum(zcoeff_local * dsindu)*np.sin(phi))
            nz = -r*dphidv*np.sum(rcoeff_local*dcosdu)
            nvec2 = np.array([nx,ny,nz])
            nvec2 = nvec2 / np.linalg.norm(nvec2)
            assert(np.allclose(nvec,nvec2))
            return nvec
        nhat = np.empty(bunit_sl.shape)
        bihat = np.empty(bunit_sl.shape)
        cosn = np.empty(ngrid)
        cosbi = np.empty(ngrid)
        for igrid in np.arange(ngrid):
            nhat[:,igrid] = eval_vmec(s_sl[igrid], u_sl[igrid], v_sl[igrid], 
                                      r_sl[igrid], z_sl[igrid])
            assert(np.allclose(np.sum(nhat[:,igrid]*bunit_sl[:,igrid])+1, 1))
            bihat[:,igrid] = np.cross(nhat[:,igrid], bunit_sl[:,igrid])
            assert(np.allclose(np.linalg.norm(bihat[:,igrid]), 1))
            bihat[:,igrid]  = bihat[:,igrid] / np.linalg.norm(bihat[:,igrid])
            cosn[igrid] = np.sum(unitvec_sl*nhat[:,igrid])
            cosbi[igrid] = np.sum(unitvec_sl*bihat[:,igrid])
        self.slhat = unitvec_sl
        self.distance = dist_sl
        self.step = step_sl
        self.xyz = xyz_sl
        self.r = r_sl
        self.z = z_sl
        self.psinorm = s_sl
        self.bangle = bangle_sl *180 / np.pi
        self.intensity = beam_intensity_sl
        self.bhat = bunit_sl
        self.nhat = nhat
        self.bihat = bihat
        self.bnorm = bnorm_sl
        self.cosn = cosn
        self.cosbi = cosbi


if __name__=='__main__':
    test_heating_beams()