#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 13:16:01 2020

@author: drsmith
"""

from pathlib import Path
import concurrent.futures
import pickle
import numpy as np
import matplotlib.pyplot as plt
import beams


class Grid(object):
    
    def __init__(self, 
                 beam=None, 
                 port='A21-lolo',
                 r_obs=5.88, 
                 z_obs=-0.39, 
                 eq_tag=None,
                 grid_shape=[8,8],  # channel layout, radial x poloidal
                 c2c_normal=1.5,  # radial-like center-to-center distance (cm)
                 c2c_binormal=None,  # poloidal-like center-to-center distance (cm)
                 no_saved_data=False, # force calculation of sightline grid data
                 ):
        if not beam:
            beam = beams.HeatingBeam(pini=2, eq_tag=eq_tag)
        assert(isinstance(beam, beams._Beam))
        if eq_tag is not None and eq_tag!=beam.eq_tag:
            beam.set_eq(eq_tag=eq_tag)
        self.eq_tag = beam.eq_tag
        self.beam = beam
        self.port = port
        self.grid_shape = grid_shape
        self.r_obs = r_obs
        self.z_obs = z_obs
        if c2c_binormal is None:
            c2c_binormal = c2c_normal
        self.identifier = '{:d}{:d}_c2c{:.0f}_P{:d}_{}_R{:.0f}_Z{:.0f}_{}'.format(
            self.grid_shape[0],
            self.grid_shape[1],
            c2c_normal*10,
            self.beam.injector,
            self.port[0:3],
            np.round(self.r_obs*1e2), 
            np.round(self.z_obs*1e2),
            self.eq_tag)
        pickle_file = Path('data') / f'grid_{self.identifier}.pickle'
        try:
            with pickle_file.open('rb') as f:
                saved_data = pickle.load(f)
                print(f'Loaded {pickle_file.as_posix()}')
        except:
            print(f'Invalid file {pickle_file.as_posix()}')
            saved_data = {}
        pickle_data = {'beam_name':beam.name, 
                       'eq_tag':self.eq_tag,
                       'port':port, 
                       'grid_shape_0':grid_shape[0], 
                       'grid_shape_1':grid_shape[1], 
                       'r_obs':r_obs, 
                       'z_obs':z_obs,
                       'c2c_normal':c2c_normal,
                       'c2c_binormal':c2c_binormal,
                       'sightlines':None,
                       'central_ray':None}
        # test saved sightline grid
        valid_saved_data = True
        for key in pickle_data.keys():
            if key in ['sightlines','central_ray']:
                continue
            if pickle_data[key] != saved_data.get(key):
                valid_saved_data = False
                break
        if no_saved_data:
            valid_saved_data = False
        self.deln_array = c2c_normal/1e2 * np.arange(-(grid_shape[0]-1)/2,
                                                (grid_shape[0]-1)/2*(1+1e-4))
        self.delbi_array = c2c_binormal/1e2 * np.arange((grid_shape[1]-1)/2,
                                                   -(grid_shape[1]-1)/2*(1+1e-4), -1)
        if valid_saved_data:
            print('Using saved sightline grid data')
            self.sightlines = saved_data['sightlines']
            self.central_ray = saved_data['central_ray']
        else:
            print('Calculating sightline grid data')
            self.central_ray = beams.Sightline(beam=beam, 
                                               port=port,
                                               r_obs=r_obs,
                                               z_obs=z_obs,
                                               eq_tag=self.eq_tag)
            pickle_data['central_ray'] = self.central_ray
            imax = self.central_ray.imax
            xyz = self.central_ray.xyz[:,imax]
            nhat = self.central_ray.nhat[:,imax]
            bihat = self.central_ray.bihat[:,imax]
            rgrid = np.empty(grid_shape)
            zgrid = np.empty(grid_shape)
            futures = np.empty(grid_shape, dtype=object)
            self.sightlines = np.empty(grid_shape, dtype=object)
            # fill sightline grid
            with concurrent.futures.ThreadPoolExecutor(4) as pool:
                for inorm in np.arange(rgrid.shape[0]-1,-1,-1):
                    for ibi in np.arange(rgrid.shape[1]-1,-1,-1):
                        xyz_disp = xyz + self.deln_array[inorm]*nhat + \
                            self.delbi_array[ibi]*bihat
                        rgrid[inorm,ibi] = np.sqrt(xyz_disp[0]**2 + xyz_disp[1]**2)
                        zgrid[inorm,ibi] = xyz_disp[2]
                        kwargs = {'beam':beam, 
                                  'port':port,
                                  'r_obs':rgrid[inorm,ibi],
                                  'z_obs':zgrid[inorm,ibi]}
                        futures[inorm,ibi] = pool.submit(beams.Sightline, **kwargs)
                for future in concurrent.futures.as_completed(futures.flatten().tolist()):
                    if future.exception():
                        raise ValueError
                    sl = future.result()
                    # print(sl.r_obs, sl.z_obs, sl.psinorm[sl.imax])
                    if sl.psinorm[sl.imax]>1.0:
                        raise ValueError
            for inorm in np.arange(grid_shape[0]):
                for ibi in np.arange(grid_shape[1]):
                    self.sightlines[inorm,ibi] = futures[inorm,ibi].result()
            pickle_data['sightlines'] = self.sightlines
            with pickle_file.open('wb') as f:
                print(f'Saving {pickle_file.as_posix()}')
                pickle.dump(pickle_data, f)
        # print('Center ray R, Z: {:.2f} {:.2f} cm'.format(self.central_ray.r_avg*1e2,
        #                                                  self.central_ray.z_avg*1e2))
        # for ir in [0,grid_shape[0]-1]:
        #     for iz in [0,grid_shape[1]-1]:
        #         print('Corner ray R, Z: {:.2f} {:.2f} cm'.format(self.sightlines[ir,iz].r_avg*1e2,
        #                                                          self.sightlines[ir,iz].z_avg*1e2))
        # for loc,rhoi in zip(['Core','Edge'], [0.4,0.1]):
        #     print('{} @ rho_i = {:.1f} mm'.format(loc, rhoi*10))
        #     krhoi = (1/2) * 2*np.pi / (c2c_normal*np.array([grid_shape[0],1])) * rhoi
        #     print('  k_rad * rho_i min/max = {:.2f} {:.2f}'.format(*krhoi.tolist()))
        #     krhoi = (1/2) * 2*np.pi / (c2c_binormal*np.array([grid_shape[1],1])) * rhoi
        #     print('  k_pol * rho_i min/max = {:.2f} {:.2f}'.format(*krhoi.tolist()))
    
    def plot(self, save=False):
        plt.figure(figsize=(14.2,7.2))
        ax1, ax2 = self.beam.plot_vertical_plane(port=self.port, sp1=234, sp2=235)
        for inorm in np.arange(self.grid_shape[0]):
            for ibi in np.arange(self.grid_shape[1]):
                sl = self.sightlines[inorm,ibi]
                for ax,color in zip([ax1,ax2],['r','b']):
                    plt.sca(ax)
                    plt.fill(sl.rseq, sl.zseq, color=color)
        ax1 = plt.subplot(231)
        ax2 = plt.subplot(233)
        ax3 = plt.subplot(232)
        for inorm in np.arange(self.grid_shape[0]):
            for ibi in np.arange(self.grid_shape[1]):
                sl = self.sightlines[inorm,ibi]
                plt.sca(ax1)
                d = 0.03
                if inorm%2!=0:
                    d = -d
                plt.errorbar(self.deln_array[inorm]*1e2,
                             self.delbi_array[ibi]*1e2+d,
                             xerr=sl.norm_half_excursion*1e2,
                             yerr=sl.binorm_half_excursion*1e2,
                             # fmt='k',
                             capsize=2)
                plt.sca(ax2)
                plt.plot(sl.psinorm[sl.imax], 
                         sl.theta[sl.imax]*180/np.pi,
                         'x')
                plt.sca(ax3)
                plt.plot(sl.norm_half_excursion*1e2*2, 
                         sl.binorm_half_excursion*1e2*2, 
                         'x')
        plt.sca(ax1)
        plt.xlim(-8,8)
        plt.ylim(-8,8)
        ax1.set_aspect('equal')
        plt.title('Beam-weighted ray localization')
        plt.xlabel('Radial distance (cm)')
        plt.ylabel('Binormal distance (cm)')
        plt.sca(ax2)
        plt.title('Beam-weighted ray locations')
        plt.xlabel('Psi-norm')
        plt.ylabel('Theta (deg)')
        plt.sca(ax3)
        ax3.set_aspect('equal')
        norm_excursion = 2*np.array([sl.norm_half_excursion for sl in self.sightlines.flatten().tolist()])
        binorm_excursion = 2*np.array([sl.binorm_half_excursion for sl in self.sightlines.flatten().tolist()])
        plt.plot(norm_excursion.mean()*1e2, 
                 binorm_excursion.mean()*1e2,
                 'ks', label='Grid avg.')
        plt.xlim(0,3)
        plt.ylim(0,3)
        plt.title('Beam-weighted ray localization')
        plt.xlabel('Radial localization (cm)')
        plt.ylabel('Binormal localization (cm)')
        plt.legend()
        plt.tight_layout()
        if save:
            fname = Path('plots') / f'grid_{self.identifier}.pdf'
            print(f'Saving {fname.as_posix()}')
            plt.savefig(fname.as_posix(), transparent=True)
        

if __name__=='__main__':
    plt.close('all')
    # grid = Grid(beam=beams.HeatingBeam(pini=2),
    #             port='A21-lolo',
    #             r_obs=5.85, 
    #             z_obs=-0.42,
    #             c2c_normal=1.5)
    # grid.plot(save=True)
    # grid = Grid(beam=beams.HeatingBeam(pini=2),
    #             port='A21-lolo',
    #             r_obs=5.82, 
    #             z_obs=-0.43,
    #             c2c_normal=1.5,
    #             eq_tag = 'w7x_ref_29')
    # grid.plot(save=True)
    # grid = Grid(beam=beams.HeatingBeam(pini=3),
    #             port='W11',
    #             r_obs=5.95, 
    #             z_obs=-0.17,
    #             c2c_normal=1.5,
    #             eq_tag='w7x_ref_29')
    # grid.plot(save=True)
    # grid = Grid(beam=beams.HeatingBeam(pini=4),
    #             port='W11',
    #             r_obs=5.98, 
    #             z_obs=-0.16,
    #             c2c_normal=1.5,
    #             eq_tag='w7x_ref_29')
    # grid.plot(save=True)
    # grid = Grid(beam=beams.HeatingBeam(pini=6),
    #             port='A21-hihi',
    #             r_obs=5.82, 
    #             z_obs=0.43,
    #             c2c_normal=1.5,
    #             eq_tag = 'w7x_ref_29')
    # grid.plot(save=True)
    grid = Grid(beam=beams.HeatingBeam(pini=7),
                port='W30',
                r_obs=5.99, 
                z_obs=0.15,
                c2c_normal=1,
                eq_tag='w7x_ref_29')
    grid.plot(save=True)
