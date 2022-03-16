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

try:
    from . import beams
    from . import sightline
except ImportError:
    from w7x_bes_tools import beams
    from w7x_bes_tools import sightline


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
                 load_file=None,
                 ):
        if load_file:
            # grid file specified
            load_file = Path(load_file)
            assert(load_file.exists())
            force_saved_data = True
            identifier='tmp'
        else:
            # grid file not specified
            assert isinstance(beam, beams._Beam)
            if eq_tag is not None and eq_tag!=beam.eq_tag:
                beam.set_eq(eq_tag=eq_tag)
            eq_tag = beam.eq_tag
            if c2c_binormal is None:
                c2c_binormal = c2c_normal
            force_saved_data = False
            identifier = '{:d}{:d}_c2c{:.0f}_P{:d}_{}_R{:.0f}_Z{:.0f}_{}'.format(
                grid_shape[0],
                grid_shape[1],
                c2c_normal*10,
                beam.injector,
                port[0:3],
                np.round(r_obs*1e2), 
                np.round(z_obs*1e2),
                beam.eq_tag,
                )
        # set data file
        if load_file:
            identifier_file = load_file
        else:
            identifier_file = Path('data') / f'grid_{identifier}.pickle'
        # try to open data file
        try:
            with identifier_file.open('rb') as f:
                saved_data = pickle.load(f)
                print(f'Loaded {identifier_file.as_posix()}')
        except:
            print(f'Invalid file {identifier_file.as_posix()}')
            saved_data = {}
        # assemble grid data from specified file, archived file, or inputs
        valid_saved_data = True
        if force_saved_data:
            grid_data = saved_data.copy()
            source = eval(grid_data['beam_name'][-1])
            eq_tag = grid_data['eq_tag']
            beam = beams.HeatingBeam(source=source, eq_tag=eq_tag)
            port = grid_data['port']
            grid_shape = [grid_data['grid_shape_0'],
                          grid_data['grid_shape_0']]
            r_obs = grid_data['r_obs']
            z_obs = grid_data['z_obs']
            c2c_normal = grid_data['c2c_normal']
            c2c_binormal = grid_data['c2c_binormal']
        else:
            # grid data from inputs
            grid_data = {'beam_name':beam.name, 
                           'eq_tag':eq_tag,
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
            for key in grid_data.keys():
                if key in ['sightlines','central_ray']:
                    continue
                if grid_data[key] != saved_data.get(key):
                    valid_saved_data = False
                    break
            if no_saved_data:
                valid_saved_data = False
        if valid_saved_data or force_saved_data:
            print('Using saved sightline grid data')
            self.sightlines = saved_data['sightlines']
            self.central_ray = saved_data['central_ray']
            self.deln_array = c2c_normal/1e2 * np.arange(-(grid_shape[0]-1)/2,
                                                (grid_shape[0]-1)/2*(1+1e-4))
            self.delbi_array = c2c_binormal/1e2 * np.arange((grid_shape[1]-1)/2,
                                               -(grid_shape[1]-1)/2*(1+1e-4), -1)
        else:
            print('Calculating sightline grid data')
            self.central_ray = sightline.Sightline(beam=beam, 
                                               port=port,
                                               r_obs=r_obs,
                                               z_obs=z_obs,
                                               eq_tag=eq_tag)
            grid_data['central_ray'] = self.central_ray
            imax = self.central_ray.imax
            xyz = self.central_ray.xyz[:,imax]
            nhat = self.central_ray.nhat[:,imax]
            bihat = self.central_ray.bihat[:,imax]
            self.deln_array = c2c_normal/1e2 * np.arange(-(grid_shape[0]-1)/2,
                                                (grid_shape[0]-1)/2*(1+1e-4))
            self.delbi_array = c2c_binormal/1e2 * np.arange((grid_shape[1]-1)/2,
                                               -(grid_shape[1]-1)/2*(1+1e-4), -1)
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
                        futures[inorm,ibi] = pool.submit(sightline.Sightline, **kwargs)
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
            grid_data['sightlines'] = self.sightlines
            with identifier_file.open('wb') as f:
                print(f'Saving {identifier_file.as_posix()}')
                pickle.dump(grid_data, f)
        self.eq_tag = beam.eq_tag
        self.beam = beam
        self.port = port
        self.grid_shape = grid_shape
        self.r_obs = r_obs
        self.z_obs = z_obs
        self.identifier = identifier
        def print_ray(ray):
            imax = ray.imax
            print(
                f'    {ray.r[imax]:.3f}, '
                f'{ray.z[imax]:.3f}, '
                f'{ray.phi[imax]*(180/np.pi):.2f}, '
                f'{ray.xyz[0,imax]:.3f}, '
                f'{ray.xyz[1,imax]:.3f}'
            )
        print('Sightline grid:  R (m), Z(m), phi (deg), X (m), Y (m)')
        print('  Central ray')
        print_ray(self.central_ray)
        print('  Grid corners')
        for ray in [self.sightlines[0,0], 
                    self.sightlines[-1,0], 
                    self.sightlines[0,-1], 
                    self.sightlines[-1,-1]]:
            print_ray(ray)

    
    def plot(self, save=False):
        plt.figure(figsize=(4.5*3,3.3*2))
        ax1, ax2 = self.beam.plot_beam_plane(port=self.port, sp1=234, sp2=235)
        for inorm in np.arange(self.grid_shape[0]):
            for ibi in np.arange(self.grid_shape[1]):
                sl = self.sightlines[inorm,ibi]
                for ax,color in zip([ax1,ax2],['r','b']):
                    plt.sca(ax)
                    plt.fill(sl.rseq, sl.zseq, color=color)
        ax1 = plt.subplot(231)
        ax2 = plt.subplot(233)
        ax3 = plt.subplot(232)
        max_bi_excursion = 0
        max_rad_excursion = 0
        # max_psi = 0
        # min_psi = 1
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
        max_rad_excursion = np.max([sl.norm_half_excursion*1e2*2 for sl in self.sightlines.flat])
        max_bi_excursion = np.max([sl.binorm_half_excursion*1e2*2 for sl in self.sightlines.flat])
        print(f'Max rad/binorm excursion (cm): {max_rad_excursion:.2f} {max_bi_excursion:.2f}')
        min_psi = np.min([sl.psinorm[sl.imax] for sl in self.sightlines.flat])
        max_psi = np.max([sl.psinorm[sl.imax] for sl in self.sightlines.flat])
        max_r = np.max([sl.r[sl.imax] for sl in self.sightlines.flat])
        min_r = np.min([sl.r[sl.imax] for sl in self.sightlines.flat])
        print(f'Min/max psinorm: {min_psi:.2f} {max_psi:.2f}')
        print(f"Min/max R (m): {min_r:.2f} {max_r:.2f}")
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
        plt.xlabel('Radial excursion (cm)')
        plt.ylabel('Binormal excursion (cm)')
        plt.annotate('Smaller sightline excursions are better', 
            (0.05,0.03), 
            xycoords='axes fraction',
            fontsize='small')
        plt.legend()
        plt.tight_layout()
        if save:
            fname = Path('plots') / f'grid_{self.identifier}.pdf'
            print(f'Saving {fname.as_posix()}')
            plt.savefig(fname.as_posix(), transparent=True)
        

if __name__=='__main__':
    plt.close('all')
    # grid = Grid(beam=beams.HeatingBeam(pini=6),
    #             port='A21-hihi',
    #             r_obs=5.82, 
    #             z_obs=0.43,
    #             c2c_normal=1.5,
    #             eq_tag = 'w7x_ref_29')
    # grid.plot(save=True)
    grid = Grid(beam=beams.HeatingBeam(source=7),
                port='W30',
                r_obs=5.96, 
                z_obs=0.22,
                c2c_normal=1,
                eq_tag='w7x_ref_29')
    # grid = Grid(load_file="data/grid_88_c2c10_P7_W30_R599_Z15_w7x_ref_29.pickle")
    grid.plot(save=False)
    plt.show()
# grid_88_c2c10_P7_W30_R596_Z22_w7x_ref_29