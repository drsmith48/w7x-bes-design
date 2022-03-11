#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:39:00 2020

@author: drsmith
"""

from pathlib import Path
import re
import csv
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from pyfidasim import load_dict

try:
    from .utilities.filters import make_filters
    from . import beams
    from .sightline_grid import Grid
except ImportError:
    from w7x_bes_tools.utilities.filters import make_filters
    from w7x_bes_tools import beams
    from w7x_bes_tools.sightline_grid import Grid


class Fida(object):
    
    def __init__(self, simdir=None):
        self.workdir = Path('data') / 'FIDASIM'
        # set fidasim results directory
        if simdir:
            if Path(simdir).exists():
                simdir = Path(simdir)
            else:
                simdir = self.workdir/simdir
        else:
            simdir = self.workdir / 'A21_HIHI_P6'
        assert(simdir.exists())
        print('FIDASIM results: {}'.format(simdir.as_posix()))
        self.simname = simdir.name
        self.beam = beams.HeatingBeam(source=int(self.simname[-1]), 
                                      eq_tag='w7x_ref_29')
        # load nbi/halo spectra file
        self.spec = load_dict(simdir/'spec.hdf5', verbose=False)
        self.grid3d = load_dict(simdir/'grid3d.hdf5', verbose=False)
        self.nlos = self.spec['nlos']
        self.lambda_array = self.spec['wavel']
        self.spectra = np.moveaxis(self.spec['intens'], 0, -1)
        self.losnames = [value.decode() for value in self.spec['losname']]
        
        self.lambda_resolution = self.lambda_array[1] - self.lambda_array[0]
        self.spectra_raw = self.spectra.copy()
        self.calc_radiance()
        
    def apply_filter(self, ifilter=-1, edge=657):
        if ifilter==-1:
            filter_array = np.ones(self.lambda_array.shape)
        else:
            filters = make_filters()
            f = filters[ifilter]
            shifted_wl = f['wl'] - f['edgewl'] + edge
            filter_interp = interp1d(shifted_wl, f['t'], 
                                     kind='cubic',
                                     fill_value=(f['t'][0], f['t'][-1]),
                                     bounds_error=False,
                                     assume_sorted=True)
            filter_array = filter_interp(self.lambda_array)
        self.spectra = self.spectra_raw * np.broadcast_to(filter_array.reshape((1,-1,1)), 
                                                          self.spectra_raw.shape)
        self.calc_radiance()
        
    def calc_radiance(self):
        self.radiance = np.sum(self.spectra, axis=1) * self.lambda_resolution
    
    def los_names(self):
        print('Lines of sight:')
        for i, los in enumerate(self.losnames):
            print('  {:2d}: {}'.format(i,los))
            
    def los_filter(self, tag):
        ilos=[]
        tag_re = re.compile(tag)
        for iname, name in enumerate(self.losnames):
            if tag_re.search(name):
                ilos.append(iname)
        if not ilos:
            print(tag)
        return np.array(ilos, dtype=int)
        
            
    def plot_spectra(self, 
                     ilos=0, 
                     ax=None, 
                     plot_all=False, 
                     save=False,
                     plot_filters=False,
                     legend_radiance=True):
        if not isinstance(ilos, np.ndarray):
            if not isinstance(ilos, (list, tuple)):
                ilos= [ilos]
            ilos = np.array(ilos, dtype=np.int)
        if plot_all:
            ilos = np.arange(self.nlos)
        if not ax:
            ncol = ilos.size
            if ncol >3:
                ncol = 3
            nrow = ilos.size // ncol
            if ilos.size % ncol:
                nrow += 1
            plt.figure(figsize=[4.5*ncol,2.7*nrow])
        if plot_filters:
            with open('Alluxa-PBvsAOI.csv') as csvfile:
                reader = csv.DictReader(csvfile, 
                                        fieldnames=['Wavelength', 'T0deg',
                                                    'T5deg', 'T10deg'],
                                        delimiter=',')
                wl, t0, t5 = [], [], []
                for irow, row in enumerate(reader):
                    if irow==0:
                        continue
                    if irow==1:
                        print(row)
                    wl.append(float(row['Wavelength']))
                    t0.append(float(row['T0deg']))
                    t5.append(float(row['T5deg']))
            wl = np.array(wl)
            t0 = np.array(t0)/1e2
            t5 = np.array(t5)/1e2
        for iplot, i in enumerate(ilos):
            # print('Plotting LOS {}: {}'.format(i, self.losnames[i]))
            if ax:
                plt.sca(ax)
            else:
                plt.subplot(nrow, ncol, iplot+1)
            if legend_radiance:
                txt = ['Full ({:.2g} Ph/s/m2/st)'.format(self.radiance[i,0]),
                        'Half ({:.2g} Ph/s/m2/st)'.format(self.radiance[i,1]),
                        'Third ({:.2g} Ph/s/m2/st)'.format(self.radiance[i,2]),
                        'Th CX ({:.2g} Ph/s/m2/st)'.format(self.radiance[i,3]),
                        # 'aFIDA ({:.2g} Ph/m2/st)'.format(self.radiance[i,4]),
                        # 'pFIDA ({:.2g} Ph/m2/st)'.format(self.radiance[i,5]),
                        ]
            else:
                txt = ['Full','Half','Third','Th CX']
            plt.plot(self.lambda_array, self.spectra[i,:,:], label=txt)
            ymax = 2e18
            if plot_filters:
                plt.plot(wl, t0 / t0.max() * ymax, '--', label='0AOI')
                plt.plot(wl, t5 / t5.max() * ymax, '--', label='5AOI')
            plt.legend(loc='upper right',
                       borderpad=0.3,
                       labelspacing=0.2,
                       handlelength=1.0,
                       handletextpad=0.4,
                       fontsize='medium')
            plt.xlabel('Wavelength (nm)')
            plt.xlim(653,658)
            plt.ylim(0,ymax)
            plt.ylabel('Spect. radiance (Ph/s/m2/st/nm)')
            plt.title(self.losnames[i])
        plt.tight_layout()
        if save:
            fname = Path('plots') / f'{self.losnames[ilos[0]]}.pdf'
            print(f'Saving {fname.as_posix()}')
            plt.savefig(fname.as_posix(), transparent=True)
                    
    def plot_radial_array(self, z=20, passband=[657.5,661], save=False):
        ilos = self.los_filter(tag=f'Z{z:d}')
        if not isinstance(ilos, np.ndarray):
            if not isinstance(ilos, (list, tuple)):
                ilos= [ilos]
            ilos = np.array(ilos, dtype=np.int)
        passband = np.array(passband)
        plt.figure(figsize=[8,3])
        for i in [0,1]:
            if i==0:
                ilam_min = 0
                ilam_max = -1
            else:
                ilam_min = np.argmin(np.abs(self.lambda_array-passband[0]))
                ilam_max = np.argmin(np.abs(self.lambda_array-passband[1]))
            beam_radiance = np.sum(self.spectra[ilos,ilam_min:ilam_max,0:3], axis=(1,2)) * self.lambda_resolution
            halo_radiance = np.sum(self.spectra[ilos,ilam_min:ilam_max,3], axis=1) * self.lambda_resolution
            radii = [eval(self.losnames[i][-7:-4]) for i in ilos]
            plt.subplot(1,2,i+1)
            plt.plot(radii, beam_radiance, label='Beam')
            plt.plot(radii, halo_radiance, label='Th CX')
            plt.annotate(f'Src {self.beam.injector}, Z = {z} cm', [0.05,0.9], xycoords='axes fraction')
            # idx = np.arange(beam_radiance.size)
            # width=0.25
            # plt.bar(idx-width/2, beam_radiance, width, label='Beam')
            # plt.bar(idx+width/2, halo_radiance, width, label='Th CX')
            # plt.xticks(ticks=idx, labels=radii)
            plt.ylabel('Radiance (Ph/s/m^2/ster)')
            plt.xlabel('Radius (cm)')
            if i==0:
                plt.title('Full spectrum')
            else:
                plt.title(f'Passband {passband[0]}-{passband[1]} nm')
                plt.ylim(0,1.4e18)
            # plt.yscale('log')
            plt.legend(fontsize='small', loc='upper right')
        plt.tight_layout()
        if save:
            fname = Path('plots') / f'fida_array_{self.losnames[ilos[0]]}.pdf'
            print(f'Saving {fname.as_posix()}')
            plt.savefig(fname.as_posix(), transparent=True)
            
    def plot_beam_plane(self, 
                          gridfile=None, 
                          passband=[653,655.3], 
                          marker=None, 
                          save=False,
                          zoom=False,
                          print_xyz=False):
        vint = self.beam.vint
        r_array = self.grid3d['R_c']/1e2
        z_array = self.grid3d['Z_c']/1e2
        density_rz = self.grid3d['density'].sum(axis=(1,-1)) * self.grid3d['dphi']
        density_rz = np.moveaxis(density_rz, 0, -1)
        plt.figure(figsize=[3*4,2*3])
        int_levels = np.array([0.7,0.8,0.9])*vint['int_values'].max()
        psi_levels = [0.2,0.4,0.6,0.8]
        def contours(title=None, 
                     data=None, 
                     r_data=r_array,
                     z_data=z_array,
                     zoom=False,
                     clabel='Density (1/cm**3)'):
            plt.contour(vint['rmaj_values'], 
                        vint['z_values'], 
                        vint['psi_values'], 
                        colors='k',
                        levels=psi_levels)
            if zoom:
                xlim = [5.8,6.05]
                ylim = [0.11,0.35]
            else:
                xlim = plt.gca().get_xlim()
                ylim = plt.gca().get_ylim()
            plt.plot(vint['lcfs_rz'][0], vint['lcfs_rz'][1], color='k')
            plt.gca().set_aspect('equal')
            plt.contourf(r_data, z_data, data, 
                         cmap='afmhot_r',
                         zorder=0,
                         vmin=0.1*data.max(),
                         vmax=data.max())
            # plt.clim(vmin=0.05*data.max(), vmax=data.max())
            cb = plt.colorbar()
            cb.ax.set_ylabel(clabel, rotation=-90, va='bottom')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.contour(vint['rmaj_values'], 
                        vint['z_values'], 
                        vint['int_values'], 
                        colors='grey', 
                        levels=int_levels)
            plt.xlabel('R (m)')
            plt.ylabel('Z (m)')
            plt.title(title)
        # contour plot beam density
        contour_axes = []
        contour_axes.append(plt.subplot(231))
        data = np.sum(density_rz[...,0:3].T, axis=0)
        contours(data=data,
                 title=f'Beam density | {self.simname}',
                 zoom=zoom,
                 clabel='Density (1/cm**3)',
                 r_data=r_array,
                 z_data=z_array)
        # contour plot halo density
        contour_axes.append(plt.subplot(232))
        data = density_rz[...,-1].T
        contours(data=data,
                 title=f'Halo density | {self.simname}',
                 zoom=zoom,
                 clabel='Density (1/cm**3)',
                 r_data=r_array,
                 z_data=z_array)
        # plot radial profile of state densities
        plt.subplot(233)
        plt.plot(r_array, density_rz.sum(axis=1)*self.grid3d['dZ'])
        plt.legend(['full', 'half', 'third', 'halo'])
        plt.xlabel('Radius (m)')
        plt.ylabel('Integ. density (1/cm)')
        plt.title(f'State densities | {self.simname}')
        # contour plot beam emission and halo emission
        rarray = np.arange(580, 606.1, 2)
        zarray = np.arange(10, 30.1, 2)
        beam_radiance = np.empty((rarray.size, zarray.size))
        halo_radiance = np.empty((rarray.size, zarray.size))
        ilam_min = np.argmin(np.abs(self.lambda_array-passband[0]))
        ilam_max = np.argmin(np.abs(self.lambda_array-passband[1]))
        for r_ind, r in enumerate(rarray):
            for z_ind, z in enumerate(zarray):
                ilos = self.los_filter(f'R{r:.0f}_Z{z:.0f}')
                beam_radiance[r_ind, z_ind] = \
                    np.sum(self.spectra[ilos[0],ilam_min:ilam_max,0:3]) \
                    * self.lambda_resolution
                halo_radiance[r_ind, z_ind] = \
                    np.sum(self.spectra[ilos[0],ilam_min:ilam_max,3]) \
                    * self.lambda_resolution
        contour_axes.append(plt.subplot(234))
        contours(data=beam_radiance.T,
                 title=f'Beam radiance | {self.simname}',
                 r_data=rarray/1e2, 
                 z_data=zarray/1e2,
                 zoom=zoom,
                 clabel='Radiance (ph/s/m**2/ster)')
        contour_axes.append(plt.subplot(235))
        contours(data=halo_radiance.T,
                 title=f'Halo radiance | {self.simname}',
                 r_data=rarray/1e2, 
                 z_data=zarray/1e2,
                 zoom=zoom,
                 clabel='Radiance (ph/s/m**2/ster)')
        # plot field alignment in beam plane
        ax1, _ = self.beam.plot_beam_plane(port='W30', 
                                           sp1=236, 
                                           sp2=None,
                                           zoom=zoom)
        contour_axes.append(ax1)
        # plot marker
        if marker:
            r, z = marker
            for ax in contour_axes:
                plt.sca(ax)
                plt.plot(r/1e2, 
                         z/1e2, 
                         color='b', 
                         markersize=4, 
                         marker='D',
                         zorder=0)
        # plot grid
        if gridfile:
            gridfile = Path(gridfile)
            print(f'Using gridfile {gridfile.as_posix()}')
            grid = Grid(load_file=gridfile)
            for ax in contour_axes:
                plt.sca(ax)
                for ibi in np.arange(grid.grid_shape[1]):
                    for inorm in np.arange(grid.grid_shape[0]):
                        sl = grid.sightlines[inorm,ibi]
                        plt.fill(np.array(sl.rseq), 
                                 np.array(sl.zseq), 
                                 color='b',
                                 zorder=0)
                        if print_xyz and ax is contour_axes[0]:
                            if ibi==0 and inorm==0:
                                print('Row, column, x (m), y (m), z (m)')
                            sl_x, sl_y, sl_z = sl.xyz[:,sl.imaxbeam]
                            print(f'{ibi}, {inorm}, {sl_x:.3f}, {sl_y:.3f}, {sl_z:.3f}')
        plt.tight_layout()
        if save:
            fname = 'fida_beam'
            if gridfile:
                fname = fname + f'_{gridfile.stem}'
            if zoom:
                fname = fname + '_zoom'
            fname = fname + '.pdf'
            fname = Path('plots') / fname
            print(f'Saving {fname.as_posix()}')
            plt.savefig(fname.as_posix(), transparent=True)
        
            
def plot_filter_comparison(ifilter=0, save=False):
    f = Fida(simdir='A21_P2')
    los = [3,21,39]
    plt.figure(figsize=(13,7))
    for icol,ilos in enumerate(los):
        ax = plt.subplot(2,3,icol+1)
        f.plot(ilos=ilos, ax=ax)
        plt.sca(ax)
        plt.annotate('No filter', (0.65, 0.9), 
                     xycoords='axes fraction', 
                     size='small')
    f.apply_filter(ifilter=0, edge=657.4)
    for icol,ilos in enumerate(los):
        ax = plt.subplot(2,3,icol+1+3)
        f.plot(ilos=ilos, ax=ax)
        plt.annotate('OD6 Ultra Narrow', (0.65, 0.9), 
                     xycoords='axes fraction', 
                     size='small')
    if save:
        fname = Path('plots') / 'fida_filter.pdf'
        print(f'Saving {fname.as_posix()}')
        plt.savefig(fname.as_posix(), transparent=True)
    
            

if __name__ == '__main__':
    plt.close('all')
    
    # A21 HIHI
    # f = Fida(simdir='A21_HIHI_P6')
    # f.plot_beam_density(gridfile='data/grid_88_c2c15_P6_A21_R582_Z43_w7x_ref_29.pickle')
    # ilos = f.los_filter(tag='Z42')
    # f.plot(ilos=ilos[::2], save=False)
    # f.plot_array(ilos=ilos, passband=[657.5,661], save=False)
    
    # W30
    f = Fida(simdir='W30_P7')

    # f.plot_beam_density(save=True)
    
    passband = [653,655.3]
    gridfile = 'data/grid_88_c2c10_P7_W30_R596_Z22_w7x_ref_29.pickle'
    grid = Grid(load_file=gridfile)
    f.plot_beam_plane(gridfile=gridfile, 
                      passband=passband, 
                      zoom=True,
                      save=True,
                      print_xyz=True)

    # ilos = f.los_filter('R(59.|600)_Z22')
    # f.plot_spectra(ilos=ilos, save=True, plot_filters=True, legend_radiance=False)
    
    # f.plot_radial_array(z=18, passband=passband, save=True)
    # f.plot_radial_array(z=22, passband=passband, save=True)
    # f.plot_radial_array(z=26, passband=passband, save=True)

    plt.show()
    