#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:39:00 2020

@author: drsmith
"""

from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from filters import make_filters
from pyfidasim import load_dict


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
            simdir = self.workdir / 'A21_HIHI_P6_Z46'
        assert(simdir.exists())
        print('FIDASIM results: {}'.format(simdir.as_posix()))
        
        # load nbi/halo spectra file
        specfile = simdir / 'spec.hdf5'
        assert(specfile.exists())
        spec = load_dict(specfile)
        self.nlos = spec['nlos']
        self.lambda_array = spec['wavel']
        self.spectra = np.moveaxis(spec['intens'], 0, -1)
        self.losnames = [value.decode() for value in spec['losname']]
        
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
            
    def plot(self, ilos=0, ax=None, plot_all=False, save=False):
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
            plt.figure(figsize=[4.5*ncol,3.3*nrow])
        for iplot, i in enumerate(ilos):
            print('Plotting LOS {}: {}'.format(i, self.losnames[i]))
            if ax:
                plt.sca(ax)
            else:
                plt.subplot(nrow, ncol, iplot+1)
            plt.plot(self.lambda_array, self.spectra[i,:,:])
            plt.legend(['Full ({:.2g} Ph/s/m2/st)'.format(self.radiance[i,0]),
                        'Half ({:.2g} Ph/s/m2/st)'.format(self.radiance[i,1]),
                        'Third ({:.2g} Ph/s/m2/st)'.format(self.radiance[i,2]),
                        'Th CX ({:.2g} Ph/s/m2/st)'.format(self.radiance[i,3]),
                        # 'aFIDA ({:.2g} Ph/m2/st)'.format(self.radiance[i,4]),
                        # 'pFIDA ({:.2g} Ph/m2/st)'.format(self.radiance[i,5]),
                        ],
                       loc='upper left',
                       borderpad=0.3,
                       labelspacing=0.2,
                       handlelength=1.0,
                       handletextpad=0.4,
                       fontsize='medium')
            plt.xlabel('Wavelength (nm)')
            plt.xlim(652,662)
            plt.ylim(0,2e18)
            plt.ylabel('Spect. radiance (Ph/s/m2/st/nm)')
            plt.title(self.losnames[i])
        plt.tight_layout()
        if save:
            fname = Path('plots') / f'{self.losnames[ilos[0]]}.pdf'
            print(f'Saving {fname.as_posix()}')
            plt.savefig(fname.as_posix(), transparent=True)
                    
    def plot_array(self, ilos=None, passband=[657.5,661], save=False):
        if not isinstance(ilos, np.ndarray):
            if not isinstance(ilos, (list, tuple)):
                ilos= [ilos]
            ilos = np.array(ilos, dtype=np.int)
        passband = np.array(passband)
        plt.figure(figsize=[10.25,4])
        for i in [0,1]:
            if i==0:
                ilam_min = 0
                ilam_max = -1
            else:
                ilam_min = np.argmin(np.abs(self.lambda_array-passband[0]))
                ilam_max = np.argmin(np.abs(self.lambda_array-passband[1]))
            beam_radiance = np.sum(self.spectra[ilos,ilam_min:ilam_max,0:3], axis=(1,2)) * self.lambda_resolution
            halo_radiance = np.sum(self.spectra[ilos,ilam_min:ilam_max,3], axis=1) * self.lambda_resolution
            losnames = [self.losnames[i] for i in ilos]
            idx = np.arange(beam_radiance.size)
            width=0.25
            plt.subplot(1,2,i+1)
            plt.bar(idx-width/2, beam_radiance, width, label='Beam')
            plt.bar(idx+width/2, halo_radiance, width, label='Th CX')
            plt.xticks(ticks=idx, labels=losnames, size='small',
                       rotation=-45, ha='left', va='top')
            plt.ylabel('Radiance (Ph/s/m^2/ster)')
            if i==0:
                plt.title('Full spectrum')
            else:
                plt.title(f'Passband {passband[0]}-{passband[1]} nm')
            plt.yscale('log')
            plt.ylim([1e15,None])
            plt.legend(fontsize='small')
        plt.tight_layout()
        if save:
            fname = Path('plots') / f'fida_array_{losnames[0]}.pdf'
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
    f = Fida(simdir='A21_HIHI_P6')
    # f = Fida(simdir='W30_P7')
    ilos=[]
    for iname,name in enumerate(f.losnames):
        # if 'Z18' in name:
        if 'Z42' in name:
            ilos.append(iname)
    f.plot(ilos=ilos, save=True)
    # f.plot_array(ilos=ilos, save=True, passband=[653,655.3])
    f.plot_array(ilos=ilos, save=True)
    # ilos = np.arange(0,42,6)
    # f.plot(ilos=ilos+3, save=True)
    # f.plot(ilos=3)
    # f.plot_array(ilos=ilos+3, save=True)
    # f.apply_filter(ifilter=0, edge=657.4)
    # f.plot(ilos=ilos+3)
    # f.plot(ilos=3)
    # plot_filter_comparison(save=True)