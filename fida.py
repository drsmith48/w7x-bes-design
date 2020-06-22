#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:39:00 2020

@author: drsmith
"""

from pathlib import Path
import struct
import numpy as np
import matplotlib.pyplot as plt


class Fida(object):
    
    def __init__(self, simdir=None):
        workdir = Path.home() / 'Documents/W7X-feasibility-study/analysis'
        # set fidasim results directory
        if not simdir:
            simdir = workdir/'fida_0046'
        if Path(simdir).exists():
            simdir = Path(simdir)
        else:
            if Path(workdir/simdir).exists():
                simdir = workdir/simdir
            else:
                raise FileNotFoundError
        print('FIDASIM results: {}'.format(simdir.as_posix()))
        
        npfp32 = np.dtype(np.float32())
        npfp64 = np.dtype(np.float64())
    
        # load nbi/halo spectra file
        nbifile = simdir / 'nbi_halo_spectra.bin'
        assert(nbifile.exists())
        with nbifile.open('rb') as f:
            f.seek(8)
            self.nlos = struct.unpack('<i', f.read(4))[0]
            nlambda = struct.unpack('<i', f.read(4))[0]
            self.spectra = np.empty((self.nlos,nlambda,0))
            self.lambda_array = np.fromfile(f, count=nlambda, dtype=npfp32)
            tmp = np.fromfile(f, count=nlambda*self.nlos, dtype=npfp32) # full energy
            self.spectra = np.append(self.spectra, tmp.reshape(self.nlos,nlambda,1), axis=2)
            tmp = np.fromfile(f, count=nlambda*self.nlos, dtype=npfp32) # half energy
            self.spectra = np.append(self.spectra, tmp.reshape(self.nlos,nlambda,1), axis=2)
            tmp = np.fromfile(f, count=nlambda*self.nlos, dtype=npfp32) # third energy
            self.spectra = np.append(self.spectra, tmp.reshape(self.nlos,nlambda,1), axis=2)
            tmp = np.fromfile(f, count=nlambda*self.nlos, dtype=npfp32) # thermal halo
            self.spectra = np.append(self.spectra, tmp.reshape(self.nlos,nlambda,1), axis=2)
        lambda_resolution = self.lambda_array[1] - self.lambda_array[0]
        print('No. lines of sight:', self.nlos)
        print('No. waveleghts:', nlambda)
        print('Min/Max wavelength: {:.1f} nm / {:.1f} nm'.format(self.lambda_array.min(),
                                                                 self.lambda_array.max()))
        print('Wavelength resolution: {:.3f} nm'.format(lambda_resolution))
        # load afida/pfida spectra file
        fidafile = simdir / 'fida_spectra.bin'
        assert(fidafile.exists())
        with fidafile.open('rb') as f:
            f.seek(8)
            self.nlos = struct.unpack('<i', f.read(4))[0]
            nlambda = struct.unpack('<i', f.read(4))[0]
            self.lambda_array = np.fromfile(f, count=nlambda, dtype=npfp32)
            tmp = np.fromfile(f, count=nlambda*self.nlos, dtype=npfp32) # act. FIDA
            self.spectra = np.append(self.spectra, tmp.reshape(self.nlos,nlambda,1), axis=2)
            tmp = np.fromfile(f, count=nlambda*self.nlos, dtype=npfp32) # pass. FIDA
            self.spectra = np.append(self.spectra, tmp.reshape(self.nlos,nlambda,1), axis=2)
        self.radiance = np.sum(self.spectra, axis=1) * lambda_resolution
        # load diag file
        diagfile = simdir / 'diag.bin'
        assert(diagfile.exists())
        with diagfile.open('rb') as f:
            # print('open', f.tell())
            self.nlos = struct.unpack('<i', f.read(4))[0]
            # print('self.nlos', f.tell())
            xyzhead = np.fromfile(f, count=self.nlos*3, dtype=npfp64)
            xyzhead = xyzhead.reshape(3,self.nlos)
            # print('xyzhead', f.tell())
            xyzlos = np.fromfile(f, count=self.nlos*3, dtype=npfp64)
            xyzlos = xyzlos.reshape(3,self.nlos)
            # print('xyzlos', f.tell())
            tmp = np.fromfile(f, count=self.nlos, dtype=npfp64)  # headsize
            tmp = np.fromfile(f, count=self.nlos, dtype=npfp64)  # opening_angle
            tmp = np.fromfile(f, count=self.nlos, dtype=npfp64)  # sigma_pi
            tmp = np.fromfile(f, count=self.nlos, dtype=npfp64)  # instfu
            read = f.read(self.nlos*20)
        form = '<'+''.join(['{}s'.format(20)]*self.nlos)
        unpack = struct.unpack(form, read)
        self.losnames = [value.decode().rstrip() for value in unpack]
    
    def los_names(self):
        print('Lines of sight:')
        for i, los in enumerate(self.losnames):
            print('  {:2d}: {}'.format(i,los))
            
    def plot(self,
             ilos=0,
             plot_all=False,
             save=False):
        if not isinstance(ilos, np.ndarray):
            if not isinstance(ilos, (list, tuple)):
                ilos= [ilos]
            ilos = np.array(ilos, dtype=np.int)
        for i in range(self.nlos):
            if i in ilos or plot_all:
                print('Plotting LOS {}: {}'.format(i, self.losnames[i]))
                plt.figure(figsize=[6,4])
                plt.plot(self.lambda_array, self.spectra[i,:,:])
                plt.legend(['Full energy ({:.2e} Ph/m^2/ster)'.format(self.radiance[i,0]),
                            'Half energy ({:.2e} Ph/m^2/ster)'.format(self.radiance[i,1]),
                            'Third energy ({:.2e} Ph/m^2/ster)'.format(self.radiance[i,2]),
                            'Therm. halo ({:.2e} Ph/m^2/ster)'.format(self.radiance[i,3]),
                            'aFIDA ({:.2e} Ph/m^2/ster)'.format(self.radiance[i,4]),
                            'pFIDA ({:.2e} Ph/m^2/ster)'.format(self.radiance[i,5])],
                           loc='upper left',
                           borderpad=0.3,
                           labelspacing=0.2,
                           handlelength=1.0,
                           handletextpad=0.4)
                plt.xlabel('Wavelength (nm)')
                plt.xlim(652,662)
                plt.ylabel('Spectral radiance (Ph/m^2/ster/nm)')
                plt.title(self.losnames[i])
                plt.tight_layout()
                if save:
                    fname = Path(self.losnames[i]+'.pdf')
                    plt.savefig(fname.as_posix(), format='pdf', transparent=True)
        

if __name__ == '__main__':
    plt.close('all')
    f = Fida(simdir='A21_P2')
    f.los_names()
    f.plot(ilos=[0,36], save=False)
    