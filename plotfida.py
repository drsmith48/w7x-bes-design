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


def plotfida(simdir=None,
             ilos=0,
             plot_all=False,
             save=False):
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
        nlos = struct.unpack('<i', f.read(4))[0]
        nlambda = struct.unpack('<i', f.read(4))[0]
        spectra = np.empty((nlos,nlambda,0))
        lambda_array = np.fromfile(f, count=nlambda, dtype=npfp32)
        tmp = np.fromfile(f, count=nlambda*nlos, dtype=npfp32) # full energy
        spectra = np.append(spectra, tmp.reshape(nlos,nlambda,1), axis=2)
        tmp = np.fromfile(f, count=nlambda*nlos, dtype=npfp32) # half energy
        spectra = np.append(spectra, tmp.reshape(nlos,nlambda,1), axis=2)
        tmp = np.fromfile(f, count=nlambda*nlos, dtype=npfp32) # third energy
        spectra = np.append(spectra, tmp.reshape(nlos,nlambda,1), axis=2)
        tmp = np.fromfile(f, count=nlambda*nlos, dtype=npfp32) # thermal halo
        spectra = np.append(spectra, tmp.reshape(nlos,nlambda,1), axis=2)
    lambda_resolution = lambda_array[1] - lambda_array[0]
    print('No. lines of sight:', nlos)
    print('No. waveleghts:', nlambda)
    print('Min/Max wavelength: {:.1f} nm / {:.1f} nm'.format(lambda_array.min(),
                                                             lambda_array.max()))
    print('Wavelength resolution: {:.3f} nm'.format(lambda_resolution))
    # load afida/pfida spectra file
    fidafile = simdir / 'fida_spectra.bin'
    assert(fidafile.exists())
    with fidafile.open('rb') as f:
        f.seek(8)
        nlos = struct.unpack('<i', f.read(4))[0]
        nlambda = struct.unpack('<i', f.read(4))[0]
        lambda_array = np.fromfile(f, count=nlambda, dtype=npfp32)
        tmp = np.fromfile(f, count=nlambda*nlos, dtype=npfp32) # act. FIDA
        spectra = np.append(spectra, tmp.reshape(nlos,nlambda,1), axis=2)
        tmp = np.fromfile(f, count=nlambda*nlos, dtype=npfp32) # pass. FIDA
        spectra = np.append(spectra, tmp.reshape(nlos,nlambda,1), axis=2)
    # load diag file
    diagfile = simdir / 'diag.bin'
    assert(diagfile.exists())
    with diagfile.open('rb') as f:
        nlos = struct.unpack('<i', f.read(4))[0]
        xyzhead = np.fromfile(f, count=nlos*3, dtype=npfp64)
        xyzhead = xyzhead.reshape(3,nlos)
        xyzlos = np.fromfile(f, count=nlos*3, dtype=npfp64)
        xyzlos = xyzlos.reshape(3,nlos)
        f.seek(404)
        losnames = str(struct.unpack('<{}s'.format(nlos*20), f.read(100))[0])[2:-1]
    losnames = [s for s in losnames.split(' ') if s != '']
    print, 'Lines of sight:'
    for i, los in enumerate(losnames):
        print('  {:2d}: {}'.format(i,los))
    radiance = np.sum(spectra, axis=1) * lambda_resolution
    # plot spectra
    for i in range(nlos):
        if i==ilos or plot_all:
            plt.figure(figsize=[6,4])
            plt.plot(lambda_array, spectra[i,:,:])
            plt.legend(['Full energy ({:.2e} Ph/sr/m^2)'.format(radiance[i,0]),
                        'Half energy ({:.2e} Ph/sr/m^2)'.format(radiance[i,1]),
                        'Third energy ({:.2e} Ph/sr/m^2)'.format(radiance[i,2]),
                        'Therm. halo ({:.2e} Ph/sr/m^2)'.format(radiance[i,3]),
                        'aFIDA ({:.2e} Ph/sr/m^2)'.format(radiance[i,4]),
                        'pFIDA ({:.2e} Ph/sr/m^2)'.format(radiance[i,5])],
                       loc='upper left',
                       borderpad=0.3,
                       labelspacing=0.2,
                       handlelength=1.0,
                       handletextpad=0.4)
            plt.xlabel('Wavelength (nm)')
            plt.xlim(652,662)
            plt.ylabel('Spectral radiance (Ph/sr/m^2/nm)')
            plt.title(losnames[i])
            plt.tight_layout()
        if save:
            fname = Path(losnames[i]+'.pdf')
            plt.savefig(fname.as_posix(), format='pdf', transparent=True)
        

if __name__ == '__main__':
    plt.close('all')
    plotfida(ilos=1)