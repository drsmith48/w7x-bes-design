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
import tkinter as tk
from tkinter import filedialog as fd

workdir = Path.home() / 'Documents/W7X-feasibility-study/analysis'
root = tk.Tk()
root.withdraw()

def plotfida(simdir=None):
    # set fidasim results directory
    if not simdir or not Path(workdir/simdir).exists():
        print('Work directory: {}'.format(workdir.as_posix()))
        simdir=fd.askdirectory(initialdir=workdir.as_posix())
        simdir = Path(simdir)
    else:
        simdir = workdir/simdir
    print(simdir)
    
    npfp32 = np.dtype(np.float32())
    npfp64 = np.dtype(np.float64())

    # load nbi/halo spectra file
    nbifile = simdir / 'nbi_halo_spectra.bin'
    assert(nbifile.exists())
    with nbifile.open('rb') as f:
        f.seek(8)
        nlos = struct.unpack('<i', f.read(4))[0]
        nlambda = struct.unpack('<i', f.read(4))[0]
        lambda_array = np.fromfile(f, count=nlambda, dtype=npfp32)
        fullsp = np.fromfile(f, count=nlambda*nlos, dtype=npfp32)
        fullsp = fullsp.reshape(nlos,nlambda)
        halfsp = np.fromfile(f, count=nlambda*nlos, dtype=npfp32)
        halfsp = halfsp.reshape(nlos,nlambda)
        thirdsp = np.fromfile(f, count=nlambda*nlos, dtype=npfp32)
        thirdsp = thirdsp.reshape(nlos,nlambda)
        halosp = np.fromfile(f, count=nlambda*nlos, dtype=npfp32)
        halosp = halosp.reshape(nlos,nlambda)
    # load afida/pfida spectra file
    fidafile = simdir / 'fida_spectra.bin'
    assert(fidafile.exists())
    with fidafile.open('rb') as f:
        f.seek(8)
        nlos = struct.unpack('<i', f.read(4))[0]
        nlambda = struct.unpack('<i', f.read(4))[0]
        lambda_array = np.fromfile(f, count=nlambda, dtype=npfp32)
        afidasp = np.fromfile(f, count=nlambda*nlos, dtype=npfp32)
        afidasp = afidasp.reshape(nlos,nlambda)
        pfidasp = np.fromfile(f, count=nlambda*nlos, dtype=npfp32)
        pfidasp = pfidasp.reshape(nlos,nlambda)
    # load diag file
    diagfile = simdir / 'diag.bin'
    assert(diagfile.exists())
    with diagfile.open('rb') as f:
        nchan = struct.unpack('<i', f.read(4))[0]
        print('nchan', nchan)
        xyzhead = np.fromfile(f, count=nchan*3, dtype=npfp64)
        xyzhead = xyzhead.reshape(3,nchan)
        print(xyzhead)
        xyzlos = np.fromfile(f, count=nchan*3, dtype=npfp64)
        xyzlos = xyzlos.reshape(3,nchan)
        print(xyzlos)
        f.seek(404)
        losname = str(struct.unpack('<{}s'.format(nchan*20), f.read(100))[0])
        print(losname, type(losname))
    # plot spectra
    plt.plot(lambda_array, fullsp[0,:], 
             lambda_array, halfsp[0,:],
             lambda_array, thirdsp[0,:],
             lambda_array, halosp[0,:],
             lambda_array, afidasp[0,:],
             lambda_array, pfidasp[0,:])
        

if __name__ == '__main__':
    plt.close('all')
    plotfida('fida_0046')