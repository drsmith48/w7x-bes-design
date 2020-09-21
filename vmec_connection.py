#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 07:57:55 2019

@author: drsmith
"""

import platform
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import zeep



def connection():
    # connect to VMEC webservice
    if platform.system().lower() == 'windows':
        count_str = '-n'
        timeout_str = '-w'
    else:
        count_str = '-c'
        timeout_str = '-t'
    ping = subprocess.run(f'ping {count_str} 1 {timeout_str} 2 esb.ipp-hgw.mpg.de'.split(' '),
                          capture_output=True)
    ping.check_returncode()
    connection_url = "http://esb.ipp-hgw.mpg.de:8280/services/vmec_v5?wsdl"
    return zeep.Client(connection_url)


def write_wout_nc(eqtag='w7x_ref_9'):
    vmec = connection()
    wout_nc = vmec.service.getVmecOutputNetcdf(eqtag)
    wout_nc_file = open('wout.nc', 'wb')
    wout_nc_file.write(wout_nc)
    wout_nc_file.close()


def test():
    plt.close('all')
    
    vmec = connection()
    
    s = [0.0,0.2,0.4,0.6,0.8,1.0]
    numPoints = 100
    fs0 = vmec.service.getFluxSurfaces('w7x_ref_9', 0.0, s, numPoints)
    fs36 = vmec.service.getFluxSurfaces('w7x_ref_9', np.pi/5, s, numPoints)
    
    plt.figure(figsize=(7.8,3.8))
    
    plt.subplot(121)
    plt.plot(fs0[0].x1, fs0[0].x3)
    for i in range(len(fs0)):
        plt.plot(fs0[i].x1, fs0[i].x3)
    plt.title('flux surfaces at phi = 0°')
    plt.xlabel('R')
    plt.ylabel('Z')
    ax1 = plt.gca()
    plt.axis([4.5, 6.5, -1.0, 1.0])
    
    plt.subplot(122, sharex=ax1, sharey=ax1)
    plt.plot(fs36[0].x1, fs36[0].x3)
    for i in range(len(fs36)):
        plt.plot(fs36[i].x1, fs36[i].x3)
    plt.title('flux surfaces at phi = 36°')
    plt.xlabel('R')
    plt.ylabel('Z')
    
    for ax in plt.gcf().axes:
        ax.set_aspect('equal')
    
    plt.tight_layout()


if __name__=='__main__':
    test()
    # write_wout_nc()