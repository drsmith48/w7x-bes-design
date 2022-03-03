#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 09:45:07 2020

@author: drsmith
"""

from pathlib import Path
import numpy as np
import scipy.constants as pc
import matplotlib.pyplot as plt


ppd_list = [
       {'name':'API PDB-C164',
        'dark_current':1e-9,  # V_b = 10 V, T=20C
        'junction_cap':7e-12,  # V_b = 10 V, T=20C
        'responsivity':0.43,
        },
       # {'name':'API PDB-C160SM',
       #  'dark_current':2e-9,  # V_b = 10 V, T=20C
       #  'junction_cap':15e-12,  # V_b = 10 V, T=20C
       #  'responsivity':0.37,
       #  },
       # {'name':'API SD-076',
       #  'dark_current':8e-9,  # V_b = 50 V, T=20C
       #  'junction_cap':2.5e-12,  # V_b = 50 V, T=20C
       #  'responsivity':0.35,
       #  },
       # {'name':'API SD-057',
       #  'dark_current':0.1e-9,  # V_b = 50 V, T=20C
       #  'junction_cap':9e-12,  # V_b = 50 V, T=20C
       #  'responsivity':0.4,
       #  },
       # {'name':'First Sensor X7-F',
       #  'dark_current':0.25e-9,  # V_b = 10 V, T=20C
       #  'junction_cap':12e-12,  # V_b = 10 V, T=20C
       #  'qe':0.71,
       #  },
       #  {'name':'Hamm S3805',
       #   'dark_current':0.1e-9,  # V_b = 10 mV, T=20C
       #   'junction_cap':15e-12,  # V_b = 0 V, T=20C
       #   'responsivity':0.45,
       #   },
       # {'name':'Vishay T1170P',
       #  'dark_current':1e-9,  # V_b = 10 V, T=20C
       #  'junction_cap':3e-12,  # V_b = 3 V, T=20C
       #  'responsivity':0.48,
       #  },
       # {'name':'Osram SFH 203 P',
       #  'dark_current':1e-9,  # V_b = 10 V, T=20C
       #  'junction_cap':2.5e-12,  # V_b = 3 V, T=20C
       #  'responsivity':0.46,
       #  },
       {'name':'Typical',
        'dark_current':0.1e-9,  # V_b = 10 V, T=20C
        'junction_cap':15e-12,  # V_b = 3 V, T=20C
        'responsivity':0.45,
        },
    ]

apd_list = [
    # {'name':'Hamm S8550-2',
    #   'qe':0.85,
    #   'junction_cap':9e-12,
    #   'gain':50,
    #   'noise_figure':0.2,
    #   },
    ]


bandwidth = 1e6  # signal bandwidth [Hz]
p_ref = 10e-9  # reference power [W]
t_ref = 20  # reference temp [C]
vb_ref = 10  # reference reverse bias [V]
    

class _Diode(object):
    
    def __init__(self, 
                 name='', 
                 responsivity=0, 
                 qe=0, 
                 dark_current=0, 
                 junction_cap=0,
                 verbose=False
                 ):
        self.name = name
        self.gain = None
        self.noise_factor = None
        if responsivity:
            # responsivity at 650 nm [A/W]
            self.responsivity = responsivity
            self.qe = self.responsivity * 1240/656
        elif qe:
            self.qe = qe
            self.responsivity = self.qe * 656/1240
        else:
            raise ValueError
        
        # dark current at V_b=10V, T=20C [nA]
        # drops by factor of 10 for 20 C decrease
        # slight increase with V_b
        self.dark_current_ref = dark_current
        # junction cap at V_b=10V, T=20C [pF]
        # sharp decrease with V_b; C_j ~ 1/sqrt(V_b)
        # C_j(10V) ~ C_j(0V)/5
        self.junction_cap_ref = junction_cap
        if verbose:
            print(self.name)
            print(f'  QE = {self.qe*1e2:.1f}')
            print(f'  Resp. = {self.responsivity:.2f} A/W')
        
    def photocurrent(self, p_inc=p_ref):
        return self.gain * self.responsivity * p_inc
    
    def darkcurrent(self, t=t_ref):
        return self.dark_current_ref * 2**((t-t_ref)/10)
    
    def junction_cap(self, vb=vb_ref):
        return self.junction_cap_ref * np.sqrt(vb_ref)/np.sqrt(vb)
    
    def sigmasq_thermal(self, t=t_ref, rload=100e6):
        return 4*pc.k*(t+273)/rload * bandwidth
    
    def sigmasq_shot(self, p_inc=p_ref, t=t_ref):
        photocurrent = self.photocurrent(p_inc=p_inc)
        darkcurrent = self.darkcurrent(t=t)
        return 2*pc.e * self.gain**2 * self.noise_factor \
            * (photocurrent+darkcurrent) * bandwidth
    
    def SNR(self, p_inc=p_ref, t=t_ref, rload=100e6):
        photocurrent = self.photocurrent(p_inc=p_inc)
        sigmasq = self.sigmasq_thermal(t=t, rload=rload) \
            + self.sigmasq_shot(p_inc=p_inc, t=t)
        return photocurrent / np.sqrt(sigmasq)
    

class PinDiode(_Diode):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = self.name + ' (PIN)'
        self.gain = 1
        self.noise_factor = 1
    

class ApdDiode(_Diode):
    
    def __init__(self, 
                 gain=0, 
                 noise_factor=0,
                 noise_figure=0,
                 noise_current=0, 
                 **kwargs):
        super().__init__(**kwargs)
        self.name = self.name + ' (APD)'
        self.gain = gain
        if noise_factor:
            # noise factor specified
            self.noise_factor = noise_factor
        elif noise_figure:
            # calc noise factor from noise figure
            self.noise_factor = 10**(noise_figure/10)
        elif noise_current:
            # calc noise factor from noise current
            sigma_sq_ideal = self.sigmasq_shot(p_inc=0)/bandwidth
            self.noise_factor = noise_current**2 / sigma_sq_ideal
        else:
            raise ValueError
        # assert noise factor > 1
        assert(self.noise_factor>1)
    

def plot_diodes(save=False):
    plt.close('all')
    diodes = []
    diodes.extend([PinDiode(verbose=True, **ppd_kw) for ppd_kw in ppd_list])
    diodes.extend([ApdDiode(verbose=True, **apd_kw) for apd_kw in apd_list])
    # prange = np.array([1,3,10])*1e-9
    prange = np.arange(3,16)*1e-9
    vbrange = np.arange(1,30,1)
    trange = np.arange(-40,26,5)
    rload = 10e6
    plt.figure(figsize=[11.5,8])
    # photocurrent
    plt.subplot(3,3,1)
    for d in diodes:
        plt.plot(prange*1e9, d.photocurrent(prange)*1e9, label=d.name)
    plt.xlabel('Incident Power (nW)')
    plt.ylabel('Photocurrent (nA)')
    plt.title('I_photo vs. inc. power')
    # junction capacitance
    plt.subplot(3,3,2)
    for d in diodes:
        plt.semilogy(vbrange, d.junction_cap(vb=vbrange)*1e12, label=d.name)
    plt.xlabel('Reverse bias (V)')
    plt.ylabel('Diode junction cap. (pF)')
    plt.title('C_j vs. bias')
    # dark current
    plt.subplot(3,3,3)
    for d in diodes:
        plt.semilogy(trange, d.darkcurrent(t=trange)*1e9, label=d.name)
    plt.xlabel('Temp. (C)')
    plt.ylabel('Dark current (nA)')
    plt.title('I_dark vs. temp.')
    # thermal noise
    plt.subplot(3,3,4)
    for d in diodes:
        sigmasq = d.sigmasq_thermal(t=trange, rload=rload)
        plt.plot(trange, np.sqrt(sigmasq)*1e9, label=d.name)
    plt.xlabel('Temp. (C)')
    plt.ylabel('sqrt(<i_th^2>*B) (nA)')
    plt.title('Thermal noise vs. temp.')
    plt.annotate(f'BW = {bandwidth/1e6} MHz', (0.55,0.15),
                 xycoords='axes fraction', size='small')
    plt.annotate(f'R_L = {rload/1e6} MOhm', (0.55,0.05),
                 xycoords='axes fraction', size='small')
    # shot nosie
    plt.subplot(3,3,5)
    for d in diodes:
        sigmasq = d.sigmasq_shot(p_inc=prange)
        plt.plot(prange*1e9, np.sqrt(sigmasq)*1e9/d.gain, label=d.name)
    plt.xlabel('Incident Power (nW)')
    plt.ylabel('sqrt(<i_sh^2>*B)/G (nA)')
    plt.title('Shot noise/G vs. inc. power')
    plt.annotate(f'BW = {bandwidth/1e6} MHz', (0.55,0.05),
                 xycoords='axes fraction', size='small')
    # shot nosie
    plt.subplot(3,3,6)
    p_inc=1e-9
    for d in diodes:
        sigmasq = d.sigmasq_shot(p_inc=p_inc, t=trange)
        plt.plot(trange, np.sqrt(sigmasq)*1e9/d.gain, label=d.name)
    plt.xlabel('Temp. (C)')
    plt.ylabel('sqrt(<i_shot^2>*B)/G (nA)')
    plt.title('Shot noise/G vs. temp.')
    plt.annotate(f'BW = {bandwidth/1e6} MHz', (0.05,0.5),
                 xycoords='axes fraction', size='small')
    plt.annotate(f'P_inc = {p_inc*1e9} nW', (0.05,0.4),
                 xycoords='axes fraction', size='small')
    # total noise
    plt.subplot(3,3,7)
    for d in diodes:
        sigmasq = d.sigmasq_shot(p_inc=prange) + d.sigmasq_thermal(t=-20, rload=rload)
        plt.plot(prange*1e9, np.sqrt(sigmasq)*1e9/d.gain, label=d.name)
    plt.xlabel('Incident Power (nW)')
    plt.ylabel('sqrt(<i_n^2>*B)/G (nA)')
    plt.title('Noise/G vs. inc. power')
    plt.annotate(f'BW = {bandwidth/1e6} MHz', (0.55,0.05),
                 xycoords='axes fraction', size='small')
    plt.annotate('T = -20 C', (0.05,0.7),
                 xycoords='axes fraction', size='small')
    # SNR
    for it, temp in enumerate([-20,-190]):
        plt.subplot(3,3,8+it)
        for d in diodes:
            plt.plot(prange*1e9, 2./d.SNR(p_inc=prange, t=temp, rload=rload), label=d.name)
        plt.xlabel('Incident power (nW)')
        plt.ylabel('del-n/n = 2 * I_n/I_ph')
        plt.title('del-n/n vs. inc. power')
        plt.annotate(f'BW = {bandwidth/1e6} MHz', (0.05,0.9),
                     xycoords='axes fraction', size='small')
        plt.annotate(f'R_L = {rload/1e6} MOhm', (0.05,0.8),
                     xycoords='axes fraction', size='small')
        plt.annotate(f'T = {temp} C', (0.05,0.7),
                     xycoords='axes fraction', size='small')
    # all plots
    for ax in plt.gcf().axes[3:4]:
        ax.legend(fontsize='small', labelspacing=0.2, handlelength=1)
    plt.tight_layout()
    if save:
        fname = Path('diodes.pdf')
        print(f'Saving {fname.as_posix()}')
        plt.savefig(fname.as_posix(), transparent=True)
    return diodes
    
    
if __name__=='__main__':
    plt.close('all')
    diodes = plot_diodes(save=False)
    plt.show(block=True)
