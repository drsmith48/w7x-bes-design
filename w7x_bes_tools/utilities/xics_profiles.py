#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:51:23 2019

@author: drsmith
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import hdf5

np.random.seed()

# shotlist for profiles
profiles_list = [
#                 [180904027, 1.0, 'pre-pellet'],
                 [180904027, 1.9, 'post-pellet', 
                  'w7x_ar16_180904027_xics_100ms_3cm_inverted.hdf5'],
#                 [180906040, 2.0, '5.0 MW ECH'],
#                 [180906040, 4.5, '4.2 MW ECH'],
#                 [180906040, 7.0, '2.5 MW ECH'],
#                 [180906040, 9.5, '1.5 MW ECH'],
#                 [180919007, 1.4, 'pre-NBI'],
                 [180919007, 2.4, 'during NBI', 
                  'w7x_ar16_180919007_xics_100ms_3cm_inverted.hdf5'],
#                 [180919007, 3.2, 'post-NBI'],
#                 [180925033, 0.65, 'low density'],
#                 [180925033, 3.9, 'high density'],
                ]
profiles_default = [{'shot':p[0], 'time':p[1], 'desc':p[2], 'datafile':p[3]} 
                    for p in profiles_list]

def fit_xics(profiles=profiles_default,   # list of dictionaries, like profiles_default
             noplot=False,                # plot results to screen?
             save_figs=False,               # save EPS figures?
             save_profile_data=False,     # save TS profile data?
             save_fits=False,              # save best fits?
             nohollow=False):             # prohibit hollow core?
    def feval(c, x, nohollow):
        if nohollow:
            yfit = c[0] * (c[1] + (1-c[1])*(1-np.abs(x)**c[2])**c[3])
        else:
            yfit = c[0] * (c[1] - c[4] + (1-c[1]+c[4])*(1-np.abs(x)**c[2])**c[3] + \
                         c[4]*(1-np.exp(-(x**2)/(c[5]**2))))
        return yfit
    def residuals(c, x, yresample, nohollow):
        # input: 1d array of n fit parameters
        # output: 1d array of m residuals: y - yfit
        yfit = feval(c, x/x.max(), nohollow=nohollow)
        res = yfit.reshape(-1,1) - yresample
        return res.reshape(res.size)
    nfits=60
    nresample=5
    for shottime in profiles:
        shotin = shottime['shot']
        timein = shottime['time']
        # load data from hdf5 datafile
        h5data = hdf5.hdf5ToDict(shottime['datafile'])
        time = np.array(h5data['dim']['time'])  # s ; 1D
        rho = np.array(h5data['dim']['rho'])  # norm. ; 1D
        reff = np.array(h5data['coord']['reff'])  # m ; [time,rho]
        ti = np.array(h5data['value']['T_ion_Ar'])  # keV ; [time,rho]
        tierr = np.array(h5data['sigma']['T_ion_Ar'])  # keV ; [time,rho]
        mask = np.array(h5data['mask']['T_ion_Ar'])  # bool ; [time,rho]
        # time slice
        tind = np.argmin(np.abs(time - timein))
        mask = np.squeeze(mask[tind,:])
        # mask[:] = True
        rho = rho[mask]
        ti = ti[tind,mask]
        tierr = tierr[tind,mask]
        plt.figure(figsize=(7.66,3.25))
        plt.subplot(1,2,1)
        plt.errorbar(rho, ti, yerr=tierr, fmt='x')
        plt.xlabel('Psi-norm')
        plt.ylabel('Ti (keV)')
        plt.title('{} | {:.2f} s'.format(shotin, time[tind]))
        # fit profile
        ymax = np.max(ti[rho<0.5])
        c0 = np.array([  ymax, 0.01, 1.1, 1.5,  0,   0.25])
        lb = np.array([ymax/2, 1e-3, 0.3, 0.3, -0.25, 0.2])
        ub = np.array([ymax*2, 0.4, 6.0, 4.0,  0.25, 0.6])
        x_scale = np.array([1,1e-2,0.1,0.1,5e-2,1e-2])
        cresults = np.empty((c0.size,0))
        wtsqerr = np.empty(0)
        for ifit in np.arange(nfits):
            yresample = ti.reshape(-1,1) + tierr.reshape(-1,1) * np.random.normal(size=[ti.size, nresample])
            result = least_squares(residuals, c0, 
                                   jac='3-point', 
                                   bounds=(lb,ub),
                                   method='trf',
                                   x_scale=x_scale,
                                   loss='arctan',
                                   f_scale=1.0,
                                   verbose=0,
                                   args=(rho,yresample),
                                   kwargs={'nohollow':nohollow})
            if result.status > 0:
                cf = result.x
                cresults = np.append(cresults, cf[...,np.newaxis], axis=1)
                tifit = feval(cf, rho/rho.max(), nohollow=nohollow)
                err = np.sqrt(np.mean(np.arctan((tifit-ti)/tierr/2)**2))
                wtsqerr = np.append(wtsqerr, err.reshape((-1)), axis=0)
                plt.plot(rho, tifit, color='C1', linewidth=0.5)
        ibestfit = np.argsort(wtsqerr)[:3]
        # fieldfits = []
        for i in ibestfit:
            ybestfit = feval(cresults[:,i], rho/rho.max(), nohollow=nohollow)
            # plot best resampled fits
            plt.plot(rho, ybestfit, color='C2', linewidth=2)
            # fieldfits.append({'rho':rho, 'y':ybestfit, 'coeff':cresults[:,i]})
        plt.subplot(1,2,2)
        for i in [0,1,2,3,4,5]:
            plt.plot([i,i], 
                     [lb[i],ub[i]], 
                     '_', 
                     color='k', 
                     mew=2,
                     ms=10)
            plt.plot(np.zeros(cresults.shape[1])+i, 
                     np.abs(cresults[i,:]), 'o', 
                     color='C1',
                     linewidth=0.5,
                     ms=3)
            # for ii in ibestfit:
            #     plt.plot(i, np.abs(cresults[i,ii]), 'o', 
            #              color='C2',
            #              linewidth=0.5,
            #              ms=3)
        plt.yscale('log')
        plt.xlabel('fit coefficients')
        plt.tight_layout()

        
if __name__=='__main__':
    plt.close('all')
    fit_xics()

# def fit_xics(conn=None,      # must pass valid mds.Connection()
#              profiles=profiles_default,
#              save_profile_data=True):
#     # if not isinstance(conn, mds.Connection):
#     #     print('Must provide valid mds.Connection()')
#     #     print("conn = mds.Connection('ssh://dvs@mds-trm-1.ipp-hgw.mpg.de')")
#     #     return
#     # try loading pickled data
#     datafile = 'xics_data.pickle'
#     try:
#         with open(datafile, 'rb') as f:
#             profile_data = pickle.load(f)
# #        assert(isinstance(profile_data, dict))
#         print('Pickle load successful')
#     except:
#         profile_data = {}
#         print('Pickle load unsuccessful, continuing')
#     # MDS connection
# #    try:
# #        conn = mds.Connection('ssh://dvs@mds-trm-1.ipp-hgw.mpg.de')
# #        mdsok = True
# #    except:
# #        mdsok = False
#     previous_shot = -1
#     xics_data = None
#     for ipro,pro in enumerate(profiles):
#         # load/fetch data
#         if pro['shot'] != previous_shot:
#             try:
#                 xics_data = profile_data[pro['shot']]
#                 print('** Loading XICS data from {}, shot {} ...'.
#                       format(datafile, pro['shot']))
#             except KeyError:
#                 print('** Getting XICS data, shot {} ...'.format(pro['shot']))
# #                if not mdsok:
# #                    raise RuntimeError('Need MDSplus connection')
#                 try:
#                     conn.openTree('qsw_eval', pro['shot'])
#                     time = conn.get('xics:ti:time')
#                     reff = conn.get('xics:ti:reff')
#                     ti = conn.get('xics:ti')
#                     tierr = conn.get('xics:ti:sigma')
#                     a = gl._get_minor_radius(pro['shot'])
#                 except:
#                     print('** MDSPLUS error **')
#                     continue
#                 xics_data = {'ti':ti,
#                              'tierr':tierr,
#                              'time':time,
#                              'roa':None,
#                              'reff':reff,
#                              'a':a}
#                 profile_data[pro['shot']] = xics_data
#     # end loop over profiles    
#     # pickle profile data
#     if save_profile_data:
#         with open(datafile, 'wb') as f:
#             pickle.dump(profile_data, f)