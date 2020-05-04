#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 08:00:17 2019

@author: drsmith
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import FIT
from FIT import model_spec, derivatives
import PCIanalysis.gradientlengths as gl
import pybaseutils.utils as ut

plt.close('all')

verbose = True

datafile = 'profiles.pickle'
try:
    with open(datafile, 'rb') as f:
        profile_data = pickle.load(f)
    print('Pickle load successful')
except:
    profile_data = None
    print('Pickle load unsuccessful, continuing')

profiles_list = [
#        [180904027, 1.0, 'pre-pellet'],
        [180904027, 1.9, 'post-pellet'],
#        [180906040, 2.0, '5.0 MW ECH'],
#        [180906040, 4.5, '4.2 MW ECH'],
#        [180906040, 7.0, '2.5 MW ECH'],
#        [180906040, 9.5, '1.5 MW ECH'],
#        [180919007, 1.4, 'pre-NBI'],
#        [180919007, 2.4, 'during NBI'],
#        [180919007, 3.2, 'post-NBI'],
#        [180925033, 0.65, 'low density'],
#        [180925033, 3.9, 'high density'],
    ]

profiles = [{'shot':p[0], 'time':p[1], 'desc':p[2]}
            for p in profiles_list]

def myqparab(x=None, af=None, **kwargs):
    if 'XX' in kwargs:
        kwargs.pop('XX')
    return model_spec.model_qparab(x, af=af, **kwargs)

def feval(af, x):
    y = af[0] * (af[1] - af[4] + (1-af[1]+af[4])*(1-np.abs(x)**af[2])**af[3] + \
                 af[4]*(1-np.exp(-(x**2)/(af[5]**2))))
    return y

previous_shot = -1
tsdata = None
shotdata = {}
nresample = 1
chi2_limit = 80.0
nohollow = False
for pro in profiles:
    # load/fetch data
    if pro['shot'] != previous_shot:
        if profile_data:
            print('** Loading MPTS data from {}, shot {} ...'.
                  format(datafile, pro['shot']))
            tsdata = profile_data[pro['shot']]
        else:
            print('** Getting MPTS data, shot {} ...'.format(pro['shot']))
            tsdata = gl.get_thomsondata(pro['shot'])
        shotdata[pro['shot']] = tsdata
    print('**** Performing ne profile fit, shot {}, time {:.2f} s ...'.
          format(pro['shot'], pro['time']))
    # index for time slice
    index = np.argmin(np.abs(tsdata['time'] / 1e9 - pro['time']))
    # cut edge channels
    edgecut=1
    if edgecut:
        x = tsdata['roa'][:-edgecut].copy()
        y = tsdata['ne'][index, :-edgecut].copy()
        yerr = tsdata['ne_err'][index, :-edgecut].copy()
    else:
        x = tsdata['roa'].copy()
        y = tsdata['ne'][index, :].copy()
        yerr = tsdata['ne_err'][index, :].copy()
    # remove any y=0 data points
    if np.count_nonzero(y==0):
        mask = y == 0
        x = x[~mask]
        y = y[~mask]
        yerr = yerr[~mask]
    # adjust edge to ensure zero or near-zero value
    if (np.nanmin(y)>0.01*np.nanmax(y)):
        # First try linearly interpolating the last few points to 0
        xedge, vedge = ut.interp(y[-3:], x[-3:], ei=np.sqrt(yerr[-3:]), xo=0.0)
        if (xedge < x[-1]) or xedge>1.15:
            xedge = 1.05*np.max((np.nanmax(x), 1.10))
            vedge = yerr[-1]
        x = np.insert(x,-1, xedge)
        y = np.insert(y,-1, 0.0)
        yerr = np.insert(yerr,-1,vedge)
    # add x=0 point
    x = np.insert(x,0,0.001)
    y = np.insert(y,0,y[0])
    yerr = np.insert(yerr,0,yerr[0])
    # reflect arrays (why??)
#    x = ut.cylsym_odd(x) # anti-reflect x about x=0
#    y = ut.cylsym_even(y) # reflect y about x=0
#    yerr = ut.cylsym_even(yerr)
    # y variance
    yvar = yerr*yerr
    # compress x so max(x) = 1.0
    xinitial = x.copy()
    xmax = x.max()
    x /= xmax
    # x grid for interpolation
    xinterp = np.linspace(0, 1.0, 50)
    # default fit to get initial parameter values
    info = myqparab(x=None, af=None, nohollow=nohollow, nprint=1) #, nohollow=nohollow)
    af0 = info.af.copy()
    # adjust initial parameters based on data
    af0[0] = y.max()
    af0[1] = np.min((np.max((0.02, y.min()/af0[0])), 0.99))
    # setup y resample loop
    nparameters = len(af0)
    af = np.zeros((nresample, nparameters), dtype=np.float64)
    af0_copy = np.zeros((nresample, nparameters), dtype=np.float64)
    vaf = np.zeros((nresample, nparameters), dtype=np.float64)
    covmat = np.zeros((nresample, nparameters, nparameters), dtype=np.float64)
    chi2_reduced = np.zeros((nresample,), dtype=np.float64)

    nn = 0
    chi2_min = 1e18
    np.random.seed()
    for mm in range(nresample):
        if verbose:
            print('****** {:03d} of {:03d} calls to profilefit'.format(mm+1, nresample))
        # resample y data according to y uncertainty
        ydat = y.copy() + np.sqrt(yvar/2)*np.random.normal(0.0,1.0,y.shape)
        # enforce y>=0
        ydat[ydat<0] = 0.0
        # variance w.r.t. true y data
        vary = (ydat-y)*(ydat-y)
        vary[vary==0] = np.nanmean(vary)
        # adust x
#        xslope = 1.05*np.nanmax((np.nanmax(x), np.nanmax(xinterp))) # shrink so maximum is less than 1.0
#        xoffset = -1e-4  # prevent 0 from being in problem
#        XXt = (xinterp.copy()-xoffset)/xslope
#        xt = (x.copy()-xoffset)/xslope
        af_mod = af0.copy()
        af_mod[0] = y.max() + np.sqrt(yvar[0])*np.random.normal()
        af_mod[1] = np.min((np.max((0.02, y.min()/af_mod[0])), 0.99))
        print(af_mod[0:2])
        info = FIT.fitNL.fit_mpfit(x, ydat, np.sqrt(vary), xinterp, myqparab, 
                                   fkwargs={'nohollow':False},
                                   plotit=False,
                                   quiet=False,
                                   verbose=verbose,
                                   nohollow=nohollow,
                                   scalex=False,
                                   nprint=1,
                                   errx=0.0,
#                                   af0=af_mod.copy(),
                                   af0=None,
                                   perpchi2=False,
                                   scale_problem=False,
                                   autoderivative=1,
                                   scale_update=False,
                                   Method='Bootstrap',
                                   )
        setattr(info, 'af0', af_mod)

#        info.XX = info.XX*xslope + xoffset
#        info.dprofdx *= (1.0/xslope)
#        info.vardprofdx *= 1.0/(xslope*xslope)
#        info.d2profdx2 *= 1.0/(xslope*xslope)
        # plot fit
        plt.figure()
        plt.errorbar(xinitial, y, yerr=yerr, linestyle='', marker='x')
        plt.plot(xinitial, ydat, 'x')
        plt.plot(0, af_mod[0], 'X')
        plt.plot(xinterp*xmax, feval(af_mod, xinterp), label='input params')
        plt.plot(xinterp*xmax, feval(info.af, xinterp), label='final params')
        plt.legend()
        if info.chi2_reduced<chi2_limit:
            print('******** valid chi2_reduced {:.3f}'.
                  format(info.chi2_reduced))
        else:
            print('******** chi2_reduced {:.3f} exceeds limit {:.1f}'.
                  format(info.chi2_reduced, chi2_limit))
            continue
        if info.chi2_reduced<chi2_min:
            af0 = info.af.copy()
            chi2_min = np.copy(info.chi2_reduced)
        af[nn, :] = info.af.copy()
        af0_copy[nn, :] = info.af0.copy()
        covmat[nn,:,:] = info.covmat.copy()
        vaf[nn, :] = np.power(info.perror.copy(), 2.0)
        chi2_reduced[nn] = np.copy(info.chi2_reduced)
        nn += 1
        
        
    # end resample loop

    print('**** %i of %i runs valid with chi_nu**2 <= %3.1f'%(nn, nresample, chi2_limit))
    af = af[:nn, :]
    af0_copy = af0_copy[:nn, :]
    vaf = vaf[:nn,:]
    covmat = covmat[:nn,:,:]
    chi2_reduced = chi2_reduced[:nn]

    previous_shot = pro['shot']
    
    print('\n*** Results ***\n')
    for ip in range(nparameters):
        print('af[{}]'.format(ip))
        for n in range(nn):
            print('  Init {:10.3e}  Final {:10.3e} +/- {:10.3e}'.
                  format(af0_copy[n,ip], af[n,ip], vaf[n,ip]))
    
# end loop over profiles
    
print('Saving MPTS data in {}'.format(datafile))
with open(datafile, 'wb') as f:
    pickle.dump(shotdata, f)

