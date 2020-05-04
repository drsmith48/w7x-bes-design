#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 08:00:17 2019

@author: drsmith
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from FIT.fitNL import fit_mpfit
from FIT.model_spec import model_qparab
import PCIanalysis.gradientlengths as gl
import pybaseutils.utils as ut

plt.close('all')


def feval(af, x):
    y = af[0] * (af[1] - af[4] + (1-af[1]+af[4])*(1-np.abs(x)**af[2])**af[3] + \
                 af[4]*(1-np.exp(-(x**2)/(af[5]**2))))
    return y

# shotlist for profiles
profiles_list = [
        [180904027, 1.0, 'pre-pellet'],
        [180904027, 1.9, 'post-pellet'],
        [180906040, 2.0, '5.0 MW ECH'],
        [180906040, 4.5, '4.2 MW ECH'],
        [180906040, 7.0, '2.5 MW ECH'],
        [180906040, 9.5, '1.5 MW ECH'],
        [180919007, 1.4, 'pre-NBI'],
        [180919007, 2.4, 'during NBI'],
        [180919007, 3.2, 'post-NBI'],
        [180925033, 0.65, 'low density'],
        [180925033, 3.9, 'high density'],
    ]
profiles = [{'shot':p[0], 'time':p[1], 'desc':p[2]} for p in profiles_list]
nprofiles = len(profiles)

# try loading pickled data
datafile = 'ts_data.pickle'
try:
    with open(datafile, 'rb') as f:
        profile_data = pickle.load(f)
    assert(isinstance(profile_data, dict))
    print('Pickle load successful')
except:
    profile_data = {}
    print('Pickle load unsuccessful, continuing')


# controls
nattempts = 50
chi2_cutoff = 15
maxiter = 40
nsample = 1
tol = 1.0e-6
nohollow = True
zerosol = True
noplot = False
lmfit_quiet = True
edgecut=2
savefig = True
np.random.seed()

# default fit to get initial parameter values
info = model_qparab(nohollow=nohollow)
af_default = info.af.copy()
nparameters = len(af_default)

# output containers
af = np.empty((nprofiles, nattempts, nparameters)) * np.nan
af_weighted = np.empty((nprofiles, nparameters)) * np.nan
vaf = np.empty((nprofiles, nattempts, nparameters)) * np.nan
vaf_weighted = np.empty((nprofiles, nparameters)) * np.nan
covmat = np.empty((nprofiles, nattempts, nparameters, nparameters)) * np.nan
fit_chi2_reduced = np.empty((nprofiles, nattempts)) * np.nan
chi2_reduced = np.empty((nprofiles, nattempts)) * np.nan
iterations = np.empty((nprofiles, nattempts), dtype=np.int) * np.nan

# loop over shotlist
previous_shot = -1
tsdata = None
for ipro,pro in enumerate(profiles):
    # load/fetch data
    if pro['shot'] != previous_shot:
        try:
            tsdata = profile_data[pro['shot']]
            print('** Loading MPTS data from {}, shot {} ...'.
                  format(datafile, pro['shot']))
        except KeyError:
            print('** Getting MPTS data, shot {} ...'.format(pro['shot']))
            tsdata = gl.get_thomsondata(pro['shot'])
            profile_data[pro['shot']] = tsdata
    print('**** Performing ne profile fit, shot {}, time {:.2f} s ...'.
          format(pro['shot'], pro['time']))
    # unpack data
    time = tsdata['time']/1e9
    index = np.argmin(np.abs(time - pro['time']))
    x = tsdata['roa'].copy()
    y = tsdata['ne'][index,:]
    yerr = tsdata['ne_err'][index,:]
    # cut edge channels
    if edgecut:
        x = x[:-edgecut]
        y = y[:-edgecut]
        yerr = yerr[:-edgecut]
    # remove any y=0 data points
    if np.count_nonzero(y==0):
        mask = y == 0
        x = x[~mask]
        y = y[~mask]
        yerr = yerr[~mask]
    # adjust edge to ensure zero or near-zero value
    xedge, vedge = ut.interp(y[-3:], x[-3:], ei=np.sqrt(yerr[-3:]), xo=0.0)
    if (xedge < x[-1]) or xedge>1.15:
        xedge = 1.05*np.max((np.nanmax(x), 1.10))
        vedge = yerr[-1]
    x = np.insert(x,-1, xedge)
    y = np.insert(y,-1, 0.0)
    yerr = np.insert(yerr,-1,vedge/10)
    # reflect arrays (why??)
    x = ut.cylsym_odd(x) # anti-reflect x about x=0
    y = ut.cylsym_even(y) # reflect y about x=0
    yerr = ut.cylsym_even(yerr)
    # y variance
    yvar = yerr*yerr
    # compress x so max(x) = 1.0
    xmax = np.max(np.abs(x))
    xscaled = x / xmax
    # adjust initial parameters based on data
    af_default[0] = y.max()
    af_default[1] = np.min((np.max((0.02, y.min()/af_default[0])), 0.99))
    # loop over fit attempts
    for iatt in range(nattempts):
        # tile true data for resampling
        yresample = np.tile(y,nsample)
        xresample = np.tile(xscaled, nsample)
        # resample y data according to y uncertainty
        ydat = np.empty((0,))
        for i in range(nsample):
            ydat = np.append(ydat, y.copy() + 
                             np.sqrt(yvar)*np.random.normal(0.0,1.0,y.shape))
        # enforce y>=0
        ydat[ydat<0] = 0
        # variance w.r.t. true y data
        vary = (ydat-yresample)*(ydat-yresample)
        vary[vary==0] = np.nanmean(vary)
        af_initial = af_default.copy()
        af_initial[0] = y.max() + np.sqrt(yvar[0])*np.random.normal()
        af_initial[1] = 0.0
        try:
            info = fit_mpfit(xresample, ydat, np.sqrt(vary), np.linspace(-1,1), model_qparab, 
                             fkwargs={'nohollow':nohollow},
                             plotit=False,
                             quiet=lmfit_quiet,
                             verbose=False,
                             scalex=False,
                             af0=af_initial,
                             autoderivative=1,     # default is 0
                             nprint=1,             # default is 10
                             perpchi2=False,       # default is True
                             scale_problem=False,  # default is True
                             scale_update=False,
                             atol=tol,
                             ftol=tol,
                             gtol=tol,
                             maxiter=maxiter,
                             damp=2,
                             LB=np.array([ 0,    0, 0, 0, -0.3, 0]),
                             UB=np.array([20, 0.05, 3, 3,  0.3, 1]),
                             fixed=np.array([0,zerosol,0,0,nohollow,nohollow]),
                             )
            print('  Run {:02d} of {:02d}  chi2_reduced {:8.1f} with {:3d} iterations'.
                  format(iatt+1, nattempts, info.chi2_reduced, info.niter))
        except:
            
            continue
        # capture output
        af[ipro,iatt, :] = info.af
        covmat[ipro,iatt,:,:] = info.covmat
        perr = info.perror
        perr[perr==0.0] = 1.0e8
        vaf[ipro,iatt, :] = perr**2
        fit_chi2_reduced[ipro,iatt] = info.chi2_reduced
        iterations[ipro,iatt] = info.niter
        # residual with input data
        residual = (y-feval(info.af, xscaled)) / yerr
        residual_limit= np.percentile(np.abs(residual), 96)
        residual[residual>residual_limit] = residual_limit
        residual[residual<-residual_limit] = -residual_limit
        chi2_reduced[ipro,iatt] = np.sum(residual**2) / info.dof
        
    # end resample loop

    print('\n*** Results ***\n')
    mask = np.isfinite(np.squeeze(iterations[ipro,:]))
    chi2_limit = np.percentile(chi2_reduced[ipro,mask], chi2_cutoff)
    valid = np.squeeze(np.argwhere(np.logical_and(chi2_reduced[ipro,:] <= chi2_limit, mask)))
    for iparam in range(nparameters):
        print('af[{}]'.format(iparam))
        for ivalid in valid:
            print('  fit {:02d}: {:10.3e} +/- {:10.3e}'.
                  format(ivalid+1, 
                         af[ipro,ivalid,iparam], 
                         np.sqrt(vaf[ipro,ivalid,iparam])))
        weights = 1/vaf[ipro,valid,iparam]
        af_weighted[ipro,iparam] = np.average(af[ipro,valid,iparam],
                                              weights=weights)
        v1 = np.sum(weights)
        v2 = np.sum(weights**2)
        delsq = (af[ipro,valid,iparam] - af_weighted[ipro,iparam])**2
        vaf_weighted[ipro,iparam] = (v1/(v1**2-v2)) * np.sum(weights*delsq)
    print('\n*** Weighted parameters ***')
    for iparam in range(nparameters):
        print('  af[{}] = {:.3g} +/- {:.3g}'.format(iparam, 
                                             af_weighted[ipro,iparam], 
                                             np.sqrt(vaf_weighted[ipro,iparam])))
    
    if not noplot:
        title = '{} | {:.1f} s | {}'.format(pro['shot'],pro['time'],pro['desc'])
        # plot coefficients and variances
        fig,axes = plt.subplots(2,3, figsize=(12,5.5))
        for i,ax,coeff,vcoeff in zip(range(len(af_default)),
                                   axes.flat, 
                                   af[ipro,valid,:].T.tolist(),
                                   vaf[ipro,valid,:].T.tolist()):
            plt.sca(ax)
            for ic in range(len(coeff)):
                plt.errorbar(ic, coeff[ic], 
                             yerr=np.sqrt(vcoeff[ic]),
                             label='fit {}'.format(valid[ic]+1),
                             marker='x',
                             capsize=2)
            plt.errorbar(len(coeff), af_weighted[ipro,i],
                         yerr=np.sqrt(vaf_weighted[ipro,i]),
                         color='k',
                         marker='x',
                         capsize=2,
                         label='wt avg')
            ax.axhline(af_weighted[ipro,i], color='k')
            plt.ylabel('a[{}]'.format(i))
            plt.title(title)
            if i==0:
                plt.legend(ncol=2, loc='upper left', fontsize='small')
        plt.tight_layout()
        if savefig:
            plt.savefig('fit_params_{:02d}.eps'.format(ipro+1))
        # plot fits and weighted-average fit
        xgrid = np.linspace(-1.0, 1.0, 50)
        plt.figure(figsize=(10,4.5))
        plt.subplot(121)
        plt.errorbar(x, y, yerr=yerr, linestyle='', marker='x')
        for ivalid in valid:
            plt.plot(xgrid*xmax, feval(af[ipro,ivalid,:], xgrid), 
                     label='fit {}'.format(ivalid+1))
        plt.plot(xgrid*xmax, feval(af_weighted[ipro,:], xgrid), 
                 linewidth=2, 
                 label='wt avg')
        plt.legend()
        plt.xlim([0,None])
        plt.ylim([0,None])
        plt.title(title)
        # sample fits and plot
        plt.subplot(122)
        nfits = 15
        saf = np.zeros((nfits, nparameters))
        fits = np.zeros((nfits, xgrid.size))
        for iparam in range(nparameters):
            var = vaf_weighted[ipro,iparam]
            if var==0.0:
                var = 1e-16
            saf[:,iparam] = af_weighted[ipro,iparam] +\
                np.sqrt(var) * np.random.normal(size=nfits)
        plt.errorbar(x, y, yerr=yerr, linestyle='', marker='x')
        for ifit in range(nfits):
            fits[ifit,:] = feval(saf[ifit,:], xgrid)
            plt.plot(xgrid*xmax, fits[ifit,:], color='C2', linewidth=0.5)
        fitmin = np.min(fits, axis=0)
        fitmax = np.max(fits, axis=0)
        plt.fill_between(xgrid*xmax, fitmin, fitmax, color='C1')
        plt.xlim([0,None])
        plt.ylim([0,None])
        plt.title(title)
        plt.tight_layout()
        if savefig:
            plt.savefig('fit_{:02d}.eps'.format(ipro+1))
    # end plot section
    
    previous_shot = pro['shot']
    
# end loop over profiles

# pickle profile data
with open(datafile, 'wb') as f:
    pickle.dump(profile_data, f)

# pickle fit coefficients
fits = []
for ipro,pro in enumerate(profiles):
    fit = pro.copy()
    fit['coeff'] = af_weighted[ipro,:]
    fit['var'] = vaf_weighted[ipro,:]
    fits.append(fit.copy())
with open('ts_fits.pickle', 'wb') as f:
    pickle.dump(fits, f)