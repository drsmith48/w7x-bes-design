#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 08:00:17 2019

@author: drsmith
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import MDSplus as mds
from FIT.fitNL import fit_mpfit
from FIT.model_spec import model_qparab
import PCIanalysis.gradientlengths as gl
import pybaseutils.utils as ut

plt.close('all')
np.random.seed()

def feval(af, x):
    assert(af.size==6)
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
profiles_default = [{'shot':p[0], 'time':p[1], 'desc':p[2]} for p in profiles_list]

def fit_xics(conn=None,      # must pass valid mds.Connection()
             profiles=profiles_default,
             save_profile_data=True):
    if not isinstance(conn, mds.Connection):
        print('Must provide valid mds.Connection()')
        print("conn = mds.Connection('ssh://dvs@mds-trm-1.ipp-hgw.mpg.de')")
        return
    # try loading pickled data
    datafile = 'xics_data.pickle'
    try:
        with open(datafile, 'rb') as f:
            profile_data = pickle.load(f)
#        assert(isinstance(profile_data, dict))
        print('Pickle load successful')
    except:
        profile_data = {}
        print('Pickle load unsuccessful, continuing')
    # MDS connection
#    try:
#        conn = mds.Connection('ssh://dvs@mds-trm-1.ipp-hgw.mpg.de')
#        mdsok = True
#    except:
#        mdsok = False
    previous_shot = -1
    xics_data = None
    for ipro,pro in enumerate(profiles):
        # load/fetch data
        if pro['shot'] != previous_shot:
            try:
                xics_data = profile_data[pro['shot']]
                print('** Loading XICS data from {}, shot {} ...'.
                      format(datafile, pro['shot']))
            except KeyError:
                print('** Getting XICS data, shot {} ...'.format(pro['shot']))
#                if not mdsok:
#                    raise RuntimeError('Need MDSplus connection')
                try:
                    conn.openTree('qsw_eval', pro['shot'])
                    time = conn.get('xics:ti:time')
                    reff = conn.get('xics:ti:reff')
                    ti = conn.get('xics:ti')
                    tierr = conn.get('xics:ti:sigma')
                    a = gl._get_minor_radius(pro['shot'])
                except:
                    print('** MDSPLUS error **')
                    continue
                xics_data = {'ti':ti,
                             'tierr':tierr,
                             'time':time,
                             'roa':None,
                             'reff':reff,
                             'a':a}
                profile_data[pro['shot']] = xics_data
    # end loop over profiles
    
    # pickle profile data
    if save_profile_data:
        with open(datafile, 'wb') as f:
            pickle.dump(profile_data, f)

def fit_ts(field='ne',                  # 'ne' or 'te'
           profiles=profiles_default,   # list of dictionaries, like profiles_default
           nattempts=50,                # typ. 15-100
           chi2_perc_limit=15,          # typ. 10-40
           maxiter=40,                  # max iter. for NL fit; typ. 20-100
           nohollow=True,               # allow hollow core?
           zerosol=True,                # allow non-zero in far-SOL?
           noplot=False,                # plot results to screen?
           savefig=True,                # save EPS figures?
           save_profile_data=False,     # save TS profile data?
           save_coeff=True):            # save fit coefficients?

    # other controls
    tol = 1.0e-6
    lmfit_quiet = True
    edgecut=2
    
    # default fit to get initial parameter values
    info = model_qparab(nohollow=nohollow)
    af_default = info.af.copy()
    nparameters = len(af_default)
    
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

    # output containers
    nprofiles = len(profiles)
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
        # unpack data
        time = tsdata['time']/1e9
        index = np.argmin(np.abs(time - pro['time']))
        ts_time = tsdata['time'][index]/1e9
        x = tsdata['roa'].copy()
        if field.lower().startswith(('ne','den')):
            field = 'ne'
        elif field.lower().startswith('te'):
            field = 'te'
        y = tsdata[field][index,:]
        yerr = tsdata['{}_err'.format(field)][index,:]
        print('**** Performing {} profile fit, shot {}, time {:.2f} s ...'.
              format(field, pro['shot'], ts_time))
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
        # compress x so max(x) = 1.0
        xmax = np.max(np.abs(x))
        xscaled = x / xmax
        # adjust initial parameters based on data
        af_default[0] = y.max()
        af_default[1] = np.min((np.max((0.02, y.min()/af_default[0])), 0.99))
        # loop over fit attempts
        for iatt in range(nattempts):
            # sample y data according to y uncertainty
            ysample = y + yerr*np.random.normal(0.0,1.0,y.shape)
            # enforce y>=0
            ysample[ysample<0] = 0
            y_ub = 1.5 * np.max(ysample[x<=0.5])
            # variance w.r.t. true y data
            vary = (ysample-y)*(ysample-y)
            vary[vary==0] = np.nanmean(vary)
            af_initial = af_default.copy()
            af_initial[0] = y.max() + yerr[0]*np.random.normal()
            af_initial[1] = 0.0
            LB = np.array([   0,    0,  0,  0,  -0.3,  0])
            UB = np.array([y_ub, 0.05,  5,  5,   0.3,  1])
            fixed = np.array([0,zerosol,0,0,nohollow,nohollow])
            try:
                info = fit_mpfit(xscaled, ysample, np.sqrt(vary), np.linspace(-1,1), model_qparab, 
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
                                 LB=LB,
                                 UB=UB,
                                 fixed=fixed,
                                 )
                print('  Run {:02d} of {:02d}  chi2_reduced {:8.1f} with {:3d} iterations'.
                      format(iatt+1, nattempts, info.chi2_reduced, info.niter))
            except:
                print('  Run {:02d} of {:02d} failed'.format(iatt+1,nattempts))
                continue
            # capture output
            af[ipro,iatt, :] = info.af
            covmat[ipro,iatt,:,:] = info.covmat
            vaf[ipro,iatt, :] = info.perror**2
            fit_chi2_reduced[ipro,iatt] = info.chi2_reduced
            iterations[ipro,iatt] = info.niter
            # residual with input data
            residual = (y-feval(info.af, xscaled)) / yerr
            # limit large residuals
            residual_limit= np.percentile(np.abs(residual), 96)
            residual[residual>residual_limit] = residual_limit
            residual[residual<-residual_limit] = -residual_limit
            chi2_reduced[ipro,iatt] = np.sum(residual**2) / info.dof
            
        # end resample loop
    
        print('\n*** Results ***\n')
        mask = np.isfinite(np.squeeze(iterations[ipro,:]))
        if not np.any(mask):
            print('No valid fits, continuing')
            continue
        chi2_limit = np.percentile(chi2_reduced[ipro,mask], chi2_perc_limit)
        valid = np.squeeze(np.argwhere(np.logical_and(chi2_reduced[ipro,:] <= chi2_limit, mask)))
        for iparam in range(nparameters):
            print('af[{}]'.format(iparam))
            for ivalid in valid:
                print('  fit {:02d}: {:10.3e} +/- {:10.3e}'.
                      format(ivalid+1, 
                             af[ipro,ivalid,iparam], 
                             np.sqrt(vaf[ipro,ivalid,iparam])))
            vaff = vaf[ipro,valid,iparam]
            vaff[vaff==0] = 1e16
            weights = 1/vaff
            af_weighted[ipro,iparam] = np.average(af[ipro,valid,iparam],
                                                  weights=weights)
            v1 = np.sum(weights)
            v2 = np.sum(weights**2)
            delsq = (af[ipro,valid,iparam] - af_weighted[ipro,iparam])**2
            vaf_weighted[ipro,iparam] = (v1/(v1**2-v2)) * np.sum(weights*delsq)
        print('\n*** Weighted parameters ***')
        for iparam in range(nparameters):
            print('  af[{}] = {:.3g} +/- {:.3g}'.
                  format(iparam, af_weighted[ipro,iparam], 
                         np.sqrt(vaf_weighted[ipro,iparam])))
        
        if not noplot:
            title = '{} | {:.2f} s | {}'.format(pro['shot'],ts_time,pro['desc'])
            xgrid = np.linspace(-1.0, 1.0, 50)
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
                plt.ylabel('{} a[{}]'.format(field, i))
                plt.title(title)
                if i==0:
                    plt.legend(ncol=2, loc='upper left', fontsize='small')
            plt.tight_layout()
            if savefig:
                plt.savefig('fit_params_{}_{:02d}.eps'.
                            format(field,ipro+1))
            # plot fits and weighted-average fit
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
            yuplim = np.ceil(y.max()/2) * 2 + 2
            plt.ylim([0,yuplim])
            plt.xlabel('reff')
            plt.ylabel(field.lower())
            plt.title(title)
            # sample fits and plot
            nfits = 15
            saf = np.zeros((nfits, nparameters))
            fits = np.zeros((nfits, xgrid.size))
            for iparam in range(nparameters):
                saf[:,iparam] = af_weighted[ipro,iparam] + \
                    np.sqrt(vaf_weighted[ipro,iparam]) * np.random.normal(size=nfits)
            for ifit in range(nfits):
                fits[ifit,:] = feval(saf[ifit,:], xgrid)
            fitmin = np.min(fits, axis=0)
            fitmax = np.max(fits, axis=0)
            plt.subplot(122)
            plt.errorbar(x, y, yerr=yerr, linestyle='', marker='x')
            for ifit in range(nfits):
                plt.plot(xgrid*xmax, fits[ifit,:], color='C2', linewidth=0.5)
            plt.fill_between(xgrid*xmax, fitmin, fitmax, color='C1')
            plt.xlim([0,None])
            plt.ylim([0,yuplim])
            plt.xlabel('reff')
            plt.ylabel(field.lower())
            plt.title(title)
            plt.tight_layout()
            if savefig:
                plt.savefig('fit_{}_{:02d}.eps'.
                            format(field,ipro+1))
        # end plot section
        
        previous_shot = pro['shot']
        
    # end loop over profiles
    
    # pickle profile data
    if save_profile_data:
        with open(datafile, 'wb') as f:
            pickle.dump(profile_data, f)
    
    # pickle fit coefficients
    if save_coeff:
        fits = []
        for ipro,pro in enumerate(profiles):
            fit = pro.copy()
            fit['coeff'] = af_weighted[ipro,:]
            fit['var'] = vaf_weighted[ipro,:]
            fits.append(fit.copy())
        filename = '{}_fits.pickle'.format(field.lower())
        with open(filename, 'wb') as f:
            pickle.dump(fits, f)
        
# end fit_ts()

    
if __name__=='__main__':
#    pass
#    fit_xics()
    fit_ts('ne', nattempts=80, chi2_perc_limit=10, maxiter=50)
    fit_ts('te', nattempts=80, chi2_perc_limit=10, maxiter=50)