#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 08:00:17 2019

@author: drsmith
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import PCIanalysis.gradientlengths as gl

plt.close('all')
np.random.seed()

# shotlist for profiles
profiles_list = [
#                 [180904027, 1.0, 'pre-pellet'],
                 [180904027, 1.9, 'post-pellet'],
#                 [180906040, 2.0, '5.0 MW ECH'],
#                 [180906040, 4.5, '4.2 MW ECH'],
#                 [180906040, 7.0, '2.5 MW ECH'],
#                 [180906040, 9.5, '1.5 MW ECH'],
#                 [180919007, 1.4, 'pre-NBI'],
                [180919007, 2.4, 'during NBI'],
#                 [180919007, 3.2, 'post-NBI'],
#                 [180925033, 0.65, 'low density'],
#                 [180925033, 3.9, 'high density'],
                ]
profiles_default = [{'shot':p[0], 'time':p[1], 'desc':p[2]} for p in profiles_list]


"""
Y/Y0 = aa[1]+(1-aa[1])*(1-XX^aa[2])^aa[3]
    XX - r/a

a - aa[0] - Y0 - function value on-axis
e - aa[1] - gg - Y1/Y0 - function value at edge over core
c, d - aa[2],aa[3]-  pp, qq - power scaling parameters

Y/Y0 = aa[1]-aa[4]+(1-aa[1]+aa[4])*(1-XX^aa[2])^aa[3]+aa[4]*(1-exp(-XX^2/aa[5]^2))
    XX - r/a

h, w - aa[4],aa[5]-  hh, ww - hole depth and width

f/a = prof1/a + prof2

    prof1 = a*( e+(1-e)*(1-XX^c)^d )    # ModelTwoPower
{x element R: (d>0 and c=0 and x>0) or (d>0 and x=1)
        or (c>0 and 0<=x<1) or (c<0 and x>1) }

    prof2 = h*(1-exp(-XX^2/w^2))        # EdgePower
   {x element R}
"""

def fit_ts(profiles=profiles_default,   # list of dictionaries, like profiles_default
           noplot=False,                # plot results to screen?
           save_figs=False,               # save EPS figures?
           save_profile_data=False,     # save TS profile data?
           save_fits=False,              # save best fits?
           nohollow=False):             # prohibit hollow core?
    # sub-functions
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
        res = yfit - yresample
        return res.reshape(res.size)
    # other settings
    edgecut=2
    nfits=60
    nresample=5
    # try loading pickled data
    ts_datafile = 'ts_data.pickle'
    try:
        with open(ts_datafile, 'rb') as f:
            profile_data = pickle.load(f)
        assert(isinstance(profile_data, dict))
        print('Pickle load successful')
        for key in profile_data.keys():
            print('  {}'.format(key))
    except:
        profile_data = {}
        print('Pickle load unsuccessful, continuing')
    # loop over profile instances
    previous_shot = -1
    tsdata = None
    goodfits = []
    for shottime in profiles:
        shot = shottime['shot']
        time = shottime['time']
        description = shottime['desc']
        # load/fetch data
        if shot != previous_shot:
            try:
                # raise KeyError
                tsdata = profile_data[shot]
                print('Extracting MPTS data from {}, shot {}'.
                      format(ts_datafile, shot))
            except KeyError:
                print('Fetching MPTS data, shot {}'.format(shot))
                tsdata = gl.get_thomsondata(shot)
                profile_data[shot] = tsdata
        # unpack data
        index = np.argmin(np.abs(tsdata['time']/1e9 - time))
        ts_time = tsdata['time'][index]/1e9
        goodfits_shot = shottime.copy()
        # begin loop over ne/te fields
        for field in ['ne','te']:
            x = tsdata['roa'].copy()
            y = tsdata[field][index,:]
            yerr = tsdata['{}_err'.format(field)][index,:]
            print('{} profile fit, shot {}, time {:.2f} s ...'.
                  format(field, shot, ts_time))
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
            edgelim = 0.02*y.max()
            if (y[-1]>edgelim) or (yerr[-1]>edgelim):
                x = np.append(x, np.max([1.1,x.max()+0.05]))
                y = np.append(y, edgelim)
                yerr = np.append(yerr, edgelim)
            # make 2d
            x = x[:,np.newaxis]
            y = y[:,np.newaxis]
            yerr = yerr[:,np.newaxis]
            goodfits_shot[field+'rawdata'] = {'x':x, 'y':y, 'yerr':yerr}
            # plot raw data and uncertainties
            plt.figure(figsize=(7.66,3.25))
            plt.subplot(1,2,1)
            plt.errorbar(x, y, yerr=yerr, fmt='x')
            plt.xlabel('r/a')
            plt.ylabel(field)
            plt.ylim(0,None)
            plt.title('{} | {:.2f} s | {}'.format(shot,ts_time,description))
            # least squares fitting, optimized with Numba
            ymax = np.max(y[x<=0.5])
            c0 = np.array([  ymax, 0.01, 1.1, 1.5,  0,   0.25])
            lb = np.array([ymax/2, 1e-3, 0.3, 0.3, -0.25, 0.2])
            ub = np.array([ymax*2, 0.06, 4.0, 4.0,  0.25, 0.5])
            x_scale = np.array([1,1e-2,0.1,0.1,5e-2,1e-2])
            if nohollow:
                c0 = c0[0:4]
                lb = lb[0:4]
                ub = ub[0:4]
                x_scale = x_scale[0:4]
            cresults = np.empty((c0.size,0))
            wtsqerr = np.empty(0)
            for i in range(nfits):
                # resample profile and fit
                yresample = y + yerr*np.random.normal(size=(y.size,nresample))
                result = least_squares(residuals, c0, 
                                       jac='3-point', 
                                       bounds=(lb,ub),
                                       method='trf',
                                       x_scale=x_scale,
                                       loss='arctan',
                                       f_scale=1.0,
                                       verbose=0,
                                       args=(x,yresample),
                                       kwargs={'nohollow':nohollow})
                if result.status > 0:
                    cf = result.x
                    cresults = np.append(cresults, cf[...,np.newaxis], axis=1)
                    yfit = feval(cf, x/x.max(), nohollow=nohollow)
                    # weighted squared error with raw data
                    err = np.sqrt(np.mean(np.arctan((yfit-y)/yerr/2)**2))
                    wtsqerr = np.append(wtsqerr, err.reshape((-1)), axis=0)
                    # plot fit to resampled data
                    plt.plot(x, yfit, color='C1', linewidth=0.5)
            ibestfit = np.argsort(wtsqerr)[:3]
            fieldfits = []
            for i in ibestfit:
                ybestfit = feval(cresults[:,i], x/x.max(), nohollow=nohollow)
                # plot best resampled fits
                plt.plot(x, ybestfit, color='C2', linewidth=2)
                fieldfits.append({'x':x, 'y':ybestfit, 'coeff':cresults[:,i]})
            goodfits_shot[field+'fits'] = fieldfits
            if field=='te':
                plt.ylim([0,6])
            plt.subplot(1,2,2)
            for i in [0,1,2,3,4,5]:
                plt.plot([i,i], [lb[i],ub[i]], '_', 
                         color='k', 
                         mew=2,
                         ms=10)
                plt.plot(np.zeros(cresults.shape[1])+i, np.abs(cresults[i,:]), 'o', 
                         color='C1',
                         linewidth=0.5,
                         ms=3)
                for ii in ibestfit:
                    plt.plot(i, np.abs(cresults[i,ii]), 'o', 
                             color='C2',
                             linewidth=0.5,
                             ms=3)
            plt.yscale('log')
            plt.xlabel('fit coefficients')
            plt.tight_layout()
            if save_figs:
                filename = 'fit_{}_{:.0f}ms_{}.pdf'.format(shot,ts_time*1e3, field)
                print('saving file: {}'.format(filename))
                plt.savefig(filename, transparent=True)
        # end loop over fields
        previous_shot = shot
        goodfits.append(goodfits_shot)
    # end loop over profile instances
    
    if save_fits:
        fits_datafile = 'ts_fits.pickle'
        print('saving {}'.format(fits_datafile))
        with open(fits_datafile, 'wb') as f:
            pickle.dump(goodfits, f)
    
    # pickle profile data
    if save_profile_data:
        with open(ts_datafile, 'wb') as f:
            pickle.dump(profile_data, f)
    
    return goodfits


if __name__=='__main__':
    plt.close('all')
    fits = fit_ts(save_figs=False, save_fits=False)