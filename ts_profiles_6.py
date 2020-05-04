#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 08:00:17 2019

@author: drsmith
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
#from scipy.interpolate import InterpolatedUnivariateSpline as spline
#import MDSplus as mds
#from FIT.fitNL import fit_mpfit
#from FIT.model_spec import model_qparab
import PCIanalysis.gradientlengths as gl
#import pybaseutils.utils as ut

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
#                 [180919007, 2.4, 'during NBI'],
#                 [180919007, 3.2, 'post-NBI'],
#                 [180925033, 0.65, 'low density'],
####                 [180925033, 3.9, 'high density'],
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

def feval(c, x, nohollow=True):
    if nohollow:
        yfit = c[0] * (c[1] + (1-c[1])*(1-np.abs(x)**c[2])**c[3])
    else:
        yfit = c[0] * (c[1] - c[4] + (1-c[1]+c[4])*(1-np.abs(x)**c[2])**c[3] + \
                     c[4]*(1-np.exp(-(x**2)/(c[5]**2))))
    return yfit

def residuals(c, x, yresample, nohollow=True):
    # input: 1d array of n fit parameters
    # output: 1d array of m residuals: y - yfit
    yfit = feval(c, x/x.max(), nohollow=nohollow)
    res = yfit - yresample
    return res.reshape(res.size)

def fit_ts(profiles=profiles_default,   # list of dictionaries, like profiles_default
           noplot=False,                # plot results to screen?
           savefig=False,                # save EPS figures?
           save_profile_data=False,     # save TS profile data?
           ):            # save fit coefficients?

    # other controls
    edgecut = 2
    nresamples = 5
    nfits = 60
    nohollow = False
    
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

    # loop over profile instances
    previous_shot = -1
    tsdata = None
    for ipro,pro in enumerate(profiles):
        # load/fetch data
        if pro['shot'] != previous_shot:
            try:
                raise KeyError
                tsdata = profile_data[pro['shot']]
                print('Extracting MPTS data from {}, shot {}'.
                      format(datafile, pro['shot']))
            except KeyError:
                print('Fetching MPTS data, shot {}'.format(pro['shot']))
                tsdata = gl.get_thomsondata(pro['shot'])
                profile_data[pro['shot']] = tsdata
        # unpack data
        time = tsdata['time']/1e9
        index = np.argmin(np.abs(time - pro['time']))
        ts_time = tsdata['time'][index]/1e9
        # begin loop over ne/te fields
        for field in ['ne','te']:
            x = tsdata['roa'].copy()
            y = tsdata[field][index,:]
            yerr = tsdata['{}_err'.format(field)][index,:]
            print('{} profile fit, shot {}, time {:.2f} s ...'.
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
            edgelim = 0.02*y.max()
            if (y[-1]>edgelim) or (yerr[-1]>edgelim):
                x = np.append(x, np.max([1.1,x.max()+0.05]))
                y = np.append(y, edgelim)
                yerr = np.append(yerr, edgelim)
            # make 2d
            x = x[:,np.newaxis]
            y = y[:,np.newaxis]
            yerr = yerr[:,np.newaxis]
            # plot profile
            plt.figure(figsize=(7.66,3.25))
            title = '{} | {:.2f} s | {}'.format(pro['shot'],ts_time,pro['desc'])
            plt.subplot(1,2,1)
            plt.errorbar(x, y, yerr=yerr, fmt='x')
            plt.xlabel('r/a')
            plt.ylabel(field)
            plt.ylim(0,None)
            plt.title(title)
            # least squares fitting, optimized with Numba
            ymax = np.max(y[x<=0.5])
            c0 = np.array([  ymax, 0.01, 1.1, 1.5,  0,   0.25])
            lb = np.array([ymax/2, 1e-3, 0.3, 0.3, -0.3, 0.2])
            ub = np.array([ymax*2, 0.05, 4.0, 4.0,  0.3, 0.5])
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
                yresample = y + yerr*np.random.normal(size=(y.size,nresamples))
                result = opt.least_squares(residuals, c0, 
                                           jac='3-point', 
                                           bounds=(lb,ub),
                                           method='trf',
                                           x_scale=x_scale,
                                           loss='arctan',
                                           f_scale=1.0,
                                           verbose=0,
                                           args=(x,yresample),
                                           kwargs={'nohollow':nohollow},
                                           )
                if result.status > 0:
                    cf = result.x
                    cresults = np.append(cresults, cf[...,np.newaxis], axis=1)
                    yfit = feval(cf, x/x.max(), nohollow=nohollow)
                    err = np.sqrt(np.mean(np.arctan((yfit-y)/yerr/2)**2))
                    wtsqerr = np.append(wtsqerr, err.reshape((-1)), axis=0)
                    plt.plot(x, yfit, 
                             color='C1', 
                             linewidth=0.5)
            ibestfit = np.argsort(wtsqerr)[:3]
            for i in ibestfit:
                print(wtsqerr[i], x.max(), cresults[:,i])
                ybestfit = feval(cresults[:,i], x/x.max(), nohollow=nohollow)
                plt.plot(x, ybestfit, color='C2', linewidth=2)
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
            if savefig:
                filename = 'fit_{}_{:.0f}ms_{}.pdf'.format(
                        pro['shot'],ts_time*1e3, field)
                print('saving file: {}'.format(filename))
                plt.savefig(filename, transparent=True)
    # end loop over fields
        previous_shot = pro['shot']
    # end loop over profile instances
    
    # pickle profile data
    if save_profile_data:
        with open(datafile, 'wb') as f:
            pickle.dump(profile_data, f)
    
    return


# ne fits
# x -> x / 1.14491
#0.3832149491929913 [10.22178262  0.0267643   1.84138753  1.62809221  0.3         0.31130153]
#0.38508911759246744 [10.21214233  0.01847742  1.62975576  1.4027948   0.29999999  0.2764953 ]
#0.386183671976114 [10.29204212  0.02788882  1.52095351  1.37012837  0.3         0.28085452]
#0.3862077999537648 [10.10663327  0.03604462  1.8833487   1.78035585  0.3         0.26743233]
#0.3868888305656996 [10.46844203  0.03319377  1.60784881  1.47541005  0.3         0.31804642]
#0.38729798392689213 [10.10015271  0.02180125  2.32219243  1.71327306  0.10866066  0.2       ]
#0.3873835115462206 [10.31760281  0.03812322  1.74278578  1.6873627   0.3         0.26225392]
#0.38752872535334737 [10.1175267   0.03805694  1.5673755   1.41982793  0.3         0.23777588]

# te fits
# x -> x / 1.09491
#0.1598875948517293 [ 3.8076127   0.00651745  2.22383115  2.85146903 -0.20842746  0.20145577]
#0.16498508346304566 [4.05836146 0.00731043 1.40823278 2.34001768 0.27358165 0.40837264]
#0.16821321133894515 [3.91150558 0.0070859  1.40659832 2.29064216 0.29999997 0.41936713]
#0.16892623670597592 [ 3.92132790e+00  1.00000000e-03  2.50898762e+00  3.02284032e+00 -3.00000000e-01  2.06457456e-01]
#0.16914170144171506 [4.27291446e+00 1.55405235e-03 1.13168927e+00 2.06825377e+00 3.00000000e-01 2.93541719e-01]
#0.17013981129440303 [3.87582258e+00 3.75811319e-03 1.39253322e+00 2.43313538e+00 3.00000000e-01 3.00983564e-01]
#0.17176957913152938 [4.14899435 0.00823581 1.26460161 2.23564824 0.3        0.34519777]
#0.17205778799237928 [4.08306349 0.00808071 1.37444638 2.39936535 0.29999964 0.33728493]
#0.17255377507747327 [ 3.72028113  0.00811113  2.88445446  3.62993618 -0.3         0.2113322 ]

if __name__=='__main__':
    plt.close('all')
    fit_ts(savefig=False)