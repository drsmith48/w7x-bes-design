#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:14:45 2020

@author: drsmith
"""

"""
Y/Y0 = aa[1]+(1-aa[1])*(1-XX^aa[2])^aa[3]
    XX - r/a

a - aa[0] - Y0 - function value on-axis
e - aa[1] - gg - Y1/Y0 - function value at edge over core
c, d - aa[2],aa[3]-  pp, qq - power scaling parameters

Y/Y0 = aa[1]-aa[4]+(1-aa[1]+aa[4])*(1-XX^aa[2])^aa[3]+aa[4]*(1-exp(-XX^2/aa[5]^2))
    XX - r/a
    
dydx = a0*( -a2*a3*x^a2*(1 - x^a2)^a3*(-a1 - a4 + 1)/(x*(1 - x^a2)) + \
           2*a4*x*exp(-x^2/a5^2)/a5^2 )

h, w - aa[4],aa[5]-  hh, ww - hole depth and width

f/a = prof1/a + prof2

    prof1 = a*( e+(1-e)*(1-XX^c)^d )    # ModelTwoPower
{x element R: (d>0 and c=0 and x>0) or (d>0 and x=1)
        or (c>0 and 0<=x<1) or (c<0 and x>1) }

    prof2 = h*(1-exp(-XX^2/w^2))        # EdgePower
   {x element R}
"""


import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import scipy.constants as pc
import PCIanalysis.gradientlengths as gl
import hdf5
from plasma_parameters import Params



np.random.seed()

# shot/times for profiles
profiles_list = [
    [180904027, 1.9, 'post-pellet'],
    [180919007, 2.4, 'during NBI'],
    ]
profiles_default = [{'shot':p[0], 'time':p[1], 'desc':p[2]} for p in profiles_list]


def feval(c, x, nohollow):
    if nohollow:
        yfit = c[0] * (c[1] + (1-c[1])*(1-np.abs(x)**c[2])**c[3])
    else:
        yfit = c[0] * (c[1] - c[4] + (1-c[1]+c[4])*(1-np.abs(x)**c[2])**c[3] + \
                     c[4]*(1-np.exp(-(x**2)/(c[5]**2))))
    return yfit


def dfeval(c, x, nohollow):
    if nohollow:
        dyfit = -c[0] * ( -c[2]*c[3]*x**c[2]*(1 - np.abs(x)**c[2])**c[3] * 
                         (1-c[1])/(x*(1+1e-6 - np.abs(x)**c[2])))
    else:
        dyfit = -c[0] * ( -c[2]*c[3]*x**c[2]*(1 - np.abs(x)**c[2])**c[3] * 
                         (1-c[1]-c[4])/(x*(1+1e-6 - np.abs(x)**c[2])) + \
                        2*c[4]*x*np.exp(-x**2/c[5]**2)/c[5]**2 )
    return dyfit


def residuals(c, x, yresample, nohollow):
    # input: 1d array of n fit parameters
    # output: 1d array of m residuals: y - yfit
    yfit = feval(c, x/x.max(), nohollow=nohollow)
    res = yfit - yresample
    return res.reshape(res.size)


def fit_profiles(profiles=profiles_default,
                 noplot=False,
                 save_data=False,
                 save_figures=False,
                 save_fits=False,
                 fit_kwargs={},
                 plot_kwargs={}):
    prodatafile = 'profiles.pickle'
    # test profile data file
    try:
        with open(prodatafile, 'rb') as f:
            prodata = pickle.load(f)
        print('Data in {}'.format(prodatafile))
        for key,value in prodata.items():
            print('  ', key)
            for key2 in value.keys():
                print('    ', key2)
    except:
        prodata = {}
    fields = ['ne','te','ti']
    labels = ['ne [1e13/cm^3]',
              'Te [keV]',
              'Ti [keV]']
    nohollow = False
    nresample=5
    ts_edgecut = 2
    nfits = 60
    fits = []
    for shottime in profiles:
        shot = shottime['shot']
        time = shottime['time']
        desc = shottime['desc']
        if shot not in prodata:
            prodata[shot] = {}
        prodata[shot].pop('fits', None)
        fit = shottime.copy()
        plt.figure(figsize=(10,8))
        for ifield,field,label in zip(range(len(fields)), fields, labels):
            print('Getting {} data for shot {:d}'.format(field, shot))
            if field in ['ne','te']:
                try:
                    # with open(prodatafile, 'rb') as f:
                        # print('  Using data in {}'.format(prodatafile))
                        # prodata_loaded = pickle.load(f)
                    tsdata = prodata[shot]['ts']
                except:
                    # try:
                    #     with open(tsdatafile, 'rb') as f:
                    #         print('  Using data in {}'.format(tsdatafile))
                    #         tsprofiledata = pickle.load(f)
                    #         tsdata = tsprofiledata[shot]
                    # except:
                    print('  Fetching data from ArchiveDB')
                    tsdata = gl.get_thomsondata(shot)
                    prodata[shot]['ts'] = tsdata
                timearray = tsdata['time']/1e9
                tindex = np.argmin(np.abs(timearray - time))
                x = tsdata['roa'].copy()
                y = tsdata[field][tindex,:]
                yerr = tsdata['{}_err'.format(field)][tindex,:]
                # cut edge channels
                if ts_edgecut:
                    x = x[:-ts_edgecut]
                    y = y[:-ts_edgecut]
                    yerr = yerr[:-ts_edgecut]
                # remove any y=0 data points
                mask = np.logical_or(y == 0.0, yerr > 4*y)
                x = x[~mask]
                y = y[~mask]
                yerr = yerr[~mask]
                edgelim = 0.02*y.max()
                if (y[-1]>edgelim) or (yerr[-1]>edgelim):
                    x = np.append(x, np.max([1.1,x.max()+0.05]))
                    y = np.append(y, edgelim)
                    yerr = np.append(yerr, edgelim)
                ymax = np.max(y[x<=0.5])
                lb = np.array([ymax/2, 1e-3, 0.3, 0.3, -0.25, 0.2])
                ub = np.array([ymax*2, 0.06, 4.0, 4.0,  0.25, 0.5])
            else:
                try:
                    # with open(prodatafile, 'rb') as f:
                    #     print('  Using data in {}'.format(prodatafile))
                    #     prodata_loaded = pickle.load(f)
                    #     prodata[shot] = prodata_loaded[shot]
                    xicsdata = prodata[shot]['xics']
                except:
                    try:
                        datafile = list(Path().glob('*{}*.hdf5'.format(shot)))[0]
                        print('  Using data in {}'.format(datafile.as_posix()))
                        xicsdata = hdf5.hdf5ToDict(datafile)
                        prodata[shot]['xics'] = xicsdata
                    except:
                        continue
                rho = np.array(xicsdata['dim']['rho'])  # norm. ; 1D
                ti = np.array(xicsdata['value']['T_ion_Ar'])  # keV ; [time,rho]
                tierr = np.array(xicsdata['sigma']['T_ion_Ar'])  # keV ; [time,rho]
                mask = np.array(xicsdata['mask']['T_ion_Ar'])  # bool ; [time,rho]
                # time slice
                timearray = np.array(xicsdata['dim']['time'])  # s ; 1D
                tindex = np.argmin(np.abs(timearray - time))
                # mask[:] = True
                x = rho[mask[tindex,:]]
                y = ti[tindex,mask[tindex,:]]
                yerr = tierr[tindex,mask[tindex,:]]
                edgelim = 0.04*y.max()
                if (y[-1]>edgelim) or (yerr[-1]>edgelim):
                    x = np.append(x, np.max([1.05,x.max()+0.05]))
                    y = np.append(y, edgelim)
                    yerr = np.append(yerr, edgelim)
                ymax = np.max(y[x<=0.5])
                lb = np.array([ymax/2, 1e-3, 0.3, 0.3, -0.15, 0.2])
                ub = np.array([ymax*2,  0.4, 6.0, 4.0,  0.15, 0.6])
            x = x.reshape(-1,1)
            y = y.reshape(-1,1)
            yerr = yerr.reshape(-1,1)
            xvals = np.linspace(0.05,x.max(),101)
            axpro = plt.subplot(3,3,ifield*3+1)
            axgrad = plt.subplot(3,3,ifield*3+2)
            plt.sca(axpro)
            plt.errorbar(x, y, yerr=yerr, fmt='x')
            plt.xlabel('r/a')
            plt.ylabel(label)
            plt.ylim(0,None)
            plt.xlim(0,1.2)
            plt.title('{} | {:.2f} s | {}'.format(shot,timearray[tindex],desc))
            plt.sca(axgrad)
            plt.xlabel('r/a')
            plt.ylabel('-d ln( {} )/dx'.format(label))
            plt.title('{} | {:.2f} s | {}'.format(shot,timearray[tindex],desc))
            plt.xlim(0,1.2)
            plt.ylim(-5,15)
            # least squares fit
            c0 = np.array([  ymax, 0.01, 1.1, 1.5,  0,   0.25])
            x_scale = np.array([1,1e-2,0.1,0.1,5e-2,1e-2])
            if nohollow:
                c0 = c0[0:4]
                lb = lb[0:4]
                ub = ub[0:4]
                x_scale = x_scale[0:4]
            params = np.empty((c0.size,0))
            wtsqerr = np.empty(0)
            for i in range(nfits):
                # resample profile and fit
                yresample = y + yerr*np.random.normal(size=(y.size,nresample))
                result = least_squares(residuals, 
                                       c0, 
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
                    params = np.append(params, result.x.reshape(-1,1), axis=1)
                    yfit = feval(params[:,-1], x/x.max(), nohollow=nohollow)
                    # weighted squared error with raw data
                    err = np.sqrt(np.mean(np.arctan((yfit-y)/yerr/2)**2))
                    wtsqerr = np.append(wtsqerr, err.reshape((-1)))
            igood = np.nonzero(wtsqerr>=(np.median(wtsqerr)+wtsqerr.std()/2))[0]
            for i in igood:
                # plot fit to resampled data
                yfit = feval(params[:,i], xvals/xvals.max(), nohollow=nohollow)
                dyfit = dfeval(params[:,i], xvals/xvals.max(), nohollow=nohollow)
                plt.sca(axpro)
                plt.plot(xvals, yfit, color='C1', linewidth=0.5)
                plt.sca(axgrad)
                plt.plot(xvals, dyfit/yfit, color='C1', linewidth=0.5)
            plt.subplot(3,3,ifield*3+3)
            for icoeff in [0,1,2,3,4,5]:
                plt.plot([icoeff,icoeff], [lb[icoeff],ub[icoeff]], '_', 
                         color='k', 
                         mew=2,
                         ms=10)
                for i in igood:
                    plt.plot(icoeff, np.abs(params[icoeff,i]), 'o', 
                             color='C1',
                             linewidth=0.5,
                             ms=3)
            plt.title('{} | {:.2f} s | {}'.format(shot,timearray[tindex],desc))
            plt.yscale('log')
            plt.xlabel('fit coefficient index')
            plt.ylabel('{} fit coefficients'.format(field))
            ibest = np.argmax(wtsqerr)
            fit[field] = {'params':params[:,ibest], 'xmax':x.max()}
        plt.tight_layout()
        fits.append(fit)
        if save_figures:
            fname = Path('profiles_{}_{:.2f}s.pdf'.format(shot, timearray[tindex]))
            plt.savefig(fname.as_posix(), format='pdf', transparent=True)
    if save_fits:
        fitfile = 'fits.pickle'
        print('Saving fits in {}'.format(fitfile))
        with open(fitfile, 'wb') as f:
            pickle.dump(fits, f)
    if save_data:
        print('Saving profile data in {}'.format(prodatafile))
        with open(prodatafile, 'wb') as f:
            pickle.dump(prodata, f)


def profile_calculations():
    # load data and print contents
    prodatafile = 'fits.pickle'
    with open(prodatafile, 'rb') as f:
        fits = pickle.load(f)
    print('Available data in {}'.format(prodatafile))
    for fit in fits:
        for key,value in fit.items():
            print(key, value)
    nohollow=False
    # loop over input shots
    for fullfit in fits:
        # plot profiles and gradients
        plt.figure(figsize=[10,8])
        xmax = np.min([fullfit['ne']['xmax'],
                       fullfit['te']['xmax'],
                       fullfit['ti']['xmax']])
        x = np.linspace(0.05, xmax, 80)
        profiles = {}
        dprofiles = {}
        for ifield,field in enumerate(['ne','te','ti']):
            plt.subplot(4,4,1+ifield*2)
            fit = fullfit[field]
            params = fit['params']
            profiles[field] = feval(params, x/xmax, nohollow=nohollow)
            plt.plot(x, profiles[field])
            plt.xlabel('r/a')
            plt.ylabel(field)
            plt.title('{} | {:.2g} s'.format(fullfit['shot'],fullfit['time']))
            plt.subplot(4,4,2+ifield*2)
            dprofiles[field] = dfeval(params, x/x.max(), nohollow=nohollow)
            plt.plot(x, dprofiles[field] / profiles[field])
            plt.xlabel('r/a')
            plt.ylabel('d/drho ln({})'.format(field))
            plt.title('{} | {:.2g} s'.format(fullfit['shot'],fullfit['time']))
        params = []
        for ix in np.arange(x.size):
            params.append(Params(ne=profiles['ne'][ix]*1e13,
                                 Te=profiles['te'][ix],
                                 Ti=profiles['ti'][ix],
                                 ))
        rhoi = np.array([param.rho_i for param in params])
        k = np.array([param.k for param in params])
        lam = np.array([param.lam for param in params])
        Ti_J = np.array([param.Ti_J for param in params])
        omega_star_e = (k / pc.e / 2.6) * Ti_J * dprofiles['ne'] / profiles['ne']
        plt.subplot(4,4,7)
        plt.plot(x, rhoi*1e3)
        plt.ylim([0,None])
        plt.xlabel('r/a')
        plt.ylabel('rho-i (mm)')
        plt.title('{} | {:.2g} s'.format(fullfit['shot'],fullfit['time']))
        plt.subplot(4,4,8)
        plt.plot(x, lam*1e2)
        plt.ylim([0,None])
        plt.xlabel('r/a')
        plt.ylabel('2pi/k (cm) @ k*rho-i=0.3')
        plt.title('{} | {:.2g} s'.format(fullfit['shot'],fullfit['time']))
        plt.subplot(4,4,9)
        plt.plot(x, omega_star_e/(2*np.pi)/1e3)
        plt.xlabel('r/a')
        plt.ylabel('omega_star_e (kHz)')
        plt.title('{} | {:.2g} s'.format(fullfit['shot'],fullfit['time']))
        plt.tight_layout()
        break


if __name__=='__main__':
    plt.close('all')
    # fit_profiles(save_data=False, save_fits=False)
    profile_calculations()