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
import PCIanalysis.gradientlengths as gl
import hdf5


plt.close('all')
np.random.seed()

# shot/times for profiles
profiles_list = [
                 [180904027, 1.9, 'post-pellet'],
                 [180919007, 2.4, 'during NBI'],
                ]
profiles_default = [{'shot':p[0], 'time':p[1], 'desc':p[2]} for p in profiles_list]


def fit_profiles(profiles=profiles_default,
                 noplot=False,
                 save_data=False,
                 fit_kwargs={},
                 plot_kwargs={}):
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
    tsdatafile = 'ts_data.pickle'
    prodatafile = 'profiles.pickle'
    prodata2 = {}
    fields = ['ne','te','ti']
    nohollow = False
    nresample=5
    ts_edgecut = 2
    nfits = 50
    for shottime in profiles:
        shot = shottime['shot']
        time = shottime['time']
        desc = shottime['desc']
        prodata2[shot] = {}
        plt.figure(figsize=(10,8))
        for ifield,field in enumerate(fields):
            print('Getting {} data for shot {:d}'.format(field, shot))
            if field in ['ne','te']:
                try:
                    with open(prodatafile, 'rb') as f:
                        print('  Using data in {}'.format(prodatafile))
                        prodata = pickle.load(f)
                        tsdata = prodata[shot]['ts']
                except:
                    try:
                        with open(tsdatafile, 'rb') as f:
                            print('  Using data in {}'.format(tsdatafile))
                            tsprofiledata = pickle.load(f)
                            tsdata = tsprofiledata[shot]
                    except:
                        print('  Fetching data from ArchiveDB')
                        tsdata = gl.get_thomsondata(shot)
                prodata2[shot]['ts'] = tsdata
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
                    with open(prodatafile, 'rb') as f:
                        print('  Using data in {}'.format(prodatafile))
                        prodata = pickle.load(f)
                        xicsdata = prodata[shot]['xics']
                except:
                    try:
                        raise Exception
                    except:
                        try:
                            datafile = list(Path().glob('*{}*.hdf5'.format(shot)))[0]
                            print('  Using data in {}'.format(datafile.as_posix()))
                            xicsdata = hdf5.hdf5ToDict(datafile)
                        except:
                            continue
                prodata2[shot]['xics'] = xicsdata
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
                ymax = np.max(y[x<=0.5])
                lb = np.array([ymax/2, 1e-3, 0.3, 0.3, -0.25, 0.2])
                ub = np.array([ymax*2,  0.4, 6.0, 4.0,  0.25, 0.6])
            x = x.reshape(-1,1)
            y = y.reshape(-1,1)
            yerr = yerr.reshape(-1,1)
            plt.subplot(3,3,ifield*3+1)
            plt.errorbar(x, y, yerr=yerr, fmt='x')
            plt.xlabel('r/a')
            plt.ylabel(field)
            plt.ylim(0,None)
            plt.xlim(0,1.2)
            plt.title('{} | {:.2f} s | {}'.format(shot,timearray[tindex],desc))
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
                    # plot fit to resampled data
                    plt.plot(x, yfit, color='C1', linewidth=0.5)
            ibest = np.argsort(wtsqerr)[:3]
            for i in ibest:
                yfit = feval(params[:,i], x/x.max(), nohollow=nohollow)
                # plot best resampled fits
                plt.plot(x, yfit, color='C2', linewidth=2)
            plt.subplot(3,3,ifield*3+2)
            for i in [0,1,2,3,4,5]:
                plt.plot([i,i], [lb[i],ub[i]], '_', 
                         color='k', 
                         mew=2,
                         ms=10)
                plt.plot(np.zeros(params.shape[1])+i, np.abs(params[i,:]), 'o', 
                         color='C1',
                         linewidth=0.5,
                         ms=3)
                for ii in ibest:
                    plt.plot(i, np.abs(params[i,ii]), 'o', 
                             color='C2',
                             linewidth=0.5,
                             ms=3)
            plt.yscale('log')
            plt.xlabel('fit coefficients')
        plt.tight_layout()
    if save_data:
        print('Saving profile data in {}'.format(prodatafile))
        with open(prodatafile, 'wb') as f:
            pickle.dump(prodata2, f)


if __name__=='__main__':
    plt.close('all')
    fit_profiles(save_data=False)