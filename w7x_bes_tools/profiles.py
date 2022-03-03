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

import hdf5
import PCIanalysis.gradientlengths as gl

try:
    from .utilities.plasma_parameters import Params
except ImportError:
    from w7x_bes_tools.utilities.plasma_parameters import Params


np.random.seed()

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


# shot/times for profiles
profiles_list = [
    [180904027, 1.9, 'post-pellet'],
    [180919007, 2.4, 'during NBI'],
    ]


class Profiles(object):
    
    _fields = ['ne','te','ti']
    _labels = ['ne (1e13/cm^3)',
               'Te (keV)',
               'Ti (keV)']
    
    def __init__(self, 
                 iprofile=0,
                 nohollow=True,
                 no_saved_data=False):
        self.shot, self.time, self.desc = profiles_list[iprofile]
        self.nohollow = nohollow
        self.nfits = 50
        self.fit_inputs = {}
        self.fit_results = {}
        self.best_fits = {}
        self.get_data(no_saved_data=no_saved_data)
        self.do_ts_fits()
        self.do_xics_fits()
        
    def get_data(self, no_saved_data=False):
        datafile = Path('data') / f'profile_data_{self.shot}.pickle'
        try:
            with datafile.open('rb') as f:
                self.prodata = pickle.load(f)
            assert('ts' in self.prodata.keys())
            assert('xics' in self.prodata.keys())
            print('Available profile data in {}'.format(datafile.as_posix()))
        except:
            print('Saved profile data not found')
            self.prodata = {}
        if 'ts' not in self.prodata.keys() or no_saved_data:
            print('Fetching TS data from ArchiveDB')
            self.prodata['ts'] = gl.get_thomsondata(self.shot)
        if 'xics' not in self.prodata.keys() or no_saved_data:
            xicsfile = next(Path('data').glob(f'*{self.shot}*.hdf5'))
            assert(xicsfile.exists())
            print('Using XICS data in {}'.format(xicsfile.as_posix()))
            self.prodata['xics'] = hdf5.hdf5ToDict(xicsfile)
        print('Saving profile data in {}'.format(datafile.as_posix()))
        with datafile.open('wb') as f:
            pickle.dump(self.prodata, f)
                    
    def do_ts_fits(self):
        tsdata = self.prodata['ts']
        timearray = tsdata['time']/1e9
        tindex = np.argmin(np.abs(timearray - self.time))
        ts_edgecut = 2
        for field in ['ne','te']:
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
            lb = np.array([ymax/2, 1e-3, 0.3, 0.3, 1e-3, 0.2])
            ub = np.array([ymax*2, 5e-2, 4.0, 4.0,  0.15, 0.5])
            self.fit_inputs[field] = (x, y, yerr, lb, ub, timearray[tindex])
            self.do_fits(field, x, y, yerr, lb, ub)
            
    def do_xics_fits(self):
        xicsdata = self.prodata['xics']
        rho = np.array(xicsdata['dim']['rho'])  # norm. ; 1D
        ti = np.array(xicsdata['value']['T_ion_Ar'])  # keV ; [time,rho]
        tierr = np.array(xicsdata['sigma']['T_ion_Ar'])  # keV ; [time,rho]
        mask = np.array(xicsdata['mask']['T_ion_Ar'])  # bool ; [time,rho]
        # time slice
        timearray = np.array(xicsdata['dim']['time'])  # s ; 1D
        tindex = np.argmin(np.abs(timearray - self.time))
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
        lb = np.array([ymax/2, 1e-3, 0.3, 0.3, 1e-3, 0.2])
        ub = np.array([ymax*2, 5e-2, 6.0, 4.0,  0.15, 0.6])
        field = 'ti'
        self.fit_inputs[field] = (x, y, yerr, lb, ub, timearray[tindex])
        self.do_fits(field, x, y, yerr, lb, ub)

    def do_fits(self, field, x, y, yerr, lb, ub):
            print(f'Doing {field} fits')
            params, wtsqerr = self.calc_fits(x, y, yerr, lb, ub)
            self.fit_results[field] = (params, wtsqerr)
            bestparams = self.find_best_params(params, wtsqerr)
            self.best_fits[field] = {'params':bestparams, 
                                     'xmax':x.max(),
                                     'label':field}
    
    def calc_fits(self, x, y, yerr, lb, ub):
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        yerr = yerr.reshape(-1,1)
        # least squares fit
        ymax = np.max(y[x<=0.5])
        c0 = np.array([  ymax, 0.01, 1.1, 1.5,  1e-2,   0.25])
        x_scale = np.array([1,1e-2,0.1,0.1,5e-2,1e-2])
        if self.nohollow:
            c0 = c0[0:4]
            lb = lb[0:4]
            ub = ub[0:4]
            x_scale = x_scale[0:4]
        params = np.empty((c0.size,0))
        wtsqerr = np.empty(0)
        nresample=5
        for i in range(self.nfits):
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
                                   kwargs={'nohollow':self.nohollow})
            if result.status > 0:
                params = np.append(params, result.x.reshape(-1,1), axis=1)
                yfit = feval(params[:,-1], x/x.max(), nohollow=self.nohollow)
                # weighted squared error with raw data
                err = np.sqrt(np.mean(np.arctan((yfit-y)/yerr/2)**2))
                wtsqerr = np.append(wtsqerr, err.reshape((-1)))
        return (params, wtsqerr)
    
    def find_best_params(self, params, wtsqerr):
        isort = np.argsort(wtsqerr)[0:self.nfits//10]
        bestfits = params[:,isort]
        bestparams = np.empty(bestfits.shape[0])
        lesszero = np.count_nonzero(bestfits<0, axis=1)
        for iparam,zerocount in enumerate(lesszero):
            if zerocount==0:
                bestparams[iparam] = np.exp(np.median(np.log(bestfits[iparam,:])))
            elif zerocount==self.nfits//10:
                bestparams[iparam] = -np.exp(np.median(np.log(-bestfits[iparam,:])))
            else:
                raise ValueError
        return bestparams
    
    def plot_fits(self, save=False):
        plt.figure(figsize=(10,8))
        for ifield,field in enumerate(self._fields):
            x, y, yerr, lb, ub, time = self.fit_inputs[field]
            params, wtsqerr = self.fit_results[field]
            label = self._labels[ifield]
            xvals = np.linspace(0.05,x.max(),101)
            axpro = plt.subplot(3,3,ifield*3+1)
            axgrad = plt.subplot(3,3,ifield*3+2)
            title = f'{self.shot} | {time:.2f} s | {self.desc}'
            plt.sca(axpro)
            plt.errorbar(x, y, yerr=np.squeeze(yerr), fmt='x')
            plt.xlabel('r/a')
            plt.ylabel(label)
            plt.ylim(0,None)
            plt.xlim(0,1.2)
            plt.title(title)
            plt.sca(axgrad)
            plt.xlabel('r/a')
            plt.ylabel('-d ln( {} )/dx'.format(label))
            plt.title(title)
            plt.xlim(0,1.2)
            plt.ylim(-5,15)
            igood = np.nonzero(wtsqerr>=(np.median(wtsqerr)+wtsqerr.std()/2))[0]
            for i in igood:
                # plot fit to resampled data
                yfit = feval(params[:,i], xvals/xvals.max(), nohollow=self.nohollow)
                dyfit = dfeval(params[:,i], xvals/xvals.max(), nohollow=self.nohollow)
                plt.sca(axpro)
                plt.plot(xvals, yfit, color='C1', linewidth=0.5)
                plt.sca(axgrad)
                plt.plot(xvals, dyfit/yfit, color='C1', linewidth=0.5)
            plt.subplot(3,3,ifield*3+3)
            for icoeff in np.arange(params.shape[0]):
                plt.plot([icoeff,icoeff], np.abs([lb[icoeff],ub[icoeff]]), '_', 
                         color='k', 
                         mew=2,
                         ms=10)
                for i in igood:
                    plt.plot(icoeff, np.abs(params[icoeff,i]), 'o', 
                             color='C1',
                             linewidth=0.5,
                             ms=3)
            plt.title(title)
            plt.yscale('log')
            plt.xlabel('fit coefficient index')
            plt.ylabel('{} fit coefficients'.format(field))
            isort = np.argsort(wtsqerr)[0:self.nfits//10]
            bestfits = params[:,isort]
            bestparams = np.empty(bestfits.shape[0])
            lesszero = np.count_nonzero(bestfits<0, axis=1)
            for iparam,zerocount in enumerate(lesszero):
                if zerocount==0:
                    bestparams[iparam] = np.exp(np.median(np.log(bestfits[iparam,:])))
                elif zerocount==self.nfits//10:
                    bestparams[iparam] = -np.exp(np.median(np.log(-bestfits[iparam,:])))
                else:
                    raise ValueError
        plt.tight_layout()
        if save:
            fname = Path('plots') / f'profiles_{self.shot}_{self.time*1e3:.0f}s.pdf'
            print(f'Saving {fname.as_posix()}')
            plt.savefig(fname.as_posix(), transparent=True)
    # if save_fits:
    #     fitfile = Path('data') / 'fits.pickle'
    #     print('Saving fits in {}'.format(fitfile.as_posix()))
    #     with fitfile.open('wb') as f:
    #         pickle.dump(fits, f)
    
    def plot_profiles(self, mu=1, save=False):
        x = np.linspace(0.01, 1.0, 80)
        c2c = [1.5,2]
        # plot profiles and gradients
        profiles = {}
        dprofiles = {}
        for ifield,field in enumerate(['ne','te','ti']):
            fit = self.best_fits[field]
            profiles[field] = feval(fit['params'], 
                                    x/fit['xmax'], 
                                    nohollow=self.nohollow)
            dprofiles[field] = dfeval(fit['params'], 
                                      x/fit['xmax'], 
                                      nohollow=self.nohollow)
        print(f'Using atomic mass = {mu} AMU for rho-i')
        params = []
        for ix in np.arange(x.size):
            params.append(Params(ne=profiles['ne'][ix]*1e13,
                                 Te=profiles['te'][ix],
                                 Ti=profiles['ti'][ix],
                                 mu=mu
                                 ))
        rhoi = np.array([param.rho_i for param in params])
        rhos = np.array([param.rho_s for param in params])
        title = f'{self.shot} | {self.time:.2f} s | {self.desc}'
        legend_kw = {'loc':'upper left', 
                     'labelspacing':0.2, 
                     'fontsize':'small', 
                     }
        plt.figure(figsize=[8,6.25])
        plt.subplot(2,2,1)
        plt.plot(x, rhoi*1e3, label='rho-i')
        plt.plot(x, rhos*1e3, label='rho-s')
        plt.legend()
        plt.ylim([0,None])
        plt.xlabel('r/a')
        plt.ylabel('rho-i,s (mm)')
        plt.title(title)
        # k values for k*rhoi= X
        krhoi_values = [0.2,0.4,0.6]
        k = np.matmul(1/rhoi.reshape(-1,1),
                      np.array(krhoi_values).reshape(1,-1))
        klabels = [f'k*rhoi = {kval}' for kval in krhoi_values]
        plt.subplot(2,2,2)
        plt.plot(x, k/1e2)
        plt.xlabel('r/a')
        plt.ylabel('k (1/cm)')
        plt.ylim(0,3)
        for c in c2c:
            plt.axhline(np.pi/c, c='k', ls=':')
            plt.annotate(f'k_max with C2C={c:.1f} cm', (0,np.pi/c), 
                         xytext=(1,3),
                         textcoords='offset points')
        plt.title(title)
        plt.legend(klabels,**legend_kw)
        # omega-star
        Ti_J = np.array([param.Ti_J for param in params])
        Te_J = np.array([param.Te_J for param in params])
        gradpi_over_n = np.array(Ti_J * dprofiles['ne'] / profiles['ne']).reshape(-1,1)
        gradpe_over_n = np.array(Te_J * dprofiles['ne'] / profiles['ne']).reshape(-1,1)
        omega_star_i = (k / pc.e / 2.6) * gradpi_over_n
        omega_star_e = (k / pc.e / 2.6) * gradpe_over_n
        plt.subplot(2,2,3)
        plt.plot(x, omega_star_i/(2*np.pi)/1e3)
        plt.legend(klabels,**legend_kw)
        plt.xlabel('r/a')
        plt.ylabel('omega_star_i (kHz)')
        plt.title(title)
        plt.subplot(2,2,4)
        plt.plot(x, omega_star_e/(2*np.pi)/1e3)
        plt.legend(klabels,**legend_kw)
        plt.xlabel('r/a')
        plt.ylabel('omega_star_e (kHz)')
        plt.title(title)
        plt.tight_layout()
        if save:
            fname = Path('plots') / f'k-omega_{self.shot}_{self.time*1e3:.0f}ms.pdf'
            print(f'Saving {fname.as_posix()}')
            plt.savefig(fname.as_posix(), transparent=True)
            
    def plot_profiles2(self, mu=1, save=False):
        x = np.linspace(0.01, 1.0, 80)
        c2c = [1,1.4] # center-to-center spacings in cm
        profiles = {}
        dprofiles = {}
        for ifield,field in enumerate(['ne','te','ti']):
            fit = self.best_fits[field]
            profiles[field] = feval(fit['params'], 
                                    x/fit['xmax'], 
                                    nohollow=self.nohollow)
            dprofiles[field] = dfeval(fit['params'], 
                                      x/fit['xmax'], 
                                      nohollow=self.nohollow)
        print(f'Using atomic mass = {mu} AMU for rho-i')
        params = []
        for ix in np.arange(x.size):
            # profiles['ti'][ix] = np.min([profiles['ti'][ix],profiles['te'][ix]])
            params.append(Params(ne=profiles['ne'][ix]*1e13,
                                 Te=profiles['te'][ix],
                                 Ti=profiles['ti'][ix],
                                 mu=mu,
                                 ))
        rhoi = np.array([param.rho_i for param in params]) # rho-i in m
        rhos = np.array([param.rho_s for param in params]) # rho-s in m
        # legend_kw = {'loc':None, 
        #              'labelspacing':0.2, 
        #              'fontsize':'small', 
        #              }
        title = f'{self.shot} | {self.time:.2f} s | {self.desc}'
        # k values for k*rhoi= X
        # krhoi_values = [0.2,0.4,0.6,0.8]
        # k = np.matmul(1/rhoi.reshape(-1,1),
        #               np.array(krhoi_values).reshape(1,-1)) # k in 1/m
        # klabels = [f'k*rhoi = {kval}' for kval in krhoi_values]
        for c in c2c:
            kmax = np.pi/c  # 1/cm
            print(f'C2C = {c:.02f} cm  -> kmax = {kmax:.02f} 1/cm')
            for rova in [0.55,0.75,0.95]:
                xind = np.argmin(np.abs(x-rova))
                max_krhoi = kmax * rhoi[xind]*1e2
                print(f'  max(k*rhoi) = {max_krhoi:.02f} at r/a = {rova:.02f}')
        # omega-star
        Ti_J = np.array([param.Ti_J for param in params])
        Te_J = np.array([param.Te_J for param in params])
        gradpi_over_n = np.array(Ti_J * dprofiles['ne'] / profiles['ne']).reshape(-1,1)
        gradpe_over_n = np.array(Te_J * dprofiles['ne'] / profiles['ne']).reshape(-1,1)
        # omega_star_i = (k / pc.e / 2.6) * gradpi_over_n
        # omega_star_e = (k / pc.e / 2.6) * gradpe_over_n
        plt.figure(figsize=(6.6,5.4))
        # ne profile
        # plt.subplot(2,3,2)
        # plt.plot(x, profiles['ne'])
        # plt.ylabel(self._labels[0])
        # plt.xlabel('r/a')
        # plt.ylim(0,None)
        # plt.title(title)
        # ne, Te, Ti profiles
        plt.subplot(2,2,1)
        plt.plot(x, profiles['ti'], label='Ti')
        plt.plot(x, profiles['te'], label='Te')
        plt.plot(x, profiles['ne']/2, label='ne/2')
        plt.ylabel('Te, Ti (keV), ne/2 (1e13/cm^3)')
        plt.xlabel('r/a')
        plt.ylim(0,None)
        plt.legend()
        plt.title(title)
        # # k*rho-i
        # plt.subplot(2,3,3)
        # plt.plot(x, k/1e2)
        # plt.xlabel('r/a')
        # plt.ylabel('k (1/cm)')
        # plt.ylim(0,4)
        # for c in c2c:
        #     plt.axhline(np.pi/c, c='k', ls=':')
        #     plt.annotate(f'k_max with C2C={c:.1f} cm', (0,np.pi/c), 
        #                  xytext=(1,3),
        #                  textcoords='offset points')
        # plt.title(title)
        # plt.legend(klabels,**legend_kw)
        # rhoi ,rhos profiles
        plt.subplot(2,2,2)
        plt.plot(x, rhoi*1e3, label='rho-i')
        plt.plot(x, rhos*1e3, label='rho-s')
        plt.legend()
        plt.ylim([0,None])
        plt.xlabel('r/a')
        plt.ylabel('rho-i,s (mm)')
        plt.title(title)
        # omega_star_i
        plt.subplot(2,2,3)
        kmax = np.pi/c2c[1]*1e2  # 1/m
        omega_star_i = (kmax / pc.e / 2.6) * gradpi_over_n
        omega_star_e = (kmax / pc.e / 2.6) * gradpe_over_n
        plt.plot(x, omega_star_i/(2*np.pi)/1e3,
                 label='omega_star_i')
        plt.plot(x, omega_star_e/(2*np.pi)/1e3,
                 label='omega_star_e')
        plt.annotate(f'{c2c[1]:.1f} cm grid spacing',
                     (0.05,0.9), xycoords='axes fraction')
        plt.legend(loc='lower right')
        plt.xlabel('r/a')
        plt.ylabel('omega_star (kHz)')
        plt.title(title)
        # omega_star_e
        # plt.subplot(2,3,5)
        # plt.plot(x, omega_star_e/(2*np.pi)/1e3)
        # plt.legend(klabels,**legend_kw)
        # plt.xlabel('r/a')
        # plt.ylabel('omega_star_e (kHz)')
        # plt.title(title)
        # max k*rhoi
        plt.subplot(2,2,4)
        for spacing,linestyle in zip(c2c, ['-','--']):
            kmax = np.pi/spacing  # 1/cm
            max_krhoi = kmax*rhoi*1e2
            plt.plot(x, max_krhoi,
                     label=f'{spacing:.1f} cm grid spacing',
                     linestyle=linestyle)
        plt.legend(loc='lower left')
        plt.xlabel('r/a')
        plt.ylabel('max k*rhoi')
        plt.ylim(0,1)
        plt.title(title)
        plt.tight_layout()
        if save:
            fname = Path('plots') / f'k-omega_{self.shot}_{self.time*1e3:.0f}ms.pdf'
            print(f'Saving {fname.as_posix()}')
            plt.savefig(fname.as_posix(), transparent=True)
            
            


if __name__=='__main__':
    plt.close('all')
    pro = Profiles()
    plt.ioff()  #  the `PCIanalysis` library sets `plt.ion()``
    pro.plot_profiles2(save=False)
    plt.show()
