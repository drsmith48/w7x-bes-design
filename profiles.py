#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 12:56:07 2019

@author: drsmith
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import FIT
import PCIanalysis.gradientlengths as gl
import pybaseutils.utils as ut

plt.close('all')

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

#profiles = [{'shot':180904027, 'time':1.9, 'desc':'test'}]

previous_shot = -1
tsdata = None
shotdata = {}
for pro in profiles:
    if pro['shot'] != previous_shot:
        if profile_data:
            print('Loading MPTS data from {}, shot {} ...'.
                  format(datafile, pro['shot']))
            tsdata = profile_data[pro['shot']]
        else:
            print('Getting MPTS data, shot {} ...'.format(pro['shot']))
            tsdata = gl.get_thomsondata(pro['shot'])
        shotdata[pro['shot']] = tsdata
    print('*** Performing ne profile fit, shot {}, time {:.2f} s ...'.
          format(pro['shot'], pro['time']))
    edgecut=2; rescale=False
    index = np.argmin(np.abs(tsdata['time'] / 1e9 - pro['time']))
    scalefactor = tsdata['roa'][-1] if rescale else 1
    if edgecut == 0:
        x = tsdata['roa'].copy() / scalefactor
        y = tsdata['ne'][index, :].copy()
        yerr = tsdata['ne_err'][index, :].copy()
    else:
        x = tsdata['roa'][:-edgecut].copy() / scalefactor
        y = tsdata['ne'][index, :-edgecut].copy()
        yerr = tsdata['ne_err'][index, :-edgecut].copy()
    mask = y == 0
    x = x[~mask]
    y = y[~mask]
    yerr = yerr[~mask]

    norm = max(y)
    y /= norm
    yerr /= norm
    xinterp = np.linspace(-1.1, 1.1, 50)
    x = ut.cylsym_odd(x) # anti-reflect x about x=0
    y = ut.cylsym_even(y) # reflect y about x=0
    yerr = ut.cylsym_even(yerr)
    qparaboptions = {
        'plotit': False,
        'nohollow':True,
#        'nprint':1,
        'quiet':False,
        'verbose':True,
        'resampleit':50,
        'autoderivative':1,
    }
    print('Calling FIT.qparabfit()')
    fit = FIT.qparabfit(x, y, yerr, xinterp, **qparaboptions)
    nefit = {
        'time': tsdata['time'][index] / 1e9,
        'xinterp': xinterp,
        'index': index,
        'fit': fit,
        'norm': norm,
        'scalefactor': scalefactor,
    }
#    print('  Performing Te profile fit, shot {}, time {:.2f} s ...'.
#          format(pro['shot'], pro['time']))
#    tefit = gl.fit_thomson_te(tsdata, pro['time'], edgecut=2)
    gl.plot_nefit(tsdata, nefit)
#    gl.plot_tefit(tsdata, tefit)
    previous_shot = pro['shot']

print('Saving MPTS data in {}'.format(datafile))
with open(datafile, 'wb') as f:
    pickle.dump(shotdata, f)


#shot = 180904027
#time = 1.9
#
#mpts_data = gl.get_thomsondata(shot)
##print(timeit(stmt='gl.get_thomsondata(shot)', number=1, globals=globals()))
#nefit = gl.fit_thomson_ne(mpts_data, time, edgecut=2, nohollow=True)
##print(timeit(stmt='gl.fit_thomson_ne(mpts_data, time, edgecut=2, nohollow=True)', number=1, globals=globals()))
#gl.plot_nefit(mpts_data, nefit)