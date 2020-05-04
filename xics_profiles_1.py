#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:51:23 2019

@author: drsmith
"""

import MDSplus as m
import PCIanalysis.gradientlengths as gl

shot = 180904027

#c=m.Connection('ssh://dvs@mds-trm-1.ipp-hgw.mpg.de')
#c=m.Connection('mds-trm-1.ipp-hgw.mpg.de')

m.setenv('qsw_eval_path', 'mds-trm-1.ipp-hgw.mpg.de::/w7x/eval/~t')
t = m.Tree('qsw_eval', shot, mode='READONLY')

#xdata = gl.get_xics_data(shot)