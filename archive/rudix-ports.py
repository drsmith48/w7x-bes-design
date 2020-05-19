#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:03:02 2019

@author: drsmith
"""

import numpy as np

G41 = np.array([[-3880.6, -3975.5, -943.6],
                [-3937.4, -4108.4, -1072.9],
                [-4063.7, -4077.9, -1141.1],
                [-3978.1, -3949.5, -1071.5],
                [-3880.6, -3975.5, -943.6],
                ]).transpose() / 1e3

ports = {'G41':G41}