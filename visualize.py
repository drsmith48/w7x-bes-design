#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:24:04 2019

@author: drsmith
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from coords import Point, NBPoint


#plt.close('all')

plt.figure()
mngr = plt.get_current_fig_manager()
rect = mngr.window.geometry().getRect()
mngr.window.setGeometry(30,30,rect[2],rect[3])

ax = plt.axes(projection='3d')
ax.set_xlabel('Machine X (m)')
ax.set_ylabel('Machine Y (m)')
ax.set_zlabel('Machine Z (m)')

    
# plot machine origin and x,y,z lines
machine_origin = Point()
ax.scatter(*machine_origin._xyz, color='b')


# plot NB origins and grid centers
grid_u = 6.5
grid_v = 0.47
grid_z = 0.6
NB_center_grids_uvz = [[grid_u,  grid_v,  grid_z],
                       [grid_u, -grid_v,  grid_z],
                       [grid_u, -grid_v, -grid_z],
                       [grid_u,  grid_v, -grid_z]]
s1234_u = -0.5
s14_v = -0.0362
s23_v = -s14_v

for i,c in enumerate(['g','r']):
    # plot uvz origin and u,v axes
    uvz_origin = NBPoint(box=i)
    print('Box {:d} origin: {:4f} {:4f} {:4f}'.
          format(i, *uvz_origin._xyz))
    ax.scatter(*uvz_origin._xyz, color='k')
    for j,ax_point in enumerate([NBPoint(box=i, u=2), NBPoint(box=i, v=2)]):
        ax.plot([uvz_origin._xyz[0], ax_point._xyz[0]],
                [uvz_origin._xyz[1], ax_point._xyz[1]],
                [uvz_origin._xyz[2], ax_point._xyz[2]], color='k')
        tmp = ax_point._xyz + np.array([0,0,0.05])
        if j==0:
            ax.text(*tmp, 'u')
        else:
            ax.text(*tmp, 'v')
    for j,uvz in enumerate(NB_center_grids_uvz):
        # plot source center points
        grid = NBPoint(*uvz, box=i)
        ax.scatter(*grid._xyz, color=c)
        # plot line from source cent point to u,v plane intersection
        if j==0 or j==3:
            s = NBPoint(u=s1234_u, v=s14_v, box=i)
        elif j==1 or j==2:
            s = NBPoint(u=s1234_u, v=s23_v, box=i)
        tmp = s - grid
        uv = tmp / abs(tmp)
        s = grid + uv*10
        ax.plot([grid._xyz[0], s._xyz[0]],
                [grid._xyz[1], s._xyz[1]],
                [grid._xyz[2], s._xyz[2]], color=c)
        print('  Box {:d}  Source {:d} origin: {:4f} {:4f} {:4f}'.
              format(i, j, *grid._xyz))
        if j==0:
            tmp = grid._xyz + np.array([0,0,0.05])
            ax.text(*tmp, 'NB2{:1d} Src 1'.format(i),
                    horizontalalignment='center')

# scale axes and draw axis lines
r = 0.0
for axis in [ax.xaxis, ax.yaxis]:
    lim = axis.get_data_interval()
    r = np.max(np.array([r,lim.max()-lim.min()]))
    
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    mid = axis.get_data_interval().sum()/2
    axis.set_data_interval(mid-r/2, mid+r/2)

ax.plot([-2,12],[0,0],[0,0], 'b')
ax.plot([0,0],[0,14],[0,0], 'b')
ax.plot([0,0],[0,0],[-1,1], 'b')

#for i,axis in enumerate([ax.xaxis, ax.yaxis, ax.zaxis]):
#    data = np.zeros((2,3))
#    data[:,i] = axis.get_data_interval()
#    ax.plot(data[:,0], data[:,1], data[:,2], color='b')
