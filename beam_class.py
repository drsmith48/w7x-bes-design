#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 20:24:57 2019

@author: drsmith
"""

import numpy as np


# All coord. are xyz unless noted with a suffix
# All angles are radians unless noted with a suffix

# NB coord. sys. origins and rotations
ni20_origin_rpz = {'r':6.75,
                   'phi':56.9059*np.pi/180,
                   'z':-0.305}
ni20_rotation = (90 - 36 + 2.9059 + 7.4441)*np.pi/180

ni21_origin_rpz = {'r':6.75,
                   'phi':87.0941*np.pi/180,
                   'z':0.305}
ni21_rotation = (90 - 36 + 2.9059 + 2*15.0941 - 7.4441)*np.pi/180

# source displacements from uvw origin
u_disp = 6.5
v_disp = 0.47
w_disp = 0.6

# vertical displacement of beam crossing at uvw origin
# (S12 in NBI_Geometry.pdf)
w_cross_disp = 0.0429

# source size
source_height = 0.506
source_width = 0.228

# source divergence angle
source_divergence = 1.0


def xyz_to_rpz(xyz):
    rpz = np.empty(xyz.shape)
    rpz[0,...] = np.linalg.norm(xyz[0:2,...], axis=0)
    rpz[1,...] = np.arctan2(xyz[1,...], xyz[0,...])
    rpz[2,...] = xyz[2,...]
    return rpz

def rpz_to_xyz(rpz):
    xyz = np.empty(rpz.shape)
    xyz[0,...] = rpz[0,...] * np.cos(rpz[1,...])
    xyz[1,...] = rpz[0,...] * np.sin(rpz[1,...])
    xyz[2,...] = rpz[2,...]
    return xyz

class Beam(object):
    
    def __init__(self, injector=0, beam=0, axis_spacing=0.05):
        
        self.injector = injector
        self.beam = beam
        
        self.axis = None
        self.axis_rpz = None
        
        if (self.injector not in [0,1]) or (self.beam not in [0,1,2,3]):
            raise ValueError('Invalid injector={} and beam={}'.
                             format(self.injector, self.beam))
        self._set_injector_geometry()
        self._set_source_geometry()
        self._set_axis(axis_spacing=axis_spacing)
        self._set_axis_rpz()
        
    def _set_injector_geometry(self):
        # set injector origin
        if self.injector==0:
            # injector NI20
            injector_origin_rpz = ni20_origin_rpz
            self.uv_rotation = ni20_rotation
            self.name = 'K20'
        elif self.injector==1:
            # injector NI21
            injector_origin_rpz = ni21_origin_rpz
            self.uv_rotation = ni21_rotation
            self.name = 'K21'
        # coord. of injector uvw origin
        self.injector_origin_rpz = np.array([injector_origin_rpz['r'],
                                             injector_origin_rpz['phi'],
                                             injector_origin_rpz['z']])
        self.injector_origin = rpz_to_xyz(self.injector_origin_rpz)
        # u/v unit vectors
        angle = self.uv_rotation
        rot_mat = np.array([[np.cos(angle), -np.sin(angle), 0],
                            [np.sin(angle), np.cos(angle), 0],
                            [0, 0, 1]])
        self.u_hat = np.matmul(rot_mat, np.array([1,0,0]))
        self.v_hat = np.matmul(rot_mat, np.array([0,1,0]))
        
    def _set_source_geometry(self):
        # source coord.
        u = u_disp
        if self.beam in [0,3]:
            v = v_disp
        else:
            v = -v_disp
        if self.beam in [0,1]:
            w = w_disp
        else:
            w = -w_disp
        self.source = self.injector_origin + \
                      u*self.u_hat + \
                      v*self.v_hat + \
                      w*np.array([0,0,1])
        self.source_rpz = xyz_to_rpz(self.source)
        # source crossing at u=v=0
        if self.beam in [0,1]:
            w_crossing = w_cross_disp
        else:
            w_crossing = -w_cross_disp
        self.crossing = self.injector_origin + np.array([0,0,w_crossing])
        # r-hat is direction perpendicular to source
        beam_vec = self.crossing - self.source
        self.r_hat = beam_vec / np.linalg.norm(beam_vec)
        # s-hat = unit vector(z-hat cross r-hat)
        # s-hat is horizontal direction of source extent
        s_direction = np.cross(np.array([0,0,1]), self.r_hat)
        self.s_hat = s_direction / np.linalg.norm(s_direction)
        # t-hat = r-hat cross s-hat
        # t-hat is near-vertical direction of source extent
        t_direction = np.cross(self.r_hat, self.s_hat)
        self.t_hat = t_direction / np.linalg.norm(t_direction)
        # take s-hat and t-hat to be like x and y; r-hat to be like z
        self.metric = np.array([self.s_hat, self.t_hat, self.r_hat])
        
    def _set_axis(self, axis_spacing=0.01, rmin=4.5, rmax=6.75):
        """
        Return x,y,z of points [m] on beam axis with specified spacing
        
        spacing - array spacing [m]
        rmax - maximum major radius [m]
        rmin - minimum major radius [m]
        """
        beamaxis = np.empty((3,0))
        xyz = self.source.copy()
        while True:
            xyz += axis_spacing * self.r_hat
            rmajor = np.linalg.norm(xyz[0:2])
            if rmajor > rmax:
                continue
            elif rmajor > rmin:
                beamaxis = np.append(beamaxis, 
                                     np.reshape(xyz, (3,1)), 
                                     axis=1)
            else:
                break
        self.axis = beamaxis
    
    def _set_axis_rpz(self):
        self.axis_rpz = xyz_to_rpz(self.axis)
    
    def get_str_coords(self, xyz=None):
        """
        Return str coords. for xyz input
        """
        if xyz is None:
            xyz = self.source + \
                6.5 * self.r_hat + \
                0.3 * self.s_hat + \
                0.4 * self.t_hat
        xyz_shift = xyz.reshape((3,)) - self.source
        str_coords = np.matmul(self.metric, xyz_shift)
        return str_coords

#    def get_beam_grid(self, spacing=0.01, svec=None, tvec=None, rmax=6, rmin=3):
#        if svec is None:
#            half_height = 1.5*source_height/2
#            svec = np.linspace(-half_height,half_height,7)
#        if tvec is None:
#            half_width = 1.5*source_width/2
#            tvec = np.linspace(-half_width,half_width,7)
#        ns = svec.size
#        nt = tvec.size
#        beamaxis = self.get_beam_axis_array(spacing=spacing, rmax=rmax, rmin=rmin)
#        nbeam = beamaxis.shape[1]
#        grid = np.empty((3,ns,nt,nbeam))
#        return grid

def plot_beams():
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d
#    plt.close('all')
    plt.figure()
    mngr = plt.get_current_fig_manager()
    rect = mngr.window.geometry().getRect()
    mngr.window.setGeometry(30,30,rect[2],rect[3])
    ax = plt.axes(projection='3d')
    ax.plot([-2,12],[0,0],[0,0], color='b')
    ax.plot([0,0],[0,14],[0,0], color='b')
    ax.plot([0,0],[0,0],[-1,1], color='b')
    ax.set_xlabel('Machine X (m)')
    ax.set_ylabel('Machine Y (m)')
    ax.set_zlabel('Machine Z (m)')
    for inj,col in zip([0,1], ['g','r']):
        for src in range(4):
            beam=Beam(injector=inj, beam=src)
            if src==0:
#                print('Box {} origin: {:4f} {:4f} {:4f}'.
#                      format(inj, *beam.injector_origin))
                ax.scatter(*beam.injector_origin, color='k')
                for vec in [beam.u_hat, beam.v_hat]:
                    axpt = beam.injector_origin+2*vec
                    l = list(zip(beam.injector_origin.tolist(), axpt.tolist()))
                    ax.plot(*l, color='k')
            ax.scatter(*beam.source, color=col)
            rpt = beam.source+10*beam.r_hat
            l = list(zip(beam.source.tolist(), rpt.tolist()))
            ax.plot(*l, color=col)
#            print('  Box {:d}  Source {:d} origin: {:4f} {:4f} {:4f}'.
#                  format(inj, src, *beam.source))
            

test_sources = np.array([[6.075594, 11.717889, 0.295000],
                         [6.922962, 11.310989, 0.295000],
                         [6.922962, 11.310989, -0.905000],
                         [6.075594, 11.717889, -0.905000],
                         [1.047639, 13.219997, 0.905000],
                         [1.972344, 13.051116, 0.905000],
                         [1.972344, 13.051116, -0.295000],
                         [1.047639, 13.219997, -0.295000]]).transpose()
test_origins = np.array([[3.685606, 5.654981, -0.305000],
                         [0.342197, 6.741320, 0.305000]]).transpose()

def test_beams():
    beam = []
    for inj in [0,1]:
        for src in range(4):
            beam.append(Beam(injector=inj, beam=src))
    for i in [0,4]:
        assert(np.allclose(beam[i].injector_origin, test_origins[:,i//4]))
    for i,b in enumerate(beam):
        assert(np.allclose(b.source, test_sources[:,i]))
        # ensure r/s/t axes are orthogonal
        assert(np.isclose(np.sum(b.r_hat*b.s_hat)+1,1))
        assert(np.isclose(np.sum(b.t_hat*b.s_hat)+1,1))
        # ensure axis point is on r axis
        x2source = b.axis[:,5] - b.source
        x2scomp = np.sum(x2source * b.s_hat)
        x2tcomp = np.sum(x2source * b.t_hat)
        assert(np.isclose(x2scomp+1,1))
        assert(np.isclose(x2tcomp+1,1))
        assert(np.isclose(np.linalg.norm(x2source+b.r_hat),
                          np.linalg.norm(x2source)+
                              np.linalg.norm(b.r_hat)))


if __name__=='__main__':
    import pytest, sys
    pytest.main(['-v', sys.modules[__name__].__file__])
    plot_beams()
    