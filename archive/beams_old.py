#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on Mon Apr 15 21:10:11 2019

import numpy as np


def xyz_to_rpz(xyz):
    if xyz.shape[0] != 3: raise ValueError('{}'.format(xyz.shape))
    rpz = np.empty(xyz.shape)
    rpz[0,...] = np.linalg.norm(xyz[0:2,...], axis=0)
    rpz[1,...] = np.arctan2(xyz[1,...], xyz[0,...])
    rpz[2,...] = xyz[2,...]
    return rpz

def rpz_to_xyz(rpz):
    if rpz.shape[0] != 3: raise ValueError('{}'.format(rpz.shape))
    xyz = np.empty(rpz.shape)
    xyz[0,...] = rpz[0,...] * np.cos(rpz[1,...])
    xyz[1,...] = rpz[0,...] * np.sin(rpz[1,...])
    xyz[2,...] = rpz[2,...]
    return xyz


class Beam(object):
    
    def __init__(self, injector=0, source=0, axis_spacing=0.06):
        """
        injector
            0/1: heating beams @ K20/K21
            2: Rudix @ T40
        source
            1-4: PINI sources for heating beams; n/a for Rudix
        """
        self._injector = injector
        self._source = source
        self.axis = None    # xyz axis points
        self.axis_rpz = None    # r,phi,z axis points
        self.source = None      # xyz of source
        self.source_rpz = None # r,phi,z of source
        self.r_hat = None   # xyz vector along beam axis
        self.s_hat = None   # xyz vector perp to axis, in xy plane
        self.t_hat = None   # xyz vector perp to axis and s_hat
        self.name = None
        
        if self._injector in [0,1]:
            self.load_heating_beams()
        elif self._injector == 2:
            self.load_rudix()
        else:
            raise ValueError('invalid self._injector {}'.
                             format(self._injector))
            
        self.make_axis(axis_spacing=axis_spacing)
            
    def load_heating_beams(self):
        self.set_nbi_geometry()
        self.set_pini_geometry()
    
    def load_rudix(self):
        self.name = 'Rudix'
        self.source = np.array([-2.963, -5.648, -0.165])
        self.source_rpz = xyz_to_rpz(self.source)
        self.injector_origin = self.source
        self.injector_origin_rpz = self.source_rpz
        # r-hat is direction perpendicular to source
        self.r_hat = np.array([0.50139767, 0.80628892, 0.31384478])
        # s-hat = unit vector(z-hat cross r-hat)
        # s-hat is horizontal direction of source extent
        s_direction = np.cross(np.array([0,0,1]), self.r_hat)
        self.s_hat = s_direction / np.linalg.norm(s_direction)
        # t-hat = r-hat cross s-hat
        # t-hat is near-vertical direction of source extent
        t_direction = np.cross(self.r_hat, self.s_hat)
        self.t_hat = t_direction / np.linalg.norm(t_direction)
    
    def set_nbi_geometry(self):
        if self._injector==0:
            self.name = 'K20'
            inj_phi = 56.9059 * np.pi/180
            inj_z = -0.305
            uv_rot = (90 - 36 + 2.9059 + 7.4441) * np.pi/180
        elif self._injector==1:
            self.name = 'K21'
            inj_phi = 87.0941*np.pi/180
            inj_z = 0.305
            uv_rot = (90 - 36 + 2.9059 + 2*15.0941 - 7.4441) * np.pi/180
        # injector origin
        self.injector_origin_rpz = np.array([6.75, inj_phi, inj_z])
        self.injector_origin = rpz_to_xyz(self.injector_origin_rpz)
        # u/v unit vectors
        rot_mat = np.array([[np.cos(uv_rot), -np.sin(uv_rot), 0],
                            [np.sin(uv_rot), np.cos(uv_rot), 0],
                            [0, 0, 1]])
        self.u_hat = np.matmul(rot_mat, np.array([1,0,0]))
        self.v_hat = np.matmul(rot_mat, np.array([0,1,0]))
    
    def set_pini_geometry(self):
        self.name += ' PINI {:d}'.format(self._source+1 + 4*self._injector)
        # source coord.
        v = 0.47
        if self._source in [1,2]:
            v *= -1
        w = 0.6
        if self._source in [2,3]:
            w *= -1
        self.source = self.injector_origin + 6.5 * self.u_hat + \
                      v * self.v_hat + w * np.array([0,0,1])
        self.source_rpz = xyz_to_rpz(self.source)
        # source crossing at u=v=0
        w_crossing = 0.0429
        if self._source in [2,3]:
            w_crossing *= -1
        crossing = self.injector_origin + np.array([0,0,w_crossing])
        # r-hat is direction perpendicular to source
        beam_vec = crossing - self.source
        self.r_hat = beam_vec / np.linalg.norm(beam_vec)
        # s-hat = unit vector(z-hat cross r-hat)
        # s-hat is horizontal direction of source extent
        s_direction = np.cross(np.array([0,0,1]), self.r_hat)
        self.s_hat = s_direction / np.linalg.norm(s_direction)
        # t-hat = r-hat cross s-hat
        # t-hat is near-vertical direction of source extent
        t_direction = np.cross(self.r_hat, self.s_hat)
        self.t_hat = t_direction / np.linalg.norm(t_direction)
    
    def make_axis(self, axis_spacing=0.06, rmin=4.5, rmax=6.75):
        beamaxis = np.empty((3,0))
        xyz = self.source.copy()
        while True:
            xyz += axis_spacing * self.r_hat
            rmajor = np.linalg.norm(xyz[0:2])
            if rmajor > rmax:
                continue
            elif rmajor > rmin:
                beamaxis = np.append(beamaxis, xyz.reshape((3,1)), axis=1)
            else:
                break
        self.axis = beamaxis
        self.axis_rpz = xyz_to_rpz(self.axis)


def plot_heating_beams():
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d
    plt.close('all')
    plt.figure()
    mngr = plt.get_current_fig_manager()
    rect = mngr.window.geometry().getRect()
    mngr.window.setGeometry(30,30,rect[2],rect[3])
    ax = plt.axes(projection='3d')
    ax.plot([-2,12],[0,0],[0,0], color='k')
    ax.plot([0,0],[0,14],[0,0], color='k')
    ax.plot([0,0],[0,0],[-1,1], color='k')
    ax.set_xlabel('Machine X (m)')
    ax.set_ylabel('Machine Y (m)')
    ax.set_zlabel('Machine Z (m)')
    for inj,col in zip([0,1,2], ['g','r','b']):
        for src in range(4):
            beam=Beam(injector=inj, source=src)
            if inj in [0,1] and src==0:
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
#            print('  Box {} Source {}'.format(inj, src))
#            print('    Origin: {:4f}, {:4f}, {:4f}'.format(*beam.source))
#            print('    r-hat:  {:4f}, {:4f}, {:4f}'.format(*beam.r_hat))
            if inj==2: break

def test_heating_beams():
    test_origins = np.array([[3.685606, 5.654981, -0.305000],
                             [0.342197, 6.741320, 0.305000]]).transpose()
    test_sources = np.array([[6.075594, 11.717889, 0.295000],
                             [6.922962, 11.310989, 0.295000],
                             [6.922962, 11.310989, -0.905000],
                             [6.075594, 11.717889, -0.905000],
                             [1.047639, 13.219997, 0.905000],
                             [1.972344, 13.051116, 0.905000],
                             [1.972344, 13.051116, -0.295000],
                             [1.047639, 13.219997, -0.295000]]).transpose()
    test_rhat =    np.array([[-0.365400, -0.926946, -0.085174],
                             [-0.494953, -0.864735, -0.085174],
                             [-0.494953, -0.864735, 0.085174],
                             [-0.365400, -0.926946, 0.085174],
                             [-0.107854, -0.990511, -0.085174],
                             [-0.249230, -0.964692, -0.085174],
                             [-0.249230, -0.964692, 0.085174],
                             [-0.107854, -0.990511, 0.085174]]).transpose()
    beam = []
    for inj in [0,1]:
        for src in range(4):
            beam.append(Beam(injector=inj, source=src))
    for i in [0,4]:
        assert(np.allclose(beam[i].injector_origin, test_origins[:,i//4]))
    for i,b in enumerate(beam):
        assert(np.allclose(b.source, test_sources[:,i]))
        assert(np.allclose(b.r_hat, test_rhat[:,i]))
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
                          np.linalg.norm(x2source) + np.linalg.norm(b.r_hat)))


if __name__=='__main__':
    import pytest, sys
    pytest.main(['-v', sys.modules[__name__].__file__])
    plot_heating_beams()
    