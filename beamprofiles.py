#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on Tue Apr 16 17:26:14 2019

import numpy as np
from scipy import integrate
import scipy.special as sp
import matplotlib.pyplot as plt

plt.close('all')

# half-angle divergence
theta = 1.0 * np.pi/180
tantheta = np.tan(theta)

### square emitter with half-length S
def diff_intensity(yp=0, xp=0, x=0, y=0, z=4.5, focal=np.inf):
    """
    Differential intensity at x,y,z due to emitter at xp,yp (zp=0)
    
    See eq. 3 in J. Kim et al, Nucl. Instrum. Meth. 141, 187 (1977)
    """
    a_sq = (z * tantheta)**2
    s_sq = (1-z/focal)**2
    exp_x = np.exp(-np.square(xp-x)/a_sq)
    exp_y = np.exp(-np.square(yp-y)/a_sq)
    return exp_x * exp_y / (np.pi * a_sq * s_sq)

def obs_point_intensity(x=0, y=0, z=4.5, S=0.1, focal=np.inf):
    """
    Total intensity at the observation point x,y,z
    Perform 2D numerical integration over square emitter with half-length `S`
    """
    intensity = integrate.dblquad(diff_intensity, 
                                  -S, S, -S, S, 
                                  (x, y, z, focal))
    return intensity[0] / (2*S)**2

def obs_point_intensity_erf(x=0, y=0, z=4.5, S=0.1, focal=np.inf):
    """
    Total intensity at the observation point x,y,z
    """
    a = z * tantheta
    s_sq = (2*S)**2 * (1-z/focal)**2
    intensity = (sp.erf((S-x)/a) + sp.erf((S+x)/a)) * \
                (sp.erf((S-y)/a) + sp.erf((S+y)/a)) / (4*s_sq)
    return intensity

def test_obs_point():
    x_grid = np.linspace(-0.8,0.8,20)
    intensity1 = np.empty(x_grid.shape)
    intensity2 = np.empty(x_grid.shape)
    for i,x in enumerate(x_grid):
        intensity1[i] = obs_point_intensity(x=x)
        intensity2[i] = obs_point_intensity_erf(x=x)
    assert(np.allclose(intensity1, intensity2))

def profile_intensity(z=4.5, size=0.8, resolution=0.02, S=0.1):
    """
    Return 2D intensity distribution in plane perp. to axis
    """
    xgrid = np.arange(-size, size+1e-4, resolution)
    ygrid = xgrid.copy()
    xv, yv = np.meshgrid(xgrid, ygrid)
    intensity = np.empty(xv.shape)
    for i in np.arange(xv.size):
        intensity.flat[i] = obs_point_intensity_erf(x=xv.flat[i], y=yv.flat[i],
                                                    z=z, S=S)
    # scale by emitter size and grid resolution
    # should sum to 1
    d_area = resolution**2
    scaled_intensity = intensity * d_area
    return xv, yv, intensity, scaled_intensity

def test_profile_intensity():
    # verify scalled_intensity sums to 1
    out = profile_intensity()
    assert(np.isclose(out[3].sum(), 1))
    out = profile_intensity(resolution=0.01)
    assert(np.isclose(out[3].sum(), 1))
    out = profile_intensity(S=0.2)
    assert(np.isclose(out[3].sum(), 1))
    
def cir_obs_point_intensity(r=0, z=4.5, R0=0.1, focal=np.inf, test=False):
    r_sq = r**2
    a_sq = (z * tantheta)**2
    R_sq = R0**2 * (1 - z/focal)**2
    metric = np.sqrt(4*r_sq*R_sq/(a_sq**2))
    int1 = np.exp(-r_sq/a_sq) * (1 - np.exp(-R_sq/a_sq) + r_sq/a_sq * 
                          (1-(1+R_sq/a_sq)*np.exp(-R_sq/a_sq))) / (np.pi*R_sq)
    if metric > 0.25 or test:
        def integrand(phip, rp, rr):
            ret = rp*np.exp(-(rr**2+rp**2-2*rr*rp*np.cos(phip))/a_sq)
            ret *= 1/(np.pi**2 * a_sq * R_sq)
            return ret
        int2 = integrate.dblquad(integrand, 0, R0, 0, 2*np.pi, [np.sqrt(r_sq)])
        if test:
            #print(metric, np.abs(int1-int2[0])/int2[0])
            assert(np.isclose(int1, int2[0], rtol=5e-5))
        int1 = int2[0]
    return int1
    
def test_cir_obs_point():
    cir_obs_point_intensity(r=0.05, z=14, R0=0.1, test=True)
    cir_obs_point_intensity(r=0.05, z=14, R0=0.05, test=True)
    cir_obs_point_intensity(r=0.05, z=13, R0=0.1, test=True)
    cir_obs_point_intensity(r=0.07, z=13, R0=0.1, test=True)
    
def cir_profile_intensity(z=4.5, R0=0.1, focal=np.inf, 
                          size=0.8, resolution=0.02):
    rgrid = np.arange(0, size+1e-5, resolution)
    intensity_r = np.empty(rgrid.shape)
    for i,r in enumerate(rgrid):
        intensity_r[i] = cir_obs_point_intensity(r=r, z=z, R0=R0, focal=np.inf)
    # weighted intensities sum to 1 to reflect particle flux conservation
    d_area = rgrid * resolution * 2*np.pi
    scaled_intensity = intensity_r * d_area
    return rgrid, intensity_r, scaled_intensity

def test_cir_profile():
    _,_,int_w = cir_profile_intensity(z=15, R0=0.05, size=1, resolution=0.01)
    assert(np.isclose(int_w.sum(), 1, rtol=5e-4))
    
if __name__=='__main__':
    # run tests
    test_obs_point()
    test_profile_intensity()
    test_cir_obs_point()
    test_cir_profile()
    # radial intensity
    f1 = plt.figure()
    zgrid = np.arange(1,15,3)
    for i,z in enumerate(zgrid):
        rgrid, intensity_r, _ = \
            cir_profile_intensity(z=z, resolution=0.02, R0=0.15)
        plt.plot(rgrid, intensity_r, label='z={:.1f}'.format(z))
    plt.legend()
    plt.title('Circular source intensity')
    plt.xlabel('Axial radius (m)')
    plt.ylabel('Intensity (a.u.)')
    # intensity along z axis
    f2 = plt.figure(figsize=(9.5,3.75))
    plt.subplot(121)
    z_grid = np.linspace(4,12,num=60)
    focal = np.array([7,70,np.inf])
    for f in focal:
        intensity = np.empty(z_grid.shape)
        for i,z in enumerate(z_grid):
            intensity[i] = obs_point_intensity_erf(z=z, focal=f)
        plt.semilogy(z_grid, intensity, label='focal {:.1f}'.format(f))
    plt.title('Rect. source axial intensity')
    plt.xlabel('Source axis (m)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend()
    # intensity along x axis
    plt.subplot(122)
    x_grid = np.linspace(-0.8,0.8,50)
    z_grid = np.arange(1,15,3)
    for z in z_grid:
        intensity = np.empty(x_grid.shape)
        for i,x in enumerate(x_grid):
            intensity[i] = obs_point_intensity_erf(x=x,z=z)
        plt.plot(x_grid, intensity, label='z={:.1f}'.format(z))
    plt.title('Rect. source profile')
    plt.xlabel('x (m)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend()
    plt.tight_layout()
    # full profile intensity
    f3 = plt.figure()
    zgrid = np.arange(1,15,3)
    for i,z in enumerate(zgrid):
        xgrid, ygrid, intensity, scaled_intensity = \
            profile_intensity(z=z, size=0.8, resolution=0.01)
        plt.plot(xgrid[0,:], intensity[ygrid.shape[0]//2,:], 
                 label='z={:.1f}'.format(z))
    plt.xlabel('x (m)')
    plt.ylabel('Intensity (a.u.)')
    plt.title('Intensity distribution at y=0')
    plt.legend()
    plt.tight_layout()
