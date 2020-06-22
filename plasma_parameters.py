#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 08:44:58 2020

@author: drsmith
"""

import numpy as np
import scipy.constants as pc

class Params(object):

    def __init__(self,
                 ne=5e13,  # 1/cm**3
                 Te=1.5,  # keV
                 Ti=1.5,  # keV
                 Bmag=2.6,  # T
                 mu = 2,  # ion mass in AMUs
                 Z = 1,  # ion charge
                 ):
        self.ne = ne
        self.Te = Te
        self.Ti = Ti
        self.Bmag = Bmag
        self.mu = mu
        self.Z = Z
        
        self.q = self.Z * pc.e
        
        self.ni = ne / Z  # ion density by quasi-neutrality
        
        ev2j = pc.eV  # J/eV conversion factor
        gamma = 1.4  # adiabatic index
        rtol = 5e-3
        
        self.Te_J = self.Te * 1e3 * ev2j
        self.Ti_J = self.Ti * 1e3 * ev2j

        m_i = mu * pc.m_p  # ion mass in kg
        
        # thermal speeds; note sqrt(2)
        self.v_te = np.sqrt(2 * Te*1e3 * ev2j / pc.m_e)
        assert(np.allclose(self.v_te, 5.93e5*np.sqrt(Te*1e3), rtol=rtol))
        self.v_ti = np.sqrt(2 * Ti*1e3 * ev2j / m_i)
        assert(np.allclose(self.v_ti, 1.38e4*np.sqrt(Ti*1e3/mu), rtol=rtol))
        
        # ion sound speed
        self.c_s = np.sqrt((Z*Te + gamma*Ti) * 1e3*ev2j / m_i)
        
        # Alfven speed
        self.v_A = np.sqrt(Bmag**2/(pc.mu_0 * (self.ni*1e6) * m_i))
        # print(self.v_A, 2.18e18*Bmag/np.sqrt(mu*self.ni*1e6))
        # assert(np.allclose(self.v_A, 2.18e18*Bmag/np.sqrt(mu*self.ni*1e6), rtol=1e-2))
        
        # gyro-frequencies
        self.Omega_e = pc.e * Bmag / pc.m_e
        assert(np.allclose(self.Omega_e, 
                           1.759e11*Bmag, rtol=rtol))
        self.Omega_i = Z * pc.e * Bmag / m_i
        assert(np.allclose(self.Omega_i, 
                           9.579e7*Bmag/mu, rtol=rtol))
        
        # gyro-radii
        self.rho_e = self.v_te / self.Omega_e
        assert(np.allclose(self.rho_e, 
                           1.066e-4*np.sqrt(Te)/Bmag, rtol=rtol))
        self.rho_i = self.v_ti / self.Omega_i
        assert(np.allclose(self.rho_i, 
                           4.57e-3*np.sqrt(mu*Ti)/Bmag, rtol=rtol))
        self.rho_s = self.c_s / self.Omega_i
        
        # # k @ k*rho-i=0.3
        # self.k1 = 0.1 / self.rho_i
        # self.k3 = 0.3 / self.rho_i
        # self.k6 = 0.6 / self.rho_i
                
    def __str__(self):
        output  = "ne = {:.3g} 1/cm**3".format(self.ne) + "\n"
        output += "Te = {:.3g} keV".format(self.Te) + "\n"
        output += "Ti = {:.3g} keV".format(self.Ti) + "\n"
        output += "|B| = {:.3g} T".format(self.Bmag) + "\n"
        output += "mu = m_i/m_p = {:.1f}".format(self.mu) + "\n"
        output += "Z = {:.3g}".format(self.Z) + "\n"
        output += "ni = {:.3g} 1/cm**3 (quasi-neutrality)".format(self.ni) + "\n"
        output += "v_te = {:.3g} km/s".format(self.v_te/1e3) + "\n"
        output += "v_ti = {:.3g} km/s".format(self.v_ti/1e3) + "\n"
        output += "c_s = {:.3g} km/s".format(self.c_s/1e3) + "\n"
        output += "v_A = {:.3g} km/s".format(self.v_A/1e3) + "\n"
        output += "Omega_e = {:.3g} rad/s ({:.3g} GHz)".format(
            self.Omega_e, self.Omega_e/1e9/(2*np.pi)) + "\n"
        output += "Omega_i = {:.3g} rad/s ({:.3g} MHz)".format(
            self.Omega_i, self.Omega_i/1e6/(2*np.pi)) + "\n"
        output += "rho_e = {:.3g} mm".format(self.rho_e*1e3) + "\n"
        output += "rho_i = {:.3g} mm".format(self.rho_i*1e3) + "\n"
        output += "rho_s = {:.3g} mm".format(self.rho_s*1e3)
        return output


if __name__=='__main__':
    p = Params()
    print(p)