#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 07:20:48 2019

@author: drsmith
"""

from scipy import constants

from cherab.core import atomic
import cherab.openadas as oa


adas = oa.OpenADAS(permit_extrapolation=True)

## get CX rate like .cherab/openadas/repository/beam/cx/h/c/6.json
#c_cx = adas.beam_cx_rate(deuterium, carbon, 6, (8,7))
## get beam emission rates like .cherab/openadas/repository/beam/emission/h/h/1.json
#d_emission = adas.beam_emission_rate(deuterium, deuterium, 1, (3,2))
## D0(n=3) beam population due to D1 plasma
#balmer_alpha_pop = adas.beam_population_rate(deuterium, 3, deuterium, 1)
## get beam stopping rates like .cherab/openadas/repository/beam/stopping/h/h/1.json
#d_stopping = adas.beam_stopping_rate(deuterium, deuterium, 1)
#c_stopping = adas.beam_stopping_rate(deuterium, carbon, 6)
## get excitation rate like .cherab/openadas/repository/pec/excitation/h/0.json
#d_impact = adas.impact_excitation_rate(deuterium, 0, (3,2))
#he_rec = adas.recombination_rate(helium,1,(3,2))


beam_species = atomic.lookup_isotope('hydrogen', number=1)
target_species = atomic.lookup_isotope('hydrogen', number=1)
impurity_species = atomic.lookup_isotope('carbon12')

beam_voltage_ev = 60e3
plasma_density = 1e19
impurity_density = plasma_density / 100
ti_ev = 1e3

beam_ev_amu = beam_voltage_ev / beam_species.mass_number

# beam stopping rates
# (beam atom, target atom, target ionization state)
stop_rate_h = adas.beam_stopping_rate(beam_species, target_species, 1)
stop_rate_c = adas.beam_stopping_rate(beam_species, impurity_species, 6)
# evaluate(beam eV/amu, target #/m**3, target eV)
# rate is m**3/s
rate_h = stop_rate_h(beam_ev_amu, plasma_density, ti_ev)
rate_c = stop_rate_c(beam_ev_amu, impurity_density, ti_ev)
print('{} stopping rate on {}: {:.4g} m**3/s'.
      format(beam_species.symbol, target_species.symbol, rate_h))
print('{} stopping rate on {}: {:.4g} m**3/s'.
      format(beam_species.symbol, impurity_species.symbol, rate_c))

# beam emission rate
# (beam atom, target atom, target ionization state, beam transition)
beam_emission_rate = adas.beam_emission_rate(beam_species, target_species, 1, (3, 2))
# evaluate(beam eV/amu, target #/m**3, target eV)
# rate is W-m**3/s
be_rate = beam_emission_rate(beam_ev_amu, plasma_density, ti_ev)
print('{} beam 3->2 emission rate: {:.4g} W-m**3/s'.
      format(beam_species.symbol, be_rate))

# transition wavelength
# (atom, ionization state, transition)
wavelength = adas.wavelength(beam_species, 0, (3, 2))
print('{} 3->2 wavelength: {:.2f} nm'.
      format(beam_species.symbol, wavelength))
# convert rate to phot-m**3/s
photon_rate = be_rate*(wavelength*1e-9)/(constants.Planck * constants.speed_of_light)
print('{} beam 3->2 photon rate: {:.4g} phot-m**3/s'.
      format(beam_species.symbol, photon_rate))