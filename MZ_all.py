"""
This program plots the MZ relation for galaxies from different simulations,
compared to the values in Tremonti et al (2004).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels
from iccpy.gadget.subfind import SubfindCatalogue
from iccpy.utils import match

from functions import get_massive, get_subh_from_CM


# These are the included simulations with the number of the snapshot to be used:

sims = [
        ['gecko_A/run_01', 100],
        ['gecko_C/run_01', 135],
        ['Aq5', 127]
        ]
# Here we'll store the data for each simulation:
dict = {}

for sim in sims:
    path = '/home/luis/Documents/simus/{}/outputs'.format(sim[0])
    snap_num = sim[1]  # number of the snapshot to be loaded

    snap = load_snapshot(directory=path, snapnum=snap_num, label_table=cecilia_labels)
    hubble0 = snap.header.hubble0[0]
    cat = SubfindCatalogue(path, snap_num) # get a catalogue of subhaloes
    threshold_mass = 1e10
    subhaloes = get_massive(snap, cat, threshold_mass)
    num = np.size(subhaloes)

    print(  'There are {} subhaloes with mass higher than {} solar masses in {}, snap {}'.format(num, threshold_mass, sim[0], sim[1]))

    # We will save these quantities for each subhalo:
    OH_gas = np.zeros(num)
    OH_stars = np.zeros(num)
    stellar_mass = np.zeros(num)
    SFR = np.zeros(num)

    subfind = False
    i = 0
    for subh in subhaloes:
        print(i)

        # To track the gas particles, I search for matching IDs and matching IDs+2**31 (for the particles that have formed stars)
        if subfind:
            ind_stars   = match(subh.ids, snap['ID  '][4])
            ind_stars   = ind_stars[ind_stars != -1]
            ind_gas     = np.concatenate([match(subh.ids, snap['ID  '][0]), match(subh.ids + 2**31, snap['ID  '][0])])
            ind_gas     = ind_gas[ind_gas != -1]
        else:
            ind_stars   = get_subh_from_CM(snap, subh, 4)
            ind_gas     = get_subh_from_CM(snap, subh, 0)

        mass_stars = snap['MASS'][4][ind_stars] * 1e10 / hubble0        # masses of the stars in the subhalo
        stellar_mass[i] = np.sum(mass_stars)                        # total subhalo mass
        SFR[i] = np.sum(mass_stars[snap['AGE '][4][ind_stars] < .1])    # SFR calculated from the total mass of young stars (time interval missing to correct units)
        # We want to consider gas within r_eff:
        # r_eff = cat.subhalo[i].half_mass_radius
        CM = subh.com

        star_pos = snap['POS '][4][ind_stars] - CM
        star_x = star_pos[:,0]
        star_y = star_pos[:,1]
        star_z = star_pos[:,2]
        star_r = np.sqrt(star_x**2 + star_y**2 + star_z**2)
        r_eff, m_r = 0, 0
        while m_r < stellar_mass[i] / 2:
            r_eff += .1
            m_r = np.sum(mass_stars[star_r < r_eff])


        gas_pos = snap['POS '][0][ind_gas] - CM
        gas_x = gas_pos[:,0]
        gas_y = gas_pos[:,1]
        gas_z = gas_pos[:,2]
        gas_r = np.sqrt(gas_x**2 + gas_y**2 + gas_z**2)
        ind_gas = ind_gas[gas_r < r_eff]


        Zgas = snap['Z   '][0][ind_gas]
        O_gas = np.sum(Zgas[:, 3])
        H_gas = np.sum(Zgas[:, 6])
        OH_gas[i] = 12 + np.log10(O_gas / 16 / H_gas)

        # If we want to consider only cold gas:
        cold_gas = True
        if cold_gas:
            XH = Zgas[:, 6] / snap['MASS'][0][ind_gas]
            yHelium = (1-XH) / (4*XH)
            mu = (1 + 4*yHelium) / (1 + yHelium + snap['NE  '][0][ind_gas])  # Mean molecular weight
            ugas = snap['U   '][0][ind_gas]
            temp = 2/3 * ugas * mu * 1.6726 / 1.3806 * 1e-8
            temp_gas = temp * 1e10
            ind_gas = ind_gas[temp_gas < 2e4]
            Zgas = snap['Z   '][0][ind_gas]
            O_gas = np.sum(Zgas[:, 3])
            H_gas = np.sum(Zgas[:, 6])
            OH_gas[i] = 12 + np.log10(O_gas / 16 / H_gas)

        # When we look at stars, we care only about recent stars of this age:
        age = .5
        Zstars = snap['Z   '][4][ind_stars][snap['AGE '][4][ind_stars] < age]
        O_stars = np.sum(Zstars[:, 3])
        H_stars = np.sum(Zstars[:, 6])
        OH_stars[i] = 12 + np.log10(O_stars / 16 / H_stars)
        i += 1

    # Here we filter out nan values:
    stellar_mass = stellar_mass[OH_gas < 100]
    OH_gas = OH_gas[OH_gas < 100]

    # We save everything to an entry in dict using the name of the corresponding
    # simulation as a key:
    dict[sim[0]] = [stellar_mass, OH_gas, OH_stars, SFR]


# Data from Tremonti et al 2004:
Trem_mass =         np.array([8.57, 8.67, 8.76, 8.86, 8.96, 9.06, 9.16, 9.26,
                    9.36, 9.46, 9.57, 9.66, 9.76, 9.86, 9.96, 10.06, 10.16,
                    10.26, 10.36, 10.46, 10.56, 10.66, 10.76, 10.86, 10.95,
                    11.05, 11.15, 11.25])
P_2_sigma_down =    np.array([8.18, 8.11, 8.13, 8.14, 8.21, 8.26, 8.37, 8.39,
                    8.46, 8.53, 8.59, 8.60, 8.63, 8.67, 8.71, 8.74, 8.77, 8.80,
                    8.82, 8.85, 8.87, 8.89, 8.91, 8.93, 8.93, 8.92, 8.94, 8.93])
P_1_sigma_down =    np.array([8.25, 8.28, 8.32, 8.37, 8.46, 8.56, 8.59, 8.60,
                    8.63, 8.66, 8.69, 8.72, 8.76, 8.80, 8.83, 8.85, 8.88, 8.92,
                    8.94, 8.96, 8.98, 9.00, 9.01, 9.02, 9.03, 9.03, 9.04, 9.03])
median         =    np.array([8.44, 8.48, 8.57, 8.61, 8.63, 8.66, 8.68, 8.71,
                    8.74, 8.78, 8.82, 8.84, 8.87, 8.90, 8.94, 8.97, 8.99, 9.01,
                    9.03, 9.05, 9.07, 9.08, 9.09, 9.10, 9.11, 9.11, 9.12, 9.12])
P_1_sigma_up   =    np.array([8.64, 8.65, 8.70, 8.73, 8.75, 8.82, 8.82, 8.86,
                    8.88, 8.92, 8.94, 8.96, 8.99, 9.01, 9.05, 9.06, 9.09, 9.10,
                    9.11, 9.12, 9.14, 9.15, 9.15, 9.16, 9.17, 9.17, 9.18, 9.18])
P_2_sigma_up   =    np.array([8.77, 8.84, 8.88, 8.89, 8.95, 8.97, 8.95, 9.04,
                    9.03, 9.07, 9.08, 9.09, 9.10, 9.12, 9.14, 9.15, 9.16, 9.17,
                    9.18, 9.21, 9.21, 9.23, 9.24, 9.25, 9.26, 9.27, 9.29, 9.29])


fig = plt.figure()

plt.loglog(dict[sims[0][0]][0], dict[sims[0][0]][1], '^', color='m', label='Gecko A') # gecko_A, run_01
plt.loglog(dict[sims[1][0]][0], dict[sims[1][0]][1], 'o', color='m', label='Gecko C') # gecko_C, run_01
plt.loglog(dict[sims[2][0]][0], dict[sims[2][0]][1], '*', color='b', label='Aquarius 5') # Aq5

plt.semilogy(10**Trem_mass, P_2_sigma_down, 'yellow')
plt.semilogy(10**Trem_mass, P_2_sigma_up, 'yellow')
plt.semilogy(10**Trem_mass, P_1_sigma_down, 'orange')
plt.semilogy(10**Trem_mass, P_1_sigma_up, 'orange')
plt.semilogy(10**Trem_mass, median, 'red')

# plt.xlim(10**Trem_mass[0], 10**Trem_mass[-1])
plt.xlabel(r'$M_{\star}\ [M_{\odot}]$', fontsize=16)
plt.ylabel(r'$12 + log[O/H]_{gas}$', fontsize=16)
plt.grid()
plt.legend()
plt.show()
