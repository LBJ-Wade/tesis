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

path = '/media/luis/82A8355FA83552C1/CLUES_Gustavo'

m_stars_0 = []
m_stars_33 = []
OH0 = []
OH33 = []
FeH0 = []
FeH33 = []

merg_tree_0 = path + '/postproc/Prog_0.dat'
merg_tree_0 = np.flip(np.loadtxt(merg_tree_0), axis=0)

merg_tree_33 = path + '/postproc/Prog_33.dat'
merg_tree_33 = np.flip(np.loadtxt(merg_tree_33), axis=0)

starting_snap = int(merg_tree_0[0][0])

for j in range(starting_snap, 127):
    print('Analysing snapshot {}'.format(j))
    snap_num = j
    snap = load_snapshot(directory=path+'/outputs', snapnum=snap_num, label_table=cecilia_labels)
    cat = SubfindCatalogue(path+'/outputs', snap_num) # get a catalogue of subhaloes
    hubble0 = snap.header.hubble0[0]

    big_two = [cat.subhalo[int(merg_tree_0[j - starting_snap][1])], cat.subhalo[int(merg_tree_33[j - starting_snap][1])]]

    i=0
    for subh in big_two:
        # To track the gas particles, I search for matching IDs and matching IDs+2**31 (for the particles that have formed stars)
        ind_stars   = match(subh.ids, snap['ID  '][4])
        ind_stars   = ind_stars[ind_stars != -1]
        ind_gas     = np.concatenate([match(subh.ids, snap['ID  '][0]), match(subh.ids + 2**31, snap['ID  '][0])])
        ind_gas     = ind_gas[ind_gas != -1]

        mass_stars = snap['MASS'][4][ind_stars] * 1e10 / hubble0        # masses of the stars in the subhalo
        stellar_mass = np.sum(mass_stars)                        # total subhalo mass

        # We want to consider gas within r_eff:
        CM = subh.pot_min
        star_pos = snap['POS '][4][ind_stars] - CM
        star_r = np.linalg.norm(star_pos, axis=1)

        r_eff, m_r = 0, 0
        while m_r < stellar_mass / 2:
            r_eff += .1
            m_r = np.sum(mass_stars[star_r < r_eff])

        gas_pos = snap['POS '][0][ind_gas] - CM
        gas_r = np.linalg.norm(gas_pos, axis=1)
        ind_gas = ind_gas[gas_r < r_eff]

        Zgas = snap['Z   '][0][ind_gas]

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
        Fe_gas = np.sum(Zgas[:, 4])
        H_gas = np.sum(Zgas[:, 6])
        OH_gas = 12 + np.log10(O_gas / 16 / H_gas)
        FeH_gas = 4.55 + np.log10(Fe_gas / 56 / H_gas)

        # # When we look at stars, we care only about recent stars of this age:
        # age = .5
        # Zstars = snap['Z   '][4][ind_stars][snap['AGE '][4][ind_stars] < age]
        # O_stars = np.sum(Zstars[:, 3])
        # H_stars = np.sum(Zstars[:, 6])
        # OH_stars = 12 + np.log10(O_stars / 16 / H_stars)

        if i == 0:
            m_stars_0.append([stellar_mass])
            OH0.append(OH_gas)
            FeH0.append(FeH_gas)
        if i == 1:
            m_stars_33.append([stellar_mass])
            OH33.append(OH_gas)
            FeH33.append(FeH_gas)
        i += 1

# plt.semilogx(m_stars_1, OH0, label=r'$[O/H]_{2}$')
# plt.semilogx(m_stars_0, FeH0, label=r'$[Fe/H]_{0}$')
# plt.semilogx(m_stars_1, FeH1, label=r'$[Fe/H]_{1}$')
# plt.semilogx(m_stars_2, FeH2, label=r'$[Fe/H]_{2}$')

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


plt.semilogy(10**Trem_mass, P_2_sigma_down, 'yellow')
plt.semilogy(10**Trem_mass, P_2_sigma_up, 'yellow')
plt.semilogy(10**Trem_mass, P_1_sigma_down, 'orange')
plt.semilogy(10**Trem_mass, P_1_sigma_up, 'orange')
plt.semilogy(10**Trem_mass, median, 'red')

t = np.array(range(starting_snap, 127))

plt.scatter(m_stars_0, OH0, label=r'$[O/H]_{0}$', c=t)
plt.scatter(m_stars_33, OH33, label=r'$[O/H]_{33}$', c=t)
plt.semilogx(m_stars_0, OH0, label=r'$[O/H]_{0}$')
plt.semilogx(m_stars_33, OH33, label=r'$[O/H]_{33}$')



# plt.xlim(10**Trem_mass[0], 10**Trem_mass[-1])
plt.xlabel(r'$M_{\star}\ [M_{\odot}]$', fontsize=16)
plt.ylabel(r'$12 + log[O/H]_{gas}$', fontsize=16)
plt.grid()
plt.legend()

plt.show()
