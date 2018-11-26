import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels
from iccpy.gadget.subfind import SubfindCatalogue
from iccpy.utils import match
import os


sim = 'gecko/gecko_C'
path = '/home/luis/Documents/%s/run_01/outputs' %sim
runs = os.listdir(path)

snap_num = 135  # number of the snapshot to be loaded

snap = load_snapshot(directory = path, snapnum = snap_num, label_table = cecilia_labels)
num = int(input('Cuántos subhalos incluyo?    ')) # number of subhaloes to analyze
cat = SubfindCatalogue(path, snap_num) # get a catalogue of subhaloes


# We will save these quantities for each subhalo:
OH_gas = np.zeros(num)
OH_stars = np.zeros(num)
stellar_mass = np.zeros(num)
SFR = np.zeros(num)
 
for i in range(num):
    print(i)

    # To track the gas particles, I search for matching IDs and matching IDs+2**31 (for the particles that have formed stars)
    ind_stars = match(cat.subhalo[i].ids, snap['ID  '][4])
    ind_stars = ind_stars[ind_stars != -1]

    mass_stars = snap['MASS'][4][ind_stars] * 1e10 * .73        # masses of the stars in the subhalo
    stellar_mass[i] = np.sum(mass_stars)                        # total subhalo mass
    SFR[i] = np.sum(mass_stars[snap['AGE '][4][ind_stars] < .1])                   # SFR calculated from the total mass of young stars (time interval missing to correct units)
    # We want to consider gas within r_eff:
    # r_eff = cat.subhalo[i].half_mass_radius
    CM = cat.subhalo[i].pot_min

    star_pos = snap['POS '][4][ind_stars] - CM
    star_x = star_pos[:,0]
    star_y = star_pos[:,1]
    star_z = star_pos[:,2]
    star_r = np.sqrt(star_x**2 + star_y**2 + star_z**2)
    r_eff, m_r = 0, 0
    while m_r < stellar_mass[i] / 2:
        r_eff += .1
        m_r = np.sum(mass_stars[star_r < r_eff])

    ind_gas = np.concatenate([match(cat.subhalo[i].ids, snap['ID  '][0]), match(cat.subhalo[i].ids + 2**31, snap['ID  '][0])])
    ind_gas = ind_gas[ind_gas != -1]

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
    # XH = Zgas[:, 6] / snap['MASS'][0][ind_gas]
    # yHelium = (1-XH) / (4*XH)
    # mu = (1 + 4*yHelium) / (1 + yHelium + snap['NE  '][0][ind_gas])  # Mean molecular weight
    # ugas = snap['U   '][0][ind_gas]
    # temp = 2/3 * ugas * mu * 1.6726 / 1.3806 * 1e-8
    # temp_gas = temp * 1e10
    # ind_gas = ind_gas[temp_gas < 2e4]
    # Zgas = snap['Z   '][0][ind_gas]

    # When we look at stars, we care only about recent stars of this age:
    age = .5
    Zstars = snap['Z   '][4][ind_stars][snap['AGE '][4][ind_stars] < age]
    O_stars = np.sum(Zstars[:, 3])
    H_stars = np.sum(Zstars[:, 6])
    OH_stars[i] = 12 + np.log10(O_stars / 16 / H_stars)


def fitMZ(x, a, b):
    return a + b*(x - 3.5)*np.exp(3.5 - x)

popt, pcov = curve_fit(fitMZ, np.log10(stellar_mass) - 8, OH_stars)
print(popt)

fit = fitMZ(np.log10(sorted(stellar_mass)) - 8, popt[0], popt[1])

fig1 = plt.figure(1)
plt.loglog(stellar_mass, OH_stars, '*')
plt.semilogx(sorted(stellar_mass), fit, label=r'$y = a + b(x-3.5)e^{-(x-3.5)}$'+'\n'+r'$a = %s$' %np.round(popt[0], 2) +'\n'+r'$b = %s$' %np.round(popt[1], 3)) 
plt.xlabel(r'$M_{\star}\ [M_{\odot}]$', fontsize=16)
plt.ylabel(r'$[O/H]_{\star}$', fontsize=16)
plt.grid()
plt.legend()


residuals = OH_stars - fit

fig2 = plt.figure(2)
plt.semilogx(SFR, residuals, '*')
plt.xlabel(r'$SFR$', fontsize=16)
plt.ylabel(r'$[O/H]_{\star} residuals$', fontsize=16)
plt.grid()

plt.show()

