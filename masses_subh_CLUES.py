import numpy as np
import matplotlib.pyplot as plt
import os

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels
from iccpy.gadget.subfind import SubfindCatalogue
from iccpy.utils import match

from functions import get_massive


"""
This scatters Mstars vs Mvir and Mgas vs Mstar for a lot of subhaloes,
with different colors according to the number of star particles in each
subhalo
"""


path = '/media/luis/82A8355FA83552C1/CLUES_Gustavo/outputs'

snap_num = 127  # number of the snapshot to be loaded
snap = load_snapshot(directory = path, snapnum = snap_num, label_table = cecilia_labels)
hubble0 = snap.header.hubble0[0]
cat = SubfindCatalogue(path, snap_num, SAVE_MASSTAB=True) # get a catalogue of subhaloes

threshold_mass = 1e10
massives = get_massive(snap, cat, threshold_mass)

mass_stars = np.zeros(np.size(massives))
mass_gas = np.zeros(np.size(massives))
mass = np.zeros(np.size(massives))
color = []

# We iterate for each subhalo
i=0
for subh in massives:

    # print(i)
    mass_stars[i]   = subh.masstab[4] * 1e10 / hubble0
    mass_gas[i]     = subh.masstab[0] * 1e10 / hubble0
    mass[i]         = subh.mass       * 1e10 / hubble0

	# ind_stars = match(cat.subhalo[i].ids, snap['ID  '][4])
	# ind_stars = ind_stars[ind_stars != -1]
	# ind_gas = np.concatenate([match(cat.subhalo[i].ids, snap['ID  '][0]), match(cat.subhalo[i].ids + 2**31, snap['ID  '][0])])
	# ind_gas = ind_gas[ind_gas != -1]
	# mass_stars[i] = np.sum(snap['MASS'][4][ind_stars] * 1e10 / .73)
	# mass_gas[i] = np.sum(snap['MASS'][0][ind_gas] * 1e10 / .73)
	# mass[i] = mass_stars[i] + mass_gas[i]
	# We asign a color depending on the amount of star particles of the subhalo
	# if np.size(ind_stars) < 100:
	# 	color.append('g')
	# elif np.size(ind_stars) < 1000:
	# 	color.append('b')
	# else:
	# 	color.append('r')
    i+=1

fig, (ax1, ax2) = plt.subplots(1, 2)
plt.rcParams.update({'font.size': 16})

ax1.loglog(mass[mass_gas!=0], mass_stars[mass_gas!=0], '.')
ax1.set_xlabel(r'$M_{tot}\ [M_{\odot}]$')
ax1.set_ylabel(r'$M_{\star}\ [M_{\odot}]$')
ax1.grid()

ax2.loglog(mass_stars[mass_gas!=0], mass_gas[mass_gas!=0], '.')
ax2.set_xlabel(r'$M_{\star}\ [M_{\odot}]$')
ax2.set_ylabel(r'$M_{gas}\ [M_{\odot}]$')
ax2.grid()

plt.show()
