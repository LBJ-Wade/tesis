import numpy as np
import matplotlib.pyplot as plt
import os

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels
from iccpy.gadget.subfind import SubfindCatalogue
from iccpy.utils import match

"""
This scatters Mstars vs Mtot and Mgas vs Mstar for a lot of subhaloes,
with different colors according to the number of star particles in each
subhalo
"""

sim = 'gecko/gecko_C'
path = '/home/luis/Documents/%s/' %sim
RUNS = os.listdir(path)

for RUN in RUNS:
	snap_num = 135  # number of the snapshot to be loaded
	snap = load_snapshot(directory = path + RUN + '/outputs', snapnum = snap_num, label_table = cecilia_labels)
	cat = SubfindCatalogue(path + RUN + '/outputs', snap_num) # get a catalogue of subhaloes

	subh = 100 #np.size(cat.subhalo)
	mass_stars = np.zeros(subh)
	mass_gas = np.zeros(subh)
	mass = np.zeros(subh)
	color = []
	# We iterate for each subhalo
	for i in range(subh):
		print(i)
		ind_stars = match(cat.subhalo[i].ids, snap['ID  '][4])
		ind_stars = ind_stars[ind_stars != -1]
		ind_gas = np.concatenate([match(cat.subhalo[i].ids, snap['ID  '][0]), match(cat.subhalo[i].ids + 2**31, snap['ID  '][0])])
		ind_gas = ind_gas[ind_gas != -1]
		mass_stars[i] = np.sum(snap['MASS'][4][ind_stars] * 1e10 / .73)
		mass_gas[i] = np.sum(snap['MASS'][0][ind_gas] * 1e10 / .73)
		mass[i] = mass_stars[i] + mass_gas[i]
		# We asign a color depending on the amount of star particles of the subhalo
		if np.size(ind_stars) < 100:
			color.append('g')
		elif np.size(ind_stars) < 1000:
			color.append('b')
		else:
			color.append('r')

fig, (ax1, ax2) = plt.subplots(1, 2)
plt.rcParams.update({'font.size': 16})

ax1.scatter(np.log10(mass[mass_gas!=0]), np.log10(mass_stars[mass_gas!=0]), c=color)
ax1.set_xlabel(r'$M_{tot}\ [M_{\odot}]$')
ax1.set_ylabel(r'$M_{\star}\ [M_{\odot}]$')
ax1.grid()

ax2.scatter(np.log10(mass_stars[mass_gas!=0]), np.log10(mass_gas[mass_gas!=0]), c=color)
ax2.set_xlabel(r'$M_{\star}\ [M_{\odot}]$')
ax2.set_ylabel(r'$M_{gas}\ [M_{\odot}]$')
ax2.grid()
plt.legend()

plt.show()
