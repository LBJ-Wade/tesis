import numpy as np
import matplotlib.pyplot as plt
import os

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels
from iccpy.gadget.subfind import SubfindCatalogue
from iccpy.utils import match

from functions import M_vir_crit, get_massive


sim = 'gecko/gecko_C'
path = '/home/luis/Documents/%s/' %sim
RUNS = os.listdir(path)


for RUN in RUNS:
	snap_num = 135  # number of the snapshot to be loaded
	snap = load_snapshot(directory = path + RUN + '/outputs', snapnum = snap_num, label_table = cecilia_labels)
	cat = SubfindCatalogue(path + RUN + '/outputs', snap_num) # get a catalogue of subhaloes




	layers = 5000
	rmax = 150 # in kpc

	# We filter the most massive subhaloes by total mass, to then compute virial mass:
	subhaloes = get_massive(snap, cat, 1e10)

	virial_mass = np.zeros(np.size(subhaloes))
	virial_mass_stars = np.zeros(np.size(subhaloes))
	r_vir = np.zeros(np.size(subhaloes))
	# We iterate for each subhalo
	i = 0
	for subh in subhaloes:
		virial_mass[i], virial_mass_stars[i], r_vir[i] = M_vir_crit(snap, subh, layers, rmax)
		i += 1

	# We calculate fMvir:
	bins1 = np.logspace(min(np.log10(virial_mass[virial_mass!=0])), max(np.log10(virial_mass[virial_mass!=0])))
	n1 = np.zeros(np.size(bins1))
	i = 0
	for bin in bins1:
		n1[i] = np.sum(virial_mass > bin)
		i += 1

	# We calculate fMvir_stars:
	bins2 = np.logspace(min(np.log10(virial_mass_stars[virial_mass_stars!=0])), max(np.log10(virial_mass_stars[virial_mass_stars!=0])))
	n2 = np.zeros(np.size(bins2))
	i = 0
	for bin in bins2:
		n2[i] = np.sum(virial_mass_stars > bin)
		i += 1

fig, (ax1, ax2) = plt.subplots(1, 2)
plt.rcParams.update({'font.size': 16})

ax1.semilogx(bins1, n1)
ax1.set_xlabel(r'$M_{vir}\ [M_{\odot}]$')
ax1.set_ylabel(r'Galaxias con $M>M_{vir}$')
ax1.grid()

ax2.semilogx(bins2, n2)
ax2.set_xlabel(r'$M_{\star_{vir}}\ [M_{\odot}]$')
ax2.set_ylabel(r'Galaxias con $M_{\star}>M_{\star_{vir}}$')
ax2.grid()

plt.show()
