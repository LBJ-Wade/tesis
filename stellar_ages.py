import numpy as np
import matplotlib.pyplot as plt

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels
from iccpy.gadget.subfind import SubfindCatalogue
from iccpy.utils import match
from astropy.cosmology import FlatLambdaCDM

from functions import M_vir_crit, get_massive
import os

sim = 'gecko/gecko_C'
path = '/home/luis/Documents/%s/' %sim
RUNS = os.listdir(path)

for RUN in RUNS:
    snap_num = 135  # number of the snapshot to be loaded
    snap = load_snapshot(directory = path + RUN + '/outputs', snapnum = snap_num, label_table = cecilia_labels)
    cat = SubfindCatalogue(path + RUN + '/outputs', snap_num) # get a catalogue of subhaloes
    subhaloes = get_massive(snap, cat, 1e10)

    # We choose an element to calculate [X/H]_t
    elements = ['He', 'C', 'N', 'O', 'Fe', 'Mg', 'H', 'Si', 'Ba', 'Eu', 'Sr', 'Y']
    X = elements.index(str(input('Choose an element from He, C, N, O, Fe, Mg, H, Si, Ba, Eu, Sr, Y:   ')))
    Xmass = [4, 12, 14, 16, 56, 24.3, 1, 28, 137.3, 152, 87.6, 88.9][X]
    Xsolar = [10.93, 8.39, 7.78, 8.66, 7.45, 7.53, 12, 7.51, 2.17, 0.52, 2.92, 2.21][X]



    # We iterate for each subhalo
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(19.2, 12.8))
    i = 0

    for subh in subhaloes:
        r_200 = M_vir_crit(snap, subh, 5000, 300)[2]

        ind_stars = match(subh.ids, snap['ID  '][4])
        ind_stars = ind_stars[ind_stars != -1]
        # We select the inner stars:
        CM = subh.com
        pos = snap['POS '][4][ind_stars] - CM
        pos = pos / .73 # change units to kpc
        pos_x = pos[:,0]
        pos_y = pos[:,1]
        pos_z = pos[:,2]
        r = np.sqrt(pos_x**2 + pos_y**2 + pos_z**2)
        ind_stars = ind_stars[r < r_200 /3]

        # We skip the subhalo if it has no stars:
        if np.size(ind_stars) > 0:

            # Here we turn the age of the stars from scale factor to Gyr
            cosmo = FlatLambdaCDM(H0=70, Om0=.3)
            age_stars = snap['AGE '][4][ind_stars]				 	# Age in scale factor
            age_stars = (1 - age_stars) / age_stars				 	# Age in redshift
            age_stars = np.array(cosmo.lookback_time(age_stars)) 	# Lookback time in Gyr
            time_stars = float(np.array(cosmo.lookback_time(9999))) - age_stars		# Time at which the star was born in Gyr


            # We calculate [X/H]
            H_stars = snap['Z   '][4][ind_stars][:, 6]
            X_stars = snap['Z   '][4][ind_stars][:, X]
            XH_stars = -Xsolar + 12 + np.log10(X_stars / Xmass / H_stars)


            bins = 50 # Number of bins for SFR
            ages, agebins = np.histogram(age_stars, bins=bins)
            XH, XHbins = np.histogram(XH_stars, bins=50, range=(-5, .5))


            # We plot ages in the upper row
            plt.subplot(2, 4, (i%4) + 1)
            plt.bar(agebins[:-1], ages, width=(agebins[-1] - agebins[0]) / (np.size(agebins) - 1))
            plt.xlabel(r'$Age\ (Gyr)$', fontsize=16)
            plt.ylabel(r'$Number\ of\ star\ particles$', fontsize=16)
            plt.grid()

            # We plot rel_abunds in the lower row
            plt.subplot(2, 4, (i%4) + 4 + 1)
            plt.bar(XHbins[:-1], XH, width=(XHbins[-1] - XHbins[0]) / (np.size(XHbins) - 1))
            plt.xlabel(r'$[%s/H]_{\star int}$' % elements[X], fontsize=16)
            plt.ylabel(r'$Number\ of\ star\ particles$', fontsize=16)
            plt.grid()

            if (i + 1) % 4 == 0:
                plt.suptitle('Ages and [Fe/H] for subhaloes %s to %s' %(i//4 * 4 + 1, i + 1))
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig('/home/luis/Pictures/Tesis/gecko/ages_and_FeH_C01_%s.png' % (i//4))
                plt.figure(figsize=(19.2, 12.8))
        i += 1
        print(i)
    plt.suptitle('Ages and [Fe/H] for subhaloes %s to %s' %(i//9 * 9 + 1, i + 1))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('/home/luis/Pictures/Tesis/gecko/ages_and_FeH_C01_%s.png' % (i//9))
