import numpy as np
import matplotlib.pyplot as plt

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels
from iccpy.gadget.subfind import SubfindCatalogue
from iccpy.utils import match
from astropy.cosmology import FlatLambdaCDM


# path = '/media/luis/82A8355FA83552C1/CLUES_Gustavo/outputs'
path = '/home/luis/Documents/simus/Aq5/outputs'
snap_num = 127  # number of the snapshot to be loaded

snap = load_snapshot(directory = path, snapnum = snap_num, label_table = cecilia_labels)
# num = 0 # The number of the subhalo
cat = SubfindCatalogue(path, snap_num) # get a catalogue of subhaloes
hubble0 = snap.header.hubble0[0]
omega0 = snap.header.omega0[0]

# We choose an element to calculate [X/H]_t
elements = ['He', 'C', 'N', 'O', 'Fe', 'Mg', 'H', 'Si', 'Ba', 'Eu', 'Sr', 'Y']
X = elements.index(str(input('Choose an element from He, C, N, O, Fe, Mg, H, Si, Ba, Eu, Sr, Y:   ')))
Xmass = [4, 12, 14, 16, 56, 24.3, 1, 28, 137.3, 152, 87.6, 88.9][X]
Xsolar = [10.93, 8.39, 7.78, 8.66, 7.45, 7.53, 12, 7.51, 2.17, 0.52, 2.92, 2.21][X]


def most_massives(cat, n):
    """
    This gets the n most massive subhalos in the catalogue
    """
    import heapq
    n_most_massive = []
    masses = []
    for subh in cat.subhalo[:]:
        masses.append(subh.mass)
    thresh = np.min(heapq.nlargest(n, masses))
    for subh in cat.subhalo[:]:
        if subh.mass >= thresh:
            n_most_massive.append(subh)

    return n_most_massive


n_most_massive = most_massives(cat, 3)

for subh in n_most_massive:
    ind_stars = match(subh.ids, snap['ID  '][4])
    ind_stars = ind_stars[ind_stars != -1]

    # Here we turn the age of the stars from scale factor to Gyr
    cosmo = FlatLambdaCDM(H0=hubble0*100, Om0=omega0)
    age_stars = snap['AGE '][4][ind_stars]				 	# Age in scale factor
    age_stars = (1 - age_stars) / age_stars				 	# Age in redshift
    age_stars = np.array(cosmo.lookback_time(age_stars)) 	# Lookback time in Gyr
    time_stars = float(np.array(cosmo.lookback_time(9999))) - age_stars		# Time at which the star was born in Gyr

    bins = 200 # Number of bins for SFR
    mass_stars = snap['MASS'][4][ind_stars] * 1e10 / hubble0        # masses of the stars in the subhalo in solar masses

    # We calculate [X/H]
    H_stars = snap['Z   '][4][ind_stars][:, 6]
    X_stars = snap['Z   '][4][ind_stars][:, X]
    XH_stars = -Xsolar + 12 + np.log10(X_stars / Xmass / H_stars)

    SFR = np.histogram(time_stars, bins=bins, weights=mass_stars)
    # XH_t = np.histogram(age_stars[XH_stars > -10E5], bins=bins, weights=XH_stars[XH_stars > -10E5]) # we remove -inf


    fig = plt.figure()
    plt.semilogy(SFR[1][0:-1], SFR[0] / (13.8 / bins), 'r')		# Units are solar masses per Gyr
    plt.xlabel(r'$Time\ (Gyr)$', fontsize=24)
    plt.ylabel(r'$SFR\ (M_{\odot} Gyr^{-1})$', fontsize=24)
    plt.grid()

    plt.show()
    # plt.savefig('/home/luis/Pictures/Tesis/SFR_t_mainAq.png', transparent=True, bbox_inches='tight')


    """
    age_stars = age_stars[XH_stars > -5]
    mass_stars = mass_stars[XH_stars > -5]
    XH_stars = XH_stars[XH_stars > -5]

    hist = np.histogram2d(age_stars, XH_stars, bins = 500, weights=mass_stars)

    fig = plt.figure()
    plt.imshow(np.transpose(np.log(hist[0])), origin='lower', extent=(min(age_stars), max(age_stars), min(XH_stars), max(XH_stars)), aspect='auto')
    # plt.scatter(age_stars, XH_stars, s=.1)		# Units are solar masses per Gyr
    plt.xlabel(r'$Edad\ estelar\ (Gyr)$', fontsize=16)
    plt.ylabel(r'$[Fe/H]$', fontsize=16)
    # plt.ylabel('[%s/H]' %elements[X], fontsize=16)
    plt.grid()
    plt.savefig('/home/luis/Pictures/Tesis/Z_age.png', transparent=True, bbox_inches='tight')
    plt.show()
    """
