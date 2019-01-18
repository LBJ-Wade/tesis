"""
Here we will
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 32})

import os
import time

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels
from iccpy.gadget.subfind import SubfindCatalogue
from iccpy.utils import match

from functions import PCA_matrix, Vcm, most_massives, grid_maker_SPH, get_inner_inds



def v_circular(snap, subh, layers, rmax):
    """
    This function returns the circular velocity profile and the corresponding radii.
    The profile goes from r = 0 up to r = r_200 for a given subhalo.


    snap:		GADGET snapshot
    subh:		Subhalo object
    layers:		number of radial bins
    rmax:		maximum radius in kpc
    """
    start=time.clock()

    CM = subh.pot_min
    # Now using CLUES, so unit is Mpc/h
    pos = (np.concatenate((snap['POS '][0], snap['POS '][4], snap['POS '][1])) - CM) * 1000 / hubble0

    rot_matrix = PCA_matrix(snap, subh)
    pos = np.dot(pos, rot_matrix)

    r = np.linalg.norm(pos, axis=1)

    ind = r < 200
    r = r[ind]

    masses = np.concatenate((snap['MASS'][0], snap['MASS'][4], snap['MASS'][1])) * 1e10 / hubble0
    masses = masses[ind]

    # We make a mass histogram with radial bins and calculate the inner mass at each radius:
    mass, radii = np.histogram(r, bins=layers, range=(0, rmax), weights=masses)
    inner_mass = np.cumsum(mass)

    # We calculate r_200:
    rho = inner_mass / (4/3 * np.pi * radii[1:]**3)
    rho_crit = 126.7 # solar masses per kpc^3, from Planck
    ind_200 = (np.abs(rho - 200*rho_crit)).argmin() # This gives the index of the bin where rho is closest to 200*rho_crit
    r_200 = radii[ind_200]
    print('r_200 = %s kpc' %r_200)

    # Finally, we calculate v_circ with the newtonian expression:
    G = 43007.1 # gravitational constant in code units
    v_circ = np.sqrt(G * 1e-10 * inner_mass / radii[1:]) # I use 1e-10 to turn mass back to code units, h in mass and radii cancel out.
    													# Velocity comes out in km/s

    print('v_circular takes %s seconds' %(time.clock() - start))
    return v_circ[0:ind_200], radii[1:ind_200+1], r_200


def epsilon(snap, subh, v_circ, radii):
    """
    Returns an array with all epsilons (j_z / j_circ)
    given an array of v_circ previously computed and
    the corresponding radii and mass of the particle
    """

    ind_stars = get_inner_inds(snap, subh, 4)

    # We will rotate positions and velocities so that z axis is aligned with the total angular momentum of the subhalo:
    rot_matrix = PCA_matrix(snap, subh)

    CM = subh.pot_min

    pos = snap['POS '][4][ind_stars] - CM
    pos = pos * 1000 / hubble0 # change units to kpc
    r = np.linalg.norm(pos, axis=1)
    # We will avoid particles too near to CoM because fof_group.pot_min is the position of the most bound particle
    ind_stars = ind_stars[r > .1]
    pos = (snap['POS '][4][ind_stars] - CM) * 1000 / hubble0
    r = np.linalg.norm(pos, axis=1)

    pos = np.dot(pos, rot_matrix)
    pos_x = pos[:,0]
    pos_y = pos[:,1]
    pos_z = pos[:,2]

    V_cm = Vcm(snap, subh)


    vel = snap['VEL '][4][ind_stars]
    vel = vel - V_cm
    vel = np.dot(vel, rot_matrix)
    vel_x = vel[:,0]
    vel_y = vel[:,1]
    vel_z = vel[:,2]

    masses = snap['MASS'][4][ind_stars] * 1e10 / hubble0

    jz = pos_x * vel_y - pos_y * vel_x # we don't multiply by masses because they will cancel out with j_circ's
    epsilon = np.zeros(np.size(ind_stars))
    # for each star, we round it's radius to the nearest in "radii" and calculate epsilon:
    i=0
    for radius in r:
        nearest_radius = np.abs(radii - radius).argmin()  	# returns the index of the closest radius in "radii"
        v_circ_star = v_circ[nearest_radius]			# this gives the corresponding v_circ to that star
        epsilon[i] = jz[i] / (radius * v_circ_star)
        i+=1

    # We use the radius in the disc plane:
    r = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)

    return epsilon, r, masses


path = '/media/luis/82A8355FA83552C1/CLUES_Gustavo'
layers = 5000
rmax = 300
resolution = 50 # for rho projection

# elements = ['He', 'C', 'N', 'O', 'Fe', 'Mg', 'H', 'Si', 'Ba', 'Eu', 'Sr', 'Y']
# X = elements.index(str(input('Choose an element from He, C, N, O, Fe, Mg, H, Si, Ba, Eu, Sr, Y:   ')))
# Xmass = [4, 12, 14, 16, 56, 24.3, 1, 28, 137.3, 152, 87.6, 88.9][X]
# Xsolar = [10.93, 8.39, 7.78, 8.66, 7.45, 7.53, 12, 7.51, 2.17, 0.52, 2.92, 2.21][X] # solar abundances relative to log(NH)=12


merg_tree_0 = path + '/postproc/Prog_0.dat'
merg_tree_0 = np.flip(np.loadtxt(merg_tree_0), axis=0)

merg_tree_33 = path + '/postproc/Prog_33.dat'
merg_tree_33 = np.flip(np.loadtxt(merg_tree_33), axis=0)


starting_snap = 127
# This loop visits each snapshot:
for j in range(starting_snap, 129):
    print('Analysing snapshot {}'.format(j))
    snap_num = j
    snap = load_snapshot(directory=path + '/outputs', snapnum=snap_num, label_table=cecilia_labels)
    cat = SubfindCatalogue(path + '/outputs', snap_num) # get a catalogue of subhaloes
    hubble0 = snap.header.hubble0[0]


    big_two = [cat.subhalo[int(merg_tree_0[j - 36][1])], cat.subhalo[int(merg_tree_33[j - 36][1])]]

    fig, axes = plt.subplots(2, 6, figsize=(38.4, 21.6))

    i=0 # i will count the row in which we plot the subhalo
    for subh in big_two:

        ######################################################################################################
        # Here we compute v_circ and plot it
        ######################################################################################################

        v_circ, radii, r_200 = v_circular(snap, subh, layers, rmax)
        axes[i, 0].plot(radii, v_circ)
        axes[i, 0].axvline(x=r_200, linestyle='--', color='r', label=r'$r_{{200}} = ${} kpc'.format(r_200))
        axes[i, 0].grid()
        axes[i, 0].set_xlabel(r'$r\ [kpc]$')
        axes[i, 0].set_ylabel(r'$v_{circ}\ [km/s]$')

        ######################################################################################################
        # Here we compute epsilon_r map and plot it
        ######################################################################################################

        eps, r, masses = epsilon(snap, subh, v_circ, radii)

        disc_mass_05 =      np.sum(masses[eps > .5])
        total_mass =        np.sum(masses)
        disc_mass_excess =  total_mass - 2 * np.sum(masses[eps < 0])
        DT_05 =     disc_mass_05 / total_mass
        DT_excess = disc_mass_excess / total_mass

        inner = False # we are now taking inner stars in the epsilon function, this is deprecated
        if inner:
            eps = eps[r < r_200 / 3]
            r = r[r < r_200 / 3]

        if np.sum(abs(eps)) > 0:
            hist_eps_r = np.histogram2d(eps, r, bins=100, range=[[-3, 3], [0, np.max(r)]])
        else:
            hist_eps_r = 0
        hist_eps_r = np.array(hist_eps_r)
        hist_eps_r[0] = np.log10(hist_eps_r[0])
        hist_eps_r[0][hist_eps_r[0] < -10000] = np.min(hist_eps_r[0][hist_eps_r[0] > -10000])
        eps, eps_bins = np.histogram(eps, bins=100, range=(-5, 5))

        axes[i, 1].imshow(hist_eps_r[0], aspect='auto', extent=(0, hist_eps_r[2][-1], -3, 3), origin='lower')
        axes[i, 1].set_xlabel('r (kpc)')
        axes[i, 1].set_ylabel(r'$\epsilon\ =\ j_z / j_{circ}$')
        axes[i, 1].plot(eps / np.max(eps) * hist_eps_r[2][-1], eps_bins[:-1], color='white')
        axes[i, 1].text(hist_eps_r[2][-1] / 3, -2.8, r'$D/T_{{>.5}} = {}$'.format(DT_05), color='white')
        axes[i, 1].text(hist_eps_r[2][-1] / 3, -2.0, r'$D/T_{{XS}} = {}$'.format(DT_excess), color='white')
        axes[i, 1].text(hist_eps_r[2][-1] / 2, 2.3, r'$r_{{200}} = {}$'.format(r_200), color='white')
        axes[i, 1].set_ylim((-3, 3))

        ######################################################################################################
        # Here we make the rho projections and plot them
        ######################################################################################################

        box_size = r_200 / 4.5

        rho_xy = grid_maker_SPH(snap, subh, 'MASS', 4, 1, 0, box_size/1000, resolution, False) / (box_size**2 / resolution**2)
        rho_xy = gaussian_filter(rho_xy, 1)
        rho_xy = np.log10(rho_xy)
        rho_xy[rho_xy < -10000] = np.min(rho_xy[rho_xy > -10000])

        rho_xz = grid_maker_SPH(snap, subh, 'MASS', 4, 2, 0, box_size/1000, resolution, False) / (box_size**2 / resolution**2)
        rho_xz = gaussian_filter(rho_xz, 1)
        rho_xz = np.log10(rho_xz)
        rho_xz[rho_xz < -10000] = np.min(rho_xz[rho_xz > -10000])

        rho_yz = grid_maker_SPH(snap, subh, 'MASS', 4, 2, 1, box_size/1000, resolution, False) / (box_size**2 / resolution**2)
        rho_yz = gaussian_filter(rho_yz, 1)
        rho_yz = np.log10(rho_yz)
        rho_yz[rho_yz < -10000] = np.min(rho_yz[rho_yz > -10000])

        vmax = np.max(rho_xy)
        vmin = vmax - 5
        axes[i, 2].imshow(rho_xy, extent=(-r_200/9, r_200/9, -r_200/9, r_200/9), vmin=vmin, vmax=vmax)
        circle = plt.Circle((0, 0), radius=r_200 / 10, fill=False, color='white')
        axes[i, 2].add_artist(circle)
        axes[i, 2].set_xlabel('x (kpc)')
        axes[i, 2].set_ylabel('y (kpc)')

        vmax = np.max(rho_xz)
        vmin = vmax - 5
        axes[i, 3].imshow(rho_xz, extent=(-r_200/9, r_200/9, -r_200/9, r_200/9), vmin=vmin, vmax=vmax)
        circle = plt.Circle((0, 0), radius=r_200 / 10, fill=False, color='white')
        axes[i, 3].add_artist(circle)
        axes[i, 3].set_xlabel('x (kpc)')
        axes[i, 3].set_ylabel('z (kpc)')

        vmax = np.max(rho_yz)
        vmin = vmax - 5
        axes[i, 4].imshow(rho_yz, extent=(-r_200/9, r_200/9, -r_200/9, r_200/9), vmin=vmin, vmax=vmax)
        circle = plt.Circle((0, 0), radius=r_200 / 10, fill=False, color='white')
        axes[i, 4].add_artist(circle)
        axes[i, 4].set_xlabel('y (kpc)')
        axes[i, 4].set_ylabel('z (kpc)')

        ######################################################################################################
        # Here we calculate the abundances and plot them
        ######################################################################################################

        ind_gas = np.concatenate([match(subh.ids, snap['ID  '][0]), match(subh.ids + 2**31, snap['ID  '][0])])
        ind_gas = ind_gas[ind_gas != -1]
        ind_stars = get_inner_inds(snap, subh, 4)

        # These are the masses of each element for each particle:
        Fe_gas = snap['Z   '][0][ind_gas][:, 4]
        H_gas = snap['Z   '][0][ind_gas][:, 6]
        # X_gas = snap['Z   '][0][ind_gas][:, X]
        Fe_stars = snap['Z   '][4][ind_stars][:, 4]
        H_stars = snap['Z   '][4][ind_stars][:, 6]
        # X_stars = snap['Z   '][4][ind_stars][:, X]

        # And these are the relative abundances:
        FeH_gas = 4.55 + np.log10(Fe_gas / 56 / H_gas)    			# [Fe/H]_solar = 7.45 - 12
        FeH_stars = 4.55 + np.log10(Fe_stars / 56 / H_stars)
        # XFe_gas = - Xsolar + 7.45 + np.log10(X_gas / Xmass / Fe_gas * 56)   	# [X/Fe]_solar = [X] - 7.45
        # XFe_stars = - Xsolar + 7.45 + np.log10(X_stars / Xmass / Fe_stars * 56)

        hist, bins = np.histogram(FeH_stars, bins=15, range=(-5,1), density=True)
        normed = hist / hist.sum() # we normalize so the bars add to 1
        widths = bins[:-1] - bins[1:]
        axes[i, 5].bar(bins[1:], normed, width=widths)
        axes[i, 5].set_xlabel(r'$[Fe / H]_{{\star}}$')
        axes[i, 5].set_ylabel(r'$Fracción$')

        i += 1

    plt.tight_layout(pad=.5, w_pad=0, h_pad=0)
    plt.savefig('/home/luis/Pictures/Tesis/CLUES/two_main/two_main_{:03d}.png'.format(j))
    plt.close()
