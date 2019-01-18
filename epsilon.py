import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
import os

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels
from iccpy.gadget.subfind import SubfindCatalogue
from iccpy.utils import match

from functions import PCA_matrix, Vcm, get_massive, grid_maker_SPH
from scipy.ndimage import gaussian_filter

import time
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
    # r = np.sqrt(pos[:, 0]**2, pos[:, 1]**2)

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
    ind_stars = match(subh.ids, snap['ID  '][4])
    ind_stars = ind_stars[ind_stars != -1]

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


if __name__ == "__main__":

    path = '/media/luis/82A8355FA83552C1/CLUES_Gustavo/outputs'

    layers = 5000
    rmax = 300

    snap_num = 127  # number of the snapshot to be loaded
    snap = load_snapshot(directory=path, snapnum=snap_num, label_table=cecilia_labels)
    cat = SubfindCatalogue(path, snap_num) # get a catalogue of subhaloes
    threshold_mass = 1e10
    massives = get_massive(snap, cat, threshold_mass)
    print('There are {} subhaloes with mass greater than {} solar masses'.format(np.size(massives), threshold_mass))
    hubble0 = snap.header.hubble0[0]
    resolution = 250 # for rho projection


    plt.figure(figsize=(19.2, 12.8))
    epsilons = [] # in each element of this list we will store a list of 2 elements containing the counts and bins for the epsilon histogram, respectively
    # (We do this to plot it all at the end of the subhalo loop)
    projs = [] # here we will store the rho grids in xy, xz, yz
    r_200s = []

    # We iterate for each subhalo
    i = 0
    for subh in massives:
        v_circ, radii, r_200 = v_circular(snap, subh, layers, rmax)
        r_200s.append(r_200)


        # This plots V_circ and saves the figures
        plt.subplot(3, 3, (i%9) + 1)
        plt.plot(radii, v_circ)
        plt.axvline(x=r_200, linestyle='--', color='r', label=r'$r_{200} = $%s kpc'%r_200)
        # plt.xlim(0, r_200 / 10)
        plt.grid()
        plt.xlabel(r'$r\ [kpc]$')
        plt.ylabel(r'$v_{circ}\ [km/s]$')
        plt.legend()
        # This will save the figure once 9 subplots are made and will create a new figure:
        if (i + 1) % 9 == 0:
            plt.suptitle(r'$v_{{circ}}$ for subhaloes {} to {}'.format(i//9 * 9 + 1, i + 1))
            plt.savefig('/home/luis/Pictures/Tesis/CLUES/v_circ_C01_{}.png'.format((i//9)))
            plt.close()
            plt.figure(figsize=(19.2, 12.8))


        # We compute epsilons here but will store them in a list to plot them in a separate loop to access figures easier
        eps, r, masses = epsilon(snap, subh, v_circ, radii)

        disc_mass_05 =      np.sum(masses[eps > .5])
        total_mass =        np.sum(masses)
        disc_mass_excess =  total_mass - 2 * np.sum(masses[eps < 0])
        DT_05 =     disc_mass_05 / total_mass
        DT_excess = disc_mass_excess / total_mass

        inner = True
        if inner:
            eps = eps[r < r_200 / 10]
            r = r[r < r_200 / 10]

        print(i)
        if np.sum(abs(eps)) > 0:
            hist_eps_r = np.histogram2d(eps, r, bins=100, range=[[-3, 3], [0, np.max(r)]])
        else:
            hist_eps_r = 0
        hist_eps_r = np.array(hist_eps_r)
        hist_eps_r[0] = np.log10(hist_eps_r[0])
        hist_eps_r[0][hist_eps_r[0] < -10000] = np.min(hist_eps_r[0][hist_eps_r[0] > -10000])
        eps, eps_bins = np.histogram(eps, bins=100, range=(-5, 5))


        epsilons.append([eps_bins, eps, hist_eps_r, DT_05, DT_excess])

        # We plot the density projections for the star component (actual plots are done outside this loop)
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

        projs.append([rho_xy, rho_xz, rho_yz])

        i += 1

    # Now we plot and save the epsilon histograms:
    fig, axes = plt.subplots(4, 4, figsize=(19.2, 12.8))


    for j in range(np.size(epsilons)):
        # We only plot if the subhalo has stars:
        if np.size(epsilons[j][2]) > 1:

            axes[0, j%4].imshow(epsilons[j][2][0], aspect='auto', extent=(0, epsilons[j][2][2][-1], -3, 3), origin='lower')
            axes[0, j%4].set_xlabel('r (kpc)')
            axes[0, j%4].set_ylabel(r'$\epsilon\ =\ j_z / j_{circ}$')
            axes[0, j%4].plot(epsilons[j][1] / np.max(epsilons[j][1]) * epsilons[j][2][2][-1], epsilons[j][0][:-1], color='white')
            axes[0, j%4].text(epsilons[j][2][2][-1] / 2, -2.8, r'$D/T_{>0.5} = %s$' % str(epsilons[j][3]), color='white')
            axes[0, j%4].text(epsilons[j][2][2][-1] / 2, -2.0, r'$D/T_{excess} = %s$' % str(epsilons[j][4]), color='white')
            axes[0, j%4].text(epsilons[j][2][2][-1] / 2, 2.3, r'$r_{200} = %s$' % str(r_200s[j]), color='white')
            axes[0, j%4].set_ylim((-3, 3))


            vmax = np.max(projs[j][0])
            vmin = vmax - 5
            axes[1, j%4].imshow(projs[j][0], extent=(-r_200s[j]/9, r_200s[j]/9, -r_200s[j]/9, r_200s[j]/9), vmin=vmin, vmax=vmax)
            circle = plt.Circle((0, 0), radius=r_200s[j] / 10, fill=False, color='white')
            axes[1, j%4].add_artist(circle)
            axes[1, j%4].set_xlabel('x (kpc)')
            axes[1, j%4].set_ylabel('y (kpc)')

            vmax = np.max(projs[j][1])
            vmin = vmax - 5
            axes[2, j%4].imshow(projs[j][1], extent=(-r_200s[j]/9, r_200s[j]/9, -r_200s[j]/9, r_200s[j]/9), vmin=vmin, vmax=vmax)
            circle = plt.Circle((0, 0), radius=r_200s[j] / 10, fill=False, color='white')
            axes[2, j%4].add_artist(circle)
            axes[2, j%4].set_xlabel('x (kpc)')
            axes[2, j%4].set_ylabel('z (kpc)')

            vmax = np.max(projs[j][2])
            vmin = vmax - 5
            axes[3, j%4].imshow(projs[j][2], extent=(-r_200s[j]/9, r_200s[j]/9, -r_200s[j]/9, r_200s[j]/9), vmin=vmin, vmax=vmax)
            circle = plt.Circle((0, 0), radius=r_200s[j] / 10, fill=False, color='white')
            axes[3, j%4].add_artist(circle)
            axes[3, j%4].set_xlabel('y (kpc)')
            axes[3, j%4].set_ylabel('z (kpc)')



            # plt.bar(epsilons[j][0][:-1], epsilons[j][1], width=.1)
            # plt.grid()
            # plt.xlabel(r'$\epsilon\ =\ j_z / j_{circ}$')
            # plt.ylabel('Counts')

            # This will save the figure once 9 subplots are made and will create a new figure:
            print(j)
            if (j + 1) % 4 == 0:
                plt.suptitle(r'$\epsilon\ =\ j_z / j_{{circ}}$ for subhaloes {} to {}'.format(j//4 * 4 + 1, j + 1))
                plt.savefig('/home/luis/Pictures/Tesis/CLUES/inner_epsilon_r_C01_{}.png'.format(j//4))
                plt.close()
                fig, axes = plt.subplots(4, 4, figsize=(19.2, 12.8))
    # And this will save the last one, given the amount of subh may not be a multiple of 9:
    # plt.suptitle(r'$\epsilon\ =\ j_z / j_{{circ}}$ for subhaloes %s to %s' %(j//4 * 4 + 1, j + 1))
    plt.savefig('/home/luis/Pictures/Tesis/CLUES/inner_epsilon_r_C01_{}.png'.format(j//4))
    # plt.close()
