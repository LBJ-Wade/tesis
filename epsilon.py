import numpy as np
import matplotlib.pyplot as plt
import os

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels
from iccpy.gadget.subfind import SubfindCatalogue
from iccpy.utils import match

from functions import PCA_matrix, Vcm, get_massive, grid_maker
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
	ind_gas = np.concatenate([match(subh.ids, snap['ID  '][0]), match(subh.ids + 2**31, snap['ID  '][0])])
	ind_gas = ind_gas[ind_gas != -1]
	ind_stars = match(subh.ids, snap['ID  '][4])
	ind_stars = ind_stars[ind_stars != -1]
	ind_DM = match(subh.ids, snap['ID  '][1])
	ind_DM = ind_DM[ind_DM != -1]

	CM = subh.com

	pos = np.concatenate((snap['POS '][0][ind_gas] - CM, snap['POS '][4][ind_stars] - CM, snap['POS '][1][ind_DM] - CM))
	pos = pos / .73 # change units to kpc
	pos_x = pos[:,0]
	pos_y = pos[:,1]
	pos_z = pos[:,2]
	r = np.sqrt(pos_x**2 + pos_y**2 + pos_z**2)


	masses = np.concatenate((snap['MASS'][0][ind_gas], snap['MASS'][4][ind_stars], snap['MASS'][1][ind_DM])) * 1e10 / .73

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
	the corresponding radii
	"""
	ind_stars = match(subh.ids, snap['ID  '][4])
	ind_stars = ind_stars[ind_stars != -1]

	# We will rotate positions and velocities so that z axis is aligned with the total angular momentum of the subhalo:
	rot_matrix = PCA_matrix(snap, subh)

	CM = subh.com

	pos = snap['POS '][4][ind_stars] - CM
	pos = pos / .73 # change units to kpc
	pos = np.dot(pos, rot_matrix)
	pos_x = pos[:,0]
	pos_y = pos[:,1]
	pos_z = pos[:,2]
	r = np.sqrt(pos_x**2 + pos_y**2 + pos_z**2)

	V_cm = Vcm(snap, subh)


	vel = snap['VEL '][4][ind_stars]
	vel = vel - V_cm
	vel = np.dot(vel, rot_matrix)
	vel_x = vel[:,0]
	vel_y = vel[:,1]
	vel_z = vel[:,2]

	masses = snap['MASS'][4][ind_stars] * 1e10 / .73

	jz = pos_x * vel_y - pos_y * vel_x # we don't multiply by masses because they will cancel out with j_circ's
	epsilon = np.zeros(np.size(ind_stars))
	# for each star, we round it's radius to the nearest in "radii" and calculate epsilon:
	i=0
	for radius in r:
		nearest_radius = np.abs(radii - radius).argmin()  	# returns the index of the closest radius in "radii"
		v_circ_star = v_circ[nearest_radius]			# this gives the corresponding v_circ to that star
		epsilon[i] = jz[i] / (radii[nearest_radius] * v_circ_star)
		i+=1


	return epsilon, r


sim = 'gecko/gecko_C'
path = '/home/luis/Documents/%s/' %sim
RUNS = os.listdir(path)


layers = 5000
rmax = 200


for RUN in RUNS:
	snap_num = 135  # number of the snapshot to be loaded
	snap = load_snapshot(directory = path + RUN + '/outputs', snapnum = snap_num, label_table = cecilia_labels)
	cat = SubfindCatalogue(path + RUN + '/outputs', snap_num) # get a catalogue of subhaloes


	# We iterate for each subhalo
	plt.rcParams.update({'font.size': 16})
	plt.figure(figsize=(19.2, 12.8))
	epsilons = [] # in each element of this list we will store a list of 2 elements containing the counts and bins for the epsilon histogram, respectively
					# (We do this to plot it all at the end of the subhalo loop)


	resolution = 500
	subhaloes = get_massive(snap, cat, 1e10)
	i = 0
	for subh in subhaloes:
		v_circ, radii, r_200 = v_circular(snap, subh, layers, rmax)

		# plt.subplot(3, 3, (i%9) + 1)
		# plt.plot(radii, v_circ)
		# plt.axvline(x=r_200, linestyle='--', color='r', label=r'$r_{200} = $%s kpc'%r_200)
		# plt.grid()
		# plt.xlabel(r'$r\ [kpc]$')
		# plt.ylabel(r'$v_{circ}\ [km/s]$')
		# plt.legend()
		# This will save the figure once 9 subplots are made and will create a new figure:
		# if (i + 1) % 9 == 0:
		# 	plt.suptitle(r'$v_{circ}$ for subhaloes %s to %s' %(i//9 * 9 + 1, i + 1))
		# 	plt.savefig('/home/luis/Pictures/Tesis/gecko/v_circ_C01_%s.png' % (i//9))
		# 	plt.figure(figsize=(19.2, 12.8))

		# We compute epsilons here but will store them in a list to plot them in a separate loop to access figures easier
		eps, r = epsilon(snap, subh, v_circ, radii)

		inner = True
		if inner:
			eps = eps[r < r_200 / 3]
			r = r[r < r_200 / 3]

		print(i)
		if np.sum(abs(eps)) > 0:
			hist_eps_r = np.histogram2d(eps, r, bins=100, range=[[-3, 3], [0, np.max(r)]])
		else:
			hist_eps_r = 0
		eps, eps_bins = np.histogram(eps[(eps>-10000) & (eps<10000)], bins=100, range=(-5, 5))

		epsilons.append([eps_bins, eps, hist_eps_r])

		# We plot the edge-on density projections for the star component
		box_size = r_200
		rho = grid_maker(snap, subh, 'MASS', 4, 2, 0, box_size, resolution)
		rho = rho / (box_size**2 / resolution**2)
		rho = gaussian_filter(rho, 1)
		rho = np.log10(rho)
		rho[rho < -10000] = np.min(rho[rho > -10000])
		vmax = np.max(rho)  # Stars reach higher densities
		vmin = vmax - 5
		plt.subplot(3, 3, (i%9) + 1)
		plt.imshow(rho, extent=(-box_size/2, box_size/2, -box_size/2, box_size/2), vmin=vmin, vmax=vmax)
		plt.xlabel('x (kpc)')
		plt.ylabel('z (kpc)')
		# This will save the figure once 9 subplots are made and will create a new figure:
		if (i + 1) % 9 == 0:
			plt.suptitle(r'$\rho_{proj}$ for subhaloes %s to %s' %(i//9 * 9 + 1, i + 1))
			plt.savefig('/home/luis/Pictures/Tesis/gecko/xz_rho_proj_C01_%s.png' % (i//9))
			plt.figure(figsize=(19.2, 12.8))

		i += 1
	plt.suptitle(r'$\rho_{proj}$ for subhaloes %s to %s' %(i//9 * 9 + 1, i + 1))
	plt.savefig('/home/luis/Pictures/Tesis/gecko/xz_rho_proj_C01_%s.png' % (i//9))



	# And this will save the last one, given the amount of subh may not be a multiple of 9:
	# plt.suptitle(r'$v_{circ}$ for subhaloes %s to %s' %(i//9 * 9 + 1, i + 1))
	# plt.savefig('/home/luis/Pictures/Tesis/gecko/v_circ_C01_%s.png' % (i//9))
"""
	# Now we plot and save the epsilon histograms:
	plt.figure(figsize=(19.2, 12.8))


	for j in range(np.size(epsilons)):
		plt.subplot(3, 3, (j%9) + 1)

		if np.size(epsilons[j][2]) > 1:
			plt.imshow(epsilons[j][2][0], aspect='auto', extent=(0, epsilons[j][2][2][-1], -3, 3), origin='lower')
			plt.xlabel('r (kpc)')
			plt.ylabel(r'$\epsilon\ =\ j_z / j_{circ}$')
			plt.plot(epsilons[j][1] / np.max(epsilons[j][1]) * epsilons[j][2][2][-1], epsilons[j][0][:-1], color='white')
			plt.ylim((-3, 3))

		# plt.bar(epsilons[j][0][:-1], epsilons[j][1], width=.1)
		# plt.grid()
		# plt.xlabel(r'$\epsilon\ =\ j_z / j_{circ}$')
		# plt.ylabel('Counts')

		# This will save the figure once 9 subplots are made and will create a new figure:
		if (j + 1) % 9 == 0:
			plt.suptitle(r'$\epsilon\ =\ j_z / j_{circ}$ for subhaloes %s to %s' %(j//9 * 9 + 1, j + 1))
			plt.savefig('/home/luis/Pictures/Tesis/gecko/inner_epsilon_r_C01_%s.png' % (j//9))
			plt.figure(figsize=(19.2, 12.8))
	# And this will save the last one, given the amount of subh may not be a multiple of 9:
	plt.suptitle(r'$v_{circ}$ for subhaloes %s to %s' %(j//9 * 9 + 1, j + 1))
	plt.savefig('/home/luis/Pictures/Tesis/gecko/inner_epsilon_r_C01_%s.png' % (j//9))
"""
