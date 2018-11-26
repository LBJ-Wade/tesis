import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels
from iccpy.gadget.subfind import SubfindCatalogue
from iccpy.utils import match

import time



def M_vir_crit(snap, subh, layers, rmax):
	"""
	Returns M_vir, Mstar_vir, r_vir
	(r_vir is r_crit200)
	"""
	ind_gas = np.concatenate([match(subh.ids, snap['ID  '][0]), match(subh.ids + 2**31, snap['ID  '][0])])
	ind_gas = ind_gas[ind_gas != -1]
	ind_stars = match(subh.ids, snap['ID  '][4])
	ind_stars = ind_stars[ind_stars != -1]
	ind_DM = match(subh.ids, snap['ID  '][1])
	ind_DM = ind_DM[ind_DM != -1]

	CM = subh.com

	pos = np.concatenate((snap['POS '][0][ind_gas] - CM, snap['POS '][4][ind_stars] - CM, snap['POS '][1][ind_DM] - CM))
	pos = pos / .73
	pos_x = pos[:,0]
	pos_y = pos[:,1]
	pos_z = pos[:,2]
	r = np.sqrt(pos_x**2 + pos_y**2 + pos_z**2)


	masses = np.concatenate((snap['MASS'][0][ind_gas], snap['MASS'][4][ind_stars], snap['MASS'][1][ind_DM])) * 1e10 / .73

	# We make a mass histogram with radial bins:
	mass, radius = np.histogram(r, bins=layers, range=(0, rmax), weights=masses)

	inner_mass = np.cumsum(mass)
	rho = inner_mass / (4/3 * np.pi * radius[1:]**3)
	rho_crit = 126.7 # solar masses per kpc^3, from Planck

	ind_200 = (np.abs(rho - 200*rho_crit)).argmin() # This gives the index of the bin where rho is closest to 200*rho_crit
	r_vir = radius[ind_200]
	print(r_vir)
	M_vir = np.sum(masses[r < r_vir])

	pos = snap['POS '][4][ind_stars] - CM
	pos_x = pos[:,0]
	pos_y = pos[:,1]
	pos_z = pos[:,2]
	r = np.sqrt(pos_x**2 + pos_y**2 + pos_z**2)

	star_mass = snap['MASS'][4][ind_stars] * 1e10 / .73
	Mstar_vir = np.sum(star_mass[r < r_vir])

	return M_vir, Mstar_vir, r_vir



def get_massive(snap, cat, M):
	"""
	Returns a list with subhalo objects with more mass than M (given in solar masses)
	TAKES TOO MUCH TIME TO RUN, BUT IF WE TAKE SUBFIND MASS IT'S TOO BIG
	MAYBE FILTER OUT LOW TOTAL MASSES BEFORE FILTERING BY VIRIAL MASS????
	"""
	massives = []
	i = 0
	for subh in cat.subhalo[:]:
		mass = subh.mass
		if mass > M * .73 / 1e10:
			massives.append(subh)
		i += 1
	return massives


def PCA_matrix(snap, subh):
	CM = subh.pot_min
	ind = match(subh.ids, snap['ID  '][4])
	ind = ind[ind != -1]

	pos = snap['POS '][4][ind] - CM
	# We calculate covariance matrix and diagonalize it. The eigenvectors are the galaxy's principal axes
	covMatrix = np.cov(np.transpose(pos))
	eigenval, eigenvect = np.linalg.eig(covMatrix)

	# eigenvalues are not ordered; we make it so rot_matrix has eigenvectors as columns ordered from highest eigenvalue to lowest:
	eig1 = eigenval.argmax()
	eig3 = eigenval.argmin()
	eig2 = 3 - eig1 - eig3

	rot_matrix = np.array([eigenvect[:, eig1], eigenvect[:, eig2], eigenvect[:, eig3]])
	rot_matrix = np.transpose(rot_matrix)

	# Now we check if the total angular momentum is antiparallel to z; if it is we flip the galaxy
	vel = snap['VEL '][4][ind]
	V_cm = Vcm(snap, subh)
	vel = vel - V_cm
	vel = np.dot(vel, rot_matrix)
	pos = np.dot(pos, rot_matrix)

	pos_x = pos[:, 0]
	pos_y = pos[:, 1]
	vel_x = vel[:, 0]
	vel_y = vel[:, 1]

	jz = pos_x * vel_y - pos_y * vel_x

	if np.sum(jz) < 0:
		# We invert first and last row (x and z) from the rot_matrix which is equivalent to rotating around the y axis
		rot_matrix[:, 0] = - rot_matrix[:, 0]
		rot_matrix[:, 2] = - rot_matrix[:, 2]

	return rot_matrix


def Vcm(snap, subh):
	"""
	Computes the Vcm using only star particles
	"""
	ind = match(subh.ids, snap['ID  '][4])
	ind = ind[ind != -1]
	vel = snap['VEL '][4][ind]
	masses = snap['MASS'][4][ind]
	masses_reshaped = np.transpose(np.array([masses, masses, masses]))

	V_cm = np.sum(vel * masses_reshaped, axis=0) / np.sum(masses)

	return V_cm



def grid_maker(snap, subh, quantity, component, axis1, axis2, length, res):
	"""
	Returns a res*res 2darray with the projected quantity (e.g. 'MASS') for the desired component (0 for gas, 1 for DM, 4 for stars)
	"""
	if component == 0:
		ind = np.concatenate([match(subh.ids, snap['ID  '][0]), match(subh.ids + 2**31, snap['ID  '][0])])
	else:
		ind = match(subh.ids, snap['ID  '][component])
	ind = ind[ind != -1]

	CM = subh.com

	# We rotate the positions so that the galactic angular momentum is parallel to the z axis:
	positions = (snap['POS '][component][ind] - CM) / .73

	rot_matrix = PCA_matrix(snap, subh)
	positions = np.dot(positions, rot_matrix)

	pos_1 = positions[:, axis1]
	pos_2 = positions[:, axis2]
	# axis3 = 3 - axis2 - axis1
	# pos_3 = (snap['POS '][component][index] - CM)[axis3] * 1000 / .73
	magnitude = snap[quantity][component][ind] * 1e10 / .73  # cambio de unidades para masa
	hist = np.histogram2d(pos_1, pos_2, bins=res, range=[[-length/2, length/2], [-length/2, length/2]], weights=magnitude)
	return hist[0]

def V_i_grid(snap, subh, component, axis1, axis2, length, res, i):
	"""
	Returns a res*res 2darray with the mean velocity in the i direction (0 is x, 1 is y and 2 is z) for the desired matter component (0 for gas, 1 for DM, 4 for stars)
	"""
	if component == 0:
		ind = np.concatenate([match(subh.ids, snap['ID  '][0]), match(subh.ids + 2**31, snap['ID  '][0])])
	else:
		ind = match(subh.ids, snap['ID  '][component])
	ind = ind[ind != -1]

	CM = subh.com

	# We rotate the positions so that the galactic angular momentum is parallel to the z axis:
	positions = (snap['POS '][component][ind] - CM) / .73

	rot_matrix = PCA_matrix(snap, subh)
	positions = np.dot(positions, rot_matrix)

	pos_1 = positions[:, axis1]
	pos_2 = positions[:, axis2]
	# We rotate the velocities:
	vel = snap['VEL '][component][ind] - Vcm(snap, subh)
	vel = np.dot(vel, rot_matrix)
	# and take the i-th component:
	v_i = vel[:, i]
	hist = np.histogram2d(pos_1, pos_2, bins=res, range=[[-length/2, length/2], [-length/2, length/2]], weights=v_i)
	return hist[0]
