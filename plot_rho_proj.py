import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels
from iccpy.gadget.subfind import SubfindCatalogue
from iccpy.utils import match

from functions import PCA_matrix, Vcm


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




path = '/home/luis/Documents/gecko/gecko_C/run_01/outputs'

snap_num = 135  # number of the snapshot to be loaded
snap = load_snapshot(directory = path, snapnum = snap_num, label_table = cecilia_labels)
cat = SubfindCatalogue(path, snap_num, SAVE_MASSTAB=True) # get a catalogue of subhaloes
num = 2 #subhalo number (0 for main subhalo)
subh = cat.subhalo[num]

sigma = 1	# standard deviation for gaussian filter
resolution = 500
box_size = 40

V_res = 30     # The resolution for the velocity arrows
arrow_grid = np.arange(-box_size/2, box_size/2, box_size/V_res)

smooth = 'y'  #   input('Do you want to smooth? (y / n)		')
save = 'y'   #input('Do you want to save the projections? (y / n)		')
colormap = 'jet'


# We make a list for each grid containing the component and the 2 axis:
# gas_xy, gas_xz, gas_y_z, stars_xy, stars_xz, stars_yz
projections = [
		[0, 1, 0],
		[0, 2, 0],
		[0, 2, 1],
		[4, 1, 0],
		[4, 2, 0],
		[4, 2, 1],
]

axis_labels = [
			'x (kpc)',
			'y (kpc)',
			'z (kpc)',
]

fig, axes = plt.subplots(2, 3, dpi=300, figsize=(19.2, 12.8))
images = []

for proj, ax in zip(projections, np.concatenate(axes)):
 	# Mass projection in solar masses:
	rho = grid_maker(snap, subh, 'MASS', proj[0], proj[1], proj[2], box_size, resolution)
	# Turn into solar masses per kpc^2:
	rho = rho / (box_size**2 / resolution**2)
	if smooth == 'y':
		rho = gaussian_filter(rho, sigma)
	rho = np.log10(rho)
	# We fix null densities (-inf logarithms):
	rho[rho < -10000] = np.min(rho[rho > -10000])
	# We make the velocity arrows (X and Y stand for the axis1 and axis2 in this particular projection, not actually for x and y):
	X, Y = np.meshgrid(arrow_grid, arrow_grid)
	vel_stars_X = V_i_grid(snap, subh, proj[0], proj[1], proj[2], box_size, V_res, proj[2])
	vel_stars_Y = V_i_grid(snap, subh, proj[0], proj[1], proj[2], box_size, V_res, proj[1])

	vmax = np.max(rho)  # Stars reach higher densities
	vmin = vmax - 5
	vel_max = np.max(np.sqrt(vel_stars_X**2 + vel_stars_Y**2))


	ax.imshow(rho, cmap = colormap, extent=(-box_size/2, box_size/2, -box_size/2, box_size/2), vmin=vmin, vmax=vmax)
	ax.set_xlabel(axis_labels[proj[2]])
	ax.set_ylabel(axis_labels[proj[1]])
	ax.quiver(X, Y, vel_stars_X, vel_stars_Y, scale=5*box_size/V_res * vel_max)

# plt.tight_layout()

plt.show()


"""
if save == 'y':
	if smooth == 'y':
		plt.savefig('/home/luis/Pictures/Tesis/Aq_subh0_rho%s_smoothed.png' %component, bbox_inches='tight')
	else:
		plt.savefig('/home/luis/Pictures/Tesis/Aq_subh0_rho%s.png' %component, bbox_inches='tight')
"""
#plt.savefig('/home/luis/Pictures/Tesis/Aq_subh0_rho_both.png', bbox_inches='tight')

# plt.figure()

# plt.colorbar(im4, orientation='horizontal')
# plt.xlabel(r'$log(\rho)\ [M_{\odot}\ /\ kpc^2]$', fontsize=20)
# plt.tight_layout()

# plt.show()
