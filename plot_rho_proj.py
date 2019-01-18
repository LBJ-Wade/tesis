import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels
from iccpy.gadget.subfind import SubfindCatalogue
from iccpy.utils import match

from functions import PCA_matrix, Vcm, grid_maker_SPH, V_i_grid, get_massive


# path = '/home/luis/Documents/simus/Aq5/outputs'
path = '/media/luis/82A8355FA83552C1/CLUES_Gustavo/outputs'

snap_num = 76  # number of the snapshot to be loaded
snap = load_snapshot(directory = path, snapnum = snap_num, label_table = cecilia_labels)
cat = SubfindCatalogue(path, snap_num, SAVE_MASSTAB=True) # get a catalogue of subhaloes
num = 0 #subhalo number (0 for main subhalo)
# massives = get_massive(snap, cat, 1e11)
# subh = massives[num]
subh = cat.subhalo[num]

sigma = 1	# standard deviation for gaussian filter
resolution = 50
box_size = .050

V_res = 30     # The resolution for the velocity arrows
arrow_grid = np.arange(-box_size/2, box_size/2, box_size/V_res)

smooth = 'n'  #   input('Do you want to smooth? (y / n)		')
save = 'n'   #input('Do you want to save the projections? (y / n)		')
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
    subfind = False
    rho = grid_maker_SPH(snap, subh, 'MASS', proj[0], proj[1], proj[2], box_size, resolution, subfind)
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
    # vmin = vmax - 5
    vel_max = np.max(np.sqrt(vel_stars_X**2 + vel_stars_Y**2))


    ax.imshow(rho, cmap = colormap, extent=(-box_size/2, box_size/2, -box_size/2, box_size/2), vmax=vmax)
    ax.set_xlabel(axis_labels[proj[2]])
    ax.set_ylabel(axis_labels[proj[1]])
    # ax.quiver(X, Y, vel_stars_X, vel_stars_Y, scale=5*box_size/V_res * vel_max)

plt.tight_layout()

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
