import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels
from iccpy.gadget.subfind import SubfindCatalogue
from iccpy.utils import match



def grid_maker(snap, quantity, component, axis1, axis2, res):	
	positions = snap['POS '][component] / .73
	pos_1 = positions[:, axis1]
	pos_2 = positions[:, axis2]
	# axis3 = 3 - axis2 - axis1
	# pos_3 = (snap['POS '][component][index] - CM)[axis3] * 1000 / .73
	magnitude = snap[quantity][component] * 1e10 / .73  # cambio de unidades para masa
	hist = np.histogram2d(pos_1, pos_2, bins=res, weights=magnitude)
	return hist

sim = 'gecko/gecko_C'
path = '/home/luis/Documents/%s/run_01/outputs' %sim

snap_num = 135  # number of the load_snapshot to be loaded
snap = load_snapshot(directory = path, snapnum = snap_num, label_table = cecilia_labels)
cat = SubfindCatalogue(path, snap_num)

sigma = 1	# standard deviation for gaussian filter
resolution = 5000

smooth = 'n'  #   input('Do you want to smooth? (y / n)		')
save = 'y'   #input('Do you want to save the projections? (y / n)		')

# Mass projection in solar masses:
grid_xy_gas, x_gas, y_gas = grid_maker(snap,'MASS', 0, 1, 0, resolution)
# grid_xz_gas, x_gas, z_gas = grid_maker(snap,'MASS', 0, 2, 0, resolution)
# grid_yz_gas, y_gas, z_gas = grid_maker(snap,'MASS', 0, 2, 1, resolution)
# grid_xy_stars, x_stars, y_stars = grid_maker(snap,'MASS', 4, 1, 0, resolution)
# grid_xz_stars, x_stars, z_stars = grid_maker(snap,'MASS', 4, 2, 0, resolution)
# grid_yz_stars, y_stars, z_stars = grid_maker(snap,'MASS', 4, 2, 1, resolution)

# Turn into solar masses per kpc^2:
# ???????????


if smooth == 'y':
	grid_xy_gas = gaussian_filter(grid_xy_gas, sigma)
	grid_xz_gas = gaussian_filter(grid_xz_gas, sigma)
	grid_yz_gas = gaussian_filter(grid_yz_gas, sigma)
	grid_xy_stars = gaussian_filter(grid_xy_stars, sigma)
	grid_xz_stars = gaussian_filter(grid_xz_stars, sigma)
	grid_yz_stars = gaussian_filter(grid_yz_stars, sigma)



vmax = np.max(np.log10(grid_xy_stars))  # Stars reach higher densities
vmin = vmax - 5
colormap = 'jet'
extent_gas_xy = (np.min(x_gas), np.max(x_gas), np.min(y_gas), np.max(y_gas))
extent_gas_xz = (np.min(x_gas), np.max(x_gas), np.min(z_gas), np.max(z_gas))
extent_gas_yz = (np.min(y_gas), np.max(y_gas), np.min(z_gas), np.max(z_gas))
extent_stars_xy = (np.min(x_stars), np.max(x_stars), np.min(y_stars), np.max(y_stars))
extent_stars_xz = (np.min(x_stars), np.max(x_stars), np.min(z_stars), np.max(z_stars))
extent_stars_yz = (np.min(y_stars), np.max(y_stars), np.min(z_stars), np.max(z_stars))

grid_xy_gas = np.log10(grid_xy_gas)
grid_xz_gas = np.log10(grid_xz_gas)
grid_yz_gas = np.log10(grid_yz_gas)
grid_xy_stars = np.log10(grid_xy_stars)
grid_xz_stars = np.log10(grid_xz_stars)
grid_yz_stars = np.log10(grid_yz_stars)

# we fix null densities
grid_xy_gas[grid_xy_gas < -10000] = 0
grid_xz_gas[grid_xz_gas < -10000] = 0
grid_yz_gas[grid_yz_gas < -10000] = 0
grid_xy_stars[grid_xy_stars < -10000] = 0
grid_xz_stars[grid_xz_stars < -10000] = 0
grid_yz_stars[grid_yz_stars < -10000] = 0

"""
fig, axes = plt.subplots(2, 3, dpi=300)

im1 = axes[0,0].imshow(grid_xy_gas, cmap = colormap)#, vmin=vmin, vmax=vmax)
axes[0,0].set_xlabel('x (kpc)')
axes[0,0].set_ylabel('y (kpc)')

im2 = axes[0,1].imshow(grid_xz_gas, cmap = colormap)#, vmin=vmin, vmax=vmax)
axes[0,1].set_xlabel('x (kpc)')
axes[0,1].set_ylabel('z (kpc)')

im3 = axes[0,2].imshow(grid_yz_gas, cmap = colormap)#, vmin=vmin, vmax=vmax)
axes[0,2].set_xlabel('y (kpc)')
axes[0,2].set_ylabel('z (kpc)')

im4 = axes[1,0].imshow(grid_xy_stars, cmap = colormap)#, vmin=vmin, vmax=vmax)
axes[1,0].set_xlabel('x (kpc)')
axes[1,0].set_ylabel('y (kpc)')

im5 = axes[1,1].imshow(grid_xz_stars, cmap = colormap)#, vmin=vmin, vmax=vmax)
axes[1,1].set_xlabel('x (kpc)')
axes[1,1].set_ylabel('z (kpc)')

im6 = axes[1,2].imshow(grid_yz_stars, cmap = colormap)#, vmin=vmin, vmax=vmax)
axes[1,2].set_xlabel('y (kpc)')
axes[1,2].set_ylabel('z (kpc)')

plt.tight_layout()
"""

fig, ax = plt.subplots()
im1 = ax.imshow(grid_xy_gas, extent = extent_gas_xy, origin='lower', cmap = colormap)#, vmin=vmin, vmax=vmax)
for i in range(10):
	CM = (cat.fof_group[i].pot_min[0] / .73, cat.fof_group[i].pot_min[1] / .73)
	r = cat.fof_group[i].radius_crit200 / .73
	circle = plt.Circle(CM, r, fill=False)
	ax.add_artist(circle)
ax.set_xlabel('x (kpc)')
ax.set_ylabel('y (kpc)')





# plt.colorbar(im4, orientation='horizontal')
# plt.xlabel(r'$log(\rho)\ [M_{\odot}\ /\ kpc^2]$', fontsize=20)
# plt.tight_layout()
plt.show()