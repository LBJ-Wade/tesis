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
(grid_xy_gas, x_gas, y_gas) = grid_maker(snap,'MASS', 0, 0, 1, resolution)

if smooth == 'y':
	grid_xy_gas = gaussian_filter(grid_xy_gas, sigma)


vmax = np.max(np.log10(grid_xy_gas))  # Stars reach higher densities
vmin = vmax - 5
colormap = 'jet'
extent_gas_xy = (np.min(x_gas), np.max(x_gas), np.min(y_gas), np.max(y_gas))

grid_xy_gas = np.log10(grid_xy_gas)

# we fix null densities
grid_xy_gas[grid_xy_gas < -10000] = 0


fig, ax = plt.subplots()
im1 = ax.imshow(np.transpose(grid_xy_gas), extent = extent_gas_xy, origin='lower', cmap = colormap)#, vmin=vmin, vmax=vmax)
for i in range(10):
	CM = (cat.fof_group[i].pot_min[0] / .73, cat.fof_group[i].pot_min[1] / .73)
	r = cat.fof_group[i].radius_crit200 / .73
	circle = plt.Circle(CM, r, fill=False)
	plt.text(CM[0] + r, CM[1] + r, "n = %s" %i)
	ax.add_artist(circle)
ax.set_xlabel('x (kpc)')
ax.set_ylabel('y (kpc)')

plt.show()
