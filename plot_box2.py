import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels
from iccpy.gadget.subfind import SubfindCatalogue
from iccpy.utils import match

from functions import most_massives, grid_maker_SPH

import pickle

def grid_maker(snap, quantity, component, axis1, axis2, res):
    hubble0 = snap.header.hubble0[0]
    positions = snap['POS '][component] / hubble0
    pos_1 = positions[:, axis1]
    pos_2 = positions[:, axis2]
    # axis3 = 3 - axis2 - axis1
    # pos_3 = (snap['POS '][component][index] - CM)[axis3] * 1000 / .73
    magnitude = snap[quantity][component] * 1e10 / hubble0  # cambio de unidades para masa
    hist = np.histogram2d(pos_1, pos_2, bins=res, weights=magnitude)
    return hist


def grid_maker_SPH(snap, quantity, component, axis1, axis2, res):

    positions = (snap['POS '][component] - 50) / hubble0
    length = 10

    # max(np.max(positions[:,0] - np.min(positions[:, 0])),
    #             np.max(positions[:,1] - np.min(positions[:, 1])),
    #             np.max(positions[:,2] - np.min(positions[:, 2])))

    # We exclude all the particles laying outside the box:
    ind = np.all([[abs(positions[:, 0]) < length/2], [abs(positions[:, 1]) < length/2], [abs(positions[:, 2]) < length/2]], axis=0)[0]
    positions = positions[ind]

    if component == 0:
        hsml = snap['HSML'][0][ind] / hubble0
    else:
        hsml = []
        nbrs = NearestNeighbors(n_neighbors=32, algorithm='auto').fit(positions)
        distances = nbrs.kneighbors(positions)[0]
        for d in distances:
            hsml.append(d.mean())
        hsml = np.array(hsml)

    magnitude = snap[quantity][component] * 1e10 / hubble0  # cambio de unidades para masa

    grid3d = np.zeros((res, res, res))
    # Here we write the hsml and positions in grid units:
    h_grid = (2 * hsml * res / length).astype(int)
    # print(np.min(h_grid))
    # print(np.max(h_grid))
    pos_grid = (positions * res / length + res / 2).astype(int)
    print(np.max(pos_grid))

    # We depickle the kernels previously computed:
    pickle_in = open('kernel_list', 'rb')
    kernels = pickle.load(pickle_in)
    pickle_in.close()

    def addAtPos(mat1, mat2, pos):
        """
        Add two 3-arrays of different sizes in place, offset by xyz coordinates
        Usage:
          - mat1: base matrix
          - mat2: add this matrix to mat1
          - pos: [x,y,z] containing coordinates
        """
        x, y, z = pos[0], pos[1], pos[2]
        x1, y1, z1 = mat1.shape
        if np.size(mat2) == 1:
            mat1[x, y, z] += mat2
        else:
            x2, y2, z2 = mat2.shape

            # get slice ranges for matrix1
            x1min = max(0, x)
            y1min = max(0, y)
            z1min = max(0, z)
            x1max = max(min(x + x2, x1), 0)
            y1max = max(min(y + y2, y1), 0)
            z1max = max(min(z + z2, z1), 0)

            # get slice ranges for matrix2
            x2min = max(0, -x)
            y2min = max(0, -y)
            z2min = max(0, -z)
            x2max = min(-x + x1, x2)
            y2max = min(-y + y1, y2)
            z2max = min(-z + z1, z2)

            mat1[x1min:x1max, y1min:y1max, z1min:z1max] += mat2[x2min:x2max, y2min:y2max, z2min:z2max]
        return mat1

    l = 0
    for pos, h, mag in zip(pos_grid, h_grid, magnitude):
        if l%(int(np.size(hsml)/100)) == 0:
            print('Currently {}%'.format(int(l//(np.size(hsml)/100))))
        l += 1
        # We just add the contribution of the particle:
        if h < res:
            # If hsml exceeds the maximum kernel available, we take the maximum kernel instead:
            kernel = kernels[min(h, np.size(kernels) - 1)]
            # If h = 0 we just add a point to the grid:
            grid3d += addAtPos(np.zeros((res, res, res)), mag * kernel, pos - 2*h)

    axis3 = 3 - axis2 - axis1
    grid = np.sum(grid3d, axis=axis3)
    grid = np.transpose(grid)
    print('Finished!!!!!!!!!!!!!')
    return grid



path = '/media/luis/82A8355FA83552C1/CLUES_Gustavo/outputs'

snap_num = 127  # number of the load_snapshot to be loaded
snap = load_snapshot(directory = path, snapnum = snap_num, label_table = cecilia_labels)
cat = SubfindCatalogue(path, snap_num)
hubble0 = snap.header.hubble0[0]

sigma = 1	# standard deviation for gaussian filter
resolution = 300

smooth = 'n'  #   input('Do you want to smooth? (y / n)		')
save = 'y'   #input('Do you want to save the projections? (y / n)		')

# Mass projection in solar masses:
# (grid_xy_gas, x_gas, y_gas) = grid_maker(snap,'MASS', 0, 0, 1, resolution)
grid_xy_gas = grid_maker_SPH(snap, 'MASS', 4, 0, 1, resolution)


if smooth == 'y':
    grid_xy_gas = gaussian_filter(grid_xy_gas, sigma)


vmax = np.max(np.log10(grid_xy_gas))  # Stars reach higher densities
vmin = vmax
colormap = 'jet'

positions = (snap['POS '][4] - 50) / hubble0
length = max(np.max(positions[:,0] - np.min(positions[:, 0])),
            np.max(positions[:,1] - np.min(positions[:, 1])),
            np.max(positions[:,2] - np.min(positions[:, 2])))

length = 10
extent_gas_xy = (-length/2, length/2, -length/2, length/2)

grid_xy_gas = np.log10(grid_xy_gas)

# we fix null densities
grid_xy_gas[grid_xy_gas < -10000] = 0


fig, ax = plt.subplots()
im1 = ax.imshow(grid_xy_gas, extent = extent_gas_xy, origin='lower', cmap = colormap)#, vmin=vmin, vmax=vmax)
big_two = most_massives(cat, 2)
# for i in range(10):
    # CM = (cat.fof_group[i].pot_min[0] / hubble0, cat.fof_group[i].pot_min[1] / hubble0)
    # r = cat.fof_group[i].radius_crit200 / hubble0
    # circle = plt.Circle(CM, r, fill=False)
    # plt.text(CM[0] + r, CM[1] + r, "n = %s" %i)
    # ax.add_artist(circle)

for subh in big_two:
    CM = (subh.pot_min - 50) / hubble0
    plt.plot(CM[0], CM[1], '.', color='r')
    plt.text(CM[0] + .1, CM[1] + .1, "mass = {}".format(subh.mass), color='w')
ax.set_xlabel('x (Mpc)')
ax.set_ylabel('y (Mpc)')

plt.show()
