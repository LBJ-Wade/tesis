import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels
from iccpy.gadget.subfind import SubfindCatalogue
from iccpy.utils import match

import time, pickle


################################################################################################
################################################################################################


def M_vir_crit(snap, subh, layers, rmax):
    """
    Returns M_vir, Mstar_vir, r_200_crit
    """
    hubble0 = snap.header.hubble0[0]
    ind_gas = np.concatenate([match(subh.ids, snap['ID  '][0]), match(subh.ids + 2**31, snap['ID  '][0])])
    ind_gas = ind_gas[ind_gas != -1]
    ind_stars = match(subh.ids, snap['ID  '][4])
    ind_stars = ind_stars[ind_stars != -1]
    ind_DM = match(subh.ids, snap['ID  '][1])
    ind_DM = ind_DM[ind_DM != -1]

    CM = subh.pot_min

    pos = np.concatenate((snap['POS '][0][ind_gas] - CM, snap['POS '][4][ind_stars] - CM, snap['POS '][1][ind_DM] - CM))
    pos = pos / hubble0 * 1000
    r = np.linalg.norm(pos, axis=1)


    masses = np.concatenate((snap['MASS'][0][ind_gas], snap['MASS'][4][ind_stars], snap['MASS'][1][ind_DM])) * 1e10 / hubble0

    # We make a mass histogram with radial bins:
    mass, radius = np.histogram(r, bins=layers, range=(0, rmax), weights=masses)

    inner_mass = np.cumsum(mass)
    rho = inner_mass / (4/3 * np.pi * radius[1:]**3)
    rho_crit = 126.7 # solar masses per kpc^3, from Planck

    ind_200 = (np.abs(rho - 200*rho_crit)).argmin() # This gives the index of the bin where rho is closest to 200*rho_crit
    r_200 = radius[ind_200]
    print('r_200 is {}'.format(r_200))
    M_vir = np.sum(masses[r < r_200])

    pos = (snap['POS '][4][ind_stars] - CM) / hubble0 * 1000
    r = np.linalg.norm(pos, axis=1)

    star_mass = snap['MASS'][4][ind_stars] * 1e10 / hubble0
    Mstar_vir = np.sum(star_mass[r < r_200])

    return M_vir, Mstar_vir, r_200


################################################################################################
################################################################################################


def get_inner_inds(snap, subh, component):
    """
    Finds the indexes of the particles of
    a given component which lie within r_200 / 10 from the subhalo CM.
    This will return the indexes for the particles of the given type so one can
    then use them with snap['XXXX'][component][indexes]
    """
    CM = subh.pot_min
    hubble0 = snap.header.hubble0[0]
    pos = (snap['POS '][component] - CM) * 1000 / hubble0
    r = np.linalg.norm(pos, axis=1)
    r_200 = M_vir_crit(snap, subh, 5000, 300)[2]
    ind = r < r_200 / 10
    # r_half = .05
    print('r_200 is {} kpc'.format(r_200))
    
    ind = np.nonzero(ind)[0]

    return ind


################################################################################################
################################################################################################


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


################################################################################################
################################################################################################


def get_massive(snap, cat, M):
    """
    Returns a list with subhalo objects with more mass than M (given in solar masses)
    TAKES TOO MUCH TIME TO RUN, BUT IF WE TAKE SUBFIND MASS IT'S TOO BIG
    MAYBE FILTER OUT LOW TOTAL MASSES BEFORE FILTERING BY VIRIAL MASS????
    """
    hubble0 = snap.header.hubble0[0]
    massives = []
    i = 0
    for subh in cat.subhalo[:]:
    	if subh.mass > M * hubble0 / 1e10:
    		massives.append(subh)
    	i += 1
    return massives


################################################################################################
################################################################################################


def PCA_matrix(snap, subh):
    CM = subh.pot_min
    # ind = match(subh.ids, snap['ID  '][4])
    # ind = ind[ind != -1]

    ind = get_inner_inds(snap, subh, 4)
    print('{} stars for PCA'.format(np.size(ind)))
    pos = (snap['POS '][4][ind] - CM)
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


################################################################################################
################################################################################################


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


################################################################################################
################################################################################################


def grid_maker(snap, subh, quantity, component, axis1, axis2, length, res, use_subf_ids):
    """
    Returns a res*res 2darray with the projected quantity (e.g. 'MASS') for the
    desired component (0 for gas, 1 for DM, 4 for stars)
    subfind: True if the subfind can be trusted, False if we want to use
    get_subh_from_CM to get all the particles inside r_200 / 10
    """
    hubble0 = snap.header.hubble0[0]
    CM = subh.pot_min

    if use_subf_ids:
        if component == 0:
        	ind = np.concatenate([match(subh.ids, snap['ID  '][0]), match(subh.ids + 2**31, snap['ID  '][0])])
        else:
        	ind = match(subh.ids, snap['ID  '][component])
        ind = ind[ind != -1]
        positions = (snap['POS '][component][ind] - CM) / hubble0
    else:
        positions = (snap['POS '][component] - CM) / hubble0
        ind = list(abs(positions[:, 0]) < length/2) and list(abs(positions[:, 1]) < length/2) and list(abs(positions[:, 2]) < length/2)
        positions = positions[ind]

    # We rotate the positions so that the galactic angular momentum is parallel to the z axis:
    rot_matrix = PCA_matrix(snap, subh)
    positions = np.dot(positions, rot_matrix)

    pos_1 = positions[:, axis1]
    pos_2 = positions[:, axis2]
    # axis3 = 3 - axis2 - axis1
    # pos_3 = (snap['POS '][component][index] - CM)[axis3] * 1000 / hubble0
    magnitude = snap[quantity][component][ind] * 1e10 / hubble0  # cambio de unidades para masa

    # # Here we smooth the mass distribution, averaging with 32 nearest neighbors:
    # nbrs = NearestNeighbors(n_neighbors=32, algorithm='auto').fit(positions)
    # indices = nbrs.kneighbors(positions)[1]
    # mag_smoothed = []
    #
    # for i in range(np.size(magnitude)):
    #     # print(indices[i])
    #     mag = np.sum(magnitude[indices[i]]) / 32
    #     mag_smoothed.append(mag)

    # hist = np.histogram2d(pos_1, pos_2, bins=res, range=[[-length/2, length/2], [-length/2, length/2]], weights=mag_smoothed)
    hist = np.histogram2d(pos_1, pos_2, bins=res, range=[[-length/2, length/2], [-length/2, length/2]], weights=magnitude)

    return hist[0]


################################################################################################
################################################################################################


def grid_maker_SPH(snap, subh, quantity, component, axis1, axis2, length, res, use_subf_ids):
    """
    Returns a res*res 2darray with the projected quantity (e.g. 'MASS') for the
    desired component (0 for gas, 1 for DM, 4 for stars)
    subfind: True if the subfind can be trusted, False if we want to use
    get_subh_from_CM to get all the particles inside r_200 / 10
    """
    hubble0 = snap.header.hubble0[0]
    CM = subh.pot_min

    if use_subf_ids:
        if component == 0:
        	ind = np.concatenate([match(subh.ids, snap['ID  '][0]), match(subh.ids + 2**31, snap['ID  '][0])])
        else:
        	ind = match(subh.ids, snap['ID  '][component])
        ind = ind[ind != -1]
        positions = (snap['POS '][component][ind] - CM) / hubble0
    else:
        positions = (snap['POS '][component] - CM) / hubble0

    # We rotate the positions so that the galactic angular momentum is parallel to the z axis:
    rot_matrix = PCA_matrix(snap, subh)
    positions = np.dot(positions, rot_matrix)

    # We exclude all the particles laying outside the box post-rotation:
    ind = np.all([[abs(positions[:, 0]) < length/2], [abs(positions[:, 1]) < length/2], [abs(positions[:, 2]) < length/2]], axis=0)[0]
    positions = positions[ind]
    print('{} particles inside the grid'.format(np.sum(ind)))
    # We take HSML from snap for gas and we calculate the mean distance of the 32 nearest neighbors for other components:
    if component == 0:
        hsml = snap['HSML'][0][ind] / hubble0
    else:
        hsml = []
        nbrs = NearestNeighbors(n_neighbors=32, algorithm='auto').fit(positions)
        distances = nbrs.kneighbors(positions)[0]
        for d in distances:
            hsml.append(d.mean())
        hsml = np.array(hsml)

    magnitude = snap[quantity][component][ind] * 1e10 / hubble0  # cambio de unidades para masa

    grid3d = np.zeros((res, res, res))
    # Here we write the hsml and positions in grid units:
    h_grid = (2 * hsml * res / length).astype(int)
    # print(np.min(h_grid))
    # print(np.max(h_grid))

    pos_grid = (positions * res / length + res / 2).astype(int)

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
        l += 1
        # if l%(int(np.size(hsml)/10)) == 0:
        #     print('Currently {}0%'.format(int(l//(np.size(hsml)/10))))
        # We just add the contribution of the particle:
        if h < res:
            # If hsml exceeds the maximum kernel available, we take the maximum kernel instead:
            kernel = kernels[min(h, np.size(kernels) - 1)]
            # If h = 0 we just add a point to the grid:
            grid3d += addAtPos(np.zeros((res, res, res)), mag * kernel, pos - 2*h + 1)

    axis3 = 3 - axis2 - axis1
    grid = np.sum(grid3d, axis=axis3)
    grid = np.transpose(grid)
    print('Finished!!!!!!!!!!!!!')
    return grid


################################################################################################
################################################################################################


def V_i_grid(snap, subh, component, axis1, axis2, length, res, i):
    """
    Returns a res*res 2darray with the mean velocity in the i direction (0 is x, 1 is y and 2 is z) for the desired matter component (0 for gas, 1 for DM, 4 for stars)
    """
    hubble0 = snap.header.hubble0[0]
    if component == 0:
    	ind = np.concatenate([match(subh.ids, snap['ID  '][0]), match(subh.ids + 2**31, snap['ID  '][0])])
    else:
    	ind = match(subh.ids, snap['ID  '][component])
    ind = ind[ind != -1]

    CM = subh.pot_min

    # We rotate the positions so that the galactic angular momentum is parallel to the z axis:
    positions = (snap['POS '][component][ind] - CM) / hubble0

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
