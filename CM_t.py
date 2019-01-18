import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
plt.rcParams.update({'font.size': 32})

import os
import time

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels
from iccpy.gadget.subfind import SubfindCatalogue
from iccpy.utils import match

from functions import PCA_matrix, Vcm, most_massives, grid_maker



path = '/media/luis/82A8355FA83552C1/CLUES_Gustavo/'
CM_0_x = []
CM_0_y = []
CM_0_z = []

CM_33_x = []
CM_33_y = []
CM_33_z = []


for j in range(40, 128):
    print(j)
    snap_num = j
    # snap = load_snapshot(directory=path+'outputs', snapnum=snap_num, label_table=cecilia_labels)
    # cat = SubfindCatalogue(path+'outputs', snap_num) # get a catalogue of subhaloes
    # hubble0 = snap.header.hubble0[0]

    merg_tree_0 = path + '/postproc/Prog_0.dat'
    merg_tree_0 = np.flip(np.loadtxt(merg_tree_0), axis=0)

    merg_tree_33 = path + '/postproc/Prog_33.dat'
    merg_tree_33 = np.flip(np.loadtxt(merg_tree_33), axis=0)


    CM_0, CM_33 = merg_tree_0[j - 40][4:], merg_tree_33[j - 40][4:]
    # m_0         = cat.subhalo[0].mass
    # m_33        = cat.subhalo[33].mass
    # CM          = (CM_0 * m_0 + CM_33 * m_33) / (m_0 + m_33)
    #
    # CM_0, CM_33 = CM_0 - CM, CM_33 - CM

    CM_0_x.append(CM_0[0])
    CM_0_y.append(CM_0[1])
    CM_0_z.append(CM_0[2])

    CM_33_x.append(CM_33[0])
    CM_33_y.append(CM_33[1])
    CM_33_z.append(CM_33[2])


fig = plt.figure()
ax = plt.axes(projection='3d')

t = range(40, 128)

ax.scatter(CM_0_x, CM_0_y, CM_0_z, c=t)
ax.scatter(CM_33_x, CM_33_y, CM_33_z, c=t)
ax.text(CM_0_x[0], CM_0_y[0], CM_0_z[0], 'Snap 40')
ax.text(CM_0_x[-1], CM_0_y[-1], CM_0_z[-1], 'Snap 127')

plt.show()
