"""
Here we will plot the 3D trajectories of the most bound particles for subhaloes suspects of havinc merged
"""

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels
from iccpy.gadget.subfind import SubfindCatalogue
from iccpy.utils import match

from functions import get_massive


sim = 'gecko/gecko_C'
path = '/home/luis/Documents/%s/' %sim
RUN = 'run_01'


snap_num = 135
snap = load_snapshot(directory = path + RUN + '/outputs', snapnum = snap_num, label_table = cecilia_labels)
cat = SubfindCatalogue(path + RUN + '/outputs', snap_num) # get a catalogue of subhaloes
subhaloes = get_massive(snap, cat, 1e10)
hubble0 = snap.header.hubble0[0]


MBIDs = []
for subh in subhaloes[0:4]:
    MBIDs.append(subh.most_bound_particle_id)

MBIDs = np.array(MBIDs)
MBIDs = MBIDs + (MBIDs < 0) * 2**31 # There are some negative values which turn into the correct IDs by adding 2**31


subh0_pos = []
subh1_pos = []
subh2_pos = []
subh3_pos = []


for i in range(114, 136):
    snap_num = i  # number of the snapshot to be loaded
    snap = load_snapshot(directory = path + RUN + '/outputs', snapnum = snap_num, label_table = cecilia_labels)

    inds = match(MBIDs, snap['ID  '][4]) # Checked manually, all subh but 34th have a star particle as MBID
    inds = inds[inds != -1]


    subh0_pos.append(snap['POS '][4][inds[0]])
    subh1_pos.append(snap['POS '][4][inds[1]])
    subh2_pos.append(snap['POS '][4][inds[2]])
    subh3_pos.append(snap['POS '][4][inds[3]])

subh0_pos = np.array(subh0_pos) / hubble0
subh1_pos = np.array(subh1_pos) / hubble0
subh2_pos = np.array(subh2_pos) / hubble0
subh3_pos = np.array(subh3_pos) / hubble0

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot3D(subh0_pos[:, 0], subh0_pos[:, 1], subh0_pos[:, 2], 'blue')
ax.plot3D(subh1_pos[:, 0], subh1_pos[:, 1], subh1_pos[:, 2], 'yellow')
ax.plot3D(subh2_pos[:, 0], subh2_pos[:, 1], subh2_pos[:, 2], 'green')
ax.plot3D(subh3_pos[:, 0], subh3_pos[:, 1], subh3_pos[:, 2], 'red')
ax.text(subh0_pos[0, 0], subh0_pos[0, 1], subh0_pos[0, 2], 'Snap 114')
ax.text(subh0_pos[-1, 0], subh0_pos[-1, 1], subh0_pos[-1, 2], 'Snap 135')

plt.show()
