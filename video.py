import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels
from iccpy.gadget.subfind import SubfindCatalogue
from iccpy.utils import match

from functions import grid_maker_SPH, get_massive


dpi = 70

component = 4
box_size = .04
resolution = 100
colormap = 'jet'
use_subf_ids = False
path = '/media/luis/82A8355FA83552C1/CLUES_Gustavo'


subh_num_z0 = 0
merg_tree = path + '/postproc/Prog_{}.dat'.format(subh_num_z0)
print(merg_tree)
merg_tree = np.flip(np.loadtxt(merg_tree), axis=0)
starting_snap = 50  # int(merg_tree[0][0] + 20)
frames = int(merg_tree[-1][0] - starting_snap)


def ani_frame():

    fig = plt.figure(figsize=(19.2, 10.8))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')

    fig.set_size_inches([10,3])

    plt.tight_layout()

    def update_img(i):
        snap_num = i + starting_snap
        print('Snapshot {}'.format(snap_num))
        snap = load_snapshot(directory=path + '/outputs', snapnum=snap_num,label_table=cecilia_labels)
        cat = SubfindCatalogue(path + '/outputs', snap_num) # get a catalogue of subhaloes
        hubble0 = snap.header.hubble0[0]
        print('Subhalo number {}'.format(int(merg_tree[starting_snap - 36 + i][1])))
        subh = cat.subhalo[int(merg_tree[starting_snap - 36 + i][1])]

        rho_xy = grid_maker_SPH(snap, subh, 'MASS', component, 1, 0, box_size, resolution, use_subf_ids) / (box_size**2 / resolution**2)
        rho_xy = gaussian_filter(rho_xy, 1)
        rho_xy = np.log10(rho_xy)
        rho_xy[rho_xy < -10000] = np.min(rho_xy[rho_xy > -10000])

        rho_xz = grid_maker_SPH(snap, subh, 'MASS', component, 2, 0, box_size, resolution, use_subf_ids) / (box_size**2 / resolution**2)
        rho_xz = gaussian_filter(rho_xz, 1)
        rho_xz = np.log10(rho_xz)
        rho_xz[rho_xz < -10000] = np.min(rho_xz[rho_xz > -10000])

        rho_yz = grid_maker_SPH(snap, subh, 'MASS', component, 2, 1, box_size, resolution, use_subf_ids) / (box_size**2 / resolution**2)
        rho_yz = gaussian_filter(rho_yz, 1)
        rho_yz = np.log10(rho_yz)
        rho_yz[rho_yz < -10000] = np.min(rho_yz[rho_yz > -10000])


        ax1.clear()
        ax1.grid(True)
        ax1.imshow(rho_xy, extent=(-box_size/2, box_size/2, -box_size/2, box_size/2), origin='lower', cmap = colormap)
        ax1.set_xlim(-box_size/2,box_size/2)
        ax1.set_ylim(-box_size/2,box_size/2)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('$Snapshot = {}$'.format(i + starting_snap))

        ax2.clear()
        ax2.grid(True)
        ax2.imshow(rho_xz, extent=(-box_size/2, box_size/2, -box_size/2, box_size/2), origin='lower', cmap = colormap)
        ax2.set_xlim(-box_size/2,box_size/2)
        ax2.set_ylim(-box_size/2,box_size/2)
        ax2.set_xlabel('x')
        ax2.set_ylabel('z')
        ax2.set_title('$Snapshot = {}$'.format(i + starting_snap))

        ax3.clear()
        ax3.grid(True)
        ax3.imshow(rho_yz, extent=(-box_size/2, box_size/2, -box_size/2, box_size/2), origin='lower', cmap = colormap)
        ax3.set_xlim(-box_size/2,box_size/2)
        ax3.set_ylim(-box_size/2,box_size/2)
        ax3.set_xlabel('y')
        ax3.set_ylabel('z')
        ax3.set_title('$Snapshot = {}$'.format(i + starting_snap))

        plt.tight_layout(pad=.5, w_pad=0, h_pad=0)


        return fig

    ani = animation.FuncAnimation(fig, update_img, frames=frames, interval=200)
    writer = animation.writers['ffmpeg'](fps=12)

    ani.save('stars_proj_{}.mp4'.format(subh_num_z0),writer=writer,dpi=dpi)
    return ani

ani_frame()
