import numpy as np
import matplotlib.pyplot as plt

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels
from iccpy.gadget.subfind import SubfindCatalogue
from iccpy.utils import match

from functions import get_massive

path = '/home/luis/Documents/gecko/gecko_C/run_01/outputs'  # change between datos_Aq5 (snap127) and datos_2Mpc_LG (snap135)

snap_num = 135  # number of the snapshot to be loaded
snap = load_snapshot(directory = path, snapnum = snap_num, label_table = cecilia_labels)
cat = SubfindCatalogue(path, snap_num) # get a catalogue of subhaloes

elements = ['He', 'C', 'N', 'O', 'Fe', 'Mg', 'H', 'Si', 'Ba', 'Eu', 'Sr', 'Y']
X = elements.index(str(input('Choose an element from He, C, N, O, Fe, Mg, H, Si, Ba, Eu, Sr, Y:   ')))
Xmass = [4, 12, 14, 16, 56, 24.3, 1, 28, 137.3, 152, 87.6, 88.9][X]
Xsolar = [10.93, 8.39, 7.78, 8.66, 7.45, 7.53, 12, 7.51, 2.17, 0.52, 2.92, 2.21][X] # solar abundances relative to log(NH)=12


subhaloes = get_massive(snap, cat, 1e10)
i=0
for subh in subhaloes:
    plt.close('all')
    ind_stars = match(subh.ids, snap['ID  '][4])
    ind_stars = ind_stars[ind_stars != -1]

    # These are the masses of each element for each particle:
    Fe_stars = snap['Z   '][4][ind_stars][:, 4]
    H_stars = snap['Z   '][4][ind_stars][:, 6]
    X_stars = snap['Z   '][4][ind_stars][:, X]

    # And these are the relative abundances:
    FeH_stars = 4.55 + np.log10(Fe_stars / 56 / H_stars)
    XFe_stars = - Xsolar + 7.45 + np.log10(X_stars / Xmass / Fe_stars * 56)


    # Here we make 2D meshes for the contour maps
    FeH_XFe_stars = np.histogram2d(XFe_stars,FeH_stars,bins=25, range = [[0,1],[-5,1]])

    # We plot everything
    fig, axs = plt.subplots(2, 2)
    fig.subplots_adjust(hspace=0, wspace=0)

    """
    hist, bins = np.histogram(FeH_gas, bins=15, range=(-5,1), density=True)
    normed = hist / hist.sum() # we normalize so the bars add to 1
    widths = bins[:-1] - bins[1:]
    axs[0,0].bar(bins[1:], normed, width=widths)
    axs[0,0].set_ylabel(r'$Fracción$')
    plt.setp(axs[0,0].get_yticklabels()[0], visible=False)

    hist, bins = np.histogram(XFe_gas, bins=15, range=(0,1), density=True)
    normed = hist / hist.sum()
    widths = bins[:-1] - bins[1:]
    axs[1,1].barh(y=bins[1:], width=normed, height=widths)
    axs[1,1].set_xlabel(r'$Fracción$', fontsize=20)
    axs[1,1].set_yticklabels([])

    axs[1,0].contourf(np.linspace(-5, 1, 25), np.linspace(0, 1, 25), FeH_XFe_stars[0])
    axs[1,0].set_xlabel(r'$[Fe/H]_{\star}$', fontsize=20)
    axs[1,0].set_ylabel(r'$[%s/Fe]_{\star}$' % elements[X], fontsize=20)
    fig.delaxes(axs[0,1])
    plt.show()

    # Here we can correct the [X/Fe] range:
    XFe_min = float(input('You want to go from [%s/Fe] =   ' %elements[X]))
    XFe_max = float(input('... to [%s/Fe] =    ' %elements[X]))

    """
    XFe_min = -1
    XFe_max = 1
    # ... and we correct the [X/Fe] range:
    FeH_XFe_stars = np.histogram2d(XFe_stars,FeH_stars,bins=25, range = [[XFe_min, XFe_max],[-5,1]])


    fig, axs = plt.subplots(2, 2)
    fig.subplots_adjust(hspace=0, wspace=0)

    hist, bins = np.histogram(FeH_stars, bins=15, range=(-5,1), density=True)
    normed = hist / hist.sum()
    widths = bins[:-1] - bins[1:]
    axs[0,0].bar(bins[1:], normed, width=widths)
    axs[0,0].set_ylabel(r'$Fracción$', fontsize=20)

    hist, bins = np.histogram(XFe_stars, bins=15, range=(XFe_min, XFe_max), density=True)
    normed = hist / hist.sum()
    widths = bins[:-1] - bins[1:]
    axs[1,1].barh(y=bins[1:], width=normed, height=widths)
    axs[1,1].set_xlabel(r'$Fracción$', fontsize=20)
    axs[1,1].set_yticklabels([])


    # We logarithmize the 2d map for the countour plot:
    map2d = np.log10(FeH_XFe_stars[0])
    map2d[map2d < -10000] = np.min(map2d[map2d > -10000])

    axs[1,0].contourf(np.linspace(-5, 1, 25), np.linspace(XFe_min, XFe_max, 25), map2d, linestyles='dashed')
    axs[1,0].set_xlabel(r'$[Fe/H]_{\star}$', fontsize=20)
    axs[1,0].set_ylabel(r'$[%s/Fe]_{\star}$' % elements[X], fontsize=20)
    fig.delaxes(axs[0,1])
    fig.savefig('/home/luis/Pictures/Tesis/gecko/rel_abunds_C01/rel_%s_stars_%s.png' % (elements[X], str(i)), transparent=True, bbox_inches='tight')
    i += 1
