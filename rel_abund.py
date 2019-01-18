import numpy as np
import matplotlib.pyplot as plt

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels
from iccpy.gadget.subfind import SubfindCatalogue
from iccpy.utils import match

path = '/home/luis/Documents/datos_Aq5/outputs'  # change between datos_Aq5 (snap127) and datos_2Mpc_LG (snap135)

snap_num = 127  # number of the snapshot to be loaded
snap = load_snapshot(directory = path, snapnum = snap_num, label_table = cecilia_labels)
num = 0 #subhalo number (0 for main subhalo)
cat = SubfindCatalogue(path, snap_num) # get a catalogue of subhaloes

ind_gas = np.concatenate([match(cat.subhalo[num].ids, snap['ID  '][0]), match(cat.subhalo[num].ids + 2**31, snap['ID  '][0])])
ind_gas = ind_gas[ind_gas != -1]
ind_stars = match(cat.subhalo[num].ids, snap['ID  '][4])
ind_stars = ind_stars[ind_stars != -1]


elements = ['He', 'C', 'N', 'O', 'Fe', 'Mg', 'H', 'Si', 'Ba', 'Eu', 'Sr', 'Y']
X = elements.index(str(input('Choose an element from He, C, N, O, Fe, Mg, H, Si, Ba, Eu, Sr, Y:   ')))
Xmass = [4, 12, 14, 16, 56, 24.3, 1, 28, 137.3, 152, 87.6, 88.9][X]
Xsolar = [10.93, 8.39, 7.78, 8.66, 7.45, 7.53, 12, 7.51, 2.17, 0.52, 2.92, 2.21][X] # solar abundances relative to log(NH)=12

# These are the masses of each element for each particle:
Fe_gas = snap['Z   '][0][ind_gas][:, 4]
H_gas = snap['Z   '][0][ind_gas][:, 6]
Fe_stars = snap['Z   '][4][ind_stars][:, 4]
H_stars = snap['Z   '][4][ind_stars][:, 6]

X_gas = snap['Z   '][0][ind_gas][:, X]
X_stars = snap['Z   '][4][ind_stars][:, X]

# And these are the relative abundances:
FeH_gas = 4.55 + np.log10(Fe_gas / 56 / H_gas)    			# [Fe/H]_solar = 7.45 - 12
FeH_stars = 4.55 + np.log10(Fe_stars / 56 / H_stars)
XFe_gas = - Xsolar + 7.45 + np.log10(X_gas / Xmass / Fe_gas * 56)   	# [X/Fe]_solar = [X] - 7.45
XFe_stars = - Xsolar + 7.45 + np.log10(X_stars / Xmass / Fe_stars * 56)


# Here we make 2D meshes for the contour maps
FeH_XFe_gas = np.histogram2d(XFe_gas,FeH_gas,bins=25, range = [[0,1],[-5,1]])
FeH_XFe_stars = np.histogram2d(XFe_stars,FeH_stars,bins=25, range = [[0,1],[-5,1]])

# We plot everything
fig, axs = plt.subplots(2, 2)
fig.subplots_adjust(hspace=0, wspace=0)

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


axs[1,0].contourf(np.linspace(-5, 1, 25), np.linspace(0, 1, 25), FeH_XFe_gas[0])
axs[1,0].set_xlabel(r'$[Fe/H]_{gas}$', fontsize=20)
axs[1,0].set_ylabel(r'$[%s/Fe]_{gas}$' % elements[X], fontsize=20)
fig.delaxes(axs[0,1])
plt.show()

# Here we can correct the [X/Fe] range:
XFe_min = float(input('You want to go from [%s/Fe] =   ' %elements[X]))
XFe_max = float(input('... to [%s/Fe] =    ' %elements[X]))


# ... and we correct the [X/Fe] range:
FeH_XFe_gas = np.histogram2d(XFe_gas,FeH_gas,bins=25, range = [[XFe_min, XFe_max],[-5,1]])
FeH_XFe_stars = np.histogram2d(XFe_stars,FeH_stars,bins=25, range = [[XFe_min, XFe_max],[-5,1]])



fig, axs = plt.subplots(2, 2)
fig.subplots_adjust(hspace=0, wspace=0)

hist, bins = np.histogram(FeH_gas, bins=15, range=(-5,1), density=True)
normed = hist / hist.sum()
widths = bins[:-1] - bins[1:]
axs[0,0].bar(bins[1:], normed, width=widths)
axs[0,0].set_ylabel(r'$Fracción$', fontsize=20)

hist, bins = np.histogram(XFe_gas, bins=15, range=(XFe_min, XFe_max), density=True)
normed = hist / hist.sum()
widths = bins[:-1] - bins[1:]
axs[1,1].barh(y=bins[1:], width=normed, height=widths)
axs[1,1].set_xlabel(r'$Fracción$', fontsize=20)
axs[1,1].set_yticklabels([])


axs[1,0].contourf(np.linspace(-5, 1, 25), np.linspace(XFe_min, XFe_max, 25), FeH_XFe_gas[0])
axs[1,0].set_xlabel(r'$[Fe/H]_{gas}$', fontsize=20)
axs[1,0].set_ylabel(r'$[%s/Fe]_{gas}$' % elements[X], fontsize=20)
fig.delaxes(axs[0,1])
fig.savefig('/home/luis/Pictures/Tesis/rel_%s_gas.png' % elements[X], transparent=True, bbox_inches='tight')


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


axs[1,0].contourf(np.linspace(-5, 1, 25), np.linspace(XFe_min, XFe_max, 25), FeH_XFe_stars[0], linestyles='dashed')
axs[1,0].set_xlabel(r'$[Fe/H]_{stars}$', fontsize=20)
axs[1,0].set_ylabel(r'$[%s/Fe]_{stars}$' % elements[X], fontsize=20)
fig.delaxes(axs[0,1])
fig.savefig('/home/luis/Pictures/Tesis/rel_%s_stars.png' % elements[X], transparent=True, bbox_inches='tight')
