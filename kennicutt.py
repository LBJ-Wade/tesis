import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm, norm # For calculation of the rotation matrix
from scipy.stats import linregress
import os

from iccpy.gadget import load_snapshot
from iccpy.gadget.labels import cecilia_labels
from iccpy.gadget.subfind import SubfindCatalogue
from iccpy.utils import match

sim = 'gecko/gecko_C'
path = '/home/luis/Documents/%s/run_01/outputs' %sim
RUNS = os.listdir(path)

snap_num = 135  # number of the snapshot to be loaded
snap = load_snapshot(directory = path, snapnum = snap_num, label_table = cecilia_labels)
cat = SubfindCatalogue(path, snap_num) # get a catalogue of subhaloes
num = 0 #subhalo number (0 for main subhalo)


ind_gas = np.concatenate([match(cat.subhalo[num].ids, snap['ID  '][0]), match(cat.subhalo[num].ids + 2**31, snap['ID  '][0])])
ind_gas = ind_gas[ind_gas != -1]
ind_stars = match(cat.subhalo[num].ids, snap['ID  '][4])
ind_stars = ind_stars[ind_stars != -1]

CM = cat.subhalo[num].pot_min

# gas
pos = snap['POS '][0][ind_gas] - CM
pos_x = pos[:,0]
pos_y = pos[:,1]
pos_z = pos[:,2]
r = np.sqrt(pos_x**2 + pos_y**2)

layers = 50
rmin = 5
rmax = 30 # in kpc
#height = 30


sigma_gas, radios1 = np.histogram(r, bins=layers, range=(rmin, rmax), weights=snap['MASS'][0][ind_gas] * 1e10 / .73 )
sigma_gas = sigma_gas / (np.pi * (radios1[1:]**2 - radios1[:-1]**2)) / 1e5 # - radios1[:-1]**2))

sigma_SFR, radios2 = np.histogram(r, bins=layers, range=(rmin, rmax), weights=snap['SFR '][0][ind_gas])
sigma_SFR = sigma_SFR / (np.pi * (radios2[1:]**2 - radios2[:-1]**2)) * 1e10/.73 / 1e9 # - radios1[:-1]**2))

valid = np.array(sigma_gas!=0) * np.array(sigma_SFR!=0)
slope, intercept, r_value, p_value, std_err = linregress(np.log(sigma_gas[valid]), np.log(sigma_SFR[valid]))
print(slope)

fit = np.log(np.sort(sigma_gas)) * slope + intercept # ordeno los sigma_gas para que al plotear no quede fea la línea

fig = plt.figure()
ax = fig.add_subplot(111)
plt.loglog(sigma_gas, sigma_SFR, '.r')
plt.loglog(np.sort(sigma_gas), np.exp(fit), 'green')
plt.xlabel(r'$\Sigma_{gas}\ (M_{\odot}\ pc^{-2})$', fontsize = 16)
plt.ylabel(r'$\Sigma_{SFR}\ (M_{\odot}\ yr^{-1}\ pc^{-2})$', fontsize = 16)
plt.grid()
plt.text(.1, .85, 'Pendiente: %s' % np.round(slope, 2), bbox=dict(facecolor='wheat', alpha=0.5), transform = ax.transAxes, fontsize = 14)
plt.show()