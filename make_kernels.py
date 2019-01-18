import numpy as np
import pickle

"""
Computes a list of Monaghan-Lattanzio 3D kernels and saves it as a pickle
"""


h_max = int(input('Type in the maximum hsml (in lattice points) which you want to compute the Monaghan-Lattanzio kernel:  '))

def W(r_h, h):
    """
    Calculates the value of the Monaghan-Lattanzio kernel function given r/h and h
    """
    if 0 <= r_h <= 1:
        W = 1 - 1.5 * (r_h)**2 + .75 * (r_h)**3
    elif 1 < r_h <= 2:
        W = .25 * (2 - r_h)**3
    else:
        W = 0
    W = W / np.pi / h**3
    return W


def kernel(h):
    """
    Makes a 3D array with the 3D distribution of the Monaghan-Lattanzio kernel
    """
    ker = np.zeros((4*h-1, 4*h-1, 4*h-1))

    for i in range(4*h-1):
        for j in range(4*h-1):
            for k in range(4*h-1):
                r_grid = np.linalg.norm([i - 2*h + 1, j - 2*h + 1, k - 2*h + 1])
                ker[i, j, k] = W(r_grid / h, h)
    return ker


kernels = []

for l in range(h_max):
    if l == 0:
        kernels.append(np.array([1]))
    else:
        print('Calculating kernel for h = {}...'.format(l))
        kernels.append(kernel(l))

pickle_out = open('kernel_list', 'wb')
pickle.dump(kernels,  pickle_out)
pickle_out.close()
