# plot_2d.py
'''
Plot some 2d quantities.
'''

import numpy as np
import pylab as plt
import h5py
from scipy.interpolate import lagrange


# Define the Lagrange polynomial in 2d.
nodes = np.array([-1, -1/np.sqrt(5), 1/np.sqrt(5), 1])
e = np.identity(4)
poly1d = np.empty(4, dtype=np.poly1d)
def poly2d(x, y, z, poly1d):
    for i, row in enumerate(e):
        poly1d[i] = lagrange(nodes, row)

    r = np.zeros_like(x)
    for i in range(4):
        for j in range(4):
            r += z[i, j] * poly1d[i](x) * poly1d[j](y)
    return r
interp = lambda x, y, z: poly2d(x, y, z, poly1d)


def perform_interpolation(field):
    x = np.linspace(-0.75, 0.75, 4)
    xx, yy = np.meshgrid(x, x, indexing='ij')
    for i in range(64):
        for j in range(32):
            zz = field[4*i:4*(i+1), 4*j:4*(j+1)].copy()
            field[4*i:4*(i+1), 4*j:4*(j+1)] = interp(xx, yy, zz)
    return field


# Define the time index to plot.
t_idx = 0
# t_idx = 125

# Read the mesh files.
f = h5py.File('out/mesh_1_000000000.h5')
r_idx = f.attrs['size'][0]
f.close()
f = h5py.File('out/mesh_2_000000000.h5')
f.close()

# Read the data files.
# Polytropic left
f = h5py.File('out/solution_1_{0:09}.h5'.format(t_idx))
rho_l = f['variables_1']
rho_l = np.reshape(rho_l, [4, 4, r_idx, 32], order='F')
rho_l = np.swapaxes(rho_l, 1, 2)
rho_l = np.reshape(rho_l, [4*r_idx, 4*32], order='F')
f.close()

# Euler right
f = h5py.File('out/solution_2_{0:09}.h5'.format(t_idx))
rho_r = f['variables_1']
rho_r = np.reshape(rho_r, [4, 4, r_idx, 32], order='F')
rho_r = np.swapaxes(rho_r, 1, 2)
rho_r = np.reshape(rho_r, [4*(64 - r_idx), 4*32], order='F')
f.close()

# Get everything into one array.
rho = np.zeros([64*4, 32*4])
rho[:r_idx*4, :] = rho_l
rho[r_idx*4:, :] = rho_r

# Perform the 2d polynomial interpolation.
rho = perform_interpolation(rho)

# Prepare the plot.
width = 8
height = 5.5
plt.rc('text', usetex=True)
plt.rc('font', family='arial')
plt.rc("figure.subplot", left=0.15)
plt.rc("figure.subplot", right=0.98)
plt.rc("figure.subplot", bottom=0.12)
plt.rc("figure.subplot", top=0.85)
fig = plt.figure(figsize=(width, height))
ax = fig.add_subplot(111)
plt.ion()

plt.imshow(rho.T, origin='lower', extent=[-2, 2, -1, 1], vmin=0.993, vmax=1.007)

# Improve plot quality.
plt.tick_params(axis='both', which='major', length=8, labelsize=20)
plt.tick_params(axis='both', which='minor', length=4, labelsize=20)

plt.xlabel(r'$x$', fontsize=25)
plt.ylabel(r'$y$', fontsize=25)

for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontname('serif')
    tick.label2.set_fontname('serif')
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontname('serif')
    tick.label2.set_fontname('serif')

for label in ax.xaxis.get_ticklabels():
    label.set_position((0, -0.03))
for label in ax.yaxis.get_ticklabels():
    label.set_position((-0.03, 0))

# Add colorbar.
cax = ax.inset_axes([0.05, 1.25, 0.9, 0.1])
cb = plt.colorbar(cax=cax, orientation='horizontal')
cb.set_label(r'$\rho$', fontsize=25)
cbytick_obj = plt.getp(cb.ax.axes, 'yticklabels')
for tick in cb.ax.get_xticklabels():
    tick.set_fontsize(15)
    tick.set_fontfamily('serif')
