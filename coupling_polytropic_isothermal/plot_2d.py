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
    for i in range(48):
        for j in range(48):
            zz = field[4*i:4*(i+1), 4*j:4*(j+1)].copy()
            field[4*i:4*(i+1), 4*j:4*(j+1)] = interp(xx, yy, zz)
    return field


# Define the time index to plot
#t_idx = 0
t_idx = 150

#rho = np.zeros([48, 48])
rho = np.zeros([48*4, 48*4])

# Read the mesh files.
for i in range(3):
    for j in range(3):
        f = h5py.File('out/solution_{0}_{1:09}.h5'.format(i+j*3+1, t_idx))
        rho_part = f['variables_1']
        rho_part = np.reshape(rho_part, [4, 4, 16, 16], order='F')
        rho_part = np.swapaxes(rho_part, 1, 2)
        rho_part = np.reshape(rho_part, [64, 64], order='F')
#        rho[16*i:16*(i+1), 16*j:16*(j+1)] = np.average(rho_part, axis=(0, 1))
        rho[64*i:64*(i+1), 64*j:64*(j+1)] = rho_part[:, :]
        f.close()

# Perform the 2d polynomial interpolation.
rho = perform_interpolation(rho)

# Prepare the plot.
width = 8
height = 6
plt.rc('text', usetex=True)
plt.rc('font', family='arial')
plt.rc("figure.subplot", left=0.15)
plt.rc("figure.subplot", right=0.95)
plt.rc("figure.subplot", bottom=0.15)
plt.rc("figure.subplot", top=0.95)
fig = plt.figure(figsize=(width, height))
ax = fig.add_subplot(111)
plt.ion()

# Plot the density.
im = plt.imshow(rho.T, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5])

# Plot boundaries.
plt.plot([-1.5, 1.5], [-0.5, -0.5], color='k', linewidth=2)
plt.plot([-1.5, 1.5], [0.5, 0.5], color='k', linewidth=2)
plt.plot([-0.5, -0.5], [-1.5, 1.5], color='k', linewidth=2)
plt.plot([0.5, 0.5], [-1.5, 1.5], color='k', linewidth=2)

# Improve plot quality.
plt.tick_params(axis='both', which='major', length=8, labelsize=20)
plt.tick_params(axis='both', which='minor', length=4, labelsize=20)

plt.xlabel(r'$x$', fontsize=25)
plt.ylabel(r'$y$', fontsize=25)

plt.gca().xaxis.set_major_locator(matplotlib.ticker.LinearLocator(7))

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
cb = plt.colorbar()
cb.set_label(r'$\rho$', fontsize=25)
cbytick_obj = plt.getp(cb.ax.axes, 'yticklabels')
for tick in cb.ax.get_xticklabels():
    tick.set_fontsize(15)
    tick.set_fontfamily('serif')
cb.ax.xaxis.get_offset_text().set(size=15)
cb.ax.tick_params(labelsize=15)

plt.show()
