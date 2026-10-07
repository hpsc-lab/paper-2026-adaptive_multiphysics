# plot_2d.py
'''
Plot some 2d quantities.
'''

import numpy as np
import matplotlib
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
    for i in range(field.shape[0]//4):
        for j in range(field.shape[1]//4):
            zz = field[4*i:4*(i+1), 4*j:4*(j+1)].copy()
            field[4*i:4*(i+1), 4*j:4*(j+1)] = interp(xx, yy, zz)
    return field


# Define the derivative matrix D[i, j] = l_j'(xi_i) on the reference element.
D = np.zeros([4, 4])
for j, row in enumerate(e):
    D[:, j] = lagrange(nodes, row).deriv()(nodes)


def compute_curl(v1, v2, dx, dy):
    omega = np.zeros_like(v1)
    for i in range(v1.shape[0]//4):
        for j in range(v1.shape[1]//4):
            v1_el = v1[4*i:4*(i+1), 4*j:4*(j+1)]
            v2_el = v2[4*i:4*(i+1), 4*j:4*(j+1)]
            omega[4*i:4*(i+1), 4*j:4*(j+1)] = 2/dx * D @ v2_el - 2/dy * v1_el @ D.T
    return omega


def extract_variable(f, var_idx, nx, ny):
    var = f['variables_{0}'.format(var_idx)]
    var = np.reshape(var, [4, 4, nx, ny], order='F')
    var = np.swapaxes(var, 1, 2)
    var = np.reshape(var, [4*nx, 4*ny], order='F')
    return var


# Define the time index to plot.
# t_idx = 0
t_idx = 3829373

# Read the mesh files.
f = h5py.File('out/mesh_1_000000000.h5')
nx = f.attrs['size'][0]
ny_b = f.attrs['size'][1]
f.close()
f = h5py.File('out/mesh_2_000000000.h5')
ny_t = f.attrs['size'][1]
f.close()
dx = 1/nx
dy = 1/(ny_b + ny_t)

# Read the data files.
# MHD bottom
f = h5py.File('out/solution_1_{0:09}.h5'.format(t_idx))
b_1 = extract_variable(f, 6, nx, ny_b)
b_2 = extract_variable(f, 7, nx, ny_b)
b_3 = extract_variable(f, 8, nx, ny_b)
f.close()

# Euler top
f = h5py.File('out/solution_2_{0:09}.h5'.format(t_idx))
v_1 = extract_variable(f, 2, nx, ny_t)
v_2 = extract_variable(f, 3, nx, ny_t)
f.close()

# Compute the vorticity.
omega = compute_curl(v_1, v_2, dx, dy)

# Perform the 2d polynomial interpolation.
b_1 = perform_interpolation(b_1)
b_2 = perform_interpolation(b_2)
b_3 = perform_interpolation(b_3)
omega = perform_interpolation(omega)

b_sq = b_1**2 + b_2**2 + b_3**2

# Prepare the plot.
width = 8
height = 6
plt.rc('text', usetex=True)
plt.rc('font', family='arial')
plt.rc("figure.subplot", left=0.2)
plt.rc("figure.subplot", right=0.78)
plt.rc("figure.subplot", bottom=0.15)
plt.rc("figure.subplot", top=0.95)
fig = plt.figure(figsize=(width, height))
ax = fig.add_subplot(111)
plt.ion()

# Plot log(B^2) in the MHD domain and the vorticity in the Euler domain.
im_mhd = plt.imshow(np.log(b_sq).T, origin='lower', extent=[-0.5, 0.5, -0.5, 0])
omega_max = np.max(np.abs(omega))
im_euler = plt.imshow(omega.T, origin='lower', extent=[-0.5, 0.5, 0, 0.5],
                      cmap='RdBu_r', vmin=-omega_max, vmax=omega_max)
plt.xlim(-0.5, 0.5)
plt.ylim(-0.5, 0.5)
ax.set_aspect('equal')

# Plot boundaries.
plt.plot([-0.5, 0.5], [0, 0], color='k', linestyle=':', linewidth=2)

# Improve plot quality.
plt.tick_params(axis='both', which='major', length=8, labelsize=20)
plt.tick_params(axis='both', which='minor', length=4, labelsize=20)

plt.xlabel(r'$x$', fontsize=25)
plt.ylabel(r'$y$', fontsize=25)

plt.gca().xaxis.set_major_locator(matplotlib.ticker.LinearLocator(5))
plt.gca().yaxis.set_major_locator(matplotlib.ticker.LinearLocator(5))

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

# Add colorbars.
cax_euler = ax.inset_axes([1.05, 0.52, 0.05, 0.48])
cb_euler = fig.colorbar(im_euler, cax=cax_euler, orientation='vertical')
cb_euler.set_label(r'$\omega$', fontsize=25)
cax_mhd = ax.inset_axes([1.05, 0, 0.05, 0.48])
cb_mhd = fig.colorbar(im_mhd, cax=cax_mhd, orientation='vertical')
cb_mhd.set_label(r'$\log(B^2)$', fontsize=25)
for cb in [cb_euler, cb_mhd]:
    for tick in cb.ax.get_yticklabels():
        tick.set_fontsize(15)
        tick.set_fontfamily('serif')
    cb.ax.yaxis.get_offset_text().set(size=15)
    cb.ax.tick_params(labelsize=15)

plt.show()
