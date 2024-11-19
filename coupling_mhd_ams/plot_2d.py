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
    for i in range(96):
        for j in range(96):
            zz = field[4*i:4*(i+1), 4*j:4*(j+1)].copy()
            field[4*i:4*(i+1), 4*j:4*(j+1)] = interp(xx, yy, zz)
    return field


def extract_rho_lb(f, l_idx, b_idx):
    rho = f['variables_1']
    rho = np.reshape(rho, [4, 4, l_idx - 1, (b_idx - 1)], order='F')
    rho = np.swapaxes(rho, 1, 2)
    rho = np.reshape(rho, [4*(l_idx - 1), 4*(b_idx - 1)], order='F')
    return rho


def extract_rho_lm(f, l_idx, b_idx, t_idx):
    rho = f['variables_1']
    rho = np.reshape(rho, [4, 4, l_idx - 1, t_idx - b_idx + 1], order='F')
    rho = np.swapaxes(rho, 1, 2)
    rho = np.reshape(rho, [4*(l_idx - 1), 4*(t_idx - b_idx + 1)], order='F')
    return rho


def extract_rho_lt(f, l_idx, t_idx):
    rho = f['variables_1']
    rho = np.reshape(rho, [4, 4, l_idx - 1, 96 - t_idx], order='F')
    rho = np.swapaxes(rho, 1, 2)
    rho = np.reshape(rho, [4*(l_idx - 1), 4*(96 - t_idx)], order='F')
    return rho


def extract_rho_mb(f, l_idx, r_idx, b_idx):
    rho = f['variables_1']
    rho = np.reshape(rho, [4, 4, r_idx - l_idx + 1, (b_idx - 1)], order='F')
    rho = np.swapaxes(rho, 1, 2)
    rho = np.reshape(rho, [4*(r_idx - l_idx + 1), 4*(b_idx - 1)], order='F')
    return rho


def extract_rho_mt(f, l_idx, r_idx, t_idx):
    rho = f['variables_1']
    rho = np.reshape(rho, [4, 4, r_idx - l_idx + 1, 96 - t_idx], order='F')
    rho = np.swapaxes(rho, 1, 2)
    rho = np.reshape(rho, [4*(r_idx - l_idx + 1), 4*(96 - t_idx)], order='F')
    return rho


def extract_rho_rb(f, r_idx, b_idx):
    rho = f['variables_1']
    rho = np.reshape(rho, [4, 4, 96 - r_idx, (b_idx - 1)], order='F')
    rho = np.swapaxes(rho, 1, 2)
    rho = np.reshape(rho, [4*(96 - r_idx), 4*(b_idx - 1)], order='F')
    return rho


def extract_rho_rm(f, r_idx, b_idx, t_idx):
    rho = f['variables_1']
    rho = np.reshape(rho, [4, 4, 96 - r_idx, t_idx - b_idx + 1], order='F')
    rho = np.swapaxes(rho, 1, 2)
    rho = np.reshape(rho, [4*(96 - r_idx), 4*(t_idx - b_idx + 1)], order='F')
    return rho


def extract_rho_rt(f, r_idx, t_idx):
    rho = f['variables_1']
    rho = np.reshape(rho, [4, 4, 96 - r_idx, 96 - t_idx], order='F')
    rho = np.swapaxes(rho, 1, 2)
    rho = np.reshape(rho, [4*(96 - r_idx), 4*(96 - t_idx)], order='F')
    return rho


# Define the time index to plot
#time_idx = 0
time_idx = 14300

# Read the mesh files.
f = h5py.File('out_coupled/mesh_1_{0:09}.h5'.format(time_idx))
l_idx = f.attrs['size'][0] + 1
b_idx = f.attrs['size'][1] + 1
f.close()
f = h5py.File('out_coupled/mesh_2_{0:09}.h5'.format(time_idx))
r_idx = f.attrs['size'][0] + l_idx - 1
f.close()
f = h5py.File('out_coupled/mesh_4_{0:09}.h5'.format(time_idx))
t_idx = f.attrs['size'][1] + b_idx - 1
f.close()

# Read the data files.
# Euler left
f = h5py.File('out_coupled/solution_1_{0:09}.h5'.format(time_idx))
rho_lb = extract_rho_lb(f, l_idx, b_idx)
f.close()
f = h5py.File('out_coupled/solution_4_{0:09}.h5'.format(time_idx))
rho_lm = extract_rho_lm(f, l_idx, b_idx, t_idx)
f.close()
f = h5py.File('out_coupled/solution_7_{0:09}.h5'.format(time_idx))
rho_lt = extract_rho_lt(f, l_idx, t_idx)
f.close()

# MHD
f = h5py.File('out_coupled/solution_5_{0:09}.h5'.format(time_idx))
b_x_m = f['variables_6']
b_x_m = np.reshape(b_x_m, [4, 4, r_idx - l_idx + 1, (t_idx - b_idx + 1)], order='F')
b_x_m = np.swapaxes(b_x_m, 1, 2)
b_x_m = np.reshape(b_x_m, [4*(r_idx - l_idx + 1), 4*(t_idx - b_idx + 1)], order='F')
b_y_m = f['variables_7']
b_y_m = np.reshape(b_y_m, [4, 4, r_idx - l_idx + 1, (t_idx - b_idx + 1)], order='F')
b_y_m = np.swapaxes(b_y_m, 1, 2)
b_y_m = np.reshape(b_y_m, [4*(r_idx - l_idx + 1), 4*(t_idx - b_idx + 1)], order='F')
rho_m = f['variables_1']
rho_m = np.reshape(rho_m, [4, 4, r_idx - l_idx + 1, (t_idx - b_idx + 1)], order='F')
rho_m = np.swapaxes(rho_m, 1, 2)
rho_m = np.reshape(rho_m, [4*(r_idx - l_idx + 1), 4*(t_idx - b_idx + 1)], order='F')
f.close()

# Euler middle
f = h5py.File('out_coupled/solution_2_{0:09}.h5'.format(time_idx))
rho_mb = extract_rho_mb(f, l_idx, r_idx, b_idx)
f.close()
f = h5py.File('out_coupled/solution_8_{0:09}.h5'.format(time_idx))
rho_mt = extract_rho_mt(f, l_idx, r_idx, t_idx)
f.close()

# Euler right
f = h5py.File('out_coupled/solution_3_{0:09}.h5'.format(time_idx))
rho_rb = extract_rho_rb(f, r_idx, b_idx)
f.close()
f = h5py.File('out_coupled/solution_6_{0:09}.h5'.format(time_idx))
rho_rm = extract_rho_rm(f, r_idx, b_idx, t_idx)
f.close()
f = h5py.File('out_coupled/solution_9_{0:09}.h5'.format(time_idx))
rho_rt = extract_rho_rt(f, r_idx, t_idx)
f.close()

# MHD only
f = h5py.File('out_mhd_only/solution_{0:09}.h5'.format(time_idx))
rho_mhd = f['variables_1']
rho_mhd = np.reshape(rho_mhd, [4, 4, 96, 96], order='F')
rho_mhd = np.swapaxes(rho_mhd, 1, 2)
rho_mhd = np.reshape(rho_mhd, [4*96, 4*96], order='F')
f.close()

# Get everything into one array.
rho = np.zeros([96*4, 96*4])
b_x = np.zeros([96*4, 96*4]) + np.NaN
b_y = np.zeros([96*4, 96*4]) + np.NaN
rho[:(l_idx-1)*4, :(b_idx-1)*4] = rho_lb
rho[:(l_idx-1)*4, (b_idx-1)*4:t_idx*4] = rho_lm
rho[:(l_idx-1)*4, t_idx*4:] = rho_lt
rho[(l_idx-1)*4:r_idx*4, :(b_idx-1)*4] = rho_mb
rho[(l_idx-1)*4:r_idx*4, (b_idx-1)*4:t_idx*4] = rho_m
rho[(l_idx-1)*4:r_idx*4, t_idx*4:] = rho_mt
rho[r_idx*4:, :(b_idx-1)*4] = rho_rb
rho[r_idx*4:, (b_idx-1)*4:t_idx*4] = rho_rm
rho[r_idx*4:, t_idx*4:] = rho_rt
b_x[(l_idx-1)*4:r_idx*4, (b_idx-1)*4:t_idx*4] = b_x_m
b_y[(l_idx-1)*4:r_idx*4, (b_idx-1)*4:t_idx*4] = b_y_m

# Perform the 2d polynomial interpolation.
b_x = perform_interpolation(b_x)
b_y = perform_interpolation(b_y)
rho = perform_interpolation(rho)
rho_mhd = perform_interpolation(rho_mhd)

# Prepare the plot.
width = 8
height = 6
plt.rc('text', usetex=True)
plt.rc('font', family='arial')
plt.rc("figure.subplot", left=0.08)
plt.rc("figure.subplot", right=0.98)
plt.rc("figure.subplot", bottom=0.15)
plt.rc("figure.subplot", top=0.95)
fig = plt.figure(figsize=(width, height))
ax = fig.add_subplot(111)
plt.ion()

im = plt.imshow(rho.T, origin='lower', extent=[-3, 3, -3, 3], vmin=0.997, vmax=1.014)
#im = plt.imshow((b_x**2 + b_y**2).T, origin='lower', extent=[-3, 3, -3, 3], vmax=0.018, cmap='plasma')
#im = plt.imshow(abs(rho - rho_mhd).T, origin='lower', extent=[-3, 3, -3, 3], vmin=0)

# Plot the domain boundarier.
left = l_idx/16 - 3
right = r_idx/16 - 3
bottom = b_idx/16 - 3
top = t_idx/16 - 3
plt.plot([left, left], [-3, 3], color='r', linewidth=2)
plt.plot([right, right], [-3, 3], color='r', linewidth=2)
plt.plot([-3, 3], [bottom, bottom], color='r', linewidth=2)
plt.plot([-3, 3], [top, top], color='r', linewidth=2)

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
cax = ax.inset_axes([0.05, 1.4, 0.9, 0.1])
cb = fig.colorbar(im, orientation='vertical')
cb.set_label(r'$\rho$', fontsize=25)
#cb.set_label(r'$B^2$', fontsize=25)
#cb.set_label(r'$|\Delta\rho|$', fontsize=25)
cbytick_obj = plt.getp(cb.ax.axes, 'yticklabels')
for tick in cb.ax.get_yticklabels():
    tick.set_fontsize(15)
    tick.set_fontfamily('serif')
cb.ax.yaxis.get_offset_text().set(size=15)

plt.show()
