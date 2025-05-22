# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 11:47:27 2025

@author: karol
"""
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from numba import njit
"""Extracting, chunking and interpolating data"""

file_path_f = os.path.join(os.path.dirname(__file__), '..', 'data', 'Trans_BL_Front_22Tsteps.nc')
"""OBS remember to change L when changing from front and back!!!!"""
file_path_f = os.path.normpath(file_path_f)  # Normalize for cross-platform compatibility
ds_f = xr.open_dataset(file_path_f)
print(ds_f.dims)
print(ds_f)
print(ds_f.z.shape)

# Starting with chunks since the dataset is too large
u_chunk = ds_f['U'].isel(time=slice(0, 15), z=slice(
    0, 150), y=slice(0, 200), x=slice(0, 200))
v_chunk = ds_f['V'].isel(time=slice(0, 15), z=slice(
    0, 150), y=slice(0, 200), x=slice(0, 200))
w_chunk = ds_f['W'].isel(time=slice(0, 15), z=slice(
    0, 150), y=slice(0, 200), x=slice(0, 200))
x_chunk_front = ds_f['x'].isel(x=slice(0, 200))
y_chunk_front = ds_f['y'].isel(y=slice(0, 200))
z_chunk_front = ds_f['z'].isel(z=slice(0, 150))
# Reference 'full time' with a space
time_chunk_front = ds_f['time'].isel(**{"full time": slice(0, 15)})

# Grid for interpolating
x_new = x_chunk_front.values
y_new = y_chunk_front.values
z_new = z_chunk_front.values
time_new = time_chunk_front.values

# Periodic boundary conditions
periodic_x = False
periodic_y = False
periodic_z = False
periodic_t = False
periodic = np.array([periodic_x, periodic_y,periodic_z, periodic_t])
# Unsteady velocity field
bool_unsteady = True
## Compute meshgrid of dataset
X,Y,Z=np.meshgrid(x_new,y_new,z_new)

#Controlling the non-interpolated velocity field
plt.figure(figsize=(10, 3))
plt.contourf(x_new, z_new, u_chunk.isel(y=0, time=10).values, levels=50, cmap='inferno')
plt.title(f"u_chunk at time=0 (internal time: {u_chunk.time.values[0]})")
plt.colorbar()
plt.show()

#Manual interpolation that works with numba
@njit
def trilinear_interpolation_4D(z,y,x, t, grid_z, grid_y, grid_x, grid_t, values):
    """
    Perform trilinear interpolation in 3D with optional time interpolation (4D).
    
    Parameters:
    - x, y, z, t: Query points
    - grid_x, grid_y, grid_z, grid_t: 1D arrays representing the grid coordinates
    - values: 4D NumPy array of function values (Nz, Ny, Nx, Nt)
    
    Returns:
    - Interpolated value at (x, y, z, t)
    """
    # Find indices of lower corners
    i = np.searchsorted(grid_x, x) - 1
    j = np.searchsorted(grid_y, y) - 1
    k = np.searchsorted(grid_z, z) - 1
    l = np.searchsorted(grid_t, t) - 1

    # Clip indices to stay within bounds
    i = max(0, min(i, len(grid_x) - 2))
    j = max(0, min(j, len(grid_y) - 2))
    k = max(0, min(k, len(grid_z) - 2))
    l = max(0, min(l, len(grid_t) - 2))



    # Grid coordinates
    x1, x2 = grid_x[i], grid_x[i + 1]
    y1, y2 = grid_y[j], grid_y[j + 1]
    z1, z2 = grid_z[k], grid_z[k + 1]
    t1, t2 = grid_t[l], grid_t[l + 1]

    # Compute interpolation weights
    xd = (x - x1) / (x2 - x1) if x2 > x1 else 0.0
    yd = (y - y1) / (y2 - y1) if y2 > y1 else 0.0
    zd = (z - z1) / (z2 - z1) if z2 > z1 else 0.0
    td = (t - t1) / (t2 - t1) if t2 > t1 else 0.0

    # Get surrounding values in 4D
    c0000 = values[k, j, i, l]
    c1000 = values[k, j, i + 1, l]
    c0100 = values[k, j + 1, i, l]
    c1100 = values[k, j + 1, i + 1, l]
    c0010 = values[k + 1, j, i, l]
    c1010 = values[k + 1, j, i + 1, l]
    c0110 = values[k + 1, j + 1, i, l]
    c1110 = values[k + 1, j + 1, i + 1, l]

    c0001 = values[k, j, i, l + 1]
    c1001 = values[k, j, i + 1, l + 1]
    c0101 = values[k, j + 1, i, l + 1]
    c1101 = values[k, j + 1, i + 1, l + 1]
    c0011 = values[k + 1, j, i, l + 1]
    c1011 = values[k + 1, j, i + 1, l + 1]
    c0111 = values[k + 1, j + 1, i, l + 1]
    c1111 = values[k + 1, j + 1, i + 1, l + 1]

    # Trilinear interpolation for both time slices
    c00 = (c0000 * (1 - xd) + c1000 * xd) * (1 - yd) + (c0100 * (1 - xd) + c1100 * xd) * yd
    c01 = (c0010 * (1 - xd) + c1010 * xd) * (1 - yd) + (c0110 * (1 - xd) + c1110 * xd) * yd
    c0 = c00 * (1 - zd) + c01 * zd

    c00 = (c0001 * (1 - xd) + c1001 * xd) * (1 - yd) + (c0101 * (1 - xd) + c1101 * xd) * yd
    c01 = (c0011 * (1 - xd) + c1011 * xd) * (1 - yd) + (c0111 * (1 - xd) + c1111 * xd) * yd
    c1 = c00 * (1 - zd) + c01 * zd

    return c0 * (1 - td) + c1 * td

# Interpolating the fluid velocity field
@njit
def velocity_field(t, x, z_new, y_new, x_new, W, V, U, periodic, bool_unsteady):
    """
    Compute the velocity field using trilinear interpolation.
    """
    #x = np.array(x)
    x = np.asarray(x, dtype=np.float64)
    x_swap = np.zeros((x.shape[1], x.shape[0]))
    x_swap[:, 1] = x[1, :]
    x_swap[:, 0] = x[0, :]
    x_swap[:, 2] = x[2, :]

    # Handle periodic boundaries
    # if periodic[0]:
    #     x_swap[:, 1] = (x[0, :] - X[0]) % (X[-1] - X[0]) + X[0]

    # if periodic[1]:
    #     x_swap[:, 0] = (x[1, :] - Y[0]) % (Y[-1] - Y[0]) + Y[0]

    # if periodic[2]:
    #     x_swap[:, 2] = (x[2, :] - Z[0]) % (Z[-1] - Z[0]) + Z[0]

    # Interpolate velocity fields
    w_vals = np.array([trilinear_interpolation_4D(xi[0], xi[1], xi[2], t, z_new,y_new,x_new, time_new, W) for xi in x_swap])
    v_vals = np.array([trilinear_interpolation_4D(xi[0], xi[1], xi[2], t, z_new,y_new,x_new, time_new, V) for xi in x_swap])
    u_vals = np.array([trilinear_interpolation_4D(xi[0], xi[1], xi[2], t, z_new,y_new,x_new, time_new, U) for xi in x_swap])


    return w_vals, v_vals, u_vals

"""OBS remember to change L depending on front or back!"""
rho_f = 1.293  # Fluid density
rho_p = 80
d = 150e-6
g = 9.81
beta = (3 * rho_f) / (2 * rho_p + rho_f)
C_0 = 0.6
delta = 5.83
L = 0.095 # charac. length scale front
#L = 1.48 #length scale back
l = d/L
alpha = np.sqrt(l)
d_hat=d/L
U_re = 0.12 #Front U
nu_re = 1.48e-5 #realistic ATM nu
mu = nu_re * rho_f
g_hat = g * L/(U_re**2)
Re = U_re*L/nu_re
tau_p= rho_p*d**2/(18*mu)
tau_f=L/U_re
St=tau_p/tau_f

u_chunk = np.ascontiguousarray(u_chunk)
v_chunk = np.ascontiguousarray(v_chunk)
w_chunk = np.ascontiguousarray(w_chunk)

t0 = 0
tN = 13
dt = .01

time = np.linspace(t0, tN, int((tN - t0) / dt) + 1, dtype=np.float64)
time = np.array(time, dtype=np.float64)

num_particles = 10000

# Select equally spaced indices from x_new, y_new, z_new
x_indices = np.linspace(0, len(x_new) - 3, 100, dtype=int)  # Index positions
y_indices = np.linspace(0, len(y_new) - 3, 100, dtype=int)  # Index positions
z_indices = np.linspace(len(z_new)/2, len(z_new) +12, 1, dtype=int)  # Index positions

# Extract actual grid values
x_selected = x_new[x_indices]
y_selected = y_new[y_indices]
z_selected = z_new[-3] #start position at 7

X_grid, Y_grid, Z_grid = np.meshgrid(x_selected, y_selected, z_selected, indexing="ij")

# Flatten arrays to create (3, num_particles) shape
x0 = np.vstack((Z_grid.ravel(), Y_grid.ravel(), X_grid.ravel()))

#%%
"""Choose here the data to plot, remember to be in the correct folder depending on front or back dataset"""
data = np.load('Fmap_MR_10000_rho80_d0.00015_T13.npz')
#data = np.load('Fmap_back_MR_10000_rho140_d5e-05_T13.npz')

Fmap_MR=data['Fmap_MR']

data_ROM = np.load('ROM_test_80_150.npz')
Fmap=data_ROM['Fmap']
DFDt=data_ROM['DFDt']
#%%

X_start, Y_start, Z_start = Fmap_MR[1, 2, :], Fmap_MR[1, 1, :], Fmap_MR[1, 0, :]
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_start, Y_start, Z_start, cmap="viridis", s=2, alpha=0.7)

ax.set_xlabel("X (Streamwise)")
ax.set_ylabel("Y (Spanwise)")
ax.set_zlabel("Z (Vertical)")
ax.set_title(f'Initial positions of particles at t=0')
ax.set_zlim(z_selected-5, z_selected+0.5)
X_final, Y_final, Z_final = Fmap_MR[-1, 2, :], Fmap_MR[-1, 1, :], Fmap_MR[-1, 0, :]
#X_final, Y_final, Z_final = Fmap_MR[-1:, 2, :], Fmap_MR[-1:, 1, :], Fmap_MR[-1:, 0, :]
fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(111, projection='3d')

sc=ax.scatter(X_final, Y_final, Z_final, c=Z_final, cmap="viridis", s=1, alpha=0.7)  # Colored by Z height
cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)  # Adjust size if needed
cbar.set_label("Final Z Position")  # Label for the color bar
#ax.set_zlim(1.2,1.9)
ax.set_xlabel('X (Streamwise)')
ax.set_ylabel('Y (Spanwise)')
ax.set_zlabel('Z (Vertical)')
#ax.set_title(rf'MR Final positions of particles at t=13, $\rho_p$={80}$kg/m^3$, $\hat d=${d_hat*10**3:.3}$mm$, N=10000')
ax.set_title(rf'MR Final positions of particles at t=13, $\rho_p$={80}$kg/m^3$, $d=${d*10**3:.3}$mm$, N=10000')

plt.show()
#%%
velocities_MR= Fmap_MR[:, 3:, :]
z_settling_MR = velocities_MR[:,0,:]

#Check for NaNs
print(np.isnan(z_settling_MR).sum())



from scipy.optimize import fsolve
"""Terminal velocity for analysis"""
def solve_velocity(alpha, beta, C0, g, delta0, Re):
    """
    Solves for v in the corrected equation using numerical root finding.

    Returns:
    v_solution : float - Computed velocity
    """
    A = (2 * alpha**2 * (1 - beta) * g) / (beta * C0)

    def velocity_equation(v):
        return A - v**2 - (2 * delta0 * v**(3/2)) / (alpha * np.sqrt(Re)) - (delta0**2 * v) / (alpha**2 * Re)

    # Provide an initial guess for v (should be positive)
    v_initial_guess = np.sqrt(A)  # Approximate initial guess

    # Solve for v using numerical root-finding
    v_solution = fsolve(velocity_equation, v_initial_guess)
    
    return v_solution[0]  # Return the numerical solution


# Solve for v
v_result = solve_velocity(alpha, beta, C_0, g_hat, delta, Re)

print(f"Computed velocity v = {v_result:.6f}")
V_terminal = -v_result

"""Terminal v PDF MR"""
MR_indices = {
    #1: 0,
    2: 100,
    3: 200,
    4: 300,
    5: 400,
    6: 500,
    7: 600,
    8: 700,
    9: 800,
    10: 900,
    11: 1000,
    12: 1100,
    13: 1200,
    14: -1
}

# Extract z_settling_ROM values
z_settling_MR_dt = {key: velocities_MR[idx, 0, :] for key, idx in MR_indices.items()}

# Compute V_star_flat
V_star_flat = {key: ((z - V_terminal) / V_terminal).flatten() for key, z in z_settling_MR_dt.items()}

MR_keys = list(range(2, 15)) #original to 15
T_values = [k - 1 for k in MR_keys]  # T=0 to T=13

bin_count = 250
hist_range = (-0.25, 0.25)

cmap = cm.get_cmap("Blues")
norm = mcolors.Normalize(vmin=min(T_values), vmax=max(T_values))
sm = cm.ScalarMappable(cmap=cmap, norm=norm)

plt.figure(figsize=(8, 6))

for key, T in zip(MR_keys, T_values):
    z_data = velocities_MR[MR_indices[key], 0, :]
    v_star = ((z_data - V_terminal) / V_terminal).flatten()
    
    color = cmap(norm(T))
    plt.hist(
        v_star,
        bins=bin_count,
        range=hist_range,
        density=True,
        histtype='step',
        color=color,
        lw=2.5,           # thicker lines look smoother
        alpha=0.9         # slightly transparent lines help reduce visual clutter
    )
plt.axvspan(0, hist_range[1], color="grey", alpha=0.3)

# Adding colorbar
cbar = plt.colorbar(sm, label="T")
plt.xlabel(r"$(V_s - V_t) / V_t$")
plt.ylabel("PDF")
plt.title("PDF of Relative Difference of MR $V_s$ to $V_t$")
plt.grid(True)
plt.tight_layout()
plt.show()


mean_values_MR = []

for t in range(velocities_MR.shape[0]):
    z_data_MR = velocities_MR[t, 0, :]  # shape: (N,)
    v_star_MR = (z_data_MR - V_terminal) / V_terminal
    mean_values_MR.append(np.mean(v_star_MR))

mean_values_MR = np.array(mean_values_MR)
plt.figure(figsize=(6, 4))
plt.plot(np.arange(len(mean_values_MR[1:])), mean_values_MR[1:], linestyle='-', color='black')
plt.xlabel("T")
plt.ylabel(r"$(V_s - V_t) / V_t$")
plt.title("Mean")
plt.grid(True)
plt.tight_layout()
plt.show()


#%%
"""Finding time varying particle reynolds number and turbulence intensity"""
def compute_slip_velocity_MR(time, Fmap_MR,
                          z_new, y_new, x_new,
                          w_chunk, v_chunk, u_chunk,
                          periodic, bool_unsteady):
    """
    Compute slip velocity u_f - v_p at each time step for all particles.

    Parameters:
        time: array of length Nt
        Fmap: (Nt, 6, Np), positions and velocities:
              Fmap[:, 0:3, :] -> positions
              Fmap[:, 3:6, :] -> particle velocities
        *_chunk: velocity field interpolants
        *_new: interpolation grid
        periodic: periodicity in [z, y, x]
        bool_unsteady: whether velocity field is time-dependent
    
    Returns:
        slip_vel: (Nt, 3, Np), slip velocity = fluid velocity - particle velocity
    """
    Nt, _, Np = Fmap_MR.shape
    slip_vel = np.zeros((Nt, 3, Np))
    u_f_MR = np.zeros((Nt, 3, Np))
    for i in range(Nt):
        t = time[i]
        pos = Fmap_MR[i, 0:3, :]      # shape (3, Np)
        v_p = Fmap_MR[i, 3:6, :]      # shape (3, Np)

        # Interpolate fluid velocity at particle positions
        u_f = velocity_field(t, pos, z_new, y_new, x_new,
                             w_chunk, v_chunk, u_chunk,
                             periodic, bool_unsteady)  # shape (3, Np)

        slip_vel[i] = u_f - v_p
        u_f_MR[i] = u_f
        u_prime_x = np.std(u_f[0])  # fluctuations in x-component
        u_prime_y = np.std(u_f[1])  # y-component
        u_prime_z = np.std(u_f[2])  # z-component
        
        u_rms_MR = u_prime_z
        #u_rms_MR = np.sqrt((u_prime_x**2 + u_prime_y**2 + u_prime_z**2) / 3)
        U_mean = np.mean(np.linalg.norm(u_f, axis=0))  # mean of speed magnitudes across all particles
        I = u_rms_MR / U_mean
        
    return slip_vel, I, u_f_MR

slip_v_MR, I, u_f_MR = compute_slip_velocity_MR(time, Fmap_MR,
                          z_new, y_new, x_new,
                          w_chunk, v_chunk, u_chunk,
                          periodic, bool_unsteady)

Re_p = np.linalg.norm(slip_v_MR, axis=1) * d / nu_re
sigma_vt_MR = I/v_result

#%%

z_fluid_v=u_f_MR[:,0,:].flatten()
mean_data = np.mean(z_fluid_v)
plt.figure(figsize=(8,6))
plt.hist(z_fluid_v, bins=250, density=True, histtype='step', color='blue', linewidth=1)
plt.axvline(mean_data, color='red', linestyle='--', label=f'Mean = {mean_data:.4f}', linewidth=0.8)
plt.xlabel('Vertical fluid velocity along particle paths')
plt.ylabel('Probability density')
plt.title('Back region vertical fluid velocity')
plt.xlim(-0.25,0.25)
plt.tight_layout()
plt.grid(True)
plt.legend()
plt.show()

#%%
X_final, Y_final, Z_final = Fmap_MR[-1, 2, :], Fmap_MR[-1, 1, :], Fmap_MR[-1, 0, :]
slip_final_Z=slip_v_MR[-1, 0, :]
fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(111, projection='3d')

sc=ax.scatter(X_final, Y_final, Z_final, c=slip_final_Z, cmap="viridis", s=1, alpha=0.7)  # Colored by Z height
cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)  # Adjust size if needed
cbar.set_label("Final Z Position")  # Label for the color bar
#ax.set_zlim(1.2,1.9)
ax.set_xlabel('X (Streamwise)')
ax.set_ylabel('Y (Spanwise)')
ax.set_zlabel('Z (Vertical)')
ax.set_title(rf'MR Final positions of particles at t=13, $\rho_p$={80}$kg/m^3$, $\hat d=${d_hat*10**3:.3}$mm$, N=10000')

plt.show()
#%%
"""Plotting mean vs. Particle Reynolds number to see if loitering is due to nonlinear drag"""
mean_values = []

for t in range(velocities_MR.shape[0]):
    z_data = velocities_MR[t, 0, :]  # shape: (N,)
    v_star = (z_data - V_terminal) / V_terminal
    mean_values.append(np.mean(v_star))

mean_values = np.array(mean_values)
percent_above_zero = 100 * np.sum(mean_values > 0) / len(mean_values)
plt.figure(figsize=(6, 4))
plt.plot(np.arange(len(mean_values)), mean_values, linestyle='-', color='black')
plt.xlabel("T")
plt.ylabel(r"$(V_s - V_t) / V_t$")
plt.title("Mean")
plt.text(0.3, 0.95, f"Mean Re_p={np.mean(Re_p[1:,0]):.3f}% ", ha='right', va='bottom', 
         transform=plt.gca().transAxes, fontsize=10, color='red')
plt.text(0.85, 0.95, f"{percent_above_zero:.1f}% > 0", ha='right', va='top', 
         transform=plt.gca().transAxes, fontsize=10, color='darkgreen')
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
"""Terminal v ROM"""
rom_indices = {
    1: 0,
    2: 100,
    3: 200,
    4: 300,
    5: 400,
    6: 500,
    7: 600,
    8: 700,
    9: 800,
    10: 900,
    11: 1000,
    12: 1100,
    13: 1200,
    14: -1
}

# Extract z_settling_ROM values
z_settling_ROM = {key: DFDt[idx, 0, :] for key, idx in rom_indices.items()}

# Compute V_star_flat
#V_terminal= -0.313
V_star_flat = {key: ((z - V_terminal) / V_terminal).flatten() for key, z in z_settling_ROM.items()}

rom_keys = list(range(1, 15))
T_values = [k - 1 for k in rom_keys]  # T=0 to T=13

bin_count = 250
hist_range = (-0.65,0.65)

# Set up color map
cmap = cm.get_cmap("Greens")  # Use cividis instead of string
norm = mcolors.Normalize(vmin=min(T_values), vmax=max(T_values))
sm = cm.ScalarMappable(cmap=cmap, norm=norm)

plt.figure(figsize=(8, 6))

for key, T in zip(rom_keys, T_values):
    z_data = DFDt[rom_indices[key], 0, :]
    v_star = ((z_data - V_terminal) / V_terminal).flatten()
    
    color = cmap(norm(T))
    plt.hist(
        v_star,
        bins=bin_count,
        range=hist_range,
        density=True,
        histtype='step',
        color=color,
        lw=2.5,           # thicker lines look smoother
        alpha=0.9 
    )
plt.axvspan(0, hist_range[1], color="grey", alpha=0.3)

# Add colorbar
cbar = plt.colorbar(sm, label="T")
#cbar.set_ticks(np.linspace(min(T_values), max(T_values), 7))  # optional: adjust tick spacing

plt.xlabel(r"$(V_s - V_t) / V_t$")
plt.ylabel("PDF")
plt.title("PDF of Relative Difference of ROM $V_s$ to $V_t$")
plt.grid(True)
plt.tight_layout()
plt.show()


mean_values_ROM = []

for t in range(DFDt.shape[0]):
    z_data_ROM = DFDt[t, 0, :]  # shape: (N,)
    v_star_ROM = (z_data_ROM - V_terminal) / V_terminal
    mean_values_ROM.append(np.mean(v_star_ROM))

mean_values_ROM = np.array(mean_values_ROM)



def compute_slip_velocity(time, Fmap, dFdt,
                          z_new, y_new, x_new,
                          w_chunk, v_chunk, u_chunk,
                          periodic, bool_unsteady):
    """
    Compute slip velocity u_f - v_p at each time step for all particles,
    and the fluid velocity RMS at the particle positions.

    Returns:
        slip_vel: (Nt-1, 3, Np), fluid velocity - particle velocity
        u_rms:    (Nt-1, Np), RMS fluid velocity magnitude at particle positions
    """
    Nt_1, _, Np = dFdt.shape
    slip_vel = np.zeros((Nt_1, 3, Np))
    u_rms = np.zeros((Nt_1, Np))  # store RMS velocity magnitudes
    u_f_all = np.zeros((Nt_1, 3, Np))
    for i in range(Nt_1):
        t = time[i]
        pos = Fmap[i]  # shape (3, Np)

        u_f = velocity_field(t, pos, z_new, y_new, x_new,
                             w_chunk, v_chunk, u_chunk,
                             periodic, bool_unsteady)  # shape: (3, Np)

        slip_vel[i] = dFdt[i] - u_f
        u_f_all[i] = u_f
        # Compute fluid velocity magnitude at each particle position
        #u_rms[i] = np.linalg.norm(u_f, axis=0)  # shape (Np,)
        u_prime_x = np.std(u_f[0])  # fluctuations in x-component
        u_prime_y = np.std(u_f[1])  # y-component
        u_prime_z = np.std(u_f[2])  # z-component
        
        u_rms = u_prime_z
        #u_rms = np.sqrt((u_prime_x**2 + u_prime_y**2 + u_prime_z**2) / 3)
        U_mean = np.mean(np.linalg.norm(u_f, axis=0))  # mean of speed magnitudes across all particles
        I = u_rms / U_mean
    return slip_vel, I, u_f_all

slip_v, I_ROM, u_f = compute_slip_velocity(time, Fmap, DFDt,
                          z_new, y_new, x_new,
                          w_chunk, v_chunk, u_chunk,
                          periodic, bool_unsteady)

Re_p_ROM = np.linalg.norm(slip_v, axis=1) * d / nu_re
sigma_vt_ROM= I_ROM/v_result

print(f'ROM slip={np.abs(np.mean(slip_v))}')
print(f'MR slip={np.abs(np.mean(slip_v_MR))}')
#%%
plt.rcParams.update({
    'font.size': 15,          # Base font size (ticks)
    'axes.titlesize': 14,     # Figure title
    'axes.labelsize': 14,     # Axis labels
    'xtick.labelsize': 14,    # X tick labels
    'ytick.labelsize': 14,    # Y tick labels
    'legend.fontsize': 14,    # Legend text
    'legend.title_fontsize': 14
})
z_fluid_v=u_f[:,0,:].flatten()
mean_data = np.mean(z_fluid_v)
plt.figure(figsize=(8,6))
plt.hist(z_fluid_v, bins=750, density=True, histtype='step', color='blue', linewidth=1.2)
plt.axvline(mean_data, color='red', linestyle='--', label=f'Mean = {mean_data:.4f}', linewidth=1)
plt.xlabel('Vertical fluid velocity along particle path')
plt.ylabel('Probability density')
plt.title('Front region vertical fluid velocity along particle path')
plt.xlim(-0.25,0.25)
plt.tight_layout()
plt.grid(True)
plt.legend()
plt.show()

#%%
bias = mean_values_ROM - mean_values_MR[1:]
print(f'sigma/v_t=ROM_case!{I_ROM/0.564}')
print(f'sigma/v_t_MR: {sigma_vt_MR}')
print(f'sigma/v_t_ROM: {sigma_vt_ROM}')
print(f'BIAS ROM vs MR: {np.mean(bias)*100:.2f} %')
print(f'Mean $Re_p$ ROM ={np.mean(Re_p_ROM[1:,0]):.2f}')
print(f'Mean $Re_p$ MR ={np.mean(Re_p[1:,0]):.2f}')
plt.figure(figsize=(6, 4))
plt.plot(np.arange(len(mean_values_ROM)), mean_values_ROM, linestyle='-', color='green', label= 'ROM')
plt.plot(np.arange(len(mean_values_ROM[1:])), mean_values_MR[2:], linestyle='-', color='blue', label = 'MR')
plt.xlabel("T")
plt.ylabel(r"$(V_s - V_t) / V_t$")
plt.title("Mean relative difference ROM vs MR")
# plt.text(0.4, 0.75, rf"Mean $Re_p$ MR ={np.mean(Re_p[1:,0]):.2f}% ", ha='right', va='bottom', 
#          transform=plt.gca().transAxes, fontsize=10, color='blue')
# plt.text(0.4, 0.7, rf"Mean $Re_p$ ROM ={np.mean(Re_p_ROM[1:,0]):.2f}% ", ha='right', va='bottom', 
#          transform=plt.gca().transAxes, fontsize=10, color='red')
plt.grid(True)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.legend()
plt.tight_layout()
plt.show()

#%%
#%%



fig, ax1 = plt.subplots(figsize=(8, 5))

percent_above_zero = 100 * np.sum(mean_values > 0) / len(mean_values)
color1 = 'tab:blue'
ax1.set_xlabel('T')
ax1.set_ylabel(r'$(V_s - V_z) / V_t$', color=color1)
ax1.plot(np.arange(len(mean_values)), mean_values, color=color1, label='Mean Relative Difference')
ax1.tick_params(axis='y', labelcolor=color1)
plt.text(0.95, 0.95, f"{percent_above_zero:.1f}% > 0", ha='right', va='top', 
         transform=plt.gca().transAxes, fontsize=10, color='darkgreen')
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel(r'$Re_p$', color=color2)
ax2.plot(np.arange(len(Re_p_ROM)), slip_v[:, 0,0], color=color2, label=r'$Re_p$')
ax2.tick_params(axis='y', labelcolor=color2)

fig.tight_layout()
plt.title(rf"Mean Relative Difference and $Re_p$ d={d*10**6}$\mu m$, $\rho$={rho_p}")
plt.grid(True)
plt.show()
#%%
"""This data is already pre-calculated in this code and saved in a text file manually
Here I am just plotting the data"""
import pandas as pd


data = [
    (80, 50, 'Back', 0.01738, 0.01744),
    (80, 50, 'Front', 0.01738, 0.01744),
    (80, 100, 'Back', 0.06268, 0.06427),
    (80, 100, 'Front', 0.06268, 0.06427),
    (80, 150, 'Back', 0.1210, 0.13170),
    (80, 150, 'Front', 0.1210, 0.13170),
    (140, 50, 'Back', 0.03009, 0.03026),
    (140, 50, 'Front', 0.03009, 0.03026),
    (140, 100, 'Back', 0.10433, 0.10903),
    (140, 100, 'Front', 0.10433, 0.10903),
    (140, 150, 'Back', 0.1880, 0.21853),
    (140, 150, 'Front', 0.1880, 0.21777),
]

Re_p_data = [
    (80, 50, 'Back', 0.18, 0.18),
    (80, 50, 'Front', 0.18, 0.18),
    (80, 100, 'Back', 1.30, 1.27),
    (80, 100, 'Front', 1.30, 1.27),
    (80, 150, 'Back', 4.00, 4.00),
    (80, 150, 'Front', 4.35, 4.00),
    (140, 50, 'Back', 0.32, 0.31),
    (140, 50, 'Front', 0.32, 0.31),
    (140, 100, 'Back', 2.37, 2.21),
    (140, 100, 'Front', 2.37, 2.21),
    (140, 150, 'Back', 7.27, 6.64),
    (140, 150, 'Front', 7.27, 6.64)
]


df = pd.DataFrame(data, columns=['rho', 'd', 'region', 'ROM', 'MR'])

# Create a DataFrame from Re_p data
re_p_df = pd.DataFrame(
    Re_p_data,
    columns=['rho', 'd', 'region', 'Re_p_ROM', 'Re_p_MR']
)

# Merge into your existing df by matching rho, d, region
df = pd.merge(df, re_p_df, on=['rho', 'd', 'region'])

# Create label for each case
df['label'] = df.apply(lambda row: f"{int(row['rho'])}/{int(row['d'])} {row['region']}", axis=1)

x = np.arange(len(df))  # label positions
width = 0.35  # bar width

fig, ax = plt.subplots(figsize=(12,6))
rects1 = ax.bar(x - width/2, df['ROM'], width, label='ROM', color='blue')
rects2 = ax.bar(x + width/2, df['MR'], width, label='MR', color='orange')

ax.set_ylabel('Mean absolute slip velocity $|u_s|$')
ax.set_title('Comparison of slip velocities')
ax.set_xticks(x)
ax.set_xticklabels(df['label'], rotation=45, ha='right')
ax.legend()
ax.grid(True, axis='y')

plt.tight_layout()
plt.show()

Re_p_data = [
    (80, 50, 'ROM', 0.18),
    (80, 50, 'MR', 0.18),
    (80, 100, 'ROM', 1.27),
    (80, 100, 'MR', 1.30),
    (80, 150, 'ROM', 3.68),
    (80, 150, 'MR', 4.00),
    (140, 50, 'ROM', 0.30),
    (140, 50, 'MR', 0.31),
    (140, 100, 'ROM', 2.11),
    (140, 100, 'MR', 2.21),
    (140, 150, 'ROM', 5.71),
    (140, 150, 'MR', 6.64),
]
df = pd.DataFrame(Re_p_data, columns=['rho', 'd', 'model', 'Re_p'])

density_color = {80: 'blue', 140: 'orange'}
model_style = {'ROM': '-', 'MR': '--'}

plt.figure(figsize=(8,6))

for rho in [80, 140]:
    for model in ['ROM', 'MR']:
        subset = df[(df['rho'] == rho) & (df['model'] == model)].sort_values(by='d')
        plt.plot(
            subset['d'],
            subset['Re_p'],
            linestyle=model_style[model],
            color=density_color[rho],
            marker='o' if model == 'ROM' else 's',
            markersize=7,
            label=f"$\\rho_p$={rho} {model}"
        )

plt.xlabel('Particle diameter $d_p$ [$\mu$m]', fontsize=12)
plt.ylabel('Mean particle Reynolds number $Re_p$', fontsize=12)
plt.title('Mean $Re_p$ vs Particle diameter', fontsize=14)
plt.xticks([50, 100, 150])
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title="Density / Model", fontsize=10)
plt.tight_layout()
plt.show()
#%%
# Create the meshgrid of your domain
X, Y, Z = np.meshgrid(x_new, y_new, z_new, indexing='ij')
X = X.astype(np.float64)
Y = Y.astype(np.float64)
Z = Z.astype(np.float64)

positions = np.vstack((Z.ravel(), Y.ravel(), X.ravel()))  # shape (3, N)

# Call your velocity_field function
a = velocity_field(
    time,
    positions,          # pass flattened positions
    z_new, y_new, x_new,
    w_chunk, v_chunk, u_chunk,
    periodic, bool_unsteady
)
#%%
W_field = w_chunk[:, :, :, :13]  # vertical velocity at time t

values = W_field.flatten()
mean_values=np.mean(values)
plt.figure(figsize=(8,6))
plt.hist(values, bins=750, density=True, histtype='step', color='blue', linewidth=1.5)
plt.axvline(mean_values, color='red', linestyle='--', label=f'Mean = {mean_values:.4f}', linewidth=0.8)
plt.xlabel('Vertical velocity $W$')
plt.ylabel('Probability Density')
plt.title('PDF of vertical velocity field back region')
plt.legend()
plt.grid(True)
plt.show()
