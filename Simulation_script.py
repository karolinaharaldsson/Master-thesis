# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 16:59:27 2025

@author: karol
"""

"""Script to get the positions and velocities of the simulations, stored and later used in plotting_data.py"""

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from numba import njit

"""The trajecotires and velocities for ROM are stored in Fmap and DFDt, respectively
while the trajectories and velocities for MR are stored in Fmap_MR together. """


"""Extracting, chunking and interpolating data"""

file_path_f = os.path.join(os.path.dirname(__file__), '..', 'data', 'Trans_BL_Back_22Tsteps.nc')
"""OBS remember to change L, dt and saved_timesteps when changing from front and back!!!!"""
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

#%%
"""This section tests the manual interpolation with xarrays built in interpolator (not working with numba)
to make sure its done correctly"""
U_interp = u_chunk.interp(x=x_new, y=y_new, z=z_new, time=time_new)
V_interp = v_chunk.interp(x=x_new, y=y_new, z=z_new, time=time_new)
W_interp = w_chunk.interp(x=x_new, y=y_new, z=z_new, time=time_new)
U_manual_interp = np.zeros((len(z_new), len(y_new), len(x_new), len(time_new)))

# Loop through all grid points and interpolate
for k in range(len(z_new)):       # Nz
    for j in range(len(y_new)):   # Ny
        for i in range(len(x_new)):  # Nx
            for l in range(len(time_new)):  # Nt
                U_manual_interp[k, j, i, l] = trilinear_interpolation_4D(
                    z_new[k], y_new[j], x_new[i], time_new[l],  # Query points (z, y, x, t)
                    z_new, y_new, x_new, time_new,  # Grid points
                    u_chunk # Velocity field array (Nz, Ny, Nx, Nt)
                )

print("Manual interpolation shape:", U_manual_interp.shape)

t_index = 13
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.contourf(x_new, y_new, U_interp.isel(time=t_index, z=0).values, levels=50, cmap='inferno')
plt.title("Xarray Interpolation (U_interp)")

plt.subplot(1, 2, 2)
plt.contourf(x_new, y_new, U_manual_interp[0, :, :, t_index], levels=50, cmap='inferno')
plt.title("Manual Interpolation (U_manual_interp)")

plt.show()

"""This has been tested and works"""
#%%

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
rho_p = 140
d = 150e-6
g = 9.81
beta = (3 * rho_f) / (2 * rho_p + rho_f)
C_0 = 0.6
delta = 5.83
#L = 0.095 # charac. length scale front
L = 1.48 #length scale back
l = d/L
alpha = np.sqrt(l)
d_hat=d/L
U_re = 0.12 
nu_re = 1.48e-5 #realistic ATM nu
mu = nu_re * rho_f
g_hat = g * L/(U_re**2)
Re = U_re*L/nu_re

# Making sure the fluid velocities works with numba
u_chunk = np.ascontiguousarray(u_chunk)
v_chunk = np.ascontiguousarray(v_chunk)
w_chunk = np.ascontiguousarray(w_chunk)

# Finite differences fore the material derivative
@njit
def fluid_acceleration(t, state):
    dx = (X[:, 1:, :] - X[:, :-1, :]).mean()
    dy = (Y[1:, :, :] - Y[:-1, :, :]).mean()
    dz = (Z[:, :, 1:] - Z[:, :, :-1]).mean()

    # Ensure `state` is shape (3, N)
    z = state[0, :]  # Keep as arrays
    y = state[1, :]
    x = state[2, :]

    # Get velocity components at the current position
    w_, v_, u_ = velocity_field(t, state, z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady)

    # Compute shifted positions for ALL particles at once
    w_x_prim_plus, v_x_prim_plus, u_x_prim_plus = velocity_field(
        t, np.vstack((z, y, x + dx)), z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady
    )

    w_x_prim_minus, v_x_prim_minus, u_x_prim_minus = velocity_field(
        t, np.vstack((z, y, x - dx)), z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady
    )

    w_y_prim_plus, v_y_prim_plus, u_y_prim_plus = velocity_field(
        t, np.vstack((z, y + dy, x)), z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady
    )

    w_y_prim_minus, v_y_prim_minus, u_y_prim_minus = velocity_field(
        t, np.vstack((z, y - dy, x)), z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady
    )

    w_z_prim_plus, v_z_prim_plus, u_z_prim_plus = velocity_field(
        t, np.vstack((z + dz, y, x)), z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady
    )

    w_z_prim_minus, v_z_prim_minus, u_z_prim_minus = velocity_field(
        t, np.vstack((z - dz, y, x)), z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady
    )

    # Compute spatial derivatives using central differences
    u_x = (u_x_prim_plus - u_x_prim_minus) / (2 * dx)
    u_y = (u_y_prim_plus - u_y_prim_minus) / (2 * dy)
    u_z = (u_z_prim_plus - u_z_prim_minus) / (2 * dz)

    v_x = (v_x_prim_plus - v_x_prim_minus) / (2 * dx)
    v_y = (v_y_prim_plus - v_y_prim_minus) / (2 * dy)
    v_z = (v_z_prim_plus - v_z_prim_minus) / (2 * dz)

    w_x = (w_x_prim_plus - w_x_prim_minus) / (2 * dx)
    w_y = (w_y_prim_plus - w_y_prim_minus) / (2 * dy)
    w_z = (w_z_prim_plus - w_z_prim_minus) / (2 * dz)

    # Compute time derivatives using central differences
    w_t_prim_plus, v_t_prim_plus, u_t_prim_plus = velocity_field(
        t + dt, state, z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady
    )

    w_t_prim_minus, v_t_prim_minus, u_t_prim_minus = velocity_field(
        t - dt, state, z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady
    )

    # Compute derivatives
    u_t = (u_t_prim_plus - u_t_prim_minus) / (2 * dt)
    v_t = (v_t_prim_plus - v_t_prim_minus) / (2 * dt)
    w_t = (w_t_prim_plus - w_t_prim_minus) / (2 * dt)

    # Compute the material derivative
    Du_Dt = u_t + u_ * u_x + v_ * u_y + w_ * u_z
    Dv_Dt = v_t + u_ * v_x + v_ * v_y + w_ * v_z
    Dw_Dt = w_t + u_ * w_x + v_ * w_y + w_ * w_z

    return Dw_Dt, Dv_Dt, Du_Dt

@njit
def particle_dynamics(t, x0, z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady):
    """
    The ROM model!
    Computes the particle dynamics at time t.
    
    Parameters:
        t: float, current time
        x0: array of shape (3,), the current position of the particle [z, y, x]
        X, Y, Z: Mesh grids for interpolation
        W_values, V_values, U_values: Interpolant objects for velocity components
        periodic: list of 3 booleans indicating periodic boundary conditions
        bool_unsteady: bool, whether the velocity field is unsteady
    
    Returns:
        dydt: array of shape (3,), the derivatives [dz/dt, dy/dt, dx/dt]
    """
    
    z, y, x = x0

    #FLuid velocity at current position
    u_f = np.vstack(velocity_field(t, x0, z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady))    
    # Material derivative
    Du_Dt = np.vstack(fluid_acceleration(t, x0))
    gravity_term = np.array([-g_hat, 0, 0]).reshape(3, 1)
    
    u_4 = (2 * Re / (beta * C_0 * delta**2)) * (1 - beta) * (gravity_term - Du_Dt)
    u_7 = (-2*np.sqrt(Re) / delta) * np.sqrt(np.abs(u_4)) * u_4
    v_p_new = u_f + alpha**4 * u_4 + alpha**7 * u_7
    
    dydt = v_p_new[0], v_p_new[1], v_p_new[2]  # Shape (3,)

    return dydt

"""The intergration part, here RK4 and intergration are from the TBarrier 
https://github.com/haller-group/TBarrier/blob/main/TBarrier/3D/subfunctions/integration/integration_dFdt.ipynb
proper citations can be found in the thesis section 3.2.2"""
@njit
def RK4_step(t, x1, dt, z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady):
    '''
    Advances the flow field by a single step given by u, v, w velocities, starting from given initial conditions. 
    The initial conditions can be specified as an array. 
    
    Parameters:
        time: array (Nt,),  time array  
        x: array (3,Npoints),  array of ICs
        X: array (NY, NX, NZ)  X-meshgrid
        Y: array (NY, NX, NZ)  Y-meshgrid 
        Z: array (NY, NX, NZ)  Z-meshgrid
        Interpolant_u: Interpolant object for u(x, t)
        Interpolant_v: Interpolant object for v(x, t)
        Interpolant_w: Interpolant object for w(x, t)
        periodic: list of 3 bools, periodic[i] is True if the flow is periodic in the ith coordinate.
        bool_unsteady:  specifies if velocity field is unsteady/steady
    
    Returns:

        y_update: array (3,Npoints), integrated trajectory (=flow map) 
        y_prime_update: array (3,Npoints), velocity along trajectories (=time derivative of flow map) 
    '''
    t0 = t # float
    
    # Compute x_prime at the beginning of the time-step by re-orienting and rescaling the vector field
    x_prime=np.vstack(particle_dynamics(t, x1, z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady)) # array(3, Nx*Ny*Nz)
    
    # compute derivative
    k1 = dt * x_prime # array(3, Nx*Ny*Nz)
    
    # Update position at the first midpoint.
    x2 = x1 + .5 * k1 # array(3, Nx*Ny*Nz)
     
    # Update time
    t = t0 + .5*dt # float
    
    # Compute x_prime at the first midpoint.
    x_prime = np.vstack(particle_dynamics(t, x2, z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady)) # array(3, Nx*Ny*Nz)
    
    # compute derivative
    k2 = dt * x_prime # array(3, Nx*Ny*Nz)

    # Update position at the second midpoint.
    x3 = x1 + .5 * k2 # array(3, Nx*Ny*Nz)
    
    # Update time
    t = t0 + .5*dt # float
    
    # Compute x_prime at the second midpoint.
    x_prime = np.vstack(particle_dynamics(t, x3, z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady)) # array(3, Nx*Ny*Nz)
    
    # compute derivative
    k3 = dt * x_prime # array(3, Nx*Ny*Nz)
    
    # Update position at the endpoint.
    x4 = x1 + k3 # array(3, Nx*Ny*Nz)
    
    # Update time
    t = t0+dt # float
    
    # Compute derivative at the end of the time-step.
    x_prime = np.vstack(particle_dynamics(t, x4, z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady)) # array(3, Nx*Ny*Nz)
    
    # compute derivative
    k4 = dt * x_prime
    
    # Compute RK4 derivative
    y_prime_update = 1.0 / 6.0*(k1 + 2 * k2 + 2 * k3 + k4) # array(3, Nx*Ny*Nz)
    
    # Integration y <-- y + y_primeupdate
    y_update = x1 + y_prime_update # array(3, Nx*Ny*Nz)
    #print(y_update.shape)
    #print(y_prime_update.shape)
    return y_update, y_prime_update/dt


@njit
def integration_dFdt(time, x0, z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady):
    '''
    Wrapper for RK4_step(). Advances the flow field given by u, v, w velocities, starting from given initial conditions. 
    The initial conditions can be specified as an array. 
    
    Parameters:
        time: array (Nt,),  time array  
        x: array (3, Npoints),  array of ICs
        X: array (NY, NX, NZ)  X-meshgrid
        Y: array (NY, NX, NZ)  Y-meshgrid 
        Z: array (NY, NX, NZ)  Z-meshgrid
        Interpolant_u: Interpolant object for u(x, t)
        Interpolant_v: Interpolant object for v(x, t)
        Interpolant_w: Interpolant object for w(x, t)
        periodic: list of 3 bools, periodic[i] is True if the flow is periodic in the ith coordinate.
        bool_unsteady:  specifies if velocity field is unsteady/steady
        verbose: bool, if True, function reports progress
    
    Returns:
        Fmap: array (Nt, 3, Npoints), integrated trajectory (=flow map)
        dFdt: array (Nt-1, 3, Npoints), velocity along trajectories (=time derivative of flow map) 
    '''
    # reshape x
    x = x0.reshape(3, -1) # reshape array (3, Nx*Ny*Nz)
    #x = np.asarray(x0, dtype=np.float64).reshape(3,-1)  # Ensure correct dtype

    # Initialize arrays for flow map and derivative of flow map
    Fmap = np.zeros((len(time), 3, x.shape[1])) # array (Nt, 3, Nx*Ny*Nz)
    dFdt = np.zeros((len(time)-1, 3, x.shape[1])) # array (Nt-1, 3, Nx*Ny*Nz)
    
    # Step-size
    dt = time[1]-time[0] # float
    
    counter = 0 # int

    # initial conditions
    Fmap[counter,:,:] = x
    
    # Runge Kutta 4th order integration with fixed step size dt
    for t in time[:-1]:
        # if verbose == True:
        #     if np.around((t-time[0])/(time[-1]-time[0])*1000,4)%10 == 0:
        #         print('Percentage completed: ', np.around(np.around((t-time[0])/(time[-1]-time[0]), 4)*100, 2))
        
        Fmap[counter+1,:, :], dFdt[counter,:,:] = RK4_step(t, Fmap[counter,:, :], dt, z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady)[:2]
    
        counter += 1
    
    return Fmap, dFdt

t0 = 0
tN = 13 #simulation time in DNS time units
dt = .01
# NOTE: For computing the backward trajectories set: tN < t0 and dt < 0.

time = np.linspace(t0, tN, int((tN - t0) / dt) + 1, dtype=np.float64)
time = np.array(time, dtype=np.float64)

num_particles = 10000

# An uniform grid of initial particle positions
x_indices = np.linspace(0, len(x_new) - 3, 100, dtype=int)  # Index positions
y_indices = np.linspace(0, len(y_new) - 3, 100, dtype=int)  # Index positions


# Extract actual grid values
x_selected = x_new[x_indices]
y_selected = y_new[y_indices]
z_selected = z_new[-3] #start position at 7

X_grid, Y_grid, Z_grid = np.meshgrid(x_selected, y_selected, z_selected, indexing="ij")

# Flatten arrays to create (3, num_particles) shape
x0 = np.vstack((Z_grid.ravel(), Y_grid.ravel(), X_grid.ravel()))
# arrays of the ROM
Fmap, DFDt = integration_dFdt(time, x0, z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady)
#%%
# Saving the files to later be plotted
sfilename = f"ROM_test_140_150.npz"

# Save data
np.savez(sfilename, Fmap=Fmap, DFDt=DFDt)
#%%
"""Full MR"""
@njit
def particle_dynamics_MR(t, x0, z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady):
    """
    Computes derivatives for position and velocity for multiple particles.
    Input: x0 shape (6, N)
    Output: shape (6, N)
    """
    z, y, x, vz, vy, vx = x0  # Each is shape (N,)
    pos = x0[:3,:]
    # Fluid velocity and acceleration
    u_f_MR = np.vstack(velocity_field(t, pos, z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady))  # shape (3, N)
    Du_Dt_MR = np.vstack(fluid_acceleration(t, pos))  # shape (3, N)

    # Particle velocity and relative velocity
    v_s = np.stack((vz, vy, vx))  # shape (3, N)
    relative_velocity = v_s - u_f_MR  # shape (3, N)
    # relative_velocity = np.array([[1.0],  # z-component
    #                [2.0],  # y-component
    #                [3.0]]) # x-component
    # assert relative_velocity.shape == (3, 1)
    gravity_term = (1 - beta) * np.array([-g_hat, 0, 0]).reshape(3, 1)  # shape (3, 1)
    relative_velocity_norm = np.sqrt(relative_velocity[0]**2 + relative_velocity[1]**2 + relative_velocity[2]**2)
    
    #relative_velocity_norm = np.where(relative_velocity_norm > 1e-6, relative_velocity_norm, 1e-6)

    acceleration = (
        beta * Du_Dt_MR
        + gravity_term
        - (beta * C_0) / (2 * alpha**2) * (
            1 + (2 * delta / (alpha * np.sqrt(Re * relative_velocity_norm)))
            + delta**2 / (alpha**2 * Re * relative_velocity_norm)
        ) * relative_velocity_norm * relative_velocity
    )  # shape (3, N)
    if t == 0 or t == dt:
        print("acceleration:", acceleration)

    return v_s, acceleration
tN=13
dt = 0.0001 #Largest possible timestep for MR not to blow up :(
time = np.arange(t0, tN+dt, dt)
@njit
def RK4_step_MR(t, x1, dt, z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady):
    """
    Runge-Kutta 4th order step for integrating position and velocity.
    """
    k1 = dt * np.vstack(particle_dynamics_MR(t, x1, z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady))
    k2 = dt * np.vstack(particle_dynamics_MR(t + 0.5 * dt, x1 + 0.5 * k1, z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady))
    k3 = dt * np.vstack(particle_dynamics_MR(t + 0.5 * dt, x1 + 0.5 * k2, z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady))
    k4 = dt * np.vstack(particle_dynamics_MR(t + dt, x1 + k3, z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady))

    y_update = x1 + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_update


@njit
def integration_dFdt_MR(time, x0, z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady):
    x = x0.reshape(6, -1)
    dt = time[1] - time[0]
    total_steps = len(time)

    save_every = 100  # Save every 100th timestep (or else too large for script to work)

    num_saved_steps = (total_steps + save_every - 1) // save_every
    Fmap_MR = np.zeros((num_saved_steps, 6, x.shape[1]))

    Fmap_MR[0, :, :] = x
    save_index = 1

    for i in range(total_steps - 1):
        x = RK4_step_MR(time[i], x, dt, z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady)

        # Save only every Nth step
        if (i + 1) % save_every == 0:
            if save_index < num_saved_steps:
                Fmap_MR[save_index, :, :] = x
                save_index += 1

    return Fmap_MR




num_particles = 10000

# Select equally spaced indices from x_new, y_new, z_new
x_indices = np.linspace(0, len(x_new) - 3, 100, dtype=int)  # Index positions
y_indices = np.linspace(0, len(y_new) - 3, 100, dtype=int)  # Index positions

# Extract actual grid values
x_selected = x_new[x_indices]
y_selected = y_new[y_indices]
z_selected = z_new[-3] # start at position 7

# Create 3D grid using the selected values
X_grid, Y_grid, Z_grid = np.meshgrid(x_selected, y_selected, z_selected, indexing="ij")

# Flatten arrays to create (3, num_particles) shape
x0 = np.vstack((Z_grid.ravel(), Y_grid.ravel(), X_grid.ravel()))
x0 = np.ascontiguousarray(x0)  # Ensure NumPy-compatible format
velocities=DFDt[0,:,:] #Initial velocities is ROM at first timestep
x0_MR = np.vstack((x0, velocities))  # shape (6, num_particles)

Fmap_MR = integration_dFdt_MR(time, x0_MR, z_new, y_new, x_new, w_chunk, v_chunk, u_chunk, periodic, bool_unsteady)

#%%
sfilename = f"Fmap_back_MR_{num_particles}_rho{rho_p}_d{d}_T{tN}.npz"

# Save data
np.savez(sfilename, Fmap_MR=Fmap_MR)
#%%
data = np.load('Fmap_back_MR_10000_rho140_d5e-05_T13.npz')
Fmap_MR=data['Fmap_MR']
data = np.load('ROM_back_10000_rho140_d5e-05_T13.npz')
#%%

from scipy.optimize import fsolve

def solve_velocity(alpha, beta, C0, g, delta0, Re):
    """
    Solves for v in the corrected equation using numerical root finding.

    Returns:
    v_solution : float - Computed velocity
    """
    A = (2 * alpha**2 * (1 - beta) * g) / (beta * C0)

    def velocity_equation(v):
        return A - v**2 - (2 * delta0 * v**(3/2)) / (alpha * np.sqrt(Re)) - (delta0**2 * v) / (alpha**2 * Re)

    
    v_initial_guess = np.sqrt(A)  # Approximate initial guess

    # Solve for v using numerical root-finding
    v_solution = fsolve(velocity_equation, v_initial_guess)
    
    return v_solution[0]

v_result = solve_velocity(alpha, beta, C_0, g_hat, delta, Re)

# Print the result
print(f"Computed velocity v = {v_result:.6f}")
V_terminal = -v_result

"""Terminal velocities ROM to see relative error between the two"""

def compute_terminal_velocity_ROM(Re, delta, beta, alpha, C_0):
    """
    Compute the terminal settling velocity of a particle in a quiescent fluid.

    Returns:
        v_terminal: np.ndarray of shape (3,)
    """
    gravity_term = np.array([-g_hat, 0, 0]).reshape(3, 1)

    u_4 = (2 * Re / (beta * C_0 * delta**2)) * (1 - beta) * gravity_term
    u_7 = (-np.sqrt(Re) / delta) * np.sqrt(np.abs(u_4)) * u_4

    # Final terminal velocity expression
    v_terminal = alpha**4 * u_4 + alpha**7 * u_7

    return v_terminal.flatten()

v_t_ROM=compute_terminal_velocity_ROM(Re, delta, beta, alpha, C_0)
print(v_t_ROM)

rel_er=np.linalg.norm(v_t_ROM[0] - V_terminal) / np.linalg.norm(V_terminal)
print(f"Relative error: {rel_er:.3%}")
#%%

