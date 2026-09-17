import numpy as np
import matplotlib.pyplot as plt

# Parameters
mu = 1.0
E = 1.0
eta = 5.0
delta = 1.0 / (4.0 * np.pi)
F_ext_mag = 0.05
t1 = 2.5
dt = 0.01
t_max = 5.0
no_of_prot = len(coms)

# Calculate engineering strain
def eng_strain(x_initial, x_current, P):
    displacements = np.linalg.norm(x_current - x_initial, axis=1)
    Nd = len(displacements)
    return np.sum(displacements) / (P * Nd)

# Set up node positions
positions = np.array([coms[pid] for pid in protein_ids])
x_coords = positions[:, 0]
sorted_indices = np.argsort(x_coords)
P = np.max(x_coords) - np.min(x_coords)

# Define fixed and moving nodes
fixed_nodes = sorted_indices[:50].tolist()
right_boundary = sorted_indices[-50:].tolist()
all_nodes = set(range(len(positions)))
moving_nodes = list(all_nodes - set(fixed_nodes))

N_F = len(fixed_nodes)
N_X = len(moving_nodes)

# Map protein IDs to indices
pid_to_index = {pid: i for i, pid in enumerate(protein_ids)}
connections = [(pid_to_index[pid1], pid_to_index[pid2]) for pid1, pid2 in edges]

# Store connection indices
conn_arr = np.array(connections)
idx_i = conn_arr[:, 0]
idx_j = conn_arr[:, 1]

# Initialize Maxwell element lengths
r_vecs_initial = positions[idx_j] - positions[idx_i]
e0_arr = np.linalg.norm(r_vecs_initial, axis=1)
eij_arr = e0_arr.copy()

# Store initial boundary positions
ini = positions[right_boundary].copy()

# Store 3D indices for matrix blocks
fixed_idx_3d = np.array([[3*n, 3*n+1, 3*n+2] for n in fixed_nodes]).flatten()
moving_idx_3d = np.array([[3*n, 3*n+1, 3*n+2] for n in moving_nodes]).flatten()

time_list = []
strain = []

# Run simulation
t = 0.0
while t < t_max:
    # Calculate internal spring forces
    r_vecs = positions[idx_j] - positions[idx_i]
    rs = np.linalg.norm(r_vecs, axis=1)

    rs_safe = np.where(rs == 0, 1e-10, rs)
    f_mags = E * ((e0_arr**2) / eij_arr) * ((rs / eij_arr) - 1.0)

    # Convert force magnitudes to vectors
    f_vecs = f_mags[:, np.newaxis] * (r_vecs / rs_safe[:, np.newaxis])
    f_vecs = np.where((rs == 0)[:, np.newaxis], 0, f_vecs)

    forces = np.zeros((no_of_prot, 3))
    np.add.at(forces, idx_i, f_vecs)
    np.add.at(forces, idx_j, -f_vecs)

    # Apply external boundary force
    if t < t1:
        forces[right_boundary, 0] += F_ext_mag

    # Construct the mobility matrix
    diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r_sq = np.sum(diffs**2, axis=-1)

    den = (r_sq + delta**2)**1.5
    term1 = (r_sq + 2.0 * delta**2) / den

    # Calculate Stokeslet tensor blocks
    outer_prods = diffs[:, :, :, np.newaxis] * diffs[:, :, np.newaxis, :]
    I_tensor = np.eye(3).reshape(1, 1, 3, 3)

    M_blocks = term1[:, :, np.newaxis, np.newaxis] * I_tensor + (outer_prods / den[:, :, np.newaxis, np.newaxis])

    # Set the diagonal self interaction
    diag_val = (2.0 / delta) * np.eye(3)
    idx = np.arange(no_of_prot)
    M_blocks[idx, idx] = diag_val

    # Convert blocks into the full matrix
    M = M_blocks.transpose(0, 2, 1, 3).reshape(3*no_of_prot, 3*no_of_prot)

    # Extract fixed and moving matrix blocks
    M_FF = M[np.ix_(fixed_idx_3d, fixed_idx_3d)]
    M_FX = M[np.ix_(fixed_idx_3d, moving_idx_3d)]

    g_X = forces[moving_nodes].flatten()
    B = -np.dot(M_FX, g_X)

    # Solve for the boundary forces
    lamda = np.linalg.solve(M_FF, B)
    forces[fixed_nodes] = lamda.reshape(-1, 3)

    # Calculate node velocities
    velocities = np.dot(M, forces.flatten()).reshape(-1, 3) / (8.0 * np.pi * mu)

    # Update Maxwell rest lengths
    e_dot = (E / eta) * (rs - eij_arr)
    eij_arr += e_dot * dt

    # Update positions and record strain
    positions += velocities * dt

    fin = positions[right_boundary]
    strain.append(eng_strain(ini, fin, P))
    time_list.append(t)

    t += dt

# Plot strain over time
plt.plot(time_list, strain, label='strain vs time (lammps)')
plt.title(f"Strain vs Time\n($\\mu$ = {mu}, $E$ = {E}, $\\eta$ = {eta})")
plt.xlabel("Time")
plt.ylabel("Strain")
plt.legend()
plt.tight_layout()
plt.show()
