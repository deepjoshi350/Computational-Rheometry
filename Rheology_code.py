import numpy as np
import matplotlib.pyplot as plt

# init params
mu = 1
E = 100.0
eta = 100.0
delta = 1.0 / (4.0 * np.pi)
no_of_prot = len(coms)
G_prime = []
G_dou_prime = []
ome = []

# init logging lists
time_list = []
storage_moduli = []
loss_moduli = []

# setup positions
positions = np.array([coms[pid] for pid in protein_ids])

# extract connections and lengths
pid_to_index = {pid: i for i, pid in enumerate(protein_ids)}
connections = [(pid_to_index[pid1], pid_to_index[pid2]) for pid1, pid2 in edges]
conn_arr = np.array(connections)
idx_i = conn_arr[:, 0]
idx_j = conn_arr[:, 1]
r_vecs_initial = positions[idx_j] - positions[idx_i]
e0_arr = np.linalg.norm(r_vecs_initial, axis=1)
eij_arr = e0_arr.copy()

x_coords = positions[:, 0]
sorted_indices = np.argsort(x_coords)
P = np.max(x_coords) - np.min(x_coords)

# 5 percent boundary thickness
thickness = 0.05 * P

# find left and right boundary nodes
fixed_nodes = np.where(x_coords <= np.min(x_coords) + thickness)[0].tolist()
right_boundary = np.where(x_coords >= np.max(x_coords) - thickness)[0].tolist()
print(f"Number of Fixed Nodes: {len(fixed_nodes)}")
print(f"Number of Right Boundary Nodes: {len(right_boundary)}")

# split constrained and free nodes
constrained_nodes = fixed_nodes + right_boundary
free_nodes = list(set(range(len(positions))) - set(constrained_nodes))

# prep mobility matrix
constrained_idx_3d = np.array([[3*n, 3*n+1, 3*n+2] for n in constrained_nodes]).flatten()
free_idx_3d = np.array([[3*n, 3*n+1, 3*n+2] for n in free_nodes]).flatten()

# build full 3N x 3N matrix
diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
r_sq = np.sum(diffs**2, axis=-1)
den = (r_sq + delta**2)**1.5
term1 = (r_sq + 2.0 * delta**2) / den
outer_prods = diffs[:, :, :, np.newaxis] * diffs[:, :, np.newaxis, :]
I_tensor = np.eye(3).reshape(1, 1, 3, 3)
M_blocks = term1[:, :, np.newaxis, np.newaxis] * I_tensor + (outer_prods / den[:, :, np.newaxis, np.newaxis])
diag_val = (2.0 / delta) * np.eye(3)
idx = np.arange(no_of_prot)
M_blocks[idx, idx] = diag_val
M = M_blocks.transpose(0, 2, 1, 3).reshape(3*no_of_prot, 3*no_of_prot)

# extract submatrices
M_CC = M[np.ix_(constrained_idx_3d, constrained_idx_3d)]
M_CX = M[np.ix_(constrained_idx_3d, free_idx_3d)]

# calc cross section area
right_boun = positions[right_boundary]
ymax, ymin = np.max(right_boun[:, 1]), np.min(right_boun[:, 1])
zmax, zmin = np.max(right_boun[:, 2]), np.min(right_boun[:, 2])
A = np.pi * abs(ymin-ymax) * abs(zmin-zmax) / 4
inv_M_CC = np.linalg.inv(M_CC)

# save initial states
initial_positions = positions.copy()
initial_eij = e0_arr.copy()
w = np.array([1.0, 3.0, 5.0, 10.0, 20.0])
G_prime = []
G_dou_prime = []
ome = []

# rk4 rate calculator
def compute_rates(curr_t, curr_pos, curr_eij, current_omega):
    # internal spring forces
    r_vecs = curr_pos[idx_j] - curr_pos[idx_i]
    rs = np.linalg.norm(r_vecs, axis=1)
    rs_safe = np.where(rs == 0, 1e-10, rs)
    f_mags = E * ((e0_arr**2) / curr_eij) * ((rs / curr_eij) - 1.0)
    f_vecs = f_mags[:, np.newaxis] * (r_vecs / rs_safe[:, np.newaxis])
    f_vecs = np.where((rs == 0)[:, np.newaxis], 0, f_vecs)

    spring_forces = np.zeros((no_of_prot, 3))
    np.add.at(spring_forces, idx_i, f_vecs)
    np.add.at(spring_forces, idx_j, -f_vecs)

    # time dependent boundaries
    uf = np.zeros((len(right_boundary), 3))
    uf[:, 1] = epsilon * current_omega * np.cos(current_omega * curr_t)
    U_target = np.zeros((len(constrained_nodes), 3))
    U_target[:len(fixed_nodes)] = 0.0
    U_target[len(fixed_nodes):] = uf
    U_target_flat = U_target.flatten()

    # solve net force
    g_X = spring_forces[free_nodes].flatten()
    free_influence = np.dot(M_CX, g_X)
    RHS = (8.0 * np.pi * mu) * U_target_flat - free_influence
    
    # fast solve
    lamda = np.dot(inv_M_CC, RHS)

    # assign boundary forces
    forces = spring_forces.copy()
    forces[constrained_nodes] = lamda.reshape(-1, 3)

    # calc velocities
    velocities = np.dot(M, forces.flatten()).reshape(-1, 3) / (8.0 * np.pi * mu)
    e_dot = (E / eta) * (rs - curr_eij)

    return velocities, e_dot, spring_forces, forces

# run frequency sweep
for omega in w:
    print(f"\nRunning simulation for omega = {omega}...")
    
    # reset states
    t = 0.0
    positions = initial_positions.copy()
    eij_arr = initial_eij.copy()
    time_list = []
    storage_moduli = []
    loss_moduli = []

    # freq specific params
    epsilon = 1
    time_period = (2 * np.pi) / omega
    tmax = 2 * time_period
    S_P_C = 6000
    dt = time_period / S_P_C
    total_steps = int(tmax / dt)
    start_record_step = int(0.5 * total_steps)

    # rk4 loop
    for step in range(total_steps + 1):
        k1_v, k1_e, f_springs_k1, net_forces_k1 = compute_rates(t, positions, eij_arr, omega)
        k2_v, k2_e, _, _ = compute_rates(t + 0.5*dt, positions + 0.5*dt*k1_v, eij_arr + 0.5*dt*k1_e, omega)
        k3_v, k3_e, _, _ = compute_rates(t + 0.5*dt, positions + 0.5*dt*k2_v, eij_arr + 0.5*dt*k2_e, omega)
        k4_v, k4_e, _, _ = compute_rates(t + dt, positions + dt*k3_v, eij_arr + dt*k3_e, omega)
        
        # log moduli
        ext_forces_rb =  net_forces_k1[right_boundary] - f_springs_k1[right_boundary]
        sigma_t = np.sum(ext_forces_rb[:, 1]) / A
        if step >= start_record_step:
            loss_moduli.append(sigma_t * np.cos(omega * t))
            storage_moduli.append(sigma_t * np.sin(omega * t))
            time_list.append(t)
            
        # update step
        positions += (dt / 6.0) * (k1_v + 2.0*k2_v + 2.0*k3_v + k4_v)
        eij_arr   += (dt / 6.0) * (k1_e + 2.0*k2_e + 2.0*k3_e + k4_e)
        t += dt

    # calc final moduli
    leading_term = omega / (np.pi * epsilon)
    G_pri = leading_term * np.trapezoid(storage_moduli, x=time_list)
    G_double_prime = leading_term * np.trapezoid(loss_moduli, x=time_list)
    G_prime.append(G_pri)
    G_dou_prime.append(G_double_prime)
    ome.append(omega)
    
    print(f"  G' = {G_pri:.4e} | G'' = {G_double_prime:.4e}")

# plot results
plt.figure(figsize=(8, 6))
plt.plot(ome, G_prime, marker='o', linestyle='-', label="G' (Storage Modulus)")
plt.plot(ome, G_dou_prime, marker='s', linestyle='--', label="G'' (Loss Modulus)")
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Frequency, $\omega$")
plt.ylabel("Modulus")
plt.title("Storage and Loss Moduli vs. Frequency")
plt.legend()
plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.show()
plt.legend()
plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.show()
