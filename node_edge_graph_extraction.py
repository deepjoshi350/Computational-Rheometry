import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

filename = "/content/lammps.data"
Lx = 80.0
Ly = 80.0
Lz = 70.0
box = np.array([Lx, Ly, Lz])

protein_atoms = {}
with open(filename, "r") as f:
    lines = f.readlines()

# find atoms section
start = None
for i, line in enumerate(lines):
    if line.strip().startswith("Atoms"):
        start = i + 2
        break
if start is None:
    raise ValueError("Atoms section not found.")

# parse atoms
for line in lines[start:]:
    s = line.strip()
    if s == "":
        continue
    if s.startswith(("Bonds", "Velocities", "Angles", "Dihedrals",
                     "Impropers", "Masses", "Pair", "Bond",
                     "Angle", "Dihedral", "Improper")):
        break
        
    parts = s.split()
    protein_id = int(parts[1])
    p_type = int(parts[2])
    
    # unwrap coords
    x = float(parts[3]) + int(parts[6]) * Lx
    y = float(parts[4]) + int(parts[7]) * Ly
    z = float(parts[5]) + int(parts[8]) * Lz

    if protein_id not in protein_atoms:
        protein_atoms[protein_id] = {'type': p_type, 'coords': []}
    protein_atoms[protein_id]['coords'].append([x, y, z])

# get centers of mass
protein_ids_all = sorted(protein_atoms.keys())
coms_all = {}
for pid in protein_ids_all:
    coords = np.array(protein_atoms[pid]['coords'])
    coms_all[pid] = coords.mean(axis=0)

print("Total proteins in box:", len(coms_all))

# extract dense phase
com_array = np.array([coms_all[pid] for pid in protein_ids_all])

# calc pairwise distances
deltas = com_array[:, np.newaxis, :] - com_array[np.newaxis, :, :]
deltas -= box * np.round(deltas / box)
distances = np.linalg.norm(deltas, axis=2)

# find densest node
R_search = 10.0
local_densities = np.sum(distances < R_search, axis=1)
densest_idx = np.argmax(local_densities)
core_node_com = com_array[densest_idx]

# calc core com
dense_core_indices = np.where(distances[densest_idx] < R_search)[0]
shifts = deltas[dense_core_indices, densest_idx, :]
mean_shift = np.mean(shifts, axis=0)
dense_com = core_node_com + mean_shift
dense_com = dense_com % box

# get 3d density profile
delta = com_array - dense_com
delta -= box * np.round(delta / box)
r_dist = np.linalg.norm(delta, axis=1)

dbin = 2.0
max_r = np.min(box) / 2.0
bins = np.arange(0, max_r + dbin, dbin)
r_centers = bins[:-1] + dbin / 2

density_profile = []
for i in range(len(bins)-1):
    r1, r2 = bins[i], bins[i+1]
    count = np.sum((r_dist >= r1) & (r_dist < r2))
    vol = (4.0 / 3.0) * np.pi * (r2**3 - r1**3)
    if vol > 0:
        density_profile.append(count / vol)
    else:
        density_profile.append(0.0)
density_profile = np.array(density_profile)

# find drop off cutoff
max_density = np.max(density_profile)
threshold_density = 0.2 * max_density
drop_indices = np.where(density_profile < threshold_density)[0]

if len(drop_indices) > 0:
    cutoff_radius = bins[drop_indices[0]]
else:
    cutoff_radius = max_r
print(f"Calculated cutoff distance (along X) for dense phase: {cutoff_radius:.2f} Å")

# filter to dense phase
protein_ids = []
coms = {}
for i, pid in enumerate(protein_ids_all):
    if r_dist[i] < cutoff_radius:
        protein_ids.append(pid)
        delta = com_array[i] - dense_com
        delta -= box * np.round(delta / box)
        unwrapped_coords = dense_com + delta
        coms[pid] = unwrapped_coords
print(f"Number of proteins isolated in dense phase: {len(coms)}")

# get atomic distances
threshold = 1.225
edges = set()
for pid1, pid2 in combinations(protein_ids, 2):
    atoms1 = np.array(protein_atoms[pid1]['coords'])
    atoms2 = np.array(protein_atoms[pid2]['coords'])
    delta = atoms2[:, np.newaxis, :] - atoms1[np.newaxis, :, :]
    delta -= box * np.round(delta / box)
    d = np.linalg.norm(delta, axis=2)
    if np.any(d < threshold):
        edges.add((pid1, pid2))
print(f"Number of edges = {len(edges)}")

import networkx as nx

# clean up network
G = nx.Graph()
G.add_nodes_from(protein_ids)
G.add_edges_from(edges)
largest_cc = max(nx.connected_components(G), key=len)

protein_ids = [pid for pid in protein_ids if pid in largest_cc]
coms = {pid: coms[pid] for pid in protein_ids}
edges = set((u, v) for u, v in edges if u in largest_cc and v in largest_cc)
print(f"Cleaned network! Nodes kept: {len(protein_ids)}")

# plot results
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121)
ax1.plot(r_centers, density_profile, marker='o', linestyle='-', color='b')
ax1.axvline(cutoff_radius, color='r', linestyle='--', label=f'Cutoff = {cutoff_radius:.1f}')
ax1.set_xlabel("Distance from Dense COM")
ax1.set_ylabel("Number Density")
ax1.set_title("3D Density Profile")
ax1.legend()

color_map = {1: 'blue', 2: 'red', 3: 'green', 4: 'orange'}

ax2 = fig.add_subplot(122, projection='3d')
for pid in protein_ids:
    c = coms[pid]
    p_type = protein_atoms[pid]['type']
    node_color = color_map.get(p_type, 'gray')
    ax2.scatter(c[0], c[1], c[2], color=node_color, s=15)

for pid1, pid2 in edges:
    c1 = coms[pid1]
    c2 = coms[pid2]
    ax2.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]], color='black', linewidth=0.7)

ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")
ax2.set_title(f"Dense Phase Network (Cutoff: {cutoff_radius:.1f})")

plt.tight_layout()
plt.show()
