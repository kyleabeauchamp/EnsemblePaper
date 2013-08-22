import itertools
import mdtraj, mdtraj.geometry
 
traj = mdtraj.load("./trajectories/amber96.xtc", top="pdbs/amber96.pdb")
 
top, bonds = traj.top.to_dataframe()
 
atoms = np.array(["H", "HA", "N", "CA", "C", "CB"])
 
bad_residues = np.array(["GLY", "PRO"])
 
top = top[np.in1d(top.name, atoms)] # Selected desired atoms
top = top[~np.in1d(top.resName, bad_residues)] # Remove unwanted residues
 
atom_pairs = []
for k, x in top.groupby("resSeq"):
atom_pairs.extend(list(itertools.combinations(x.index, 2)))
 
atom_pairs = np.array(atom_pairs)
 
distances = mdtraj.geometry.distance.compute_distances(traj, atom_pairs)
