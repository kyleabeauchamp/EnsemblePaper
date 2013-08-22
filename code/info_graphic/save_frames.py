import numpy as np
import scipy.stats  #Note no longe resi 0 but resi 1
import ALA3, experiment_loader
import mdtraj

ff = "amber96"
prior = "maxent"
bayesian_bootstrap_run_list = [0,1]

regularization_strength = ALA3.regularization_strength_dict[prior][ff]

predictions, measurements, uncertainties = experiment_loader.load(ff)
phi, psi, ass_raw, state_ind = experiment_loader.load_rama(ff, ALA3.stride)
p = np.vstack([np.loadtxt(ALA3.data_directory + "/frame_populations/pops_%s_%s_reg-%.1f-BB%d.dat" % (ff, prior, regularization_strength, bayesian_bootstrap_run)) for bayesian_bootstrap_run in bayesian_bootstrap_run_list]).mean(0)

traj = mdtraj.load("./trajectories/amber96.xtc", top="./pdbs/amber96.pdb")

ind0 = np.array([np.random.multinomial(1, (p ** 0) / float(len(p))).argmax() for i in xrange(3)])
ind = np.array([np.random.multinomial(1, p).argmax() for i in xrange(3)])

traj0 = traj[ind0]
traj1 = traj[ind]

traj0[1:].save(ALA3.outdir + "/info_graphic/ff96_raw_three.pdb")
traj1[1:].save(ALA3.outdir + "/info_graphic/ff96_BELT_three.pdb")

traj0[0:1].save(ALA3.outdir + "/info_graphic/ff96_raw_first_frame.pdb")
traj1[0:1].save(ALA3.outdir + "/info_graphic/ff96_BELT_first_frame.pdb")
