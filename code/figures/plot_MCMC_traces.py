from fitensemble import belt
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ALA3
import experiment_loader

grid = itertools.product(ALA3.ff_list, ALA3.prior_list)
bayesian_bootstrap_run = 0

for k, (ff, prior) in enumerate(grid):
        print(ff, prior)
        regularization_strength = ALA3.regularization_strength_dict[prior][ff]
        predictions, measurements, uncertainties = experiment_loader.load(ff)
        pymc_filename = ALA3.data_directory + "/models/model_%s_%s_reg-%.1f-BB%d.h5" % (ff, prior, regularization_strength, bayesian_bootstrap_run)
        belt_model = belt.BELT.load(pymc_filename)
        a = belt_model.mcmc.trace("alpha")[:]
        plt.figure()
        plt.title("%s - %s" % (ff, prior))
        y = a[:,0]
        x = np.arange(len(y)) * ALA3.thin
        plt.plot(x, y)
        plt.xlabel("MCMC steps")
        #plt.ylabel(r"$\alpha$:" + str(predictions.columns[0]))
        plt.ylabel(predictions.columns[0])
        plt.savefig(ALA3.outdir+"/%s-%s-MCMC_Trace.png"%(prior, ff), bbox_inches='tight')
