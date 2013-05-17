from fitensemble import lvbp
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ALA3
import experiment_loader

grid = itertools.product(ALA3.ff_list, ALA3.prior_list)

for k, (ff, prior) in enumerate(grid):
        print(ff, prior)
        regularization_strength = ALA3.regularization_strength_dict[prior][ff]
        directory = ALA3.data_dir + "/%s/" % ff
        predictions, measurements, uncertainties, phi, psi, ass_raw, state_ind = experiment_loader.load(directory)
        model_directory = "%s/models-%s/" % (directory, prior)
        lvbp_model = lvbp.LVBP.load(model_directory + "/reg-%d-BB0.h5" % regularization_strength)
        a = lvbp_model.mcmc.trace("alpha")[:]
        plt.figure()
        plt.title("%s - %s" % (ff, prior))
        y = a[:,0]
        x = np.arange(len(y)) * ALA3.thin
        plt.plot(x, y)
        plt.xlabel("MCMC steps")
        plt.ylabel(predictions.columns[0])
