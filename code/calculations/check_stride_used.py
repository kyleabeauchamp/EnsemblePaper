import numpy as np
import experiment_loader
import ALA3
from fitensemble import lvbp
import itertools
import sys

grid = itertools.product(ALA3.ff_list, ALA3.prior_list)

for k, (ff, prior) in enumerate(grid):
    print(ff, prior)
    try:
        regularization_strength = ALA3.regularization_strength_dict[prior][ff]
        directory = ALA3.data_dir + "/%s/" % ff
        predictions, measurements, uncertainties, phi, psi, ass_raw, state_ind = experiment_loader.load(directory)
        model_directory = "%s/models-%s/" % (directory, prior)
        lvbp_model = lvbp.LVBP.load(model_directory + "/reg-%d-BB0.h5" % regularization_strength)
        print(ff, prior, lvbp_model.predictions.shape)
    except:
        pass
