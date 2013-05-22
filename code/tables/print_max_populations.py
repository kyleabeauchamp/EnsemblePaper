from fitensemble import lvbp
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ALA3
import experiment_loader

data = np.zeros((len(ALA3.prior_list), len(ALA3.ff_list)))
for i, prior in enumerate(ALA3.prior_list):
    for j, ff in enumerate(ALA3.ff_list):
        regularization_strength = ALA3.regularization_strength_dict[prior][ff]
        directory = "/%s/%s/models-%s/" % (ALA3.data_dir, ff, prior)
        p = np.loadtxt(directory + "reg-%d-frame-populations.dat" % regularization_strength)
        data[i,j] = p.max()

data = pd.DataFrame(data, columns=ALA3.ff_list, index=ALA3.prior_list)
