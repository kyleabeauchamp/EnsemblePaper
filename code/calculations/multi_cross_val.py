import itertools
import experiment_loader
import ALA3
import numpy as np
from fitensemble import lvbp
import sys

ff_list = ALA3.ff_list
prior_list = ["maxent", "MVN"]
regularization_strength_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

ID = int(sys.argv[1])

for k, (ff, prior, regularization_strength) in enumerate(itertools.product(ff_list, prior_list, regularization_strength_list)):
    print(k, ff, prior, regularization_strength)
