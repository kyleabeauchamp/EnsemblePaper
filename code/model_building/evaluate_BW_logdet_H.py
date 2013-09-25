import pandas as pd
import numpy as np
from fitensemble import bayesian_weighting, belt
import experiment_loader
import ALA3

ff = "amber96"
regularization_strength = 10.0
stride = 40

thin = 500
steps = 1000000

predictions, measurements, uncertainties = experiment_loader.load(ff, stride=stride)

model = belt.MaxEntBELT(predictions.values, measurements.values, uncertainties.values, regularization_strength)
model.sample(steps, thin=thin, burn=ALA3.burn)

chi2 = []
prior = []
H_terms = []
for j, p in enumerate(model.iterate_populations()):
    mu = predictions.T.dot(p)
    chi2.append(0.5 * (((mu - measurements) / uncertainties) ** 2).sum())
    prior.append(regularization_strength * -1.0 * p.dot(np.log(p)))
    H = -np.diag(p[:-1] ** -1.) - p[-1] ** -1.
    H_terms.append(0.5 * np.linalg.slogdet(H)[1])

R = pd.DataFrame({"chi2":chi2, "prior":prior, "H":H_terms})
