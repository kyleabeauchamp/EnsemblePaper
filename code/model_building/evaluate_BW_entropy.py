import pandas as pd
import numpy as np
from fitensemble import bayesian_weighting, belt
import experiment_loader
import ALA3

prior = "BW"
ff = "amber96"
stride = 1000
regularization_strength = 10.0

thin = 400
factor = 50
steps = 1000000

predictions_framewise, measurements, uncertainties = experiment_loader.load(ff, stride=stride)
phi, psi, ass_raw0, state_ind0 = experiment_loader.load_rama(ff, stride)

num_states = len(phi)
assignments = np.arange(num_states)

prior_pops = np.ones(num_states)

predictions = pd.DataFrame(bayesian_weighting.framewise_to_statewise(predictions_framewise, assignments), columns=predictions_framewise.columns)
model = bayesian_weighting.MaxentBayesianWeighting(predictions.values, measurements.values, uncertainties.values, assignments, regularization_strength)
model.sample(steps * factor, thin=thin * factor)

model2 = belt.MaxEntBELT(predictions.values, measurements.values, uncertainties.values, regularization_strength)
model2.sample(steps, thin=thin)

pi = model.mcmc.trace("matrix_populations")[:, 0]

num_samples = len(pi)

data = np.zeros((num_samples, num_samples))
for i, p in enumerate(model.iterate_populations()):
    print(i)
    for j, p2 in enumerate(model2.iterate_populations()):
        data[i, j] = p.dot(np.log(p / p2))        


p_bw = model.accumulate_populations()
p_BELT = model2.accumulate_populations()

chi2 = []
prior = []
H_terms = []
for j, p2 in enumerate(model2.iterate_populations()):
    mu = predictions.T.dot(p2)
    chi2.append(0.5 * (((mu - measurements) / uncertainties) ** 2).sum())
    prior.append(regularization_strength * -1.0 * p2.dot(np.log(p2)))
    H = -np.diag(p2[:-1] ** -1.) - p[-1] ** -1.
    H_terms.append(0.5 * np.linalg.slogdet(H)[1])

R = pd.DataFrame({"chi2":chi2, "prior":prior, "H":H_terms})
