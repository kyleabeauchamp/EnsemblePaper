import pandas as pd
import numpy as np
from fitensemble import bayesian_weighting, belt
import experiment_loader
import ALA3

ff = "amber96"
regularization_strength = 1E-6
stride = 40

thin = 500
steps = 15000000

predictions, measurements, uncertainties = experiment_loader.load(ff, stride=stride)
phi, psi, ass_raw, state_ind = experiment_loader.load_rama(ff, stride)

model2 = belt.MaxEntBELT(predictions.values, measurements.values, uncertainties.values, regularization_strength, log_det_correction=True)
model2.sample(steps, thin=thin, burn=ALA3.burn)

regularization_strength = 10.0
model = belt.MaxEntBELT(predictions.values, measurements.values, uncertainties.values, regularization_strength)
model.sample(steps, thin=thin, burn=ALA3.burn)


ai = model.mcmc.trace("alpha")[:]
ai2 = model2.mcmc.trace("alpha")[:]

p = model.accumulate_populations()
p2 = model2.accumulate_populations()

mu = predictions.T.dot(p)
mu2 = predictions.T.dot(p2)

z = (mu - measurements) / uncertainties
z2 = (mu2 - measurements) / uncertainties


state_pops_trace = model.trace_observable(state_ind.T)
state_pops_trace2 = model2.trace_observable(state_ind.T)
