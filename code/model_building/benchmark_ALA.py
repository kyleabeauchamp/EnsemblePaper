import experiment_loader
import ALA3
import numpy as np
from fitensemble import lvbp
from fitensemble.utils import validate_pandas_columns

num_threads = 2
num_samples = 20000  # Generate 20,000 MCMC samples
thin = 25  # Subsample (i.e. thin) the MCMC traces by 25X to ensure independent samples
burn = 5000  # Discard the first 5000 samples as "burn-in"

ff = "amber99"
prior = "maxent"
regularization_strength = 3.0

directory = "%s/%s" % (ALA3.data_dir , ff)
out_dir = directory + "/cross_val/"

predictions, measurements, uncertainties = experiment_loader.load(directory)
validate_pandas_columns(predictions, measurements, uncertainties)

lvbp.ne.set_num_threads(num_threads)

lvbp_model = lvbp.MaxEnt_LVBP(predictions.values, measurements.values, uncertainties.values, regularization_strength)

%prun lvbp_model.sample(num_samples, thin=thin, burn=burn)

"""
2,2 with optimized code:

         2561713 function calls (2278913 primitive calls) in 88.531 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    80700   37.819    0.000   37.819    0.000 {method 'dot' of 'numpy.ndarray' objects}
    20350   17.375    0.001   32.767    0.002 lvbp.py:47(get_populations_from_q)
   100702   12.116    0.000   12.116    0.000 {method 'reduce' of 'numpy.ufunc' objects}
    20350   10.404    0.001   11.030    0.001 necompiler.py:667(evaluate)
    20350    3.474    0.000   28.250    0.001 lvbp.py:23(get_q)
121301/80601    0.669    0.000   76.896    0.001 PyMCObjects.py:434(get_value)
241301/120601    0.565    0.000   86.129    0.001 {method 'get' of 'pymc.LazyFunction.LazyFunction' objects}
    20000    0.557    0.000    1.259    0.000 ensemble_fitter.py:36(get_chi2)
    20000    0.416    0.000    0.711    0.000 {method 'normal' of 'mtrand.RandomState' objects}
    20350    0.379    0.000    5.218    0.000 _methods.py:42(_mean)
121051/60351    0.326    0.000   76.891    0.001 {method 'run' of 'pymc.Container_values.DCValue' objects}
    20000    0.325    0.000   86.944    0.004 StepMethods.py:434(step)
    20350    0.302    0.000    0.340    0.000 necompiler.py:462(getContext)
    20350    0.251    0.000    0.270    0.000 _methods.py:32(_count_reduce_items)
    20000    0.246    0.000    7.253    0.000 lvbp.py:211(logp_prior)
   121059    0.245    0.000    0.245    0.000 {numpy.core.multiarray.array}
    20000    0.232    0.000    0.702    0.000 linalg.py:1868(norm)
    80000    0.220    0.000   84.867    0.001 PyMCObjects.py:293(get_logp)
    20000    0.208    0.000    1.147    0.000 StepMethods.py:516(propose)





1,1
         2563078 function calls (2280230 primitive calls) in 102.982 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    80712   45.111    0.001   45.111    0.001 {method 'dot' of 'numpy.ndarray' objects}
    20356   17.069    0.001   37.650    0.002 lvbp.py:47(get_populations_from_q)
    20356   15.933    0.001   16.515    0.001 necompiler.py:667(evaluate)
   100714   11.504    0.000   11.504    0.000 {method 'reduce' of 'numpy.ufunc' objects}
    20356    6.466    0.000   33.979    0.002 lvbp.py:23(get_q)
121313/80601    0.641    0.000   90.150    0.001 PyMCObjects.py:434(get_value)
    20000    0.532    0.000    1.201    0.000 ensemble_fitter.py:36(get_chi2)
241313/120601    0.525    0.000  100.705    0.001 {method 'get' of 'pymc.LazyFunction.LazyFunction' objects}
    20000    0.395    0.000    0.668    0.000 {method 'normal' of 'mtrand.RandomState' objects}
    20356    0.359    0.000    4.870    0.000 _methods.py:42(_mean)
121069/60357    0.329    0.000   90.102    0.001 {method 'run' of 'pymc.Container_values.DCValue' objects}
    20000    0.317    0.000  101.158    0.005 StepMethods.py:434(step)
    20356    0.291    0.000    0.325    0.000 necompiler.py:462(getContext)



2,2

         2563570 function calls (2280546 primitive calls) in 91.036 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    80756   37.402    0.000   37.402    0.000 {method 'dot' of 'numpy.ndarray' objects}
    20378   17.369    0.001   32.824    0.002 lvbp.py:47(get_populations_from_q)
   100758   11.803    0.000   11.803    0.000 {method 'reduce' of 'numpy.ufunc' objects}
    20378   10.529    0.001   11.157    0.001 necompiler.py:667(evaluate)
    20378    6.635    0.000   30.884    0.002 lvbp.py:23(get_q)
121357/80601    0.667    0.000   79.525    0.001 PyMCObjects.py:434(get_value)
    20000    0.558    0.000    1.269    0.000 ensemble_fitter.py:36(get_chi2)
241357/120601    0.557    0.000   88.637    0.001 {method 'get' of 'pymc.LazyFunction.LazyFunction' objects}
    20000    0.422    0.000    0.716    0.000 {method 'normal' of 'mtrand.RandomState' objects}
    20378    0.363    0.000    4.970    0.000 _methods.py:42(_mean)
121135/60379    0.334    0.000   79.510    0.001 {method 'run' of 'pymc.Container_values.DCValue' objects}
    20000    0.325    0.000   89.299    0.004 StepMethods.py:434(step)
    20378    0.310    0.000    0.349    0.000 necompiler.py:462(getContext)
    20000    0.243    0.000    7.105    0.000 lvbp.py:211(logp_prior)
    20378    0.237    0.000    0.255    0.000 _methods.py:32(_count_reduce_items)


3,3

         2560555 function calls (2277875 primitive calls) in 91.947 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    80670   38.771    0.000   38.771    0.000 {method 'dot' of 'numpy.ndarray' objects}
    20335   17.648    0.001   32.077    0.002 lvbp.py:47(get_populations_from_q)
   100672   12.042    0.000   12.042    0.000 {method 'reduce' of 'numpy.ufunc' objects}
    20335    9.396    0.000   10.026    0.000 necompiler.py:667(evaluate)
    20335    6.848    0.000   31.664    0.002 lvbp.py:23(get_q)
121271/80601    0.654    0.000   79.951    0.001 PyMCObjects.py:434(get_value)
    20000    0.570    0.000    1.246    0.000 ensemble_fitter.py:36(get_chi2)
241271/120601    0.554    0.000   89.549    0.001 {method 'get' of 'pymc.LazyFunction.LazyFunction' objects}
    20000    0.431    0.000    0.739    0.000 {method 'normal' of 'mtrand.RandomState' objects}
    20335    0.378    0.000    5.049    0.000 _methods.py:42(_mean)
    20000    0.339    0.000   90.368    0.005 StepMethods.py:434(step)
    20335    0.315    0.000    0.352    0.000 necompiler.py:462(getContext)
121006/60336    0.304    0.000   79.885    0.001 {method 'run' of 'pymc.Container_values.DCValue' objects}
    20335    0.242    0.000    0.261    0.000 _methods.py:32(_count_reduce_items)
    20000    0.234    0.000    0.676    0.000 linalg.py:1868(norm)
   121014    0.225    0.000    0.225    0.000 {numpy.core.multiarray.array}
    20000    0.216    0.000    1.185    0.000 StepMethods.py:516(propose)
        1    0.193    0.193   91.946   91.946 MCMC.py:252(_loop)
    80000    0.182    0.000   88.268    0.001 PyMCObjects.py:293(get_logp)
    20000    0.178    0.000    7.728    0.000 lvbp.py:211(logp_prior)
    40000    0.163    0.000   88.654    0.002 Node.py:23(logp_of_set)
    20000    0.159    0.000    0.217    0.000 PyMCObjects.py:768(set_value)


4,4

         2563612 function calls (2280716 primitive calls) in 94.110 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    80724   39.526    0.000   39.526    0.000 {method 'dot' of 'numpy.ndarray' objects}
    20362   17.875    0.001   32.333    0.002 lvbp.py:47(get_populations_from_q)
   100726   12.323    0.000   12.323    0.000 {method 'reduce' of 'numpy.ufunc' objects}
    20362    9.443    0.000   10.092    0.000 necompiler.py:667(evaluate)
    20362    7.371    0.000   32.412    0.002 lvbp.py:23(get_q)
121325/80601    0.702    0.000   82.448    0.001 PyMCObjects.py:434(get_value)
241325/120601    0.585    0.000   91.603    0.001 {method 'get' of 'pymc.LazyFunction.LazyFunction' objects}
    20000    0.570    0.000    1.288    0.000 ensemble_fitter.py:36(get_chi2)
    20000    0.429    0.000    0.741    0.000 {method 'normal' of 'mtrand.RandomState' objects}
    20362    0.382    0.000    5.138    0.000 _methods.py:42(_mean)
121087/60363    0.341    0.000   82.410    0.001 {method 'run' of 'pymc.Container_values.DCValue' objects}
    20000    0.337    0.000   92.352    0.005 StepMethods.py:434(step)
    20362    0.318    0.000    0.357    0.000 necompiler.py:462(getContext)
    20362    0.246    0.000    0.264    0.000 _methods.py:32(_count_reduce_items)
    20000    0.242    0.000    7.107    0.000 lvbp.py:211(logp_prior)
    20000    0.241    0.000    0.717    0.000 linalg.py:1868(norm)
   121095    0.234    0.000    0.234    0.000 {numpy.core.multiarray.array}


"""
