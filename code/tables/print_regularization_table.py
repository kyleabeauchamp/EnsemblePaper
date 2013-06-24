import pandas as pd
import numpy as np
import ALA3
import experiment_loader

num_expt = 6.  # CHECK!

keys = ["forcefield","prior","regularization_strength", "test_chi"]
d = pd.read_csv(ALA3.cross_val_filename)[keys]

x = d.pivot_table(rows=["forcefield","prior"], cols=["regularization_strength"]).min(1)

y = pd.DataFrame([x.values, x.index.get_level_values(0), x.index.get_level_values(1)], index=["test_chi","forcefield","prior"]).T
y["test_chi"] = y["test_chi"].astype('float')
z = y.pivot_table(rows=["forcefield"],cols=["prior"]) / num_expt

print z.to_latex(float_format=(lambda x: "%.2f"%x))

for (ff, prior), data in d.groupby(["forcefield","prior"]):
    print(ff, prior)
    print(data)


