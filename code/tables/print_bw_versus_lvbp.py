import pandas as pd
import numpy as np
import ALA3
import experiment_loader

d = pd.read_csv(ALA3.chi2_filename)

x = d[d["subset"] == "test"]
y_test = x.pivot_table(rows=["method","prior"], cols=["ff"]).iloc[0:4]  # Drop the doubled raw MD row


x = d[d["subset"] == "train"]
y_train = x.pivot_table(rows=["method","prior"], cols=["ff"]).iloc[0:4]

x = d[d["subset"] == "all"]
y_all = x.pivot_table(rows=["method","prior"], cols=["ff"]).iloc[0:4]


y_all
y_train
y_test

print y_test.to_latex(float_format=(lambda x: "%.2f"%x))

print y_train.to_latex(float_format=(lambda x: "%.2f"%x))
