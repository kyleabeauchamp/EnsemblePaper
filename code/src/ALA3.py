import pandas as pd

#model = "combined_shifts"
model = "maxent"

ff_list = ["amber96","amber99","amber99sbnmr-ildn","charmm27","oplsaa"]
prior_list = ["maxent", "MVN"]

#train_keys = ['JC_2_J3_HN_Cprime', 'CS_2_CA', 'CS_2_H', 'JC_3_J2_N_CA', 'CS_2_CB', 'JC_2_J3_HN_CB']
#test_keys = ["JC_2_J3_HN_HA" , "JC_2_J3_HA_Cprime", "JC_2_J1_N_CA"]

tuples = [("JC", 2, "J3_HN_Cprime"), ("JC", 3, "J2_N_CA"), ("JC", 2, "J3_HN_CB"), ("CS", 2, "CA"), ("CS", 2, "H"), ("CS", 2, "CB")]
train_keys = pd.MultiIndex.from_tuples(tuples, names=("experiment", "resid", "name"))

tuples = [("JC", 2, "J3_HN_HA"), ("JC", 2, "J3_HA_Cprime"), ("JC", 2, "J1_N_CA")]
test_keys = pd.MultiIndex.from_tuples(tuples, names=("experiment", "resid", "name"))


#all_keys = []
#all_keys.extend(train_keys)
#all_keys.extend(test_keys)

bw_num_samples = 1000000
num_samples = 5000000
thin = 100
burn = 5000
kfold = 2
num_blocks = 10

stride = 1
cross_val_stride = 20

regularization_strength_dict = {"maxent":
{
"amber96":5,
"amber99":1,
"amber99sbnmr-ildn":500,
"charmm27":1,
"oplsaa":7
}
,
"MVN":
{
"amber96":7,
"amber99":1,
"amber99sbnmr-ildn":500,
"charmm27":1,
"oplsaa":11
}
}


old_regularization_strength_dict = {"maxent":
{
"amber96":5,
"amber99":1,
"amber99sbnmr-ildn":1,
"charmm27":1,
"oplsaa":7
}
,
"MVN":
{
"amber96":8,
"amber99":1,
"amber99sbnmr-ildn":2,
"charmm27":1,
"oplsaa":14
}
}


data_directory = "/home/kyleb/dat/ala_lvbp/"

outdir = "/home/kyleb/src/kyleabeauchamp/EnsemblePaper/paper/figures/"
cross_val_filename = outdir + "../../data/cross_val.dat"
experiment_filename = outdir + "../../data/experimental_data.csv"
