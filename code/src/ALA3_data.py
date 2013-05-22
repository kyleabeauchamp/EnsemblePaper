import numpy as np
import pandas as pd

uncertainties = pd.Series({
"J3_HN_HA":0.36,
"J3_HN_Cprime":0.30,
"J3_HA_Cprime":0.24,
"J3_Cprime_Cprime":0.13,
"J3_HN_CB":0.22,
"J1_N_CA":0.52659745254609414,
"J2_N_CA":0.4776,
#"N":2.0862,"CA":0.7743,"CB":0.8583,"C":0.8699,"H":0.3783,"HA":0.1967         # ShiftX+ V1.07
"N_2":2.4625,"CA_2":0.7781,"CB_2":1.1760,"C_2":1.1309,"H_2":0.4685,"HA_2":0.2743,  # Mean uncertainties from SPARTA+
})

measurements = pd.Series({  # Numbering from Table S3 in Schwalbe
"J3_HN_HA_2"        :5.68,  # highly correlated to other measurements, r ranges from 0.71 to 0.997
"J3_HN_Cprime_2"    :1.13,
"J3_HA_Cprime_2"    :1.84,  # highly correlated to other measurements.  r > 0.7
"J3_HN_CB_2"        :2.39,
"J1_N_CA_2"         :11.34,  #  highly correlated to other measurements.  r > 0.7
"J2_N_CA_3"         :8.45,

"H_2":8.571,
"HA_2":4.355,
"CA_2":52.38,
"CB_2":19.21
})
