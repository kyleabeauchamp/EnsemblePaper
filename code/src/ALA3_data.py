import numpy as np
import pandas as pd

uncertainties = pd.Series({
"JC_2_J3_HN_HA":0.36,
"JC_2_J3_HN_Cprime":0.30,
"JC_2_J3_HA_Cprime":0.24,
"JC_2_J3_Cprime_Cprime":0.13,
"JC_2_J3_HN_CB":0.22,
"JC_2_J1_N_CA":0.52659745254609414,
"JC_3_J2_N_CA":0.4776,
#"N":2.0862,"CA":0.7743,"CB":0.8583,"C":0.8699,"H":0.3783,"HA":0.1967         # ShiftX+ V1.07
"CS_2_N":2.4625,"CS_2_CA":0.7781,"CS_2_CB":1.1760,"CS_2_C":1.1309,"CS_2_H":0.4685,"CS_2_HA":0.2743,  # Mean uncertainties from SPARTA+
})

measurements = pd.Series({  # Numbering from Table S3 in Schwalbe
"JC_2_J3_HN_HA"        :5.68,  # highly correlated to other measurements, r ranges from 0.71 to 0.997
"JC_2_J3_HN_Cprime"    :1.13,
"JC_2_J3_HA_Cprime"    :1.84,  # highly correlated to other measurements.  r > 0.7
"JC_2_J3_HN_CB"        :2.39,
"JC_2_J1_N_CA"         :11.34,  #  highly correlated to other measurements.  r > 0.7
"JC_3_J2_N_CA"         :8.45,

"CS_2_H":8.571,
"CS_2_HA":4.355,
"CS_2_CA":52.38,
"CS_2_CB":19.21
})
