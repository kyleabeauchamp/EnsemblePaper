#!/bin/bash
#$ -pe orte 4

cmd="python /home/kyleb/src/kyleabeauchamp/EnsemblePaper/code/model_building/multi_cross_val.py ${SGE_TASK_ID}"
$cmd >& /home/kyleb/cross_val-${SGE_JOB_ID}-${SGE_TASK_ID}.log
