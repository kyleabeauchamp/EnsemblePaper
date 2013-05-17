#!/bin/bash

cmd="python /home/kyleb/src/kyleabeauchamp/EnsemblePaper/code/calculations/ALA3_fit.py amber96 maxent "
$cmd >& /home/kyleb/fit-${SGE_JOB_ID}.log
