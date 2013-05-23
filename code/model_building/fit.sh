#!/bin/bash

cmd="python /home/kyleb/src/kyleabeauchamp/EnsemblePaper/code/calculations/fit_model.py amber96 maxent "
$cmd >& /home/kyleb/fit-${JOB_ID}.log
