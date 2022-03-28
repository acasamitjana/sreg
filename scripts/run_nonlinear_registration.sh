#!/usr/bin/env bash

export PYTHONPATH='/home/acasamitjana/Repositories/sreg'

if  [ $# -eq 0 ]; then
    echo "Running all subjects in c"
    python algorithm/initialize_graph.py --reg_algorithm bidir
    python algorithm/algorithm_RegNet.py
    python segmentation/compute_ST_segmentation.py --num_cores 3
else
    echo "Running subject(s) " $@
    python algorithm/initialize_graph.py --subjects $@ --reg_algorithm bidir
    python algorithm/algorithm_RegNet.py --subjects $@
    python segmentation/compute_ST_segmentation.py --subjects $@ --num_cores 2
fi