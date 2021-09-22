#!/usr/bin/env bash

export PYTHONPATH='/home/acasamitjana/Repositories/sreg'

if  [ $# -eq 0 ]; then
    echo "Running all subjects in ALGORITHM_DIR_LINEAR/image"
    python algorithm/initialize_graph_NR.py
    python algorithm/algorithm_NR.py
else
    echo "Running subject(s) " $@
    python algorithm/initialize_graph_NR.py --subjects $@
    python algorithm/algorithm_NR.py --subjects UMA0012
fi