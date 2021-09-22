#!/usr/bin/env bash

export PYTHONPATH='/home/acasamitjana/Repositories/sreg'

if  [ $# -eq 0 ]; then
    echo "Running all subjects in IMAGES_DIR"
    python ../database/preprocess_dataset.py
    python algorithm/initialize_graph_NR_lineal.py
    python algorithm/algorithm_NR_lineal.py
else
    echo "Running subject(s) " $@
    python ../database/preprocess_dataset.py --subjects $@
    python algorithm/initialize_graph_NR_lineal.py --subjects $@
    python algorithm/algorithm_NR_lineal.py --subjects $@
fi

