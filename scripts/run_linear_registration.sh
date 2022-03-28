#!/usr/bin/env bash

export PYTHONPATH='/home/acasamitjana/Repositories/sreg'

if  [ $# -eq 0 ]; then
    echo "Running all subjects in IMAGES_DIR"
    python ../database/preprocess_dataset.py
    python algorithm/initialize_graph_lineal.py
    python algorithm/algorithm_lineal.py --num_cores 3
else
    echo "Running subject(s) " $@
    python algorithm/initialize_graph_lineal.py --subjects $@
    python algorithm/algorithm_lineal.py --subjects $@
fi

