#!/usr/bin/env bash

export PYTHONPATH='/home/acasamitjana/Repositories/sreg'

if  [ $# -eq 0 ]; then
    echo "Running all subjects in IMAGES_DIR"
    python ../database/preprocess_dataset.py
    python algorithm/initialize_graph_lineal.py
    python algorithm/algorithm_lineal.py

    echo "Running all subjects in REGISTRATION_DIR"
    python algorithm/initialize_graph.py --reg_algorithm bidir
    python algorithm/algorithm_RegNet.py
    python segmentation/compute_ST_segmentation.py

else
    echo "Running subject(s) " $@
    python ../database/preprocess_dataset.py --subjects $@
    python algorithm/initialize_graph_lineal.py --subjects $@
    python algorithm/algorithm_lineal.py --subjects $@

    echo "Running subject(s) " $@
    python algorithm/initialize_graph.py --subjects $@ --reg_algorithm bidir
    python algorithm/algorithm_RegNet.py --subjects $@
    python segmentation/compute_ST_segmentation.py --subjects $@

fi

