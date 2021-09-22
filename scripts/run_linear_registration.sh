#!/usr/bin/env bash

export PYTHONPATH='/home/acasamitjana/Repositories/sreg'

python ../database/preprocess_dataset.py #--subjects UMA0012
python algorithm/initialize_graph_NR_lineal.py #--subjects UMA0012
python algorithm/algorithm_NR_lineal.py #--subjects UMA0012