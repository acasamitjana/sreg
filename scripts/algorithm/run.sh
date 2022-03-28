#!/usr/bin/env bash

python initialize_graph.py
python algorithm_RegNet.py
python deform_RegNet.py
python ../segmentation/compute_ST_segmentation