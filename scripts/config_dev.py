import copy
from os.path import join

import numpy as np


from setup import REGISTRATION_DIR
from src.utils.image_transform import ScaleNormalization, NonLinearParams, CropParams, AffineParams

## Data characteristics

## Data dictionaries with RegNet parameters
CONFIG_REGISTRATION = {

    'TRANSFORM': None,
    'VOLUME_SHAPE': None,

    'DATA_AUGMENTATION': None,
    'NORMALIZATION': ScaleNormalization(range=[0, 1]),
    'AFFINE': AffineParams(rotation=[5]*3, scaling=[5]*3, translation=[5]*3),
    'NONLINEAR': NonLinearParams(lowres_size=[9, 9, 9], lowres_strength=[0.1, 3], distribution='uniform'),

    'ENC_NF': [16, 32, 32, 64],
    'DEC_NF': [64, 32, 32, 16, 16],

    'INT_STEPS': 7,

    'BATCH_SIZE': 1,
    'N_EPOCHS': 200,
    'LEARNING_RATE': 1e-3,
    'EPOCH_DECAY_LR': 0,
    'STARTING_EPOCH': 0,

    'USE_GPU': True,# Set to False if running to CPU
    'GPU_INDICES': [0],

    'LOG_INTERVAL': 1,
    'SAVE_MODEL_FREQUENCY': 100,

    'LOSS_REGISTRATION': {'name': 'NCC', 'params': {'kernel_var': [5, 5, 5], 'kernel_type': 'mean'},'lambda': 1},#, #
    'LOSS_REGISTRATION_SMOOTHNESS': {'name': 'Grad', 'params': {'dim': 3, 'penalty': 'l2'}, 'lambda': 0.1},

    'UPSAMPLE_LEVELS': 4,
    'FIELD_TYPE': 'velocity',

    'NEIGHBOR_DISTANCE': -1
}



def get_config_dict(volume_shape):

    config = CONFIG_REGISTRATION
    if config['LOSS_REGISTRATION']['name'] == 'NCC':
        loss_dir = config['LOSS_REGISTRATION']['name'] + str(config['LOSS_REGISTRATION']['params']['kernel_var'][0])
    else:
        loss_dir = config['LOSS_REGISTRATION']['name']

    loss_name = 'R' + str(config['LOSS_REGISTRATION']['lambda'])
    loss_name += '_S' + str(config['LOSS_REGISTRATION_SMOOTHNESS']['lambda'])
    loss_dir = join(loss_dir, loss_name)
    CONFIG_REGISTRATION['RESULTS_DIR'] = join(REGISTRATION_DIR, loss_dir, 'DownFactor_' + str(config['UPSAMPLE_LEVELS']))
    CONFIG_REGISTRATION['TRANSFORM'] = [CropParams(crop_shape=volume_shape)]
    CONFIG_REGISTRATION['VOLUME_SHAPE'] = volume_shape

    return CONFIG_REGISTRATION

