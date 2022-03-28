import copy
from os.path import join

import numpy as np

from setup import *
from src.utils.data_loader_utils import ScaleNormalization, NonLinearParams, CropParams, AffineParams
from database.FCIEN import CONFIG_REGISTRATION as config_fcien
from database.MIRIAD import CONFIG_REGISTRATION as config_miriad
from database.ADNI import CONFIG_REGISTRATION as config_adni

## Data characteristics

## Data dictionaries with RegNet parameters
if DB == 'FCIEN':
    CONFIG_REGISTRATION = config_fcien

elif DB == 'MIRIAD' or DB == 'MIRIAD_retest':
    CONFIG_REGISTRATION = config_miriad

elif DB == 'ADNI':
    CONFIG_REGISTRATION = config_adni

else:
    CONFIG_REGISTRATION = {

        'TRANSFORM': None,
        'VOLUME_SHAPE': None,

        'DATA_AUGMENTATION': None,
        'NORMALIZATION': ScaleNormalization(range=[0, 1]),
        'AFFINE': AffineParams(rotation=[2.5]*3, scaling=[0]*3, translation=[2]*3),
        'NONLINEAR': NonLinearParams(lowres_shape_factor=0.04, lowres_strength=3, distribution='uniform'),

        'ENC_NF': [16, 32, 32, 64],
        'DEC_NF': [64, 32, 32, 16, 16],

        'INT_STEPS': 7,

        'BATCH_SIZE': 1,
        'N_EPOCHS': 100,
        'LEARNING_RATE': 1e-3,
        'EPOCH_DECAY_LR': 0,
        'STARTING_EPOCH': 0,

        'USE_GPU': True,# Set to False if running to CPU
        'GPU_INDICES': [0],

        'LOG_INTERVAL': 1,
        'SAVE_MODEL_FREQUENCY': 100,

        'LOSS_REGISTRATION': {'name': 'NCC', 'params': {'kernel_var': [5]*3, 'kernel_type': 'mean'},'lambda': 1},#, #
        'LOSS_REGISTRATION_SMOOTHNESS': {'name': 'Grad', 'params': {'dim': 3, 'penalty': 'l2'}, 'lambda': 1},#0.1},

        'UPSAMPLE_LEVELS': 4,#1,#
        'FIELD_TYPE': 'velocity',
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
    CONFIG_REGISTRATION['RESULTS_DIR'] = join(REGISTRATION_DIR, 'Registration', loss_dir, 'DownFactor_' + str(config['UPSAMPLE_LEVELS']))

    outshape = []
    for vs in volume_shape:
        ratio = vs / 2**len(config['ENC_NF'])
        if ratio - np.floor(ratio) < 0.5:
            outshape.append(int(2**len(config['ENC_NF'])*np.floor(ratio)))
        else:
            outshape.append(int(2**len(config['ENC_NF'])*np.ceil(ratio)))

    if DB == 'MIRIAD' or DB == 'MIRIAD_retest' or DB == 'ADNI':
        outshape = (192, 208, 224)
    elif DB == 'FCIEN':
        outshape = (192, 208, 192)

    CONFIG_REGISTRATION['TRANSFORM'] = [CropParams(crop_shape=tuple(outshape))]
    CONFIG_REGISTRATION['VOLUME_SHAPE'] = tuple(outshape)

    return CONFIG_REGISTRATION

