from src.utils.data_loader_utils import ScaleNormalization, NonLinearParams, AffineParams

CONFIG_REGISTRATION = {

    'TRANSFORM': None,
    'VOLUME_SHAPE': None,

    'DATA_AUGMENTATION': None,
    'NORMALIZATION': ScaleNormalization(range=[0, 1]),
    'AFFINE': None,#AffineParams(rotation=[2.5]*3, scaling=[0]*3, translation=[2]*3),
    'NONLINEAR': NonLinearParams(lowres_strength=[0, 2], lowres_shape_factor=0.04, distribution='uniform'),

    'ENC_NF': [16, 32, 32, 64],
    'DEC_NF': [64, 32, 32, 16, 16],

    'INT_STEPS': 7,

    'BATCH_SIZE': 2,
    'N_EPOCHS': 100,
    'LEARNING_RATE': 2e-4,
    'EPOCH_DECAY_LR': 0,
    'STARTING_EPOCH': 0,

    'USE_GPU': True,# Set to False if running to CPU
    'GPU_INDICES': [0],

    'LOG_INTERVAL': 1,
    'SAVE_MODEL_FREQUENCY': 100,

    'LOSS_REGISTRATION': {'name': 'L2', 'params': {},'lambda': 100},#{'name': 'NCC', 'params': {'kernel_var': [5]*3, 'kernel_type': 'mean'},'lambda': 1},
    'LOSS_REGISTRATION_SMOOTHNESS': {'name': 'Grad', 'params': {'dim': 3, 'penalty': 'l2'}, 'lambda': 0.2},#0.1},#

    'UPSAMPLE_LEVELS': 4,#1,#
    'FIELD_TYPE': 'velocity',
}