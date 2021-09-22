from os.path import join, exists
from os import makedirs

# Data directories
IMAGES_DIR = '/path/to/images_original_resolution'
IMAGES_RESAMPLED_DIR = '/path/to/resampled_1x1x1_images' #optional
SEGMENTATION_DIR = '/path/to/segmentations_at_resampled_resolution'
MASKS_DIR = '/path/to/masks_at_resamples_resolution' #this will be created during the preprocessing step

# Functions pointers
NIFTY_REG_DIR = '/path/to/NiftyReg_root_dir'

# Results root directory
REGISTRATION_DIR = '/path/to/registration_results_dir'


# Output directories - automatically created
PREPROCESSING_DIR = join(REGISTRATION_DIR, 'Preprocessing')
OBSERVATIONS_DIR_LINEAL = join(REGISTRATION_DIR, 'Linear', 'Registrations')
OBSERVATIONS_DIR_REGNET = join(REGISTRATION_DIR, 'Nonlinear', 'Registrations', 'RegNet')
OBSERVATIONS_DIR_NR = join(REGISTRATION_DIR, 'Nonlinear', 'Registrations', 'NiftyReg_f3d')

ALGORITHM_DIR_LINEAR = join(REGISTRATION_DIR, 'Linear', 'ST')
ALGORITHM_DIR = join(REGISTRATION_DIR, 'Nonlinear', 'ST')

if not exists(PREPROCESSING_DIR): makedirs(PREPROCESSING_DIR)
if not exists(OBSERVATIONS_DIR_REGNET): makedirs(OBSERVATIONS_DIR_REGNET)
if not exists(OBSERVATIONS_DIR_NR): makedirs(OBSERVATIONS_DIR_NR)
if not exists(OBSERVATIONS_DIR_LINEAL):  makedirs(OBSERVATIONS_DIR_LINEAL)

if not exists(ALGORITHM_DIR): makedirs(ALGORITHM_DIR)
if not exists(ALGORITHM_DIR_LINEAR): makedirs(ALGORITHM_DIR_LINEAR)