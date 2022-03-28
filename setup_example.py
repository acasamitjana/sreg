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

if not exists(MASKS_DIR):
    makedirs(MASKS_DIR)

print('     ')
print('     ')
print('IMAGES DIR: ' + IMAGES_DIR)
print('IMAGES RESAMPLED DIR: ' + IMAGES_RESAMPLED_DIR)
print('SEGMENTATION DIR: ' + SEGMENTATION_DIR)
print('MASKS DIR: ' + IMAGES_DIR)



LABEL_DICT = {
    'Background': 0,

    'Right-Hippocampus': 53,
    'Left-Hippocampus': 17,

    'Right-Lateral-Ventricle': 43,
    'Left-Lateral-Ventricle': 4,

    'Right-Thalamus': 49,
    'Left-Thalamus': 10,

    'Right-Amygdala': 54,
    'Left-Amygdala': 18,

    'Right-Putamen': 51,
    'Left-Putamen': 12,

    'Right-Pallidum': 52,
    'Left-Pallidum': 13,

    'Right-Cerebrum-WM': 41,
    'Left-Cerebrum-WM': 2,

    'Right-Cerebellar-WM': 46,
    'Left-Cerebellar-WM': 7,

    'Right-Cerebrum-GM': 42,
    'Left-Cerebrum-GM': 3,

    'Right-Cerebellar-GM': 47,
    'Left-Cerebellar-GM': 8,

    'Right-Caudate': 50,
    'Left-Caudate': 11,

    'Brainstem': 16,
    '4th-Ventricle': 15,
    '3rd-Ventricle': 14,

    'Right-Accumbens': 58,
    'Left-Accumbens': 26,

    'Right-VentralDC': 60,
    'Left-VentralDC': 28,

    'Right-Inf-Lat-Ventricle': 44,
    'Left-Inf-Lat-Ventricle': 5,

}

LABEL_DICT = {k: LABEL_DICT[k] for k in sorted(LABEL_DICT.keys(), key=lambda x: LABEL_DICT[x])}
LABEL_DICT_REVERSE = {v: k for k,v in LABEL_DICT.items()}
LABEL_LUT = {k: it_k for it_k, k in enumerate(LABEL_DICT.values())}

def get_labels(lablist):
    return {k: v for k, v in LABEL_DICT.items() if k in lablist}


SYNTHSEG_LABELS = {
    0: 'background',
    2: 'left cerebral white matter',
    3: 'left cerebral cortex',
    4: 'left lateral ventricle',
    5: 'left inferior lateral ventricle',
    7: 'left cerebellum white matter',
    8: 'left cerebellum cortex',
    10: 'left thalamus',
    11: 'left caudate',
    12: 'left putamen',
    13: 'left pallidum',
    14: '3rd ventricle',
    15: '4th ventricle',
    16: 'brain-stem',
    17: 'left hippocampus',
    18: 'left amygdala',
    26: 'left accumbens area',
    28: 'left ventral DC',
    41: 'right cerebral white matter',
    42: 'right cerebral cortex',
    43: 'right lateral ventricle',
    44: 'right inferior lateral ventricle',
    46: 'right cerebellum white matter',
    47: 'right cerebellum cortex',
    49: 'right thalamus',
    50: 'right caudate',
    51: 'right putamen',
    52: 'right pallidum',
    53: 'right hippocampus',
    54: 'right amygdala',
    58: 'right accumbens area',
    60: 'right ventral DC',
}
SYNTHSEG_LUT = {k: it_k for it_k, k in enumerate(SYNTHSEG_LABELS.keys())}


HP_LABEL_DICT = {
    'Parasubiculum': 203,
    'HATA': 211,
    'Fimbria': 212,
    'Hippocampal_fissure': 215,
    'HP_tail':  226,
    'Presubiculum_head':  233,
    'Presubiculum_body':  234,
    'Subiculum_head': 235,
    'Subiculum_body': 236,
    'CA1-head': 237,
    'CA1-body': 238,
    'CA3-head': 239,
    'CA3-body': 240,
    'CA4-head': 241,
    'CA4-body': 242,
    'GC-ML-DG-head': 243,
    'GC-ML-DG-body': 244,
    'molecular_layer_HP-head':  245,
    'molecular_layer_HP-body': 246,
}

AM_LABEL_DICT = {
    'Lateral-nucleus': 7001,
    'Basal-nucleus': 7003,
    'Central-nucleus': 7005,
    'Medial-nucleus': 7006,
    'Cortical-nucleus': 7007,
    'Accessory-Basal-nucleus': 7008,
    'Corticoamygdaloid-transitio': 7009,
    'Anterior-amygdaloid-area-AAA': 7010,
    'Paralaminar-nucleus': 7015,
}

SUBFIELDS_LABEL_DICT = {**{'Background': 0}, **HP_LABEL_DICT, **AM_LABEL_DICT}
SUBFIELDS_LABEL_DICT = {k: SUBFIELDS_LABEL_DICT[k] for k in sorted(SUBFIELDS_LABEL_DICT.keys(), key=lambda x: SUBFIELDS_LABEL_DICT[x])}
SUBFIELDS_LABEL_DICT_REVERSE = {v: k for k,v in SUBFIELDS_LABEL_DICT.items()}
SUBFIELDS_LABEL_LUT = {k: it_k for it_k, k in enumerate(SUBFIELDS_LABEL_DICT.values())}

EXTENDED_SUBFIELDS_LABEL_DICT = SUBFIELDS_LABEL_DICT.copy()
EXTENDED_SUBFIELDS_LABEL_DICT['TotalHP'] = 10000
EXTENDED_SUBFIELDS_LABEL_DICT['TotalAM'] = 10001
EXTENDED_SUBFIELDS_LABEL_DICT = {k: EXTENDED_SUBFIELDS_LABEL_DICT[k] for k in sorted(EXTENDED_SUBFIELDS_LABEL_DICT.keys(), key=lambda x: EXTENDED_SUBFIELDS_LABEL_DICT[x])}
EXTENDED_SUBFIELDS_LABEL_DICT_REVERSE = {v: k for k,v in EXTENDED_SUBFIELDS_LABEL_DICT.items()}
EXTENDED_SUBFIELDS_LABEL_DICT_LUT = {k: it_k for it_k, k in enumerate(EXTENDED_SUBFIELDS_LABEL_DICT.values())}
