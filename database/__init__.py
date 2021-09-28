from os.path import join, exists
from os import makedirs
import csv

from setup import *
from src.utils.io import create_results_dir

def read_demo_info(demo_fields):
    data_dict = {}
    with open(SUBJECTS_DEMO_DATA_FILE, 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                tp_dict = {f: row[f] for f in demo_fields}
                data_dict[row['TIMEPOINT']] = tp_dict

    return data_dict


# Output directories class
class Results(object):
    def __init__(self, sid):
        self.base_dir = join(REGISTRATION_DIR, sid)
        create_results_dir(self.base_dir, subdirs=['Preprocessing', 'Linear', 'Nonlinear'])

        self._path_directory = {
            'preprocessing': join(self.base_dir, 'Preprocessing'),
            'preprocessing_resample_centered': join(self.base_dir, 'Preprocessing', 'resampled_centered'),
            'preprocessing_mask_centered': join(self.base_dir, 'Preprocessing', 'mask_centered'),
            'preprocessing_mask_centered_dilated': join(self.base_dir, 'Preprocessing', 'mask_centered_dilated'),
            'preprocessing_cog': join(self.base_dir, 'Preprocessing', 'cog'),

            'linear': join(self.base_dir, 'Linear'),
            'linear_registration': join(self.base_dir, 'Linear', 'Registrations'),
            'linear_st': join(self.base_dir, 'Linear', 'ST'),
            'linear_data': join(self.base_dir, 'Linear', 'Data'),
            'linear_image': join(self.base_dir, 'Linear', 'Data', 'image'),
            'linear_resample': join(self.base_dir, 'Linear', 'Data', 'resample'),
            'linear_mask': join(self.base_dir, 'Linear', 'Data', 'mask'),
            'linear_mask_dilated': join(self.base_dir, 'Linear', 'Data', 'mask'),
            'linear_seg': join(self.base_dir, 'Linear', 'Data', 'seg'),

            'linear_resampled': join(self.base_dir, 'Linear', 'Data_resampled'),
            'linear_resampled_image': join(self.base_dir, 'Linear', 'Data_resampled', 'image'),
            'linear_resampled_resample': join(self.base_dir, 'Linear', 'Data_resampled', 'resample'),
            'linear_resampled_mask': join(self.base_dir, 'Linear', 'Data_resampled', 'mask'),
            'linear_resampled_mask_dilated': join(self.base_dir, 'Linear', 'Data_resampled', 'mask'),
            'linear_resampled_seg': join(self.base_dir, 'Linear', 'Data_resampled','seg'),

            'nonlinear': join(self.base_dir, 'Nonlinear'),
            'nonlinear_registration': join(self.base_dir, 'Nonlinear', 'Registrations'),
            'nonlinear_st': join(self.base_dir, 'Nonlinear', 'ST'),
            'nonlinear_data': join(self.base_dir, 'Nonlinear', 'Data'),
            'nonlinear_image': join(self.base_dir, 'Nonlinear', 'Data', 'image'),
            'nonlinear_resample': join(self.base_dir, 'Nonlinear', 'Data', 'resample'),
            'nonlinear_mask': join(self.base_dir, 'Nonlinear', 'Data', 'mask'),
            'nonlinear_mask_dilated': join(self.base_dir, 'Nonlinear', 'Data', 'mask'),
            'nonlinear_seg': join(self.base_dir, 'Nonlinear', 'Data', 'seg'),
            # 'nonlinear_data_regnet': join(self.base_dir, 'Nonlinear', 'Data', 'RegNet'),
            # 'nonlinear_image_regnet': join(self.base_dir, 'Nonlinear', 'Data', 'RegNet', 'image'),
            # 'nonlinear_resample_regnet': join(self.base_dir, 'Nonlinear', 'Data', 'RegNet', 'resample'),
            # 'nonlinear_mask_regnet': join(self.base_dir, 'Nonlinear', 'Data', 'RegNet', 'mask'),
            # 'nonlinear_seg_regnet': join(self.base_dir, 'Nonlinear', 'Data', 'RegNet', 'seg'),
        }

        for k, v in self._path_directory.items():
            if not exists(v):
                makedirs(v)

    def get_dir(self, directory):
        return self._path_directory[directory]

    def get_available_dirs(self):
        available_dirs = []
        for k,v in self._path_directory.items():
            if exists(v):
                available_dirs.append(k)

        return available_dirs
