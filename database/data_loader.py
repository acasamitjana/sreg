from os.path import join, exists
import copy
import os

import nibabel as nib
import numpy as np

from setup import *
from database import Results

class Timepoint(object):

    def __init__(self, tid, sid, file_extension='.nii.gz'):
        self.sid = sid
        self.tid = tid
        self.file_extension = file_extension
        self.filename = tid + file_extension
        self.results_dirs = Results(sid)

        self.data_path = {
            'image': join(IMAGES_DIR, sid, str(tid) + file_extension),
            'resample': join(IMAGES_RESAMPLED_DIR, sid + '_resample', str(tid) + '_resampled' + file_extension),
            'mask': join(MASKS_DIR, sid, str(tid) + file_extension),
            'mask_dilated': join(MASKS_DIR, sid, str(tid) + '.dilated' + file_extension),
            'seg': join(SEGMENTATION_DIR, sid + '_seg', str(tid) + '_synthseg' + file_extension),
        }

    def get_filepath(self, data_type):
        if 'cog' in data_type:
            return join(self.results_dirs.get_dir(data_type), self.tid + '.cog.npy')

        if 'mask_dilated' in data_type:
            return join(self.results_dirs.get_dir(data_type), self.tid + '.dilated' + self.file_extension)
        else:
            return join(self.results_dirs.get_dir(data_type), self.filename)


    def get_cog(self):
        return np.load(join(self.results_dirs.get_dir('preprocessing_cog'), self.tid + '.cog.npy'))[:3]

        # if not exists(self.image_centered_path):
        #     raise ValueError("Please, run database/preprocess_dataset.py first.")
        #
        # proxy = nib.load(self.image_path)
        # v1 = proxy.affine
        # proxy = nib.load(self.image_centered_path)
        # v2 = proxy.affine
        #
        # return v1[:3, 3] - v2[:3, 3]

    def load_data_orig(self, *args, **kwargs):
        proxy = nib.load(join(self.data_path['image']))
        return np.asarray(proxy.dataobj)

    def load_data(self, centered=False, *args, **kwargs):
        if centered:
            proxy = nib.load(join(self.results_dirs.get_dir('preprocessing_resample'), self.tid + '.centered' + self.file_extension))
        else:
            proxy = nib.load(self.data_path['resample'])
        return np.asarray(proxy.dataobj)

    def load_mask(self, dilated=False, centered=False, *args, **kwargs):

        if dilated and centered:
            proxy = nib.load(join(self.results_dirs.get_dir('preprocessing_mask'), self.tid + '.dilated.centered' + self.file_extension))

        elif centered:
            proxy = nib.load(join(self.results_dirs.get_dir('preprocessing_mask'), self.tid + '.centered' + self.file_extension))
        else:
            proxy = nib.load(self.data_path['mask'])

        return (np.asarray(proxy.dataobj) > 0).astype('uint8')

    def load_seg(self, *args, **kwargs):
        proxy = nib.load(self.data_path['seg'])
        return np.asarray(proxy.dataobj)

    @property
    def id(self):
        return self.tid

    @property
    def image_shape(self):
        proxy = nib.load(self.data_path['resample'])
        return proxy.shape

    @property
    def vox2ras0(self):
        proxy = nib.load(self.data_path['resample'])
        return copy.copy(proxy.affine)


class Timepoint_linear(Timepoint):

    def load_data_orig(self, *args, **kwargs):
        filepath = join(self.results_dirs.get_dir('linear_image'), self.filename)
        proxy = nib.load(filepath)
        return np.asarray(proxy.dataobj)

    def load_data(self, resampled=False, *args, **kwargs):
        data_dir = 'linear_resampled_resample' if resampled else 'linear_resample'
        proxy = nib.load(join(self.results_dirs.get_dir(data_dir), self.filename))
        return np.asarray(proxy.dataobj)

    def load_mask(self, resampled=False, dilated=False, *args, **kwargs):
        data_dir = 'linear_resampled_mask' if resampled else 'linear_mask'
        filename = self.tid + '.dilated' + self.file_extension if dilated else self.filename
        proxy = nib.load(join(self.results_dirs.get_dir(data_dir), filename))
        return np.asarray(proxy.dataobj)

    def load_seg(self, resampled=False, *args, **kwargs):
        data_dir = 'linear_resampled_seg' if resampled else 'linear_seg'
        proxy = nib.load(join(self.results_dirs.get_dir(data_dir), self.filename))
        return np.asarray(proxy.dataobj)

    @property
    def vox2ras0(self):
        proxy = nib.load(join(self.results_dirs.get_dir('linear_resample'), self.filename))
        return copy.copy(proxy.affine)


class Subject(object):

    def __init__(self, sid):
        self.sid = sid
        self.slice_dict = {}
        self.subject_dir = join(REGISTRATION_DIR)

        self.results_dirs = Results(sid)

        self.linear_template = join(self.results_dirs.get_dir('linear_data'),  'linear_template.nii.gz')
        self.nonlinear_template = join(self.results_dirs.get_dir('nonlinear_data'), 'nonlinear_template.nii.gz')

    def set_timepoint(self, tid, file_extension):
        self.slice_dict[tid] = Timepoint(tid, self.sid, file_extension=file_extension)

    def get_timepoint(self, tid=None):
        if tid is None:
            return list(self.slice_dict.values())[0]
        else:
            return self.slice_dict[tid]

    @property
    def id(self):
        return self.sid

    @property
    def tid_list(self):
        return list(self.slice_dict.keys())

    @property
    def timepoints(self):
        return list(self.slice_dict.values())

    @property
    def vox2ras0(self):
        if exists(self.linear_template):
            proxy = nib.load(self.linear_template)
            return proxy.affine
        else:
            return ValueError("The Linear template has not already been created, so there is no common SUBJECT space.")

    @property
    def image_shape(self):
        max_shape = [0,0,0]
        for sl in self.timepoints:
            image_shape = sl.image_shape
            for it in range(3):
                if image_shape[it] > max_shape[it]: max_shape[it] = image_shape[it]

        return tuple(max_shape)


class Subject_linear(Subject):
    def set_timepoint(self, tid, file_extension):
        self.slice_dict[tid] = Timepoint_linear(tid, self.sid, file_extension=file_extension)

    @property
    def image_shape(self):
        proxy = nib.load(self.linear_template)
        return proxy.shape

class DataLoader(object):
    def __init__(self, **kwargs):
        self.subject_dict = {}
        self._initialize_dataset(**kwargs)

    def _initialize_dataset(self, sid_list=None, linear=False):

        subjects = os.listdir(IMAGES_DIR) if not linear else os.listdir(REGISTRATION_DIR)

        for sbj in subjects:
            if sid_list is not None:
                if sbj not in sid_list:
                            continue

            self.subject_dict[sbj] = Subject(sbj) if not linear else Subject_linear(sbj)

            timepoints = os.listdir(join(IMAGES_DIR, sbj))
            for tp in timepoints:
                if '.nii.gz' in tp:
                    tp = tp[:-7]
                    file_extension = '.nii.gz'
                elif '.nii' in tp:
                    tp = tp[:-4]
                    file_extension = '.nii'
                elif '.mgz' in tp:
                    tp = tp[:-4]
                    file_extension = '.mgz'
                else:
                    raise ValueError("Please, provide a valid .nii, .nii.gz or .mgz file")

                self.subject_dict[sbj].set_timepoint(tp, file_extension)


    @property
    def sid_list(self):
        return list(self.subject_dict.keys())

    @property
    def subject_list(self):
        return list(self.subject_dict.values())

    @property
    def image_shape(self):
        max_shape = [0, 0, 0]
        for sbj in self.subject_list:
            image_shape = sbj.image_shape
            for it in range(3):
                if image_shape[it] > max_shape[it]: max_shape[it] = image_shape[it]

        return tuple(max_shape)

    def __len__(self):
            return len(self.subject_dict.keys())

