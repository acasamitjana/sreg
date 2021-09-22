from os.path import join, exists
import copy
import os

import nibabel as nib
import numpy as np

from setup import *


class Timepoint(object):

    def __init__(self, tid, sid, file_extension='.nii.gz'):
        self.sid = sid
        self.tid = tid


        self.image_orig_path = join(IMAGES_DIR, sid, str(tid) + file_extension)
        self.image_path = join(IMAGES_RESAMPLED_DIR, sid + '_resample', str(tid) + '_resampled' + file_extension)
        self.mask_path = join(MASKS_DIR, sid, str(tid) + file_extension)
        self.mask_dilated_path = join(MASKS_DIR, sid, str(tid) + '.dilated' + file_extension)
        self.seg_path = join(SEGMENTATION_DIR, sid + '_seg' , str(tid)  + '_synthseg' + file_extension)

        if not exists(self.image_path): self.image_path = self.image_orig_path

        if not exists(join(PREPROCESSING_DIR, 'image', sid, 'tmp')): os.makedirs(join(PREPROCESSING_DIR, 'image', sid, 'tmp'))
        if not exists(join(PREPROCESSING_DIR, 'resample', sid, 'tmp')): os.makedirs(join(PREPROCESSING_DIR, 'resample', sid, 'tmp'))
        if not exists(join(PREPROCESSING_DIR, 'masks', sid, 'tmp')): os.makedirs(join(PREPROCESSING_DIR, 'masks', sid, 'tmp'))
        if not exists(join(PREPROCESSING_DIR, 'seg', sid, 'tmp')): os.makedirs(join(PREPROCESSING_DIR, 'seg', sid, 'tmp'))

        if not exists(join(ALGORITHM_DIR_LINEAR, 'image', sid, 'tmp')): os.makedirs(join(ALGORITHM_DIR_LINEAR, 'image', sid, 'tmp'))
        if not exists(join(ALGORITHM_DIR_LINEAR, 'resample', sid, 'tmp')): os.makedirs(join(ALGORITHM_DIR_LINEAR, 'resample', sid, 'tmp'))
        if not exists(join(ALGORITHM_DIR_LINEAR, 'masks', sid, 'tmp')): os.makedirs(join(ALGORITHM_DIR_LINEAR, 'masks', sid, 'tmp'))
        if not exists(join(ALGORITHM_DIR_LINEAR, 'seg', sid, 'tmp')): os.makedirs(join(ALGORITHM_DIR_LINEAR, 'seg', sid, 'tmp'))

        self.image_centered_path = join(PREPROCESSING_DIR, 'resample', sid, 'tmp', str(tid) + '.centered' + file_extension)
        self.mask_centered_path = join(PREPROCESSING_DIR, 'masks', sid, 'tmp', str(tid) + '.centered' + file_extension)
        self.seg_centered_path = join(PREPROCESSING_DIR, 'seg', sid, 'tmp', str(tid) + '.centered' + file_extension)
        self.mask_dilated_centered_path = join(PREPROCESSING_DIR, 'masks', sid, 'tmp', str(tid) + '.dilated.centered' + file_extension)

        self.image_updatedH_path = join(ALGORITHM_DIR_LINEAR, 'image', sid, str(tid) + '.updated_header' + file_extension)
        self.image_resampled_updatedH_path = join(ALGORITHM_DIR_LINEAR, 'resample', sid, 'tmp', str(tid) + '.updated_header' + file_extension)
        self.mask_updatedH_path = join(ALGORITHM_DIR_LINEAR, 'masks', sid, 'tmp', str(tid) + '.dilated.updated_header' + file_extension)
        self.mask_dilated_updatedH_path = join(ALGORITHM_DIR_LINEAR, 'masks', sid, 'tmp', str(tid) + '.dilated.updated_header' + file_extension)
        self.seg_updatedH_path = join(ALGORITHM_DIR_LINEAR, 'seg', sid, str(tid) + '.dilated.updated_header' + file_extension)

        self.image_linear_path = join(ALGORITHM_DIR_LINEAR, 'image', sid, str(tid) + '.linear' + file_extension)
        self.image_resampled_linear_path = join(ALGORITHM_DIR_LINEAR, 'resample', sid, str(tid) + '.linear' + file_extension)
        self.mask_linear_path = join(ALGORITHM_DIR_LINEAR, 'masks', sid, str(tid) + '.linear' + file_extension)
        self.mask_dilated_linear_path = join(ALGORITHM_DIR_LINEAR, 'masks', sid, str(tid) + '.linear' + file_extension)
        self.seg_linear_path = join(ALGORITHM_DIR_LINEAR, 'seg', sid, str(tid) + '.linear' + file_extension)

    def get_cog(self):

        if not exists(self.image_centered_path):
            raise ValueError("Please, run database/preprocess_dataset.py first.")

        proxy = nib.load(self.image_path)
        v1 = proxy.affine
        proxy = nib.load(self.image_centered_path)
        v2 = proxy.affine

        return v1[:3, 3] - v2[:3, 3]

    def load_data_orig(self, *args, **kwargs):
        proxy = nib.load(self.image_orig_path)
        return np.asarray(proxy.dataobj)

    def load_data(self, centered=False, *args, **kwargs):
        proxy = nib.load(self.image_path) if not centered else nib.load(self.image_centered_path)
        return np.asarray(proxy.dataobj)

    def load_mask(self, dilated=False, centered=False, *args, **kwargs):

        if dilated and centered:
            proxy = nib.load(self.mask_dilated_centered_path)
        elif dilated:
            proxy = nib.load(self.mask_dilated_path)
        else:
            proxy = nib.load(self.mask_path)

        return (np.asarray(proxy.dataobj) > 0).astype('uint8')

    def load_seg(self, centered=False, *args, **kwargs):
        proxy = nib.load(self.seg_path) if not centered else nib.load(self.seg_centered_path)
        return np.asarray(proxy.dataobj)

    @property
    def id(self):
        return self.tid

    @property
    def image_shape(self):
        proxy = nib.load(self.image_path)
        return proxy.shape

    @property
    def vox2ras0(self):
        proxy = nib.load(self.image_path)
        return copy.copy(proxy.affine)


class Timepoint_linear(Timepoint):

    def __init__(self, tid, sid, file_extension='.nii.gz'):
        
        super(Timepoint_linear, self).__init__(tid, sid, file_extension)
        
        self.image_nonlinear_path = join(ALGORITHM_DIR, 'image', sid, str(tid) + '.nonlinear' + file_extension)
        self.image_resampled_nonlinear_path = join(ALGORITHM_DIR, 'resample', sid, str(tid) + '.nonlinear' + file_extension)
        self.mask_nonlinear_path = join(ALGORITHM_DIR, 'masks', sid, str(tid) + '.nonlinear' + file_extension)
        self.mask_dilated_nonlinear_path = join(ALGORITHM_DIR, 'masks', sid, str(tid) + '.nonlinear' + file_extension)
        self.seg_nonlinear_path = join(ALGORITHM_DIR, 'seg', sid, str(tid) + '.nonlinear' + file_extension)

        if not exists(join(ALGORITHM_DIR, 'image', sid)): os.makedirs(join(ALGORITHM_DIR, 'image', sid))
        if not exists(join(ALGORITHM_DIR, 'resample', sid)): os.makedirs(join(ALGORITHM_DIR, 'resample', sid))
        if not exists(join(ALGORITHM_DIR, 'masks', sid)): os.makedirs(join(ALGORITHM_DIR, 'masks', sid))
        if not exists(join(ALGORITHM_DIR, 'seg', sid)): os.makedirs(join(ALGORITHM_DIR, 'seg', sid))


    def load_data(self, centered=False, *args, **kwargs):
        proxy = nib.load(self.image_linear_path)
        return np.asarray(proxy.dataobj)

    def load_mask(self, dilated=False, centered=False, *args, **kwargs):
        proxy = nib.load(self.mask_dilated_linear_path) if dilated else nib.load(self.mask_linear_path)
        return (np.asarray(proxy.dataobj) > 0).astype('uint8')

    def load_seg(self, centered=False, *args, **kwargs):
        proxy = nib.load(self.seg_linear_path)
        return np.asarray(proxy.dataobj)


class Subject(object):

    def __init__(self, sid):
        self.sid = sid
        self.slice_dict = {}

        registration_linear_dir = join(REGISTRATION_DIR, 'Linear')
        registration_nonlinear_dir = join(REGISTRATION_DIR, 'Nonlinear')
        if not exists(join(registration_linear_dir, 'templates', sid)): os.makedirs(join(registration_linear_dir, 'templates', sid))
        if not exists(join(registration_nonlinear_dir, 'templates', sid)): os.makedirs(join(registration_nonlinear_dir, 'templates', sid))

        self.linear_template = join(registration_linear_dir, 'templates', sid + '.nii.gz')
        self.nonlinear_template = join(registration_nonlinear_dir, 'templates', sid + '.nii.gz')

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

        subjects = os.listdir(IMAGES_DIR)
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

