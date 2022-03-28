from os.path import join, exists
import copy
import time

import nibabel as nib
import numpy as np
from skimage.morphology import ball
from scipy.ndimage.morphology import binary_dilation

from setup import *
from database import Results
from src.utils import image_utils, io_utils

class Timepoint(object):

    def __init__(self, tid, sid, file_extension='.nii.gz'):
        self.sid = sid
        self.tid = tid
        self.file_extension = file_extension
        self.filename = tid + '.nii.gz'
        self.results_dirs = Results(sid)

        self.init_path = {
            'image': join(IMAGES_DIR, sid, str(tid) + file_extension),
            'resample': join(IMAGES_RESAMPLED_DIR, sid, str(tid) + '_resampled' + file_extension),
            'mask': join(MASKS_DIR, sid, str(tid) + file_extension),
            'mask_dilated': join(MASKS_DIR, sid, str(tid) + '.dilated' + file_extension),
            'seg': join(SEGMENTATION_DIR, sid, str(tid) + '_synthseg' + file_extension),
            'subfields.rh': join(SEGMENTATION_DIR, sid, str(tid) + '_subfields.rh' + file_extension),
            'subfields.lh': join(SEGMENTATION_DIR, sid, str(tid) + '_subfields.lh' + file_extension),
            'posteriors': join(SEGMENTATION_DIR, sid, 'post', str(tid) + '_posteriors' + file_extension),
        }

    def get_filepath(self, data_type):
        if 'cog' in data_type:
            return join(self.results_dirs.get_dir(data_type), self.tid + '.cog.npy')

        if 'mask_dilated' in data_type:
            return join(self.results_dirs.get_dir(data_type), self.tid + '.dilated.nii.gz')
        else:
            return join(self.results_dirs.get_dir(data_type), self.filename)

    def get_cog(self):
        return np.load(join(self.results_dirs.get_dir('preprocessing_cog'), self.tid + '.cog.npy'))[:3]

    def load_data_orig(self, *args, **kwargs):
        proxy = nib.load(join(self.results_dirs.get_dir('preprocessing_image'), self.tid + '.nii.gz'))
        # proxy = nib.load(join(self.data_path['image']))
        return np.asarray(proxy.dataobj)

    def load_data(self, centered=False, *args, **kwargs):
        if centered:
            proxy = nib.load(join(self.results_dirs.get_dir('preprocessing_resample_centered'), self.filename))
        else:
            proxy = nib.load(join(self.results_dirs.get_dir('preprocessing_resample'), self.filename))
            # proxy = nib.load(self.data_path['resample'])
        return np.asarray(proxy.dataobj)

    def load_mask(self, dilated=False, centered=False, *args, **kwargs):

        if dilated and centered:
            proxy = nib.load(join(self.results_dirs.get_dir('preprocessing_mask_centered_dilated'), self.filename))
        elif centered:
            proxy = nib.load(join(self.results_dirs.get_dir('preprocessing_mask_centered'), self.filename))

        elif dilated:
            if not exists(self.init_path['mask_dilated']):
                mask = (self.load_seg() > 0).astype('uint8')
                se = ball(3)
                mask_dilated = binary_dilation(mask, se)
                proxy = nib.Nifti1Image(mask_dilated.astype('uint8'), self.vox2ras0)

            else:
                proxy = nib.load(self.init_path['mask_dilated'])

        else:
            if not exists(self.init_path['mask']):
                proxy = nib.load(self.init_path['seg'])
            else:
                proxy = nib.load(self.init_path['mask'])

        return (np.asarray(proxy.dataobj) > 0).astype('uint8')

    def load_seg(self, *args, **kwargs):
        proxy = nib.load(self.init_path['seg'])
        return np.asarray(proxy.dataobj)

    def load_posteriors(self, labels=None, *args, **kwargs):

        if exists(self.init_path['posteriors']):
            proxy = nib.load(self.init_path['posteriors'])
            if labels is not None:
                return np.asarray(proxy.dataobj[..., labels])
            else:
                return np.asarray(proxy.dataobj)

        else:
            seg = self.load_seg()
            categories = list(SYNTHSEG_LABELS.keys())
            if labels is not None:
                categories = [c for it_c, c in enumerate(categories) if it_c in labels]
            post = np.transpose(image_utils.one_hot_encoding(seg, categories=categories), axes=(1, 2, 3, 0))
            return post


    @property
    def id(self):
        return self.tid

    @property
    def image_shape_orig(self):
        proxy = nib.load(self.init_path['image'])
        return proxy.shape


    @property
    def image_shape(self):
        proxy = nib.load(self.init_path['resample'])
        return proxy.shape

    @property
    def volres(self):
        aff = self.vox2ras0
        return np.sqrt(np.sum(aff * aff, axis=0))[:-1]

    @property
    def volres_orig(self):
        aff = self.vox2ras0_orig
        return np.sqrt(np.sum(aff * aff, axis=0))[:-1]


    @property
    def vox2ras0(self):
        proxy = nib.load(self.init_path['resample'])
        return copy.copy(proxy.affine)

    @property
    def vox2ras0_orig(self):
        proxy = nib.load(self.init_path['image'])
        return copy.copy(proxy.affine)

    @property
    def vox2ras0_centered(self):
        proxy = nib.load(self.get_filepath('preprocessing_resample_centered'))
        return copy.copy(proxy.affine)


class Timepoint_linear(Timepoint):

    def load_data_orig(self, resampled=False, *args, **kwargs):
        if resampled:
            proxy = nib.load(join(self.results_dirs.get_dir('linear_resampled_image'), self.filename))
            data = np.asarray(proxy.dataobj)
            return data
        else:
            return super().load_data_orig()

    def load_data(self, resampled=False, *args, **kwargs):
        if resampled:
            proxy = nib.load(join(self.results_dirs.get_dir('linear_resampled_resample'), self.filename))
            data = np.asarray(proxy.dataobj)
            return data
        else:
            return super().load_data()

    def load_mask(self, resampled=False, dilated=False, *args, **kwargs):
        if resampled:
            proxy = nib.load(join(self.results_dirs.get_dir('linear_resampled_mask'), self.filename))
            data = np.asarray(proxy.dataobj)
            return data
        else:
            return super().load_mask()
        # data_dir = 'linear_resampled_mask' if resampled else 'linear_mask'
        # filename = self.tid + '.dilated.nii.gz' if dilated else self.filename
        # proxy = nib.load(join(self.results_dirs.get_dir(data_dir), filename))
        # data = np.asarray(proxy.dataobj)
        # return data

    def load_seg(self, resampled=False, *args, **kwargs):
        if resampled:
            proxy = nib.load(join(self.results_dirs.get_dir('linear_resampled_seg'), self.filename))
            data = np.asarray(proxy.dataobj)
            return data
        else:
            return super().load_seg()
        # data_dir = 'linear_resampled_seg' if resampled else 'linear_seg'
        # proxy = nib.load(join(self.results_dirs.get_dir(data_dir), self.filename))
        # data = np.asarray(proxy.dataobj)
        # return data

    # def load_posteriors(self, *args, **kwargs):
    #     proxy = nib.load(join(self.results_dirs.get_dir('linear_post'), self.filename))
    #     return np.asarray(proxy.dataobj)

    @property
    def vox2ras0(self):
        vox2ras0 = super().vox2ras0
        cog = self.get_cog()
        aff = io_utils.read_affine_matrix(join(self.results_dirs.get_dir('linear_st'), self.id + '.aff'), full=True)
        aff[:3, 3] += cog
        return np.matmul(np.linalg.inv(aff), vox2ras0)

    @property
    def vox2ras0_orig(self):
        vox2ras0 = super().vox2ras0_orig
        cog = self.get_cog()
        aff = io_utils.read_affine_matrix(join(self.results_dirs.get_dir('linear_st'), self.id + '.aff'), full=True)
        aff[:3, 3] += cog
        return np.matmul(np.linalg.inv(aff), vox2ras0)


class Subject(object):

    def __init__(self, sid):
        self.sid = sid
        self.slice_dict = {}
        self.subject_dir = join(REGISTRATION_DIR)

        self.results_dirs = Results(sid)

        self.linear_template = join(self.results_dirs.get_dir('linear_resampled'),  'linear_template.nii.gz')
        self.linear_template_orig = join(self.results_dirs.get_dir('linear_resampled'), 'linear_template.orig.nii.gz')

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

    @property
    def image_shape_template(self):
        proxy = nib.load(self.linear_template)
        return proxy.shape

class Subject_linear(Subject):
    def __init__(self, sid, reg_algorithm='bidir'):

        super(Subject_linear, self).__init__(sid)

        self.nonlinear_template = join(self.results_dirs.get_dir('nonlinear_data_' + reg_algorithm),
                                       'nonlinear_template.nii.gz')
        self.nonlinear_template_orig = join(self.results_dirs.get_dir('nonlinear_data_' + reg_algorithm),
                                            'nonlinear_template.orig.nii.gz')

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

    def _initialize_dataset(self, sid_list=None, linear=False, timepoints_filter=None, reg_algorithm='bidir'):

        subjects = os.listdir(IMAGES_DIR)

        if linear:
            subjects = os.listdir(REGISTRATION_DIR)
            subjects = list(filter(lambda x: exists(join(Results(x).get_dir('linear_resampled'), 'linear_template.nii.gz')), subjects))

        if EXCLUDED_SUBJECTS:
            subjects = list(filter(lambda x: x not in EXCLUDED_SUBJECTS, subjects))

        for sbj in subjects:
            if sid_list is not None:
                if sbj not in sid_list:
                            continue

            timepoints = os.listdir(join(IMAGES_DIR, sbj))
            if timepoints_filter is not None:
                timepoints = list(filter(timepoints_filter, timepoints))

            if len(timepoints) == 1:
                continue

            self.subject_dict[sbj] = Subject(sbj) if not linear else Subject_linear(sbj, reg_algorithm=reg_algorithm)
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


class DataLoaderRetest(object):

    def _initialize_dataset(self, sid_list=None, linear=False):

        subjects = os.listdir(IMAGES_DIR)

        if linear:
            subjects = os.listdir(REGISTRATION_DIR)
            subjects = list(filter(lambda x: exists(join(Results(x).get_dir('linear_data'), 'linear_template.nii.gz')), subjects))

        if EXCLUDED_SUBJECTS:
            subjects = list(filter(lambda x: x not in EXCLUDED_SUBJECTS, subjects))

        for sbj in subjects:
            if sid_list is not None:
                if sbj not in sid_list:
                            continue

            timepoints = os.listdir(join(IMAGES_DIR, sbj))
            if len(timepoints) == 1:
                continue

            self.subject_dict[sbj] = Subject(sbj) if not linear else Subject_linear(sbj)
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
