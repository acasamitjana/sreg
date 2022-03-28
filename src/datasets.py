# py
import time

# third party imports
import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
from skimage.morphology import ball
from scipy.ndimage.morphology import binary_dilation

#project imports
from src.utils import data_loader_utils as tf
from src.utils.image_utils import one_hot_encoding


class IntraModalSampler(Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)

        N = 0
        indices_dict = {} #data_loader
        for it_subject, subject in enumerate(data_source):
            tp_list = subject.tid_list
            np.random.shuffle(tp_list)
            indices_dict[it_subject] = tp_list
            N += len(tp_list)

        self.N = N
        self.indices_dict = indices_dict

    def __iter__(self):
        for tp_idx in range(12):
            for sbj, tp_list in self.indices_dict.items():
                if len(tp_list) > tp_idx:
                    tp_ref = tp_list[tp_idx]
                    tp_flo = np.random.choice([tp for tp in tp_list if tp != tp_ref], 1, replace=True)[0]
                    yield sbj, tp_ref, tp_flo

    def __len__(self):
        return self.N

class IntraModalTrainingSampler(Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)

        N = 0
        indices_dict = {} #data_loader
        for it_subject, subject in enumerate(data_source):
            tp_list = subject.tid_list
            np.random.shuffle(tp_list)
            indices_dict[it_subject] = tp_list
            N += len(tp_list)

        self.N = N
        self.indices_dict = indices_dict

    def __iter__(self):
        for sbj, tp_list in self.indices_dict.items():
            for tp_ref in tp_list:
                tp_flo = np.random.choice([tp for tp in tp_list if tp != tp_ref], 1, replace=True)[0]
                yield sbj, tp_ref, tp_flo

    def __len__(self):
        return self.N


class RegistrationDataset(Dataset):
    def __init__(self, data_loader, affine_params=None, nonlinear_params=None, tf_params=None, da_params=None,
                 norm_params=None, mask_dilation=False, to_tensor=True, num_classes=False, train=True):
        '''

        :param data_loader:
        :param affine_params:
        :param nonlinear_params:
        :param tf_params:
        :param da_params:
        :param norm_params:
        :param hist_match:
        :param mask_dilation:
        :param to_tensor:
        :param landmarks:
        :param num_classes: (int) number of classes for one-hot encoding. If num_classes=-1, one-hot is not performed.
        :param train:
        '''

        self.data_loader = data_loader
        self.N = len(self.data_loader)
        self.to_tensor = to_tensor

        self.tf_params = tf.Compose(tf_params) if tf_params is not None else None
        self.da_params = tf.Compose_DA(da_params) if da_params is not None else None
        self.norm_params = norm_params if norm_params is not None else lambda x: x
        self.mask_dilation = mask_dilation

        self.affine_params = affine_params
        self.nonlinear_params = nonlinear_params
        self.num_classes = num_classes

        image_shape = self._get_init_shape(data_loader)
        self._image_shape = self.tf_params._compute_data_shape(image_shape) if tf_params is not None else image_shape
        self.n_dims = None
        self.train = train


    def _get_init_shape(self, data):
        if isinstance(data[0], list):
            return data[0][0].image_shape
        else:
            return data[0].image_shape

    def mask_image(self, image, mask):
        raise NotImplementedError

    def get_deformation_field(self, num=2):
        raise NotImplementedError

    def get_data(self, slice, *args, **kwargs):

        t_0 = time.time()
        x = slice.load_data_orig(resampled=True, *args, **kwargs)
        t_1 = time.time()
        x_mask = slice.load_mask(resampled=True, dilated=False, *args, **kwargs)
        t_2 = time.time()
        se = ball(1)
        x_mask = binary_dilation(x_mask, se)
        if np.sum(x_mask) > 0:
            x = self.mask_image(x, x_mask)
            x = self.norm_params(x)
        t_3 = time.time()

        if self.mask_dilation:
            x_mask = slice.load_mask(resampled=True, dilated=True, *args, **kwargs)

        x_labels = slice.load_seg(resampled=True, *args, **kwargs)
        t_4 = time.time()

        # print('        - Get data ' + str(t_1 - t_0))
        # print('        - Get mask ' + str(t_2 - t_1))
        # print('        - Dilation ' + str(t_3 - t_2))
        # print('        - Get mask and labels ' + str(t_4 - t_3))

        return x, x_mask, x_labels


    def get_intramodal_data(self, slice_ref, slice_flo,  *args, **kwargs):
        t_0 = time.time()

        x_ref, x_ref_mask, x_ref_labels = self.get_data(slice_ref, *args, **kwargs)

        t_1 = time.time()

        x_flo, x_flo_mask, x_flo_labels = self.get_data(slice_flo,  *args, **kwargs)
        x_flo_orig = x_flo

        t_2 = time.time()

        if self.tf_params is not None:
            img = [x_ref, x_flo, x_flo_orig, x_ref_mask, x_flo_mask, x_ref_labels, x_flo_labels]
            img_tf = self.tf_params(img)
            x_ref, x_flo, x_flo_orig, x_ref_mask, x_flo_mask, x_ref_labels, x_flo_labels = img_tf


        data_dict = {
            'x_ref': x_ref, 'x_flo': x_flo, 'x_flo_orig': x_flo_orig,
            'x_ref_mask': x_ref_mask, 'x_flo_mask': x_flo_mask,
            'x_ref_labels': x_ref_labels, 'x_flo_labels': x_flo_labels,
        }

        # print('     - Get ref ' + str(t_1 - t_0))
        # print('     - Get flo ' + str(t_2 - t_1))

        return data_dict

    def data_augmentation(self, data_dict):
        x_ref = data_dict['x_ref']
        x_ref_mask = data_dict['x_ref_mask']
        x_ref_labels = data_dict['x_ref_labels']
        x_flo = data_dict['x_flo']
        x_flo_mask = data_dict['x_flo_mask']
        x_flo_labels = data_dict['x_flo_labels']

        if self.da_params is not None:
            img = self.da_params([x_ref, x_ref_mask, x_ref_labels], mask_flag=[False, True, True])
            x_ref, x_ref_mask, x_ref_labels = img
            x_ref = x_ref * x_ref_mask
            x_ref[np.isnan(x_ref)] = 0
            x_ref_mask[np.isnan(x_ref_mask)] = 0
            x_ref_labels[np.isnan(x_ref_labels)] = 0

            img = self.da_params([x_flo, x_flo_mask, x_flo_labels], mask_flag=[False, True, True])
            x_flo, x_flo_mask, x_flo_labels = img
            x_flo = x_flo * x_flo_mask
            x_flo[np.isnan(x_flo)] = 0
            x_flo_mask[np.isnan(x_flo_mask)] = 0
            x_flo_labels[np.isnan(x_flo_labels)] = 0

        # if self.mask_dilation is not None:
        #     ndims = len(x_ref_mask.shape)
        #     se_shape = tuple([3 for _ in range(ndims)])
        #     x_ref_mask = binary_dilation(x_ref_mask, structure=np.ones(se_shape),
        #                                  iterations=int((self.mask_dilation-3)/2 + 1))
        #     x_flo_mask = binary_dilation(x_flo_mask, structure=np.ones(se_shape),
        #                                  iterations=int((self.mask_dilation-3)/2 + 1))

        data_dict['x_ref'] = x_ref
        data_dict['x_ref_mask'] = x_ref_mask
        data_dict['x_ref_labels'] = x_ref_labels
        data_dict['x_flo'] = x_flo
        data_dict['x_flo_mask'] = x_flo_mask
        data_dict['x_flo_labels'] = x_flo_labels

        return data_dict

    def convert_to_tensor(self, data_dict):

        if not self.to_tensor:
            return data_dict

        for k, v in data_dict.items():
            if 'labels' in k and self.num_classes:
                v = one_hot_encoding(v, self.num_classes) if self.num_classes else v
                data_dict[k] = torch.from_numpy(v).float()
            elif isinstance(v, list):
                data_dict[k] = [torch.from_numpy(vl).float() for vl in v]
            else:
                data_dict[k] = torch.from_numpy(v[np.newaxis]).float()

        return data_dict

    def __getitem__(self, index):

        t_0 = time.time()

        if isinstance(index, int):
            subject = self.data_loader[index]
            if isinstance(subject, list):
                tp_ref = subject[0]
                tp_flo = subject[1]
                subject_id = ''

            else:
                tid_ref, tid_flo = np.random.choice(subject.tid_list, 2, replace=True)
                tp_ref = subject.get_timepoint(tid_ref)
                tp_flo = subject.get_timepoint(tid_flo)
                subject_id = subject.id + '.'
        else:

            sbj_idx, tid_ref, tid_flo = index
            subject = self.data_loader[sbj_idx]
            tp_ref = subject.get_timepoint(tid_ref)
            tp_flo = subject.get_timepoint(tid_flo)
            subject_id = subject.id + '.'

        t_1 = time.time()

        data_dict = self.get_intramodal_data(tp_ref, tp_flo)

        t_2 = time.time()

        data_dict = self.data_augmentation(data_dict)

        t_3 = time.time()

        affine, nonlinear_field = self.get_deformation_field(num=2)

        t_4 = time.time()

        if affine:
            data_dict['affine'] = affine
        if nonlinear_field:
            data_dict['nonlinear'] = nonlinear_field
        data_dict = self.convert_to_tensor(data_dict)

        rid = subject_id + str(tp_ref.tid) + '_to_' + str(tp_flo.tid)
        data_dict['rid'] = rid
        data_dict['ref_vox2ras0'] = tp_ref.vox2ras0
        data_dict['flo_vox2ras0'] = tp_flo.vox2ras0

        t_5 = time.time()

        # print('  - Get slice ' + str(t_1-t_0))
        # print('  - Get_intramodal_data ' + str(t_2-t_1))
        # print('  - Data_augmentation ' + str(t_3-t_2))
        # print('  - get_deformation_field ' + str(t_4-t_3))
        # print('  - convert_tensor ' + str(t_5-t_4))


        return data_dict

    def __len__(self):
        return self.N

    @property
    def image_shape(self):
        return self._image_shape


class RegistrationDataset2D(RegistrationDataset):
    '''
    Class for intermodal registration where input data, output target and the correponding transformations or data
    augmentation are specified
    '''


    def mask_image(self, image, mask):
        ndim = len(image.shape)

        if ndim == 3:
            for it_z in range(image.shape[-1]):
                image[..., it_z] = image[..., it_z] * mask
        else:
            image = image*mask

        return image

    def get_deformation_field(self, num=2):
        affine_list = []
        nonlinear_field_list = []
        for it_i in range(num):
            if self.affine_params is not None: #np.eye(4)
                affine = self.affine_params.get_affine(self._image_shape) if self.affine_params is not None else None
                affine_list.append(affine)

            if self.nonlinear_params is not None:
                nlf_xyz = self.nonlinear_params.get_lowres_strength(self._image_shape)
                svf = np.concatenate([nlf[np.newaxis] for nlf in nlf_xyz], axis=0)
                nonlinear_field_list.append(svf)

        return affine_list, nonlinear_field_list


class RegistrationDataset3D(RegistrationDataset):
    '''
    Class for intermodal registration where input data, output target and the correponding transformations or data
    augmentation are specified
    '''


    def mask_image(self, image, mask):
        image[mask<0.1] = 0
        return image

    def get_deformation_field(self, num=2):

        # if self.affine_params is None and self.nonlinear_params is None:
        #     affine_list = [np.eye(4)]*num
        #     nonlinear_field_list = [np.zeros((3,9,9,9))]*num
        #
        #     return affine_list, nonlinear_field_list

        affine_list = []
        nonlinear_field_list = []
        for it_i in range(num):
            if self.affine_params is not None: #np.eye(4)
                affine = self.affine_params.get_affine(self._image_shape)
                affine_list.append(affine)

            if self.nonlinear_params is not None:
                nlf_xyz = self.nonlinear_params.get_lowres_strength(self._image_shape)
                svf = np.concatenate([nlf[np.newaxis] for nlf in nlf_xyz], axis=0)
                nonlinear_field_list.append(svf)

        return affine_list, nonlinear_field_list


