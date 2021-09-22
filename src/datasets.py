# py
import time

# third party imports
import torch
from torch.utils.data import Dataset
from scipy.ndimage.morphology import binary_dilation
import numpy as np

#project imports
from src.utils import image_transform as tf
from src.utils.image_utils import one_hot_encoding

class RegistrationDataset(Dataset):
    def __init__(self, data_loader, affine_params=None, nonlinear_params=None, tf_params=None, da_params=None, norm_params=None,
                 mask_dilation=False, to_tensor=True, image_centered=False, num_classes=False, train=True):
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

        self.image_centered = image_centered

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

        x = slice.load_data(centered=self.image_centered, *args, **kwargs)
        x_mask = slice.load_mask(dilated=self.mask_dilation, centered=self.image_centered, *args, **kwargs)
        x_labels = slice.load_seg(centered=self.image_centered,*args, **kwargs)
        x = self.mask_image(x, x_mask)

        if np.sum(x_mask) > 0:
            x = self.norm_params(x)
            x = self.mask_image(x, x_mask)

        return x, x_mask, x_labels

    def get_intramodal_data(self, slice_ref, slice_flo,  *args, **kwargs):
        x_ref, x_ref_mask, x_ref_labels = self.get_data(slice_ref, *args, **kwargs)
        x_flo, x_flo_mask, x_flo_labels = self.get_data(slice_flo,  *args, **kwargs)
        x_flo_orig = x_flo

        if self.tf_params is not None:
            img = [x_ref, x_flo, x_flo_orig, x_ref_mask, x_flo_mask, x_ref_labels, x_flo_labels]
            img_tf = self.tf_params(img)
            x_ref, x_flo, x_flo_orig, x_ref_mask, x_flo_mask, x_ref_labels, x_flo_labels = img_tf

        data_dict = {
            'x_ref': x_ref, 'x_flo': x_flo, 'x_flo_orig': x_flo_orig,
            'x_ref_mask': x_ref_mask, 'x_flo_mask': x_flo_mask,
            'x_ref_labels': x_ref_labels, 'x_flo_labels': x_flo_labels,
        }

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

        subject = self.data_loader[index]
        if isinstance(subject, list):
            slice_ref = subject[0]
            slice_flo = subject[1]
            subject_id = ''

        else:
            sid_ref, sid_flo = np.random.choice(subject.sid_list, 2, replace=True)
            slice_ref = subject.get_timepoint(sid_flo)
            slice_flo = subject.get_timepoint(sid_ref)
            subject_id = subject.id + '.'

        data_dict = self.get_intramodal_data(slice_ref, slice_flo)
        data_dict = self.data_augmentation(data_dict)

        affine, nonlinear_field = self.get_deformation_field(num=2)
        data_dict['affine'] = affine
        data_dict['nonlinear'] = nonlinear_field
        data_dict = self.convert_to_tensor(data_dict)
        rid = subject_id + str(slice_ref.sid) + '_to_' + str(slice_flo.sid)
        data_dict['rid'] = rid
        data_dict['ref_vox2ras0'] = slice_ref.vox2ras0
        data_dict['flo_vox2ras0'] = slice_flo.vox2ras0


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
            affine = self.affine_params.get_affine(self._image_shape)
            nlf_x, nlf_y = self.nonlinear_params.get_lowres_strength(ndim=2)
            nonlinear_field = np.zeros((2,) + nlf_x.shape)
            nonlinear_field[0] = nlf_y
            nonlinear_field[1] = nlf_x

            affine_list.append(affine)
            nonlinear_field_list.append(nonlinear_field)

        return affine_list, nonlinear_field_list


class RegistrationDataset3D(RegistrationDataset):
    '''
    Class for intermodal registration where input data, output target and the correponding transformations or data
    augmentation are specified
    '''


    def mask_image(self, image, mask):
        return image*mask

    def get_deformation_field(self, num=2):

        if self.affine_params is None and self.nonlinear_params is None:
            affine_list = [np.eye(4)]*num
            nonlinear_field_list = [np.zeros((3,9,9,9))]*num

            return affine_list, nonlinear_field_list

        affine_list = []
        nonlinear_field_list = []
        for it_i in range(num):
            affine = self.affine_params.get_affine(self._image_shape) if self.affine_params is not None else np.eye(4)

            nlf_x, nlf_y, nlf_z = self.nonlinear_params.get_lowres_strength(ndim=3) if self.nonlinear_params is not None else [np.zeros((9,9,9)), np.zeros((9,9,9)), np.zeros((9,9,9))]
            nonlinear_field = np.zeros((3,) + nlf_x.shape)
            nonlinear_field[0] = nlf_x
            nonlinear_field[1] = nlf_y
            nonlinear_field[2] = nlf_z

            affine_list.append(affine)
            nonlinear_field_list.append(nonlinear_field)

        return affine_list, nonlinear_field_list


