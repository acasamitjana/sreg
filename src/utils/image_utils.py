import csv

import numpy as np

from scipy.special import softmax
from scipy.ndimage import distance_transform_edt, convolve, gaussian_filter
from scipy.interpolate import RegularGridInterpolator as rgi
from munkres import Munkres



SEG_DICT = {
    'Hippocampus': [53, 17],
    'Thalaumus': [49, 10],
    'Putamen': [51, 12],
    'Pallidum': [52, 13],
    'Caudate': [50, 11],
    'Amygdala': [54, 18],
    'Ventricles': [4, 5, 43, 44, 15, 14],
    'Accumbens': [58, 26],
    'VentralDC': [28, 60],
    'cGM': [42, 3],
    'cllGM': [47, 8],
    'cWM': [41, 2],
    'cllWM': [46, 7],
    'Brainstem': [16],
}

def convert_to_unified(seg):
    out_seg = np.zeros_like(seg)
    for it_lab, (lab_str, lab_list) in enumerate(SEG_DICT.items()):
        for lab in lab_list:
            out_seg[seg == lab] = it_lab

    return out_seg


def get_polynomial_basis_functions(image_shape, dim=3, order=2):
    '''
    :param image_shape: shape of the image/volume
    :param dim: image dimension [3]. Currently only 3D available
    :param order: order of the polynomial [1, 2].
    :return: polynomial basis functions
    '''


    assert dim in [2, 3]
    assert order in [2, 1]

    num_coeff = int(1 + dim + dim*(dim+1)/2*(order - 1))
    if dim == 3:

        x = np.linspace(0, 1, image_shape[0])[:, np.newaxis, np.newaxis]
        y = np.linspace(0, 1, image_shape[1])[np.newaxis, :, np.newaxis]
        z = np.linspace(0, 1, image_shape[2])[np.newaxis, np.newaxis]

        fx = np.tile(x, (1, image_shape[1], image_shape[2]))
        fy = np.tile(y, (image_shape[0], 1, image_shape[2]))
        fz = np.tile(z, (image_shape[0], image_shape[1], 1))

        A = np.zeros((num_coeff,) + image_shape)
        A[0] = 1
        A[1] = fx - 0.5
        A[2] = fy - 0.5
        A[3] = fz - 0.5

        if order == 2:

            A[4] = fx ** 2 - 0.5
            A[5] = fy ** 2 - 0.5
            A[6] = fz ** 2 - 0.5
            A[7] = fx * fy - 0.5
            A[8] = fx * fz - 0.5
            A[9] = fy * fz - 0.5

    else:
        x = np.linspace(0, 1, image_shape[0])[:, np.newaxis]
        y = np.linspace(0, 1, image_shape[1])[np.newaxis, :]

        fx = np.tile(x, (1, image_shape[1]))
        fy = np.tile(y, (image_shape[0], 1))

        A = np.zeros((num_coeff,) + image_shape)
        A[0] = 1
        A[1] = fx - 0.5
        A[2] = fy - 0.5

        if order == 2:

            A[3] = fx ** 2 - 0.5
            A[4] = fy ** 2 - 0.5
            A[5] = fx * fy - 0.5

    return A

def bias_field_corr(image, seg, penalty=0, patience=3):
    '''
    :param image: np array. Input image to correct
    :param seg: np.array with shape=image.shape + (num_labels). One-hot encoding of the segmentation
    :param penalty: regularization term over the coefficients.
    :param patience: int, default=3. Number indicating the maximum number of iterations where improvement < 1e-6
    :return:
    '''

    image_shape = image.shape

    image = image.reshape(-1, 1)
    image_log = np.log10(image + 1e-5)

    seg = seg[..., 1:]
    seg = seg.reshape(-1, seg.shape[-1])

    A = get_polynomial_basis_functions(image_shape, dim=3, order=2)
    num_coeff = A.shape[0]
    coeff_last = -10*np.ones((num_coeff,))
    coeff = np.zeros((num_coeff, 1))
    A = A.reshape(num_coeff, -1).T
    print('       Bias field correction')

    it_break = 0
    for it in range(30):
        bias_field = A @ coeff
        image_log_corr = image_log - bias_field

        del bias_field

        u_j = np.sum(seg * image_log_corr, axis=0) / np.sum(seg, axis=0)
        s_j = np.sum(seg * (image_log_corr-u_j)**2, axis=0) / np.sum(seg, axis=0)

        w_ij = seg / s_j
        W = np.sum(w_ij, axis=-1, keepdims=True)
        R = image_log - np.sum(w_ij*u_j, axis=-1, keepdims=True) / (W + 1e-5)

        del w_ij
        # py = 1 / np.sqrt(2 * np.pi * s_j) * np.exp(-0.5 / np.sqrt(s_j) * (image_log_corr - u_j) ** 2)
        # p_ll = np.sum(np.log(np.sum(py*seg, axis=-1) + 1e-5)) + penalty * np.sum(coeff*coeff)
        # del py

        coeff = np.linalg.inv((A * W).T @ A + penalty * np.eye(num_coeff)) @ (A * W).T @ R

        del W, R
        improv = np.max(np.abs((coeff - coeff_last) / coeff_last)) #(p_ll_last -p_ll) / p_ll_last
        print('       ' + str(it) + '. Coefficients ' + str(np.squeeze(coeff).tolist()) + '. Max change: ' + str(improv))

        coeff_last = coeff
        if improv < 1e-4:
            it_break += 1
            if it_break > patience:
                break

        else:
            it_break = 0

    bias_field = A @ coeff
    image_log_corr = image_log - bias_field

    del image_log, A, seg

    image_log_corr = 10**image_log_corr
    image_log_corr = image_log_corr.reshape(image_shape)
    bias_field = 10**(-bias_field)
    bias_field = bias_field.reshape(image_shape)
    return image_log_corr, bias_field

def align_with_identity_vox2ras0(V, vox2ras0):

    COST = np.zeros((3,3))
    for i in range(3):
        for j in range(3):

            # worker is the vector
            b = vox2ras0[:3,i]

            # task is j:th axis
            a = np.zeros((3,1))
            a[j] = 1

            COST[i, j] = - np.abs(np.dot(a.T, b))/np.linalg.norm(a, 2)/np.linalg.norm(b, 2)

    m = Munkres()
    indexes = m.compute(COST)

    v2r = np.zeros_like(vox2ras0)
    for idx in indexes:
        v2r[:, idx[1]] = vox2ras0[:, idx[0]]
    v2r[:, 3] = vox2ras0[:, 3]
    V = np.transpose(V, axes=[idx[1] for idx in indexes])

    for d in range(3):
        if v2r[d,d] < 0:
            v2r[:3, d] = -v2r[:3, d]
            v2r[:3, 3] = v2r[:3, 3] - v2r[:3, d] * (V.shape[d] -1)
            V = np.flip(V, axis=d)

    return V, v2r

def rescale_voxel_size(volume, aff, new_vox_size):
    """This function resizes the voxels of a volume to a new provided size, while adjusting the header to keep the RAS
    :param volume: a numpy array
    :param aff: affine matrix of the volume
    :param new_vox_size: new voxel size (3 - element numpy vector) in mm
    :return: new volume and affine matrix
    """

    pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
    new_vox_size = np.array(new_vox_size)
    factor = pixdim / new_vox_size
    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0  # don't blur if upsampling

    volume_filt = gaussian_filter(volume, sigmas)

    # volume2 = zoom(volume_filt, factor, order=1, mode='reflect', prefilter=False)
    x = np.arange(0, volume_filt.shape[0])
    y = np.arange(0, volume_filt.shape[1])
    z = np.arange(0, volume_filt.shape[2])

    my_interpolating_function = rgi((x, y, z), volume_filt)

    start = - (factor - 1) / (2 * factor)
    step = 1.0 / factor
    stop = start + step * np.ceil(volume_filt.shape * factor)

    xi = np.arange(start=start[0], stop=stop[0], step=step[0])
    yi = np.arange(start=start[1], stop=stop[1], step=step[1])
    zi = np.arange(start=start[2], stop=stop[2], step=step[2])
    xi[xi < 0] = 0
    yi[yi < 0] = 0
    zi[zi < 0] = 0
    xi[xi > (volume_filt.shape[0] - 1)] = volume_filt.shape[0] - 1
    yi[yi > (volume_filt.shape[1] - 1)] = volume_filt.shape[1] - 1
    zi[zi > (volume_filt.shape[2] - 1)] = volume_filt.shape[2] - 1

    xig, yig, zig = np.meshgrid(xi, yi, zi, indexing='ij', sparse=True)
    volume2 = my_interpolating_function((xig, yig, zig))

    aff2 = aff.copy()
    for c in range(3):
        aff2[:-1, c] = aff2[:-1, c] / factor[c]
    aff2[:-1, -1] = aff2[:-1, -1] - np.matmul(aff2[:-1, :-1], 0.5 * (factor - 1))

    return volume2, aff2

def gaussian_antialiasing(volume, aff, new_voxel_size):
    pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
    new_vox_size = np.array(new_voxel_size)
    factor = pixdim / new_vox_size
    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0  # don't blur if upsampling

    return gaussian_filter(volume, sigmas)

def one_hot_encoding(target, num_classes=None, categories=None):
    '''

    Parameters
    ----------
    target (np.array): target vector of dimension (d1, d2, ..., dN).
    num_classes (int): number of classes
    categories (None or list): existing categories. If set to None, we will consider only categories 0,...,num_classes

    Returns
    -------
    labels (np.array): one-hot target vector of dimension (num_classes, d1, d2, ..., dN)

    '''

    if categories is None and num_classes is None:
        raise ValueError('[ONE-HOT Enc.] You need to specify the number of classes or the categories.')
    elif categories is not None:
        num_classes = len(categories)
    else:
        categories = np.arange(num_classes)

    labels = np.zeros((num_classes,) + target.shape, dtype='int')
    for it_cls, cls in enumerate(categories):
        idx_class = np.where(target == cls)
        idx = (it_cls,) + idx_class
        labels[idx] = 1

    return labels

def grad3d(x):

    filter = np.asarray([-1,0,1])
    gx = convolve(x, np.reshape(filter, (3,1,1)), mode='constant')
    gy = convolve(x, np.reshape(filter, (1,3,1)), mode='constant')
    gz = convolve(x, np.reshape(filter, (1,1,3)), mode='constant')

    gx[0], gx[-1] = x[1] - x[0], x[-1] - x[-2]
    gy[:, 0], gy[:, -1] = x[:,1] - x[:,0], x[:, -1] - x[:, -2]
    gz[..., 0], gz[..., -1] = x[..., 1] - x[..., 0], x[..., -1] - x[..., -2]

    gmodule = np.sqrt(gx**2 + gy**2 + gz**2)
    return gmodule, gx, gy, gz

def crop_label(mask, margin=10, threshold=0):

    ndim = len(mask.shape)
    if isinstance(margin, int):
        margin=[margin]*ndim

    crop_coord = []
    idx = np.where(mask>threshold)
    for it_index, index in enumerate(idx):
        clow = max(0, np.min(idx[it_index]) - margin[it_index])
        chigh = min(mask.shape[it_index], np.max(idx[it_index]) + margin[it_index])
        crop_coord.append([clow, chigh])

    mask_cropped = mask[
                   crop_coord[0][0]: crop_coord[0][1],
                   crop_coord[1][0]: crop_coord[1][1],
                   crop_coord[2][0]: crop_coord[2][1]
                   ]

    return mask_cropped, crop_coord

def apply_crop(image, crop_coord):
    return image[crop_coord[0][0]: crop_coord[0][1],
                 crop_coord[1][0]: crop_coord[1][1],
                 crop_coord[2][0]: crop_coord[2][1]
           ]

def compute_distance_map(labelmap, soft_seg=True):
    unique_labels = np.unique(labelmap)
    distancemap = -200 * np.ones(labelmap.shape + (len(unique_labels),), dtype='float32')
    # print('Working in label: ', end='', flush=True)
    for it_ul, ul in enumerate(unique_labels):
        # print(str(ul), end=', ', flush=True)

        mask_label = labelmap == ul
        bbox_label, crop_coord = crop_label(mask_label, margin=5)

        d_in = (distance_transform_edt(bbox_label))
        d_out = -distance_transform_edt(~bbox_label)
        d = np.zeros_like(d_in)
        d[bbox_label] = d_in[bbox_label]
        d[~bbox_label] = d_out[~bbox_label]

        distancemap[crop_coord[0][0]: crop_coord[0][1],
                    crop_coord[1][0]: crop_coord[1][1],
                    crop_coord[2][0]: crop_coord[2][1], it_ul] = d


    if soft_seg:
        prior_labels = softmax(distancemap, axis=-1)
        # soft_labelmap = np.argmax(prior_labels, axis=-1).astype('uint16')
        return prior_labels
    else:
        return distancemap