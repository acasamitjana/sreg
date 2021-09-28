import csv

import numpy as np

from scipy.ndimage import convolve, gaussian_filter
from scipy.interpolate import RegularGridInterpolator as rgi
from munkres import Munkres

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

def one_hot_encoding(target, num_classes, categories=None):
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

    if categories is None:
        categories = list(range(num_classes))

    labels = np.zeros((num_classes,) + target.shape)
    for it_class in categories:
        idx_class = np.where(target == it_class)
        idx = (it_class,)+ idx_class
        labels[idx] = 1

    return labels.astype(int)

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

