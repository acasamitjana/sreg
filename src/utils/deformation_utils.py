import csv

import numpy as np
import nibabel as nib

from scipy.interpolate import interpn, RegularGridInterpolator as rgi
from scipy.ndimage import gaussian_filter

def create_template_space(linear_image_list):

    boundaries_min = np.zeros((len(linear_image_list), 3))
    boundaries_max = np.zeros((len(linear_image_list), 3))
    margin_bb = 5
    for it_lil, lil in enumerate(linear_image_list):

        proxy = nib.load(lil)
        mask = np.asarray(proxy.dataobj)
        header = proxy.affine
        idx = np.where(mask > 0)
        vox_min = np.concatenate((np.min(idx, axis=1), [1]), axis=0)
        vox_max = np.concatenate((np.max(idx, axis=1), [1]), axis=0)

        minR, minA, minS = np.inf, np.inf, np.inf
        maxR, maxA, maxS = -np.inf, -np.inf, -np.inf

        for i in [vox_min[0], vox_max[0] + 1]:
            for j in [vox_min[1], vox_max[1] + 1]:
                for k in [vox_min[2], vox_max[2] + 1]:
                    aux = np.dot(header, np.asarray([i, j, k, 1]).T)

                    minR, maxR = min(minR, aux[0]), max(maxR, aux[0])
                    minA, maxA = min(minA, aux[1]), max(maxA, aux[1])
                    minS, maxS = min(minS, aux[2]), max(maxS, aux[2])

        minR -= margin_bb
        minA -= margin_bb
        minS -= margin_bb

        maxR += margin_bb
        maxA += margin_bb
        maxS += margin_bb

        boundaries_min[it_lil] = [minR, minA, minS]
        boundaries_max[it_lil] = [maxR, maxA, maxS]
        # boundaries_min += [[minR, minA, minS]]
        # boundaries_max += [[maxR, maxA, maxS]]

    # Get the corners of cuboid in RAS space
    minR = np.mean(boundaries_min[..., 0])
    minA = np.mean(boundaries_min[..., 1])
    minS = np.mean(boundaries_min[..., 2])
    maxR = np.mean(boundaries_max[..., 0])
    maxA = np.mean(boundaries_max[..., 1])
    maxS = np.mean(boundaries_max[..., 2])

    template_size = np.asarray(
        [int(np.ceil(maxR - minR)) + 1, int(np.ceil(maxA - minA)) + 1, int(np.ceil(maxS - minS)) + 1])

    # Define header and size
    template_vox2ras0 = np.asarray([[1, 0, 0, minR],
                                    [0, 1, 0, minA],
                                    [0, 0, 1, minS],
                                    [0, 0, 0, 1]])


    # VOX Mosaic
    II, JJ, KK = np.meshgrid(np.arange(0, template_size[0]),
                             np.arange(0, template_size[1]),
                             np.arange(0, template_size[2]), indexing='ij')

    RR = II + minR
    AA = JJ + minA
    SS = KK + minS
    rasMosaic = np.concatenate((RR.reshape(-1, 1),
                                AA.reshape(-1, 1),
                                SS.reshape(-1, 1),
                                np.ones((np.prod(template_size), 1))), axis=1).T

    return rasMosaic, template_vox2ras0, template_size


def interpolate2D(image, mosaic, mode='bilinear'):
    '''
    :param image: np.array or list of np.arrays.
    :param mosaic: Nx2
    :param mode: 'nearest' or 'linear'
    :return:
    '''
    if not isinstance(image, list):
        image = [image]

    x = np.arange(0, image[0].shape[0])
    y = np.arange(0, image[0].shape[1])

    output = []
    for im in image:
        my_interpolation_function = rgi((x, y), im, method=mode, bounds_error=False, fill_value=0)
        im_resampled = my_interpolation_function(mosaic)
        output.append(im_resampled)

    return output

def interpolate3D(image, mosaic, vox2ras0=None, mode='linear'):
    '''

    :param image: np.array or list of np.arrays.
    :param mosaic: Nx3 or 4xN in case it's voxels not RAS.
    :param vox2ras0: optional if the mosaic is specified at ras space.
    :param mode: 'nearest' or 'linear'
    :return:
    '''
    if not isinstance(image, list):
        image = [image]

    if vox2ras0 is not None:
        mosaic = np.matmul(np.linalg.inv(vox2ras0), mosaic)
        mosaic = mosaic[:3].T

    x = np.arange(0, image[0].shape[0])
    y = np.arange(0, image[0].shape[1])
    z = np.arange(0, image[0].shape[2])

    output = []
    for im in image:
        my_interpolation_function = rgi((x,y,z), im, method=mode, bounds_error=False, fill_value=0)
        im_resampled = my_interpolation_function(mosaic)
        output.append(im_resampled)

    return output


def deform2D(image, deformation, mode='bilinear'):
    '''

    :param image: 3D np.array (nrow, ncol)
    :param deformation: 4D np.array (3, nrow, ncol)
    :param mode: 'bilinear' or 'nearest'
    :return:
    '''

    di = deformation[0]
    dj = deformation[1]
    output_shape = deformation.shape[1:]

    del deformation

    II, JJ = np.meshgrid(np.arange(0, output_shape[0]), np.arange(0, output_shape[1]), indexing='ij')
    IId = II + di
    JJd = JJ + dj

    del II, JJ, di, dj

    ok1 = IId >= 0
    ok2 = JJd >= 0
    ok4 = IId <= image.shape[0]
    ok5 = JJd <= image.shape[1]
    ok = ok1 & ok2 &  ok4 & ok5

    del ok1, ok2, ok4, ok5

    points = (np.arange(image.shape[0]), np.arange(image.shape[1]))
    xi = np.concatenate((IId[ok].reshape(-1, 1), JJd[ok].reshape(-1, 1)), axis=1)

    del IId, JJd

    if mode == 'bilinear':

        output_flat = interpn(points, image, xi=xi, method='linear', fill_value=0, bounds_error=False)
        output = np.zeros(output_shape)
        output[ok] = output_flat

    elif mode == 'nearest':

        output_flat = interpn(points, image, xi=xi, method='nearest', fill_value=0, bounds_error=False)
        output = np.zeros(output_shape)
        output[ok] = output_flat

    else:
        raise ValueError('Interpolation mode not available')


    return output

def deform3D(image, deformation, mode='linear'):
    '''

    :param image: 3D np.array (nrow, ncol)
    :param deformation: 4D np.array (3, nrow, ncol)
    :param mode: 'linear' or 'nearest'
    :return:
    '''

    di = deformation[0]
    dj = deformation[1]
    dk = deformation[2]
    output_shape = di.shape

    del deformation

    II, JJ, KK = np.meshgrid(np.arange(0, output_shape[0]), np.arange(0, output_shape[1]), np.arange(0, output_shape[2]), indexing='ij')
    IId = II + di
    JJd = JJ + dj
    KKd = KK + dk

    del II, JJ, KK, di, dj, dk

    mosaic = np.concatenate((IId.reshape(-1, 1), JJd.reshape(-1, 1), KKd.reshape(-1, 1)), axis=1)

    del IId, JJd, KKd

    output_flat = interpolate3D(image, mosaic,  mode=mode)
    output = []
    for it_o, out in enumerate(output_flat):
        output.append(out.reshape(output_shape))

    # ok1 = IId >= 0
    # ok2 = JJd >= 0
    # ok3 = KKd >= 0
    # ok4 = IId <= image.shape[0]
    # ok5 = JJd <= image.shape[1]
    # ok6 = KKd <= image.shape[2]
    # ok = ok1 & ok2 & ok3 & ok4 & ok5 & ok6
    #
    # del ok1, ok2, ok3, ok4, ok5, ok6
    #
    # points = (np.arange(image.shape[0]), np.arange(image.shape[1]), np.arange(image.shape[2]))
    # xi = np.concatenate((IId[ok].reshape(-1, 1), JJd[ok].reshape(-1, 1), KKd[ok].reshape(-1, 1)), axis=1)
    #
    # del IId, JJd, KKd
    #
    # if mode == 'bilinear':
    #
    #     output_flat = interpn(points, image, xi=xi, method='linear', fill_value=0, bounds_error=False)
    #     output = np.zeros(output_shape)
    #     output[ok] = output_flat
    #
    # elif mode == 'nearest':
    #
    #     output_flat = interpn(points, image, xi=xi, method='nearest', fill_value=0, bounds_error=False)
    #     output = np.zeros(output_shape)
    #     output[ok] = output_flat
    #
    # else:
    #     raise ValueError('Interpolation mode not available')
    #

    return output

def get_affine_from_rotation(angle_list):

    affine_matrix = np.zeros((len(angle_list), 2,3))
    for it_a, angle in enumerate(angle_list):
        angle_rad = angle * np.pi / 180
        affine_matrix[it_a] = np.array([
            [np.cos(angle_rad).item(), -np.sin(angle_rad).item(), 0],
            [np.sin(angle_rad).item(), np.cos(angle_rad).item(), 0],
        ])
    return affine_matrix

def affine_to_dense(affine_matrix, volshape, center=True, vox2ras0=None):
    ndims = len(volshape)
    num_voxels = int(np.prod(volshape))
    II, JJ, KK = np.meshgrid(np.arange(0, volshape[0]), np.arange(0, volshape[1]), np.arange(0, volshape[2]), indexing='ij')

    ijk = np.concatenate((II.reshape(-1, 1), JJ.reshape(-1, 1), KK.reshape(-1, 1)),axis=1)
    if center:
        ijk_center = np.asarray([(volshape[f] - 1) / 2 for f in range(ndims)])
        ijk = ijk - ijk_center

    voxMosaic = np.dot(affine_matrix, np.concatenate((ijk, np.ones((num_voxels, 1))), axis=1).T)

    IId = voxMosaic[0].reshape(volshape)
    JJd = voxMosaic[1].reshape(volshape)
    KKd = voxMosaic[2].reshape(volshape)

    field = np.zeros((3,) + volshape)
    field[0] = IId - II
    field[1] = JJd - JJ
    field[2] = KKd - KK

    return field.astype("float32")


