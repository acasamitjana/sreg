import csv
import pdb

import numpy as np
import nibabel as nib

from scipy.interpolate import interpn, RegularGridInterpolator as rgi
from skimage.transform import resize

from src.utils.image_utils import crop_label

def create_template_space(linear_image_list):

    boundaries_min = np.zeros((len(linear_image_list), 3))
    boundaries_max = np.zeros((len(linear_image_list), 3))
    margin_bb = 5
    for it_lil, lil in enumerate(linear_image_list):

        if isinstance(lil, nib.nifti1.Nifti1Image):
            proxy = lil
        else:
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

def interpolate3D(image, mosaic, vox2ras0=None, resized_shape=None, mode='linear'):
    '''

    :param image: nib.nifti1.Nifti1Image or np.array.
    :param mosaic: Nx3 or 4xN in case it's voxels not RAS.
    :param vox2ras0: optional if the mosaic is specified at ras space.
    :param mode: 'nearest' or 'linear'
    :return:
    '''

    if isinstance(image, nib.nifti1.Nifti1Image):
        image = np.asarray(image.dataobj)

    image_shape = image.shape[:3]
    nchannels = 1
    if len(image.shape) > 3:
        nchannels = image.shape[3]
    else:
        image = image[..., np.newaxis]
        nchannels = 1

    if vox2ras0 is not None:
        mosaic = np.matmul(np.linalg.inv(vox2ras0), mosaic)
        mosaic = mosaic[:3].T

    x = np.arange(0, image_shape[0])
    y = np.arange(0, image_shape[1])
    z = np.arange(0, image_shape[2])

    ok1 = mosaic[..., 0] >= 0
    ok2 = mosaic[..., 1] >= 0
    ok3 = mosaic[..., 2] >= 0
    ok4 = mosaic[..., 0] <= image_shape[0]
    ok5 = mosaic[..., 1] <= image_shape[1]
    ok6 = mosaic[..., 2] <= image_shape[2]
    ok = ok1 & ok2 & ok3 & ok4 & ok5 & ok6

    my_interpolation_function = rgi((x,y,z), image, method=mode, bounds_error=False, fill_value=0)
    im_resampled_ok = my_interpolation_function(mosaic[ok])
    output_flat = np.zeros(ok.shape + (nchannels,), dtype=image.dtype)
    for it_ch in range(nchannels): output_flat[ok, it_ch] = im_resampled_ok[..., it_ch]

    if resized_shape is not None:
        output = np.zeros(resized_shape + (nchannels,), dtype=image.dtype)
        for it_ch in range(nchannels):  output[..., it_ch] = output_flat [..., it_ch].reshape(resized_shape)

    else:
        output = output_flat

    if nchannels == 1:
        output = output[..., 0]

    return output

    if n_channels > 1:
        im_resampled = np.zeros(ok.shape + (n_channels,))
    else:
        im_resampled = np.zeros(ok.shape)

    im_resampled[ok] = im_resampled_ok

    if resized_shape is not None:
        im_resampled = im_resampled.reshape(resized_shape)

    pdb.set_trace()
    return im_resampled

def interpolate3DChannel(image, mosaic, vox2ras0=None, resized_shape=None, mode='linear'):
    '''

    :param image: nib.nifti1.Nifti1Image, np.array or list of np.arrays.
    :param mosaic: Nx3 or 4xN in case it's voxels not RAS.
    :param vox2ras0: optional if the mosaic is specified at ras space.
    :param mode: 'nearest' or 'linear'
    :return:
    '''

    if isinstance(image, list):
        image = np.concatenate([i[..., np.newaxis] for i in image], axis=-1)


    image_shape = image.shape[:3]
    if len(image.shape) == 3:
        nchannels = 1
        image = image[..., np.newaxis]

    else:
        nchannels = image.shape[3]


    if vox2ras0 is not None:
        mosaic = np.matmul(np.linalg.inv(vox2ras0), mosaic)
        mosaic = mosaic[:3].T

    x = np.arange(0, image_shape[0])
    y = np.arange(0, image_shape[1])
    z = np.arange(0, image_shape[2])

    ok1 = mosaic[..., 0] >= 0
    ok2 = mosaic[..., 1] >= 0
    ok3 = mosaic[..., 2] >= 0
    ok4 = mosaic[..., 0] <= image_shape[0]
    ok5 = mosaic[..., 1] <= image_shape[1]
    ok6 = mosaic[..., 2] <= image_shape[2]
    ok = ok1 & ok2 & ok3 & ok4 & ok5 & ok6

    my_interpolation_function = rgi((x,y,z), image, method=mode, bounds_error=False, fill_value=0)
    im_resampled_ok = my_interpolation_function(mosaic[ok])
    output_flat = np.zeros(ok.shape + (nchannels,), dtype=image.dtype)

    for it_ch in range(nchannels): output_flat[ok, it_ch] = im_resampled_ok[..., it_ch]

    if resized_shape is not None:
        output = np.zeros(resized_shape + (nchannels,), dtype=image.dtype)
        for it_ch in range(nchannels):  output[..., it_ch] = output_flat [..., it_ch].reshape(resized_shape)

    else:
        output = output_flat

    if nchannels == 1:
        output = output[..., 0]

    return output

def interpolate3DLabel(image, mosaic, vox2ras0=None, resized_shape=None, mode='linear'):
    '''

    :param image: nib.nifti1.Nifti1Image, list of np.arrays.
    :param mosaic: Nx3 or 4xN in case it's voxels not RAS.
    :param vox2ras0: optional if the mosaic is specified at ras space.
    :param mode: 'nearest' or 'linear'
    :return:
    '''
    if isinstance(image, nib.nifti1.Nifti1Image) or isinstance(image, nib.freesurfer.mghformat.MGHImage):
        is_nifti = True
        image_shape = image.shape
        num_images = 1
        if len(image_shape) == 4:
            num_images = image_shape[3]
            image_shape = image_shape[:3]

    else:
        is_nifti = False
        num_images = len(image)
        image_shape = image[0].shape

    if isinstance(mode, str):
        mode = [mode]*num_images

    if vox2ras0 is not None:
        mosaic = np.matmul(np.linalg.inv(vox2ras0), mosaic)
        mosaic = mosaic[:3].T

    output = []
    for it_im in range(num_images):
        im = np.asarray(image.dataobj[..., it_im]) if is_nifti else image[it_im]

        _, crop_coord = crop_label(im > 0.05)
        im = im[crop_coord[0][0]: crop_coord[0][1], crop_coord[1][0]: crop_coord[1][1], crop_coord[2][0]: crop_coord[2][1]]

        x = np.arange(crop_coord[0][0], crop_coord[0][1])
        y = np.arange(crop_coord[1][0], crop_coord[1][1])
        z = np.arange(crop_coord[2][0], crop_coord[2][1])

        ok1 = mosaic[..., 0] >= crop_coord[0][0]
        ok2 = mosaic[..., 1] >= crop_coord[1][0]
        ok3 = mosaic[..., 2] >= crop_coord[2][0]
        ok4 = mosaic[..., 0] <= crop_coord[0][1]
        ok5 = mosaic[..., 1] <= crop_coord[1][1]
        ok6 = mosaic[..., 2] <= crop_coord[2][1]
        ok = ok1 & ok2 & ok3 & ok4 & ok5 & ok6

        my_interpolation_function = rgi((x,y,z), im, method=mode[it_im], bounds_error=False, fill_value=0)
        im_resampled_ok = my_interpolation_function(mosaic[ok])

        im_resampled = np.zeros(ok.shape)
        im_resampled[ok] = im_resampled_ok
        im_resampled=im_resampled.reshape(resized_shape)
        output.append(im_resampled)

    if len(output) == 1:
        output = output[0]

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

def deform3D(image, deformation, mode='linear', **kwargs):
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

    output = interpolate3DChannel(image, mosaic,  mode=mode, **kwargs)

    return output

def upscale_and_deform3D(image, deformation, ref_shape, ref_vox2ras0, flo_vox2ras0, factor=1, mode='linear'):
    '''
    :param image: np.array or list of np.array to deform
    :param deformation: array [3, d1, d2, d3]
    :param ref_shape: image shape
    :param ref_vox2ras0: reference (template, etc...) vox2ras0. Used to go from IId to RRd
    :param flo_vox2ras0: float (image) vox2ras0
    :param factor:
    :return:
    '''

    ref_vox2ras0 = ref_vox2ras0.copy()
    if isinstance(factor, int):
        factor = np.asarray([factor, factor, factor])

    if any(factor > 1):
        resized_shape = tuple(np.ceil(ref_shape * factor).astype('int'))
        field = np.zeros((3,) + resized_shape)
        field[0] = factor[0] * resize(deformation[0], resized_shape, order=1)
        field[1] = factor[1] * resize(deformation[1], resized_shape, order=1)
        field[2] = factor[2] * resize(deformation[2], resized_shape, order=1)

        del deformation

        # compute new template vox2ras
        for c in range(3):
            ref_vox2ras0[:-1, c] = ref_vox2ras0[:-1, c] / factor[c]
        ref_vox2ras0[:-1, -1] = ref_vox2ras0[:-1, -1] - np.matmul(ref_vox2ras0[:-1, :-1], 0.5 * (factor - 1))

        # compute new template ras mosaic
        # VOX Mosaic
        start = - (factor - 1) / (2 * factor)
        step = 1.0
        stop = start + step * np.asarray(resized_shape)

        ii = np.arange(start=start[0], stop=stop[0], step=step)
        jj = np.arange(start=start[1], stop=stop[1], step=step)
        kk = np.arange(start=start[2], stop=stop[2], step=step)
        ii[ii < 0] = 0
        jj[jj < 0] = 0
        kk[kk < 0] = 0
        ii[ii > (resized_shape[0] - 1)] = resized_shape[0] - 1
        jj[jj > (resized_shape[1] - 1)] = resized_shape[1] - 1
        kk[kk > (resized_shape[2] - 1)] = resized_shape[2] - 1

    else:
        resized_shape = ref_shape
        field = deformation

        del deformation

        ii = np.arange(0, field.shape[1]),
        jj = np.arange(0, field.shape[2]),
        kk = np.arange(0, field.shape[3])

    II, JJ, KK = np.meshgrid(ii, jj, kk, indexing='ij')

    IId = II + field[0]
    JJd = JJ + field[1]
    KKd = KK + field[2]
    voxMosaic = np.concatenate((IId.reshape(-1, 1),
                                JJd.reshape(-1, 1),
                                KKd.reshape(-1, 1),
                                np.ones((np.prod(resized_shape), 1))), axis=1).T

    rasMosaic = np.dot(ref_vox2ras0, voxMosaic)

    # inverse of vox2ras
    image_resampled = interpolate3D(image, rasMosaic, vox2ras0=flo_vox2ras0, resized_shape=resized_shape, mode=mode)

    return image_resampled, ref_vox2ras0


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


