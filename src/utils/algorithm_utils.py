from os.path import join, exists
from os import makedirs
import subprocess
import pdb

import nibabel as nib
import torch
import numpy as np
from skimage.transform import resize
from scipy.interpolate import interpn
# import pytorch3d

from setup import NIFTY_REG_DIR
from src import models
from src.utils import image_transform as tf, deformation_utils as def_utils

ALADINcmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_aladin'
F3Dcmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_f3d'
TRANSFORMcmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_transform'
REScmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_resample'


# -------------- #
# Initialization #
# -------------- #

def initialize_graph_NR_lineal(pairwise_timepoints, results_dir, filename, tempdir='/tmp'):

    if not exists(tempdir): makedirs(tempdir)

    tp_ref, tp_flo = pairwise_timepoints

    refFile = tp_ref.image_centered_path
    floFile = tp_flo.image_centered_path
    refMaskFile = tp_ref.mask_centered_path
    floMaskFile = tp_flo.mask_centered_path
    outputFile = join(results_dir, filename + '.nii.gz')
    outputMaskFile = join(results_dir, filename + '.mask.nii.gz')
    affineFile = join(results_dir, filename + '.aff')

    # System calls
    subprocess.call([
        ALADINcmd, '-ref', refFile, '-flo', floFile, '-aff', affineFile, '-res', outputFile, '-rigOnly',
        '-rmask', refMaskFile, '-fmask', floMaskFile, '-pad', '0', '-speeeeed', '-omp', '4', '--voff'
    ])

    subprocess.call([
        REScmd, '-ref', refMaskFile, '-flo', floMaskFile, '-trans', affineFile, '-res',
        outputMaskFile, '-inter', '0', '-voff'
    ])


def initialize_graph_RegNet(model, generator_data, image_shape, device):

    num_elements = len(generator_data.dataset)
    num_batches = len(generator_data)
    batch_size = generator_data.batch_size

    downsample_factor = model.fullsize.factor
    vel_shape = tuple([int(i/d) for i,d in zip(image_shape, downsample_factor)])

    with torch.no_grad():

        registered_image = np.zeros((num_elements,) + image_shape)
        registered_mask = np.zeros((num_elements,) + image_shape)
        velocity_field = np.zeros((num_elements, 3) + vel_shape)
        deformation_field = np.zeros((num_elements, 3) + image_shape)

        for it_batch, data_dict in enumerate(generator_data):

            start = it_batch * batch_size
            end = start + batch_size
            if it_batch == num_batches - 1:
                end = num_elements

            ref_image = data_dict['x_ref'].to(device)
            flo_image = data_dict['x_flo'].to(device)
            flo_mask = data_dict['x_flo_mask'].to(device)

            r, f, v = model(flo_image, ref_image)
            r_mask = model.predict(flo_mask, f, svf=False, mode='nearest')

            registered_image[start:end] = np.squeeze(r.cpu().detach().numpy())
            registered_mask[start:end] = np.squeeze(r_mask.cpu().detach().numpy())

            velocity_field[start:end] = v.cpu().detach().numpy()
            deformation_field[start:end] = f.cpu().detach().numpy()

    velocity_field[np.isnan(velocity_field)] = 0
    deformation_field[np.isnan(deformation_field)] = 0

    return np.transpose(registered_image, [1, 2, 3, 0]), np.transpose(registered_mask, [1, 2, 3, 0]), \
           np.transpose(velocity_field, [1, 2, 3, 4, 0]), np.transpose(deformation_field, [1, 2, 3, 4, 0]),\

def initialize_graph_NR(pairwise_timepoints, results_dir, filename, vox2ras, tempdir='/tmp'):

    if not exists(tempdir): makedirs(tempdir)

    tp_ref, tp_flo = pairwise_timepoints
    refFile = tp_ref.image_linear_path
    floFile = tp_flo.image_linear_path
    refMaskFile = tp_ref.mask_linear_path
    floMaskFile = tp_flo.mask_linear_path
    outputFile = join(results_dir, filename + '.nii.gz')
    outputMaskFile = join(results_dir, filename + '.mask.nii.gz')

    nonlinearSVF = join(results_dir, filename + '.svf.nii.gz')
    dummyFileNifti = join(tempdir, 'dummyFileNifti.nii.gz')
    dummyFileSVF = join(tempdir, 'dummyFileSVF.nii.gz')

    # System calls
    subprocess.call([
        F3Dcmd, '-ref', refFile, '-flo', floFile, '-res', outputFile, '-rmask', refMaskFile,
        '-fmask', floMaskFile, '-cpp', dummyFileNifti, '-sx', str(10), '-omp', '4', '--lncc', '5', '-vel', '-voff'
    ])

    subprocess.call([
        REScmd, '-ref', refMaskFile, '-flo', floMaskFile, '-trans', dummyFileNifti, '-res',
        outputMaskFile, '-inter', '0', '-voff'
    ])

    subprocess.call([TRANSFORMcmd, '-ref', refFile, '-flow', dummyFileNifti, dummyFileSVF, '-voff'])

    proxy = nib.load(dummyFileSVF)
    svf_ras = np.transpose(np.squeeze(np.asarray(proxy.dataobj)), [3, 0, 1, 2])
    image_shape = svf_ras.shape[1:]

    II, JJ, KK = np.meshgrid(np.arange(0, image_shape[0]), np.arange(0, image_shape[1]),  np.arange(0, image_shape[2]), indexing='ij')

    rasMosaic = np.dot(proxy.affine, np.concatenate((II.reshape(-1, 1), JJ.reshape(-1, 1), KK.reshape(-1, 1), np.ones((np.prod(II.shape), 1))), axis=1).T)

    svf = np.zeros_like(svf_ras)
    svf[0] = svf_ras[0] - rasMosaic[0].reshape(image_shape)
    svf[1] = svf_ras[1] - rasMosaic[1].reshape(image_shape)
    svf[2] = svf_ras[2] - rasMosaic[2].reshape(image_shape)

    img = nib.Nifti1Image(svf, vox2ras)
    nib.save(img, nonlinearSVF)


# ----------------------------- #
# Deformation field integration #
# ----------------------------- #
def integrate_NR(svf, image_shape, nsteps=10, int_end=1):

    int_shape = svf.shape[1:]
    II, JJ, KK = np.meshgrid(np.arange(0, int_shape[0]), np.arange(0, int_shape[1]), np.arange(0, int_shape[2]), indexing='ij')
    nsteps = int(max(0, np.ceil(nsteps + np.log2(int_end))))
    flow_i = svf[0] / 2 ** nsteps * int_end
    flow_j = svf[1] / 2 ** nsteps * int_end
    flow_k = svf[2] / 2 ** nsteps * int_end
    for it_step in range(nsteps):

        inc = def_utils.deform3D([flow_i, flow_j, flow_k], [flow_i, flow_j, flow_k])
        inci, incj, inck = inc

        flow_i = flow_i + inci.reshape(int_shape)
        flow_j = flow_j + incj.reshape(int_shape)
        flow_k = flow_k + inck.reshape(int_shape)

        del inci, incj, inck

    del II, JJ, KK

    if int_shape[0] != image_shape[0] or int_shape[0] != image_shape[0] or int_shape[0] != image_shape[0]:
        flow_i = resize(flow_i, image_shape)
        flow_j = resize(flow_j, image_shape)
        flow_k = resize(flow_k, image_shape)

    flow = np.concatenate((flow_i[np.newaxis], flow_j[np.newaxis], flow_k[np.newaxis]), axis=0)
    return flow

def integrate_RegNet(svf, image_shape, parameter_dict):

    model = models.RegNet(
        nb_unet_features=[parameter_dict['ENC_NF'], parameter_dict['DEC_NF']],
        inshape=parameter_dict['VOLUME_SHAPE'],
        int_steps=7,
        int_downsize=parameter_dict['UPSAMPLE_LEVELS'],
    )

    new_svf = torch.tensor(svf[np.newaxis])
    flow = model.get_flow_field(new_svf)
    flow = np.squeeze(flow.detach().numpy())

    flow_image = np.zeros((3,) + image_shape)
    transform = tf.Compose(parameter_dict['TRANSFORM'])
    f_x, f_y, f_z = transform.inverse([flow[0], flow[1], flow[2]], img_shape=[image_shape] * 3)
    flow_image[0] = f_x
    flow_image[1] = f_y
    flow_image[2] = f_z

    return flow_image
