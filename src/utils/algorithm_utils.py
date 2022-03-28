from os.path import join, exists
from os import makedirs
import subprocess
import pdb

import nibabel as nib
import torch
import numpy as np
from skimage.transform import resize
from scipy.interpolate import interpn


from setup import *
from src import models, datasets, layers
from src.utils import data_loader_utils as tf, deformation_utils as def_utils, tensor_utils

ALADINcmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_aladin'
F3Dcmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_f3d'
TRANSFORMcmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_transform'
REScmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_resample'

f3d_parameters = ['-sx', str(4), '-omp', '4', '--lncc', '5', '-vel', '-voff']

# -------------- #
# Initialization #
# -------------- #

def initialize_graph_NR_lineal(pairwise_timepoints, results_dir, filename, tempdir='/tmp'):

    if not exists(tempdir): makedirs(tempdir)

    tp_ref, tp_flo = pairwise_timepoints

    refFile = tp_ref.get_filepath('preprocessing_resample_centered')
    floFile = tp_flo.get_filepath('preprocessing_resample_centered')
    refMaskFile = tp_ref.get_filepath('preprocessing_mask_centered')
    floMaskFile = tp_flo.get_filepath('preprocessing_mask_centered')
    outputFile = join(results_dir, filename + '.nii.gz')
    outputMaskFile = join(results_dir, filename + '.mask.nii.gz')
    affineFile = join(results_dir, filename + '.aff')

    # System calls
    subprocess.call([
        ALADINcmd, '-ref', refFile, '-flo', floFile, '-aff', affineFile, '-res', outputFile, '-rigOnly',
        '-rmask', refMaskFile, '-fmask', floMaskFile, '-cog', '-pad', '0', '-speeeeed', '-omp', '4', '--voff'
    ])

    if DEBUG:
        subprocess.call([
            REScmd, '-ref', refMaskFile, '-flo', floMaskFile, '-trans', affineFile, '-res',
            outputMaskFile, '-inter', '0', '-voff'
        ])
    else:
        os.remove(outputFile)

def predict_RegNet(pairwise_timepoints, parameter_dict, ref_aff=None, flo_aff=None, epoch='FI', use_gpu=True):
    device = 'cuda' if use_gpu else 'cpu'
    image_shape = parameter_dict['VOLUME_SHAPE']

    dataset = datasets.RegistrationDataset3D(
        [pairwise_timepoints],
        tf_params=parameter_dict['TRANSFORM'],
        norm_params=parameter_dict['NORMALIZATION'],
        mask_dilation=True,
    )
    generator = torch.utils.data.DataLoader(
        dataset,
        batch_size=parameter_dict['BATCH_SIZE'],
        shuffle=False,
    )

    int_steps = parameter_dict['INT_STEPS'] if parameter_dict['FIELD_TYPE'] == 'velocity' else 0
    if int_steps == 0: assert parameter_dict['UPSAMPLE_LEVELS'] == 1
    model = models.RegNet(
        nb_unet_features=[parameter_dict['ENC_NF'], parameter_dict['DEC_NF']],
        inshape=image_shape,
        int_steps=int_steps,
        int_downsize=parameter_dict['UPSAMPLE_LEVELS'],
    )

    model = model.to(device)
    weightsfile = 'model_checkpoint.' + str(epoch) + '.pth'
    checkpoint = torch.load(join(parameter_dict['RESULTS_DIR'], 'checkpoints', weightsfile), map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    apply_aff = layers.SpatialTransformerAffine(image_shape)

    v_list = []
    with torch.no_grad():
        for it_dd, data_dict in enumerate(generator):

            ref_image = data_dict['x_ref'].to(device)
            flo_image = data_dict['x_flo'].to(device)

            if ref_aff is not None:
                ref_image = apply_aff(ref_image, torch.from_numpy(ref_aff[np.newaxis, np.newaxis]).float())

            if flo_aff is not None:
                flo_image = apply_aff(flo_image, torch.from_numpy(flo_aff[np.newaxis, np.newaxis]).float())

            r, f, v = model(flo_image, ref_image)

            pdb.set_trace()
            velocity_field = np.squeeze(v.cpu().detach().numpy())
            velocity_field[np.isnan(velocity_field)] = 0
            v_list.append(velocity_field)

    return v_list

def initialize_graph_RegNet(pairwise_timepoints, parameter_dict, results_dir, filename, vox2ras0, subject_shape, epoch='FI', use_gpu='gpu'):
    device = 'cuda' if use_gpu else 'cpu'
    image_shape = parameter_dict['VOLUME_SHAPE']

    dataset = datasets.RegistrationDataset3D(
        [pairwise_timepoints],
        tf_params=parameter_dict['TRANSFORM'],
        norm_params=parameter_dict['NORMALIZATION'],
        mask_dilation=True,
    )
    generator = torch.utils.data.DataLoader(
        dataset,
        batch_size=parameter_dict['BATCH_SIZE'],
        shuffle=False,
    )

    int_steps = parameter_dict['INT_STEPS'] if parameter_dict['FIELD_TYPE'] == 'velocity' else 0
    if int_steps == 0: assert parameter_dict['UPSAMPLE_LEVELS'] == 1
    model = models.RegNet(
        nb_unet_features=[parameter_dict['ENC_NF'], parameter_dict['DEC_NF']],
        inshape=image_shape,
        int_steps=int_steps,
        int_downsize=parameter_dict['UPSAMPLE_LEVELS'],
    )

    model = model.to(device)
    weightsfile = 'model_checkpoint.' + str(epoch) + '.pth'
    checkpoint = torch.load(join(parameter_dict['RESULTS_DIR'], 'checkpoints', weightsfile), map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    with torch.no_grad():
        for it_dd, data_dict in enumerate(generator):

            ref_image = data_dict['x_ref'].to(device)
            flo_image = data_dict['x_flo'].to(device)
            flo_mask = data_dict['x_flo_mask'].to(device)

            r, f, v = model(flo_image, ref_image)
            r_mask = model.predict(flo_mask, f, svf=False, mode='nearest')

            registered_image = np.squeeze(r.cpu().detach().numpy())
            registered_mask = np.squeeze(r_mask.cpu().detach().numpy())

            velocity_field = np.squeeze(v.cpu().detach().numpy())
            deformation_field = np.squeeze(f.cpu().detach().numpy())

            velocity_field[np.isnan(velocity_field)] = 0
            deformation_field[np.isnan(deformation_field)] = 0

            diff_shape = [image_shape[it_d] - subject_shape[it_d] for it_d in range(3)]
            if sum(np.abs(diff_shape)) > 0:
                tx = np.eye(4)
                tx[0, -1] = -(diff_shape[0] // 2)
                tx[1, -1] = -(diff_shape[1] // 2)
                tx[2, -1] = -(diff_shape[2] // 2)
                vox2ras0 = vox2ras0 @ tx

            # Save output forward tree
            img = nib.Nifti1Image(velocity_field, vox2ras0)
            nib.save(img, join(results_dir, filename + '.svf.nii.gz'))

            img = nib.Nifti1Image(registered_mask, vox2ras0)
            nib.save(img, join(results_dir, filename + '.mask.nii.gz'))

            if DEBUG:
                img = nib.Nifti1Image(registered_image, vox2ras0)
                nib.save(img, join(results_dir, filename + '.nii.gz'))



def initialize_graph_NR(pairwise_timepoints, results_dir, filename, vox2ras, tempdir='/tmp'):

    if not exists(tempdir): makedirs(tempdir)

    tp_ref, tp_flo = pairwise_timepoints
    refFile = tp_ref.get_filepath('linear_resampled_image')
    floFile = tp_flo.get_filepath('linear_resampled_image')
    refMaskFile = tp_ref.get_filepath('linear_resampled_mask_dilated')
    floMaskFile = tp_flo.get_filepath('linear_resampled_mask_dilated')
    outputFile = join(results_dir, filename + '.nii.gz')
    outputMaskFile = join(results_dir, filename + '.mask.nii.gz')

    nonlinearSVF = join(results_dir, filename + '.svf.nii.gz')
    dummyFileNifti = join(tempdir, 'dummyFileNifti.nii.gz')
    dummyFileSVF = join(tempdir, 'dummyFileSVF.nii.gz')

    # System calls
    subprocess.call([
        F3Dcmd, '-ref', refFile, '-flo', floFile, '-res', outputFile, '-rmask', refMaskFile,
        '-fmask', floMaskFile, '-cpp', dummyFileNifti] + f3d_parameters)

    subprocess.call([
        REScmd, '-ref', refMaskFile, '-flo', floMaskFile, '-trans', dummyFileNifti, '-res',
        outputMaskFile, '-inter', '0', '-voff'
    ])

    subprocess.call([TRANSFORMcmd, '-ref', refFile, '-flow', dummyFileNifti, dummyFileSVF])

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

    if not DEBUG:
        os.remove(outputFile)


# ----------------------------- #
# Deformation field integration #
# ----------------------------- #
def integrate_NR(svf, image_shape, nsteps=10, int_end=1):

    int_shape = svf.shape[1:]
    # II, JJ, KK = np.meshgrid(np.arange(0, int_shape[0]), np.arange(0, int_shape[1]), np.arange(0, int_shape[2]), indexing='ij')
    nsteps = int(max(0, np.ceil(nsteps + np.log2(int_end))))

    if int_shape[0] != image_shape[0] or int_shape[0] != image_shape[0] or int_shape[0] != image_shape[0]:
        svf = resize(svf, svf.shape[:1] + image_shape)

    flow = svf / 2 ** nsteps * int_end
    for it_step in range(nsteps):
        inc = def_utils.deform3D(np.transpose(flow, axes=[1, 2, 3, 0]), flow, resized_shape=flow.shape[1:])
        flow = flow + np.transpose(inc, axes=[3, 0, 1, 2])

    # del II, JJ, KK
    #

    #
    # flow = np.concatenate((flow_i[np.newaxis], flow_j[np.newaxis], flow_k[np.newaxis]), axis=0)

    return flow

def integrate_RegNet(svf, image_shape, parameter_dict, device='cpu', model=None):


    if model is None:
        int_steps = parameter_dict['INT_STEPS'] if parameter_dict['FIELD_TYPE'] == 'velocity' else 0
        if int_steps == 0: assert parameter_dict['UPSAMPLE_LEVELS'] == 1

        model = models.RegNet(
            nb_unet_features=[parameter_dict['ENC_NF'], parameter_dict['DEC_NF']],
            inshape=parameter_dict['VOLUME_SHAPE'],
            int_steps=int_steps,
            int_downsize=parameter_dict['UPSAMPLE_LEVELS'],
        )

        model.to(device)

    new_svf = torch.tensor(svf[np.newaxis]).to(device)
    flow = model.get_flow_field(new_svf)
    flow = np.squeeze(flow.cpu().detach().numpy())

    flow_image = np.zeros((3,) + image_shape, dtype=svf.dtype)
    transform = tf.Compose(parameter_dict['TRANSFORM'])
    f_x, f_y, f_z = transform.inverse([flow[0], flow[1], flow[2]], img_shape=[image_shape] * 3)
    flow_image[0] = f_x
    flow_image[1] = f_y
    flow_image[2] = f_z

    del flow, new_svf, model

    return flow_image
