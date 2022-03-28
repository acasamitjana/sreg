from os.path import exists
from os import remove
import time
from argparse import ArgumentParser
from datetime import date, datetime

# project imports
from setup import *
from database.data_loader import DataLoader
from scripts import config_dev as configFile
from src.algorithm import *
from src.utils import algorithm_utils, deformation_utils as def_utils, io_utils, image_utils


print('\n\n\n\n\n')
print('# ------------------------------------ #')
print('# Deform images to the latent ST space #')
print('# ------------------------------------ #')
print('\n\n')

#####################
# Global parameters #
#####################

parameter_dict_MRI = configFile.REGISTRATION_DIR
REScmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_resample'
TRANSFORMcmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_transform'

# Parameters
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--subjects', default=None, nargs='+')
arg_parser.add_argument('--reg_algorithm', default='bidir', choices=['standard', 'bidir'])

arguments = arg_parser.parse_args()
initial_subject_list = arguments.subjects
reg_algorithm = arguments.reg_algorithm

##############
# Processing #
##############

parameter_dict = configFile.CONFIG_REGISTRATION
reg_name = reg_algorithm
if reg_algorithm == 'bidir':
    reg_name += str(parameter_dict['LOSS_REGISTRATION_SMOOTHNESS']['lambda'])
data_loader = DataLoader(sid_list=initial_subject_list, linear=True, reg_algorithm=reg_name)
subject_list = data_loader.subject_list
parameter_dict = configFile.get_config_dict(data_loader.image_shape)
parameter_dict['RESULTS_DIR'] = parameter_dict['RESULTS_DIR'] + '_' + reg_algorithm

for it_subject, subject in enumerate(subject_list):

    print('[ST Deform images] Processing Subject: ' + str(subject.id))
    print('\n')

    subject_shape = subject.image_shape
    timepoints = subject.timepoints
    results_dir_sbj = subject.results_dirs.get_dir('nonlinear_st_' + reg_name)

    check_files = [exists(join(results_dir_sbj, tp.id + '.svf.nii.gz')) for tp in timepoints]
    if sum(check_files) != len(timepoints):
        print('[ST Deform images] Subject: ' + str(subject.id) + ' has not all timepoints available.')
        continue

    ####################################################################################################
    ####################################################################################################

    if len(subject.timepoints) == 1:
        continue
    cp_shape = tuple([int(i / parameter_dict['UPSAMPLE_LEVELS']) for i in parameter_dict['VOLUME_SHAPE']])


    ####################################################################################################
    ####################################################################################################

    print('[' + str(subject.id) + ' - INTEGRATION] Computing deformation field ... ')
    t_init = time.time()
    for it_tp, tp in enumerate(timepoints):
        if not exists(join(results_dir_sbj, tp.id + '.flow.nii.gz')):
            proxy = nib.load(join(results_dir_sbj, tp.id + '.svf.nii.gz'))
            svf = np.asarray(proxy.dataobj)
            flow = algorithm_utils.integrate_RegNet(svf, subject_shape, parameter_dict=parameter_dict)
            img = nib.Nifti1Image(flow, subject.vox2ras0)
            nib.save(img, join(results_dir_sbj, tp.id + '.flow.nii.gz'))

            del flow

    print('[' + str(subject.id) + ' - INTEGRATION] Total Elapsed time: ' + str(time.time() - t_init))

    ####################################################################################################
    ####################################################################################################
    t_init = time.time()
    print('[' + str(subject.id) + ' - DEFORM] Deforming images ... ')

    mri_list = []
    for it_tp, tp in enumerate(timepoints):

        proxyflow = nib.load(join(results_dir_sbj, tp.id + '.flow.nii.gz'))
        flow = np.asarray(proxyflow.dataobj)

        if DEBUG:
            mask = tp.load_mask()
            mask_d = tp.load_mask(dilated=True)
            im_resampled, _ = def_utils.upscale_and_deform3D([mask, mask_d], flow, subject_shape,
                                                             subject.vox2ras0, tp.vox2ras0,
                                                             mode=['linear', 'linear'])

            img = nib.Nifti1Image(im_resampled[0], subject.vox2ras0)
            nib.save(img, tp.get_filepath('nonlinear_mask_' + reg_name))
            del mask

            img = nib.Nifti1Image(im_resampled[1], subject.vox2ras0)
            nib.save(img, tp.get_filepath('nonlinear_mask_dilated_' + reg_name))
            del mask_d, im_resampled

        mri = tp.load_data_orig(resampled=False)
        mri_res, _ = def_utils.upscale_and_deform3D(mri, flow, subject_shape, subject.vox2ras0, tp.vox2ras0)
        mri_list.append(mri_res)
        img = nib.Nifti1Image(mri_res, subject.vox2ras0)
        nib.save(img, tp.get_filepath('nonlinear_resample_' + reg_name))
        del mri

        seg = tp.load_seg(resampled=False)
        seg_r, _ = def_utils.upscale_and_deform3D(seg, flow, subject_shape, subject.vox2ras0, tp.vox2ras0,
                                                  mode='nearest')
        img = nib.Nifti1Image(seg_r, subject.vox2ras0)
        nib.save(img, tp.get_filepath('nonlinear_seg_' + reg_name))
        del seg

    template = np.median(mri_list, axis=0)
    img = nib.Nifti1Image(template, subject.vox2ras0)
    nib.save(img, subject.nonlinear_template)
    print('[' + str(subject.id) + ' - DEFORM] Total Elapsed time: ' + str(time.time() - t_init))

    ####################################################################################################
    ####################################################################################################

    print('[' + str(subject.id) + ' - DEFORM] Deforming original images ... ')

    for it_tp, tp in enumerate(timepoints):

        proxyflow = nib.load(join(results_dir_sbj, tp.id + '.flow.nii.gz'))
        flow = np.asarray(proxyflow.dataobj)

        proxyimage = nib.load(tp.init_path['image'])
        mri = np.asarray(proxyimage.dataobj)

        aff_dir = subject.results_dirs.get_dir('linear_st')
        cog = tp.get_cog()
        affine_matrix = read_affine_matrix(join(aff_dir, tp.id + '.aff'), full=True)
        affine_matrix[:3, 3] += cog
        v2r_image = np.matmul(np.linalg.inv(affine_matrix), proxyimage.affine)

        flow_res = np.sqrt(np.sum(proxyflow.affine * proxyflow.affine, axis=0))[:-1]
        image_res = np.sqrt(np.sum(proxyimage.affine * proxyimage.affine, axis=0))[:-1]
        factor = flow_res / image_res

        mri_resampled, template_v2r = def_utils.upscale_and_deform3D(mri, flow, subject_shape, subject.vox2ras0,
                                                                     v2r_image, factor)

        img = nib.Nifti1Image(mri_resampled, template_v2r)
        nib.save(img, tp.get_filepath('nonlinear_image_' + reg_name))

        if not DEBUG:
            remove(join(results_dir_sbj, tp.id + '.flow.nii.gz'))



