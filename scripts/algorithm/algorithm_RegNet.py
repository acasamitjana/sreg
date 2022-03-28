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
print('# ------------------------------ #')
print('# Run the nonlinear ST algorithm #')
print('# ------------------------------ #')
print('\n\n')

#####################
# Global parameters #
#####################

parameter_dict_MRI = configFile.REGISTRATION_DIR
REScmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_resample'
TRANSFORMcmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_transform'

# Parameters
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--cost', type=str, default='l1', choices=['l1', 'l2'], help='Likelihood cost function')
arg_parser.add_argument('--subjects', default=None, nargs='+')
arg_parser.add_argument('--reg_algorithm', default='bidir', choices=['standard', 'bidir'])

arguments = arg_parser.parse_args()
cost = arguments.cost
initial_subject_list = arguments.subjects
reg_algorithm = arguments.reg_algorithm

##############
# Processing #
##############

data_loader = DataLoader(sid_list=initial_subject_list, linear=True)
subject_list = data_loader.subject_list[100:]

parameter_dict = configFile.get_config_dict(data_loader.image_shape)
parameter_dict['RESULTS_DIR'] = parameter_dict['RESULTS_DIR'] + '_' + reg_algorithm

for it_subject, subject in enumerate(subject_list):

    print('[ST NONLINEAR ALGORITHM] Processing Subject: ' + str(subject.id))
    print('\n')

    subject_shape = subject.image_shape
    timepoints = subject.timepoints
    results_dir_sbj = subject.results_dirs.get_dir('nonlinear_st_' + reg_algorithm)

    ####################################################################################################
    ####################################################################################################

    date_start = date.today().strftime("%d/%m/%Y")
    time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    expWriter = io_utils.ExperimentWriter(join(subject.results_dirs.get_dir('linear_st'), 'experiment.txt'), attach=False)
    expWriter.write('Experiment date:\n')
    expWriter.write(date_start)
    expWriter.write(time_start)
    expWriter.write('Parameters of the experiment:\n\n')
    expWriter.write('cost:' + str(cost) + '\n')


    if len(subject.timepoints) == 1:
        continue

    force_downsample = 0.5 if parameter_dict['UPSAMPLE_LEVELS'] < 2 else 1 # force downsample to half resolution at least
    cp_shape = tuple([int(i * force_downsample/ parameter_dict['UPSAMPLE_LEVELS']) for i in parameter_dict['VOLUME_SHAPE']])
    input_dir = subject.results_dirs.get_dir('nonlinear_registration_' + reg_algorithm)
    if not exists(join(input_dir, timepoints[-2].id + '_to_' + timepoints[-1].id + '.svf.nii.gz')):
        print('[ST NONLINEAR ALGORITHM] -- WARNING -- No observations found for subject ' + subject.id + ' and NiftyReg ')
        continue

    if exists(join(results_dir_sbj, timepoints[-1].id + '.svf.nii.gz')):
        print('[ST NONLINEAR ALGORITHM] -- DONE -- Subject ' + subject.id + ' has already been processed')
        continue

    ####################################################################################################
    ####################################################################################################

    print('[' + str(subject.id) + ' - Init Graph] Reading SVFs ...')
    t_init = time.time()

    graph_structure = init_st2(timepoints, input_dir, cp_shape, se=None)
    R, M, W, NK = graph_structure

    print('[' + str(subject.id) + ' - Init Graph] Total Elapsed time: ' + str(time.time() - t_init))

    print('[' + str(subject.id) + ' - ALGORITHM] Running the algorithm ...')
    t_init = time.time()
    if cost == 'l2':
        Tres = st2_L2_global(R, W, len(timepoints))

    else:
        Tres = st2_L1(R, M, W, len(timepoints))

    for it_tp, tp in enumerate(timepoints):
        if force_downsample < 1:
            new_shape = tuple([int(i / force_downsample ) for i in cp_shape])

            field = [resize(Tres[0, ..., it_tp], parameter_dict['VOLUME_SHAPE'], order=1, mode='constant')[np.newaxis]]
            field += [resize(Tres[1, ..., it_tp], parameter_dict['VOLUME_SHAPE'], order=1, mode='constant')[np.newaxis]]
            field += [resize(Tres[2, ..., it_tp], parameter_dict['VOLUME_SHAPE'], order=1, mode='constant')[np.newaxis]]

            img = nib.Nifti1Image(np.concatenate(field, axis=0), subject.vox2ras0)

        else:
            img = nib.Nifti1Image(Tres[..., it_tp], subject.vox2ras0)
        nib.save(img, join(results_dir_sbj, tp.id + '.svf.nii.gz'))


    print('[' + str(subject.id) + ' - ALGORITHM] Total Elapsed time: ' + str(time.time() - t_init))

    ####################################################################################################
    ####################################################################################################

    # print('[' + str(subject.id) + ' - INTEGRATION] Computing deformation field ... ')
    # t_init = time.time()
    # for it_tp, tp in enumerate(timepoints):
    #     flow = algorithm_utils.integrate_RegNet(Tres[..., it_tp], subject_shape, parameter_dict=parameter_dict)
    #     img = nib.Nifti1Image(flow, subject.vox2ras0)
    #     nib.save(img, join(results_dir_sbj, tp.id + '.flow.nii.gz'))
    #
    #     del flow
    #
    # print('[' + str(subject.id) + ' - INTEGRATION] Total Elapsed time: ' + str(time.time() - t_init))
    #
    #
    # ####################################################################################################
    # ####################################################################################################
    #
    # t_init = time.time()
    # print('[' + str(subject.id) + ' - DEFORM] Deforming images ... ')
    #
    # mri_list = []
    # for it_tp, tp in enumerate(timepoints):
    #
    #     proxyflow = nib.load(join(results_dir_sbj, tp.id + '.flow.nii.gz'))
    #     flow = np.asarray(proxyflow.dataobj)
    #
    #     if DEBUG:
    #         mask = tp.load_mask()
    #         mask_d = tp.load_mask(dilated=True)
    #         im_resampled, _ = def_utils.upscale_and_deform3D([mask, mask_d], flow, subject_shape,
    #                                                          subject.vox2ras0, tp.vox2ras0,
    #                                                          mode=['linear', 'linear'])
    #
    #         img = nib.Nifti1Image(im_resampled[0], subject.vox2ras0)
    #         nib.save(img, tp.get_filepath('nonlinear_mask'))
    #         del mask
    #
    #         img = nib.Nifti1Image(im_resampled[1], subject.vox2ras0)
    #         nib.save(img, tp.get_filepath('nonlinear_mask_dilated'))
    #         del mask_d, im_resampled
    #
    #     mri = tp.load_data_orig()
    #     mri_res, _ = def_utils.upscale_and_deform3D(mri, flow, subject_shape, subject.vox2ras0, tp.vox2ras0_orig)
    #     mri_list.append(mri_res)
    #     img = nib.Nifti1Image(mri_res, subject.vox2ras0)
    #     nib.save(img, tp.get_filepath('nonlinear_resample'))
    #     del mri
    #
    #     seg = tp.load_seg()
    #     seg_r, _ = def_utils.upscale_and_deform3D(seg, flow, subject_shape, subject.vox2ras0, tp.vox2ras0, mode='nearest')
    #     img = nib.Nifti1Image(seg_r, subject.vox2ras0)
    #     nib.save(img, tp.get_filepath('nonlinear_seg'))
    #     del seg
    #
    # template = np.median(mri_list, axis=0)
    # img = nib.Nifti1Image(template, subject.vox2ras0)
    # nib.save(img, subject.nonlinear_template)
    # print('[' + str(subject.id) + ' - DEFORM] Total Elapsed time: ' + str(time.time() - t_init))
    #
    # ####################################################################################################
    # ####################################################################################################
    #
    # print('[' + str(subject.id) + ' - DEFORM] Deforming original images ... ')
    #
    # for it_tp, tp in enumerate(timepoints):
    #
    #     proxyflow = nib.load(join(results_dir_sbj, tp.id + '.flow.nii.gz'))
    #     flow = np.asarray(proxyflow.dataobj)
    #
    #     proxyimage = nib.load(tp.get_filepath('linear_image'))
    #     mri = np.asarray(proxyimage.dataobj)
    #
    #     flow_res = np.sqrt(np.sum(proxyflow.affine * proxyflow.affine, axis=0))[:-1]
    #     image_res = np.sqrt(np.sum(proxyimage.affine * proxyimage.affine, axis=0))[:-1]
    #     factor = flow_res / image_res
    #
    #     mri_resampled, template_v2r = def_utils.upscale_and_deform3D(mri, flow, subject_shape, subject.vox2ras0,
    #                                                                  proxyimage.affine, factor)
    #
    #     img = nib.Nifti1Image(mri_resampled, template_v2r)
    #     nib.save(img, tp.get_filepath('nonlinear_image'))
    #
    #     if not DEBUG:
    #         remove(join(results_dir_sbj, tp.id + '.flow.nii.gz'))



