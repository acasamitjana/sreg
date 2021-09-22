from os.path import exists
from os import makedirs
import time
from argparse import ArgumentParser
import subprocess

from skimage.transform import resize, rescale

# project imports
from setup import *
from database.data_loader import DataLoader
from scripts import config_dev as configFile
from src.algorithm import *
from src.utils import algorithm_utils, deformation_utils as def_utils


print('\n\n\n\n\n')
print('# ------------------------------ #')
print('# Run the nonlinear ST algorithm #')
print('# ------------------------------ #')
print('\n\n')

#####################
# Global parameters #
#####################

parameter_dict_MRI = configFile.REGISTRATION_DIR
observations_dir = OBSERVATIONS_DIR_NR
algorithm_dir = ALGORITHM_DIR
REScmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_resample'
TRANSFORMcmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_transform'

# Parameters
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--cost', type=str, default='l1', choices=['l1', 'l2'], help='Likelihood cost function')
arg_parser.add_argument('--mdil', type=int, default=7, help='Mask dilation factor')
arg_parser.add_argument('--subjects', default=None, nargs='+')

arguments = arg_parser.parse_args()
cost = arguments.cost
mdil = arguments.mdil
initial_subject_list = arguments.subjects

results_dir = join(algorithm_dir, cost)
if not exists(results_dir):
    makedirs(results_dir)


##############
# Processing #
##############

data_loader = DataLoader(sid_list=initial_subject_list, linear=True)
subject_list = data_loader.subject_list


print('[ST NONLINEAR ALGORITHM] Processing')
for it_subject, subject in enumerate(subject_list):
    print('   Subject: ' + str(subject.id))

    subject_shape = subject.image_shape
    timepoints = subject.timepoints

    if len(subject.timepoints) == 1:
        continue
    cp_shape = tuple([int(i / configFile.CONFIG_REGISTRATION['UPSAMPLE_LEVELS']) for i in subject_shape])

    input_dir = join(observations_dir, subject.id)
    # subject_dir = join(parameter_dict_MRI['DB_CONFIG']['DATA_PATH'], subject.id)

    if not exists(join(input_dir, timepoints[-2].id + '_to_' + timepoints[-1].id + '.svf.nii.gz')):
        print('[ST NONLINEAR ALGORITHM] -- WARNING -- No observations found for subject ' + subject.id + ' and NiftyReg ')
        continue

    results_dir_sbj = join(results_dir, subject.id)
    if not exists(join(results_dir_sbj)):
        makedirs(results_dir_sbj)
    elif exists(join(results_dir_sbj, timepoints[-1].id + '.nii.gz')):
        print('[ST NONLINEAR ALGORITHM] -- DONE -- Subject ' + subject.id + ' has already been processed')
        continue

    ####################################################################################################
    ####################################################################################################
    # print('[' + str(subject.id) + ' - Init Graph] Reading SVFs ...')
    # t_init = time.time()
    #
    # graph_structure = init_st2(timepoints, input_dir, cp_shape, se=np.ones((mdil, mdil, mdil)))
    #
    # R, M, W, NK = graph_structure
    # print('[' + str(subject.id) + ' - Init Graph] Total Elapsed time: ' + str(time.time() - t_init))
    #
    # print('[' + str(subject.id) + ' - ALGORITHM] Running the algorithm ...')
    # t_init = time.time()
    # if cost == 'l2':
    #     Tres = st2_L2_global(R, W, len(timepoints))
    #
    # else:
    #     Tres = st2_L1(R, M, W, len(timepoints))
    #
    # for it_tp, tp in enumerate(timepoints):
    #     img = nib.Nifti1Image(Tres[..., it_tp], subject.vox2ras0)
    #     nib.save(img, join(results_dir_sbj, tp.id + '.svf.nii.gz'))
    #
    #
    # print('[' + str(subject.id) + ' - ALGORITHM] Total Elapsed time: ' + str(time.time() - t_init))
    #
    # ####################################################################################################
    # ####################################################################################################
    #
    # print('[' + str(subject.id) + ' - INTEGRATION] Computing deformation field ... ')
    # t_init = time.time()
    # for it_tp, tp in enumerate(timepoints):
    #
    #
    #     flow = algorithm_utils.integrate_NR(Tres[..., it_tp], subject_shape)
    #     img = nib.Nifti1Image(flow, subject.vox2ras0)
    #     nib.save(img, join(results_dir_sbj, tp.id + '.flow.nii.gz'))
    #     del flow
    #
    #     # refFile = tp.linear_template
    #     # nonlinearSVF = join(results_dir_sbj, tp.id + '.svf.nii.gz')
    #     # nonlinearField = join(results_dir_sbj, tp.id + '.def.nii.gz')
    #     # subprocess.call([TRANSFORMcmd, '-ref', refFile, '-disp', nonlinearSVF, nonlinearField])
    #
    # print('[' + str(subject.id) + ' - INTEGRATION] Total Elapsed time: ' + str(time.time() - t_init))


    ####################################################################################################
    ####################################################################################################
    t_init = time.time()
    print('[' + str(subject.id) + ' - DEFORM] Deforming images ... ')

    mri_list = []
    seg_list = []
    for it_tp, tp in enumerate(timepoints):

        proxyflow = nib.load(join(results_dir_sbj, tp.id + '.flow.nii.gz'))
        flow = np.asarray(proxyflow.dataobj)

        mri = tp.load_data()
        image_deformed = def_utils.deform3D(mri, flow)
        mri_list.append(image_deformed[0])
        img = nib.Nifti1Image(image_deformed[0], subject.vox2ras0)
        nib.save(img, tp.image_nonlinear_path)
        del mri, image_deformed

        mask = tp.load_mask()
        mask_deformed = def_utils.deform3D(mask, flow, mode='nearest')
        img = nib.Nifti1Image(mask_deformed[0], subject.vox2ras0)
        nib.save(img, tp.mask_nonlinear_path)
        del mask, mask_deformed

        # labels = tp.load_seg()
        # labels_deformed = def_utils.deform3D(labels, flow, mode='nearest')
        # img = nib.Nifti1Image(labels_deformed[0], subject.vox2ras0)
        # nib.save(img, tp.seg_nonlinear_path)
        # del labels, labels_deformed, flow
        #



        # refFile = tp.linear_template
        # floFile = tp.image_linear_path
        # nonlinearSVF = join(results_dir_sbj, tp.id + '.svf.nii.gz')
        # outputFile = tp.image_nonlinear_path
        # subprocess.call([
        #     REScmd, '-ref', refFile, '-flo', floFile, '-trans', nonlinearSVF, '-res',
        #     outputFile, '-inter', '1', '-voff'
        # ])
        #
        # floFile = tp.mask_linear_path
        # outputFile = tp.mask_nonlinear_path
        # subprocess.call([
        #     REScmd, '-ref', refFile, '-flo', floFile, '-trans', nonlinearSVF, '-res',
        #     outputFile, '-inter', '0', '-voff'
        # ])
        #
        # floFile = tp.mask_dilated_linear_path
        # outputFile = tp.mask_dilated_nonlinear_path
        # subprocess.call([
        #     REScmd, '-ref', refFile, '-flo', floFile, '-trans', nonlinearSVF, '-res',
        #     outputFile, '-inter', '0', '-voff'
        # ])
        #
        # floFile = tp.seg_linear_path
        # outputFile = tp.seg_nonlinear_path
        # subprocess.call([
        #     REScmd, '-ref', refFile, '-flo', floFile, '-trans', nonlinearSVF, '-res',
        #     outputFile, '-inter', '0', '-voff'
        # ])

    template = np.median(mri_list, axis=0)
    img = nib.Nifti1Image(template, subject.vox2ras0)
    nib.save(img, subject.linear_template)
    print('[' + str(subject.id) + ' - DEFORM] Total Elapsed time: ' + str(time.time() - t_init))
