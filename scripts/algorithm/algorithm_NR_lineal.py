from os.path import exists
from os import makedirs
import time
from argparse import ArgumentParser


# project imports
from setup import *
from database.data_loader import DataLoader
from scripts import config_dev as configFile
from src.algorithm import *
from src.utils.io import write_affine_matrix
from src.utils.deformation_utils import create_template_space, interpolate3D
from src.utils.image_utils import gaussian_antialiasing

print('\n\n\n\n\n')
print('# ------------------------ #')
print('# Run the linear algorithm #')
print('# ------------------------ #')
print('\n\n')

#####################
# Global parameters #
#####################

observations_dir = OBSERVATIONS_DIR_LINEAL
algorithm_dir = ALGORITHM_DIR_LINEAR
parameter_dict_MRI = configFile.CONFIG_REGISTRATION
REScmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_resample'

# Input parameters
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--cost', type=str, default='l1', choices=['l1', 'l2'], help='Likelihood cost function')
arg_parser.add_argument('--lr', type=float, default=1.0, help='Learning rate')
arg_parser.add_argument('--max_iter', type=int, default=20, help='LBFGS')
arg_parser.add_argument('--n_epochs', type=int, default=100, help='Mask dilation factor')
arg_parser.add_argument('--subjects', default=None, nargs='+')

arguments = arg_parser.parse_args()
cost = arguments.cost
lr = arguments.lr
max_iter = arguments.max_iter
n_epochs = arguments.n_epochs
initial_subject_list = arguments.subjects

results_dir = join(algorithm_dir, cost)
if not exists(results_dir):
    makedirs(results_dir)

##############
# Processing #
##############

data_loader = DataLoader(sid_list=initial_subject_list)
subject_list = data_loader.subject_list

print('[ST LINEAR ALGORITHM] Processing')
for it_subject, subject in enumerate(subject_list):
    print('   Subject: ' + str(subject.id))

    subject_shape = subject.image_shape
    timepoints = subject.timepoints

    if len(subject.timepoints) == 1:
        #TODO small rotation and interpolation + change the header.
        continue

    input_dir = join(observations_dir, subject.id)

    if not exists(join(input_dir, timepoints[-2].id + '_to_' + timepoints[-1].id + '.aff')):
        print('[ST LINEAR ALGORITHM] -- WARING -- No observations found for subject ' + subject.id + ' and NiftyReg.')
        continue

    results_dir_sbj = join(results_dir, subject.id)
    if not exists(join(results_dir_sbj)):
        makedirs(results_dir_sbj)

    elif exists(join(results_dir_sbj, timepoints[-1].id + '.nii.gz')):
        print('[ST LINEAR ALGORITHM] -- DONE -- Subject ' + subject.id + ' has already been processed')
        continue

    # -------------------------------------------------------------------#
    # -------------------------------------------------------------------#
    print('[' + str(subject.id) + ' - Init Graph] Reading SVFs ...')
    t_init = time.time()

    graph_structure = init_st2_lineal(timepoints, input_dir)
    R_log = graph_structure

    print('[' + str(subject.id) + ' - Init Graph] Total Elapsed time: ' + str(time.time() - t_init))

    print('[' + str(subject.id) + ' - ALGORITHM] Running the algorithm ...')
    t_init = time.time()

    Tres = st2_lineal_pytorch(R_log, timepoints, n_epochs, subject_shape, cost, lr, results_dir_sbj, max_iter=max_iter)
    # Tres = st2_lineal_pytorch(R_log, model, optimizer, callbacks, n_epochs, timepoints)

    print('[' + str(subject.id) + ' - ALGORITHM] Total Elapsed time: ' + str(time.time() - t_init))

    # -------------------------------------------------------------------#
    # -------------------------------------------------------------------#

    print('[' + str(subject.id) + ' - INTEGRATION] Computing deformation field ... ')
    t_init = time.time()
    for it_tp, tp in enumerate(timepoints):
        affine_matrix = Tres[..., it_tp]
        write_affine_matrix(join(results_dir_sbj,  tp.id + '.aff'), affine_matrix)

    print('[' + str(subject.id) + ' - INTEGRATION] Total Elapsed time: ' + str(time.time() - t_init))

    # -------------------------------------------------------------------#
    # -------------------------------------------------------------------#

    t_init = time.time()
    print('[' + str(subject.id) + ' - DEFORM] Update vox2ras0  ... ')

    linear_image_list = []
    headers = []
    headers_orig = []
    for it_tp, tp in enumerate(timepoints):

        cog = tp.get_cog()
        affine_matrix = Tres[..., it_tp]
        affine_matrix[:3, 3] += cog

        mri = tp.load_data_orig()
        proxy = nib.load(tp.image_orig_path)
        v2r_mri = np.matmul(np.linalg.inv(affine_matrix), proxy.affine)

        img = nib.Nifti1Image(mri, v2r_mri)
        nib.save(img, tp.image_updatedH_path)

        header = np.matmul(np.linalg.inv(affine_matrix), tp.vox2ras0)
        mask_dilated = tp.load_mask(dilated=True)
        img = nib.Nifti1Image(mask_dilated, header)
        nib.save(img, tp.mask_dilated_updatedH_path)

        headers_orig.append(v2r_mri)
        headers.append(header)
        linear_image_list.append(tp.mask_dilated_updatedH_path)

    # -------------------------------------------------------------------#
    # -------------------------------------------------------------------#

    print('[' + str(subject.id) + ' - DEFORM] Create template space  ... ')
    rasMosaic, template_vox2ras0, template_size = create_template_space(linear_image_list)


    print('[' + str(subject.id) + ' - DEFORM] Deforming images ... ')
    mri_list = []
    for it_tp, tp in enumerate(timepoints):
        aff = headers[it_tp]
        mri = tp.load_data_orig()
        mri = gaussian_antialiasing(mri, headers_orig[it_tp], [1, 1, 1])
        mri_res = tp.load_data()
        mask = tp.load_mask(dilated=False)
        mask_dilated = tp.load_mask(dilated=True)

        image_orig_resampled = interpolate3D([mri], rasMosaic, vox2ras0=headers_orig[it_tp])
        image_orig_resampled = image_orig_resampled[0].reshape(template_size)
        img = nib.Nifti1Image(image_orig_resampled, template_vox2ras0)
        nib.save(img, tp.image_linear_path)
        mri_list.append(image_orig_resampled)
        del mri, image_orig_resampled

        im_resampled = interpolate3D([mri_res, mask, mask_dilated], rasMosaic, vox2ras0=aff)
        image_resampled, mask_resampled, mask_dilated_resampled = im_resampled
        image_resampled = image_resampled.reshape(template_size)
        mask_resampled = mask_resampled.reshape(template_size)
        mask_dilated_resampled = mask_dilated_resampled.reshape(template_size)

        img = nib.Nifti1Image(image_resampled, template_vox2ras0)
        nib.save(img, tp.image_resampled_linear_path)
        del mri_res, image_resampled

        img = nib.Nifti1Image(mask_resampled, template_vox2ras0)
        nib.save(img, tp.mask_linear_path)
        del mask, mask_resampled

        img = nib.Nifti1Image(mask_dilated_resampled, template_vox2ras0)
        nib.save(img, tp.mask_dilated_linear_path)
        del mask_dilated, mask_dilated_resampled

    template = np.median(mri_list, axis=0)
    img = nib.Nifti1Image(template, template_vox2ras0)
    nib.save(img, subject.linear_template)

    print('[' + str(subject.id) + ' - DEFORM] Total Elapsed time: ' + str(time.time() - t_init))
#
# mri_list = []
# for it_tp, tp in enumerate(timepoints):
#     proxy = nib.load(tp.image_resampled_linear_path)
#     data = np.asarray(proxy.dataobj)
#     mri_list.append(data)
# template = np.median(mri_list, axis=0)
# img = nib.Nifti1Image(template, proxy.affine)
# nib.save(img, subject.linear_template)