from os.path import exists
from argparse import ArgumentParser
import time

from joblib import delayed, Parallel
import torch
import torch.nn.functional as F

# project imports
from setup import *
from database.data_loader import DataLoader
from scripts import config_dev as configFile
from src.algorithm import *
from src.utils import algorithm_utils, deformation_utils as def_utils, io_utils, image_utils, results_utils
from database import read_demo_info
from src import models, layers

device = 'cpu'#'cuda:0'
labels_to_write = ['Thalamus', 'Lateral-Ventricle', 'Hippocampus', 'Amygdala', 'Caudate', 'Pallidum', 'Putamen', 'Accumbens', 'Inf-Lat-Ventricle']
keep_labels = ['Right-' + l for l in labels_to_write] + ['Left-' + l for l in labels_to_write]
UNIQUE_LABELS = np.asarray([0] + [lab for labstr, lab in LABEL_DICT.items() if labstr in keep_labels], dtype=np.uint8)
# UNIQUE_LABELS = np.array([  0,   2,   3,   4,   5,   7,   8,  10,  11,  12,  13,  14,  15,
#         16,  17,  18,  24,  26,  28,  30,  31,  41,  42,  43,  44,  46,
#         47,  49,  50,  51,  52,  53,  54,  58,  60,  62,  63,  77,  80,
#         85, 251, 252, 253, 254, 255], dtype=np.uint8)


class FS_Lin(results_utils.LabelFusion):

    def compute_p_label(self, timepoints, image_list, svf_list, age_list, tp, subject, parameter_dict, regnet_model):
    # def compute_p_label(self, timepoints, image_list, age_list, tp, subject):


        interp_func = layers.SpatialInterpolation(padding_mode='zeros').to(device)

        results_dir_sbj = subject.results_dirs.get_dir('linear_st')
        p_data = {t_var: {sp_var: np.zeros(tp.image_shape + (len(timepoints),), dtype='float32') for sp_var in spatial_variance} for t_var in temp_variance}
        p_label = {t_var: {sp_var: np.zeros(tp.image_shape + (len(UNIQUE_LABELS),), dtype='float32') for sp_var in spatial_variance} for t_var in temp_variance}
        ref_shape = image_list[tp.id].shape

        ii = np.arange(0, ref_shape[0], dtype='int32')
        jj = np.arange(0, ref_shape[1], dtype='int32')
        kk = np.arange(0, ref_shape[2], dtype='int32')

        II, JJ, KK = np.meshgrid(ii, jj, kk, indexing='ij')

        voxMosaic_orig = np.concatenate((II.reshape(-1, 1), JJ.reshape(-1, 1), KK.reshape(-1, 1), np.ones((int(np.prod(ref_shape[:3])), 1), dtype='int32')), axis=1).T
        rasMosaic_orig = np.dot(tp.vox2ras0, voxMosaic_orig).astype('float32')

        del voxMosaic_orig, ii, jj, kk, II, JJ, KK

        for it_tp_flo, tp_flo in enumerate(timepoints):
            fileparts = tp_flo.id.split('_')
            filepath = join(FS_DIR, fileparts[0] + '_' + fileparts[1] + '_' + str(int(fileparts[4])) + '_' + fileparts[5] + '_' + fileparts[6] + '.cross.aseg.mgz')
            proxyseg = nib.load(filepath)

            cog = tp_flo.get_cog()
            affine_matrix = read_affine_matrix(join(results_dir_sbj, tp_flo.id + '.aff'), full=True)
            affine_matrix[:3, 3] += cog
            vox2ras0_fs = np.dot(np.linalg.inv(affine_matrix), proxyseg.affine).astype('float32')

            if tp_flo.id == tp.id:
                for t_var in temp_variance:
                    for s_var in spatial_variance:
                        p_data[t_var][s_var][..., it_tp_flo] = 1

                rasMosaic_targ = rasMosaic_orig.copy()
                voxMosaic_targ = np.matmul(np.linalg.inv(vox2ras0_fs), rasMosaic_targ).astype('float32')#np.dot(np.linalg.inv(vox2ras0_fs), np.dotnp.dot(tp.vox2ras0, voxMosaic_targ))
                voxMosaic_targ = voxMosaic_targ[:3]


                del rasMosaic_targ

            else:

                rasMosaic_targ = rasMosaic_orig.copy()
                voxMosaic_targ = np.matmul(np.linalg.inv(tp_flo.vox2ras0), rasMosaic_targ).astype('float32')
                voxMosaic_targ = voxMosaic_targ[:3]

                voxMosaic_targ = voxMosaic_targ.reshape((1,3) + ref_shape)
                voxMosaic_targ = torch.from_numpy(voxMosaic_targ).float().to(device)

                im = torch.from_numpy(image_list[tp_flo.id][np.newaxis, np.newaxis]).float().to(device)
                im_resampled = interp_func(im, voxMosaic_targ)
                im_resampled = im_resampled[0, 0].cpu().detach().numpy()

                del voxMosaic_targ

                mean_im_2 = (im_resampled - image_list[tp.id]) ** 2

                del im_resampled

                mean_age_2 = (age_list[tp_flo.id] - age_list[tp.id]) ** 2
                for t_var in temp_variance:
                    for s_var in spatial_variance:
                        p_data[t_var][s_var][..., it_tp_flo] = np.exp(-0.5 / s_var * mean_im_2) \
                                                             * np.exp(-0.5 / t_var * mean_age_2)

                voxMosaic_targ = np.matmul(np.linalg.inv(vox2ras0_fs), rasMosaic_targ).astype('float32')
                voxMosaic_targ = voxMosaic_targ[:3]

                del rasMosaic_targ

            voxMosaic_targ = voxMosaic_targ.reshape((3,) + ref_shape)
            voxMosaic_targ_torch = voxMosaic_targ.copy()
            voxMosaic_targ_torch = torch.from_numpy(voxMosaic_targ_torch[np.newaxis]).float().to(device)

            del voxMosaic_targ

            seg = np.asarray(proxyseg.dataobj)
            seg_onehot = image_utils.one_hot_encoding(seg, categories=UNIQUE_LABELS).astype('float32')

            del seg

            seg_onehot = torch.from_numpy(seg_onehot[np.newaxis]).float().to(device)
            seg_resampled = interp_func(seg_onehot, voxMosaic_targ_torch)

            del seg_onehot, voxMosaic_targ_torch

            seg_resampled = np.transpose(seg_resampled[0].cpu().detach().numpy(), (1, 2, 3, 0))

            for t_var in temp_variance:
                for s_var in spatial_variance:
                    p_label[t_var][s_var] += p_data[t_var][s_var][..., it_tp_flo, np.newaxis] * seg_resampled

            del seg_resampled

        for t_var in temp_variance:
            for s_var in spatial_variance:
                p_label[t_var][s_var] = p_label[t_var][s_var] / np.sum(p_data[t_var][s_var], axis=-1, keepdims=True)

        return p_label

# def write_volume_results(volume_dict, filepath, fieldnames=None, attach_overwrite='a'):
#     if fieldnames is None:
#         fieldnames = ['TID'] + list(LABEL_DICT.keys())
#
#     with open(filepath, attach_overwrite) as csvfile:
#         csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         if attach_overwrite == 'w':
#             csvwriter.writeheader()
#         csvwriter.writerows(volume_dict)
#
# def get_vols_post(post):
#     n_labels = post.shape[-1]
#     vols = {}
#     for l in range(n_labels):
#         mask_l = post[..., l]
#         mask_l[post[..., l] < 0.05] = 0
#         vols[l] = np.sum(mask_l)
#
#     return vols
#
#
# def label_fusion(subject):
#     print('Subject: ' + str(subject.id))
#     timepoints = subject.timepoints
#     results_dir_sbj = subject.results_dirs.get_dir('longitudinal_segmentation_st_freesurfer') + '_AFF'
#
#     last_dir = join(results_dir_sbj, str(temp_variance[-1]) + '_' + str(spatial_variance[1]))
#     timepoints_to_run = timepoints[0:1]#list(filter(lambda x: not exists(join(last_dir, x.id + '.nii.gz')), timepoints))
#     attach_overwrite = 'a' if exists(join(last_dir, 'vols_st.txt')) else 'w'
#
#
#     # if not timepoints_to_run:
#     #     print('Subject: ' + str(subject.id) + '. DONE')
#     #     return
#
#     print('  o Reading the input files')
#     image_list = {}
#     age_list = {}
#     for tp in timepoints:
#         # Age
#         age_list[tp.id] = float(demo_dict[subject.id][tp.id]['AGE'])
#
#         # Data
#         seg = tp.load_seg()
#         image = tp.load_data()
#
#         # Normalize image
#         wm_mask = (seg == 2) | (seg == 41)
#         m = np.mean(image[wm_mask])
#         image = 110 * image / m
#         image_list[tp.id] = image
#
#         del image, seg, wm_mask
#
#     # Model
#
#     print('  o Computing the segmentation')
#     st_vols = {t_var: {sp_var: [] for sp_var in spatial_variance} for t_var in temp_variance}
#     for tp in timepoints_to_run:
#         print('        - Timepoint ' + tp.id, end=':', flush=True)
#
#         t_0 = time.time()
#         p_label_dict = compute_p_label(timepoints, image_list, age_list, tp, subject)
#         t_1 = time.time()
#         print(str(t_1-t_0) + ' seconds.')
#
#         proxy = nib.load(tp.init_path['resample'])
#         for t_var in temp_variance:
#             for s_var in spatial_variance:
#                 if not exists(join(results_dir_sbj, str(t_var) + '_' + str(s_var))):
#                     os.makedirs(join(results_dir_sbj, str(t_var) + '_' + str(s_var)))
#
#                 p_label = p_label_dict[t_var][s_var]
#                 p_label[np.isnan(p_label)] = 0
#                 mask = np.sum(p_label, axis=-1) > 0.5
#                 fake_vol = np.argmax(p_label, axis=-1).astype('uint16')
#                 true_vol = np.zeros_like(fake_vol)
#                 for it_ul, ul in enumerate(UNIQUE_LABELS): true_vol[fake_vol == it_ul] = ul
#                 true_vol = true_vol * mask
#
#                 img = nib.Nifti1Image(p_label, proxy.affine)
#                 nib.save(img, join(results_dir_sbj, str(t_var) + '_' + str(s_var), tp.id + '.post.nii.gz'))
#
#                 img = nib.Nifti1Image(true_vol, proxy.affine)
#                 nib.save(img, join(results_dir_sbj, str(t_var) + '_' + str(s_var), tp.id + '.nii.gz'))
#
#                 vols = get_vols_post(p_label)
#                 st_vols_dict = {k: vols[[it_ul for it_ul, ul in enumerate(UNIQUE_LABELS) if ul == val][0]] for k, val in LABEL_DICT.items() if val in UNIQUE_LABELS}
#                 st_vols_dict['TID'] = tp.id
#                 st_vols[t_var][s_var].append(st_vols_dict)
#
#                 del p_label, fake_vol, true_vol, mask
#
#         del p_label_dict
#
#     fieldnames = ['TID'] + list(LABEL_DICT.keys())
#     for t_var in temp_variance:
#         for s_var in spatial_variance:
#             write_volume_results(st_vols[t_var][s_var], join(results_dir_sbj, str(t_var) + '_' + str(s_var), 'vols_st.txt'),
#                                  fieldnames=fieldnames, attach_overwrite=attach_overwrite)
#
#     print('Subject: ' + str(subject.id) + '. DONE')
#

print('\n\n\n\n\n')
print('# -------------------------------------------- #')
print('# Running the longitudinal segmentation script #')
print('# -------------------------------------------- #')
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
arg_parser.add_argument('--num_cores', default=1, type=int)
arg_parser.add_argument('--reg_algorithm', default='bidir', choices=['standard', 'bidir', 'niftyreg'])


arguments = arg_parser.parse_args()
initial_subject_list = arguments.subjects
num_cores = arguments.num_cores
reg_algorithm = arguments.reg_algorithm

FS_DIR = '/home/acasamitjana/Results/Registration/BrainAging/miriad_Eugenio/longitudinal'

##############
# Processing #
##############

demo_dict = read_demo_info(demo_fields=['AGE'])
data_loader = DataLoader(sid_list=initial_subject_list, linear=True)
parameter_dict = configFile.get_config_dict(data_loader.image_shape)

subject_list = data_loader.subject_list
subject_list = list(filter(lambda x: x.id in demo_dict.keys(), subject_list))

print('[LONGITUDINAL SEGMENTATION] Start processing.')
spatial_variance = [3**2, 5**2, 7**2, 9**2] # in grayscale
temp_variance = [100000]#[1/4, 1/2, 1, 2] #in years^2

segmenter = FS_Lin(demo_dict=demo_dict, parameter_dict=parameter_dict, reg_algorithm=reg_algorithm, device=device,
                   long_seg_algo='freesurfer_lineal', temp_variance=temp_variance, spatial_variance=spatial_variance)

if num_cores == 1:
    for subject in subject_list:
        segmenter.label_fusion(subject)

elif len(subject_list) == 1:
    segmenter.label_fusion(subject_list[0])

else:
    results = Parallel(n_jobs=num_cores)(delayed(segmenter.label_fusion)(subject) for subject in subject_list)








