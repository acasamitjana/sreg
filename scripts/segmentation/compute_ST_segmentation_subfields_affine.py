import pdb
from os.path import exists
from argparse import ArgumentParser
import time

import numpy as np
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

device = 'cpu' #'cuda:0'#

class FS_LinNonlin(results_utils.LabelFusionSubfields):


    def compute_1x1x1_inputs_old(self, tp, tp_flo, ref_shape):
        results_dir_sbj = subject.results_dirs.get_dir('linear_st')
        cog = tp.get_cog()
        affine_matrix_ref = read_affine_matrix(join(results_dir_sbj, tp.id + '.aff'), full=True)
        affine_matrix_ref[:3, 3] += cog

        ii = np.arange(0, ref_shape[0], dtype='int32')
        jj = np.arange(0, ref_shape[1], dtype='int32')
        kk = np.arange(0, ref_shape[2], dtype='int32')

        II, JJ, KK = np.meshgrid(ii, jj, kk, indexing='ij')

        voxMosaic_orig = np.concatenate((II.reshape(-1, 1), JJ.reshape(-1, 1), KK.reshape(-1, 1),
                                         np.ones((int(np.prod(ref_shape[:3])), 1), dtype='int32')), axis=1).T
        rasMosaic_ref = np.dot(tp.vox2ras0, voxMosaic_orig).astype('float32')
        rasMosaic_lin_ref = np.dot(np.linalg.inv(affine_matrix_ref), rasMosaic_ref).astype('float32')

        del voxMosaic_orig, ii, jj, kk

        cog = tp_flo.get_cog()
        affine_matrix_flo = read_affine_matrix(join(results_dir_sbj, tp_flo.id + '.aff'), full=True)
        affine_matrix_flo[:3, 3] += cog

        if tp_flo.id == tp.id:
            rasMosaic_flo = rasMosaic_ref

        else:
            rasMosaic_flo = np.matmul(affine_matrix_flo, rasMosaic_lin_ref).astype('float32')
            rasMosaic_flo = rasMosaic_flo.reshape((4,) + ref_shape)

        return rasMosaic_flo

    def create_ref_space(self, proxyseg_ref):

        ii = np.arange(0, proxyseg_ref.shape[0], dtype='int32')
        jj = np.arange(0, proxyseg_ref.shape[1], dtype='int32')
        kk = np.arange(0, proxyseg_ref.shape[2], dtype='int32')

        II, JJ, KK = np.meshgrid(ii, jj, kk, indexing='ij')

        I = II.reshape(-1, 1)
        J = JJ.reshape(-1, 1)
        K = KK.reshape(-1, 1)
        C = np.ones((int(np.prod(proxyseg_ref.shape[:3])), 1), dtype='int32')
        voxMosaic_orig = np.concatenate((I, J, K, C), axis=1).T
        rasMosaic_ref = np.dot(proxyseg_ref.affine, voxMosaic_orig).astype('float32')

        rasMosaic_ref = rasMosaic_ref.reshape((4,) + proxyseg_ref.shape)

        return rasMosaic_ref

    def compute_1x1x1_inputs(self, tp, tp_flo, ref_shape):
        results_dir_sbj = subject.results_dirs.get_dir('linear_st')
        cog = tp.get_cog()
        affine_matrix_ref = read_affine_matrix(join(results_dir_sbj, tp.id + '.aff'), full=True)
        affine_matrix_ref[:3, 3] += cog

        ii = np.arange(0, ref_shape[0], dtype='int32')
        jj = np.arange(0, ref_shape[1], dtype='int32')
        kk = np.arange(0, ref_shape[2], dtype='int32')

        II, JJ, KK = np.meshgrid(ii, jj, kk, indexing='ij')

        voxMosaic_orig = np.concatenate((II.reshape(-1, 1), JJ.reshape(-1, 1), KK.reshape(-1, 1),
                                         np.ones((int(np.prod(ref_shape[:3])), 1), dtype='int32')), axis=1).T
        rasMosaic_ref = np.dot(tp.vox2ras0, voxMosaic_orig).astype('float32')
        rasMosaic_lin_ref = np.dot(np.linalg.inv(affine_matrix_ref), rasMosaic_ref).astype('float32')

        del voxMosaic_orig, ii, jj, kk

        cog = tp_flo.get_cog()
        affine_matrix_flo = read_affine_matrix(join(results_dir_sbj, tp_flo.id + '.aff'), full=True)
        affine_matrix_flo[:3, 3] += cog

        if tp_flo.id == tp.id:
            rasMosaic_flo = rasMosaic_ref

        else:
            rasMosaic_flo = np.matmul(affine_matrix_flo, rasMosaic_lin_ref).astype('float32')
            rasMosaic_flo = rasMosaic_flo.reshape((4,) + ref_shape)

        return rasMosaic_flo


    def resample_ras_space(self, image, v2r, rasMosaic_ref):

        ref_shape = rasMosaic_ref.shape[1:]
        voxMosaic_targ = np.matmul(np.linalg.inv(v2r), rasMosaic_ref.reshape(4, -1)).astype('float32')
        voxMosaic_targ = voxMosaic_targ[:3]

        voxMosaic_targ = voxMosaic_targ.reshape((3,) + ref_shape)
        voxMosaic_targ_torch = voxMosaic_targ.copy()
        voxMosaic_targ_torch = torch.from_numpy(voxMosaic_targ_torch[np.newaxis]).float().to(device)
        im = torch.from_numpy(image[np.newaxis]).float().to(device)
        im_resampled = self.interp_func(im, voxMosaic_targ_torch)  # , mode='nearest')

        return im_resampled[0]

    def compute_p_label(self, timepoints, image_list, flow_dir, age_list, tp, subject, parameter_dict=None, regnet_model=None):
        filepath = tp.init_path[self.filekey]
        proxyseg_ref = nib.load(filepath)
        seg_ref = np.array(proxyseg_ref.dataobj)
        seg_ref_onehot = image_utils.one_hot_encoding(seg_ref, categories=list(SUBFIELDS_LABEL_DICT.values())).astype('float32')

        p_data = {t_var: {sp_var: np.zeros(proxyseg_ref.shape + (len(timepoints),), dtype='float32') for sp_var in spatial_variance} for t_var in temp_variance}
        p_label = {t_var: {sp_var: np.zeros(proxyseg_ref.shape + (len(results_utils.SUBFIELDS_LABELS),), dtype='float32') for sp_var in spatial_variance} for t_var in temp_variance}

        ref_shape = image_list[tp.id].shape
        image_ref = image_list[tp.id]

        rasMosaic_ref = self.create_ref_space(proxyseg_ref)
        image_ref_resampled = self.resample_ras_space(image_ref[np.newaxis], tp.vox2ras0, rasMosaic_ref)
        image_ref_resampled = image_ref_resampled[0].cpu().detach().numpy()

        for it_tp_flo, tp_flo in enumerate(timepoints):

            filepath = tp_flo.init_path[self.filekey]
            proxyseg_flo = nib.load(filepath)
            seg_flo = np.array(proxyseg_flo.dataobj)
            seg_flo_onehot = image_utils.one_hot_encoding(seg_flo, categories=list(SUBFIELDS_LABEL_DICT.values())).astype('float32')

            if tp_flo.id == tp.id:
                seg_resampled = np.transpose(seg_ref_onehot, axes=(1, 2, 3, 0))
                for t_var in temp_variance:
                    for s_var in spatial_variance:
                        p_data[t_var][s_var][..., it_tp_flo] = 1
                        p_label[t_var][s_var] += seg_resampled

                if DEBUG:
                    results_dir_sbj = subject.results_dirs.get_dir('longitudinal_segmentation_st_' + self.long_seg_algo) + self.suffix
                    if not exists(join(results_dir_sbj, 'debug')): os.makedirs(join(results_dir_sbj, 'debug'))
                    img = nib.Nifti1Image(seg_resampled.astype('uint16'), proxyseg_ref.affine)
                    nib.save(img, join(results_dir_sbj, 'debug', tp.id + '_to_' + tp_flo.id + '.post.nii.gz'))

            else:

                rasDefField = self.compute_1x1x1_inputs(tp, tp_flo, ref_shape)

                ################################

                rasDefField_resampled = self.resample_ras_space(rasDefField, tp.vox2ras0, rasMosaic_ref)
                rasDefField_resampled = rasDefField_resampled.cpu().detach().numpy()

                ################################

                image_flo = image_list[tp_flo.id]
                im_resampled = self.resample_ras_space(image_flo[np.newaxis], tp_flo.vox2ras0, rasDefField_resampled)
                im_flo_resampled = im_resampled[0].cpu().detach().numpy()

                ################################

                seg_resampled = self.resample_ras_space(seg_flo_onehot, proxyseg_flo.affine, rasDefField_resampled)
                seg_resampled = seg_resampled.cpu().detach().numpy()
                seg_resampled = np.transpose(seg_resampled, axes=(1, 2, 3, 0))

                mean_im_2 = (im_flo_resampled - image_ref_resampled) ** 2
                mean_age_2 = (age_list[tp_flo.id] - age_list[tp.id]) ** 2

                for t_var in temp_variance:
                    t_ker = 1 if t_var == 'inf' else np.exp(-0.5 / t_var * mean_age_2)
                    for s_var in spatial_variance:
                        s_ker = 1 if s_var == 'inf' else np.exp(-0.5 / s_var * mean_im_2)
                        p_data[t_var][s_var][..., it_tp_flo] = s_ker * t_ker
                        p_label[t_var][s_var] += p_data[t_var][s_var][..., it_tp_flo, np.newaxis] * seg_resampled

                if DEBUG:
                    results_dir_sbj = subject.results_dirs.get_dir('longitudinal_segmentation_st_' + self.long_seg_algo)  + self.suffix
                    if not exists(join(results_dir_sbj, 'debug')): os.makedirs(join(results_dir_sbj, 'debug'))
                    img = nib.Nifti1Image(seg_resampled, proxyseg_ref.affine)
                    nib.save(img, join(results_dir_sbj, 'debug', tp.id + '_to_' + tp_flo.id + '.post.nii.gz'))

                    img = nib.Nifti1Image(im_flo_resampled, proxyseg_ref.affine)
                    nib.save(img, join(results_dir_sbj, 'debug', tp.id + '_to_' + tp_flo.id + '.image.nii.gz'))

                del im_resampled, seg_resampled, im_flo_resampled


        for t_var in temp_variance:
            for s_var in spatial_variance:
                p_label[t_var][s_var] = p_label[t_var][s_var] / np.sum(p_data[t_var][s_var], axis=-1, keepdims=True)

        return p_label




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
arg_parser.add_argument('--force', action='store_true')
arg_parser.add_argument('--reg_algorithm', default='bidir', choices=['standard', 'bidir', 'niftyreg'])


arguments = arg_parser.parse_args()
initial_subject_list = arguments.subjects
num_cores = arguments.num_cores
reg_algorithm = arguments.reg_algorithm
force_flag = arguments.force

FS_DIR = '/home/acasamitjana/Results/Registration/BrainAging/miriad_Eugenio/longitudinal'

##############
# Processing #
##############

reg_name = reg_algorithm
if reg_algorithm == 'bidir':
    parameter_dict = configFile.CONFIG_REGISTRATION
    reg_name += str(parameter_dict['LOSS_REGISTRATION_SMOOTHNESS']['lambda'])
demo_dict = read_demo_info(demo_fields=['AGE'])
data_loader = DataLoader(sid_list=initial_subject_list)#, linear=True, reg_algorithm=reg_name)
parameter_dict = configFile.get_config_dict(data_loader.image_shape)

subject_list = data_loader.subject_list
# subject_list = list(filter(lambda x: x.id in demo_dict.keys(), subject_list))[52:]
# idx = [it_s for it_s, sbj in enumerate(subject_list) if sbj.id == '100_S_0015'][0] # run 127_S_1032
# subject_list = subject_list[idx+1:]

print('[LONGITUDINAL SEGMENTATION] Start processing.')
spatial_variance = [3**2]#, 7**2] # in grayscale
temp_variance = ['inf'] #in years^2

segmenterLH = FS_LinNonlin(filekey='subfields.lh', demo_dict=demo_dict, parameter_dict=parameter_dict,
                           reg_algorithm=reg_name, device=device, long_seg_algo='linear',
                           temp_variance=temp_variance, spatial_variance=spatial_variance,  suffix='_LH')
segmenterRH = FS_LinNonlin(filekey='subfields.rh', demo_dict=demo_dict, parameter_dict=parameter_dict,
                           reg_algorithm=reg_name, device=device, long_seg_algo='linear',
                           temp_variance=temp_variance, spatial_variance=spatial_variance,  suffix='_RH')

if num_cores == 1:
    for subject in subject_list:
        # res_dir_lh = subject.results_dirs.get_dir('longitudinal_segmentation_st_linear') + '_LH'
        # res_dir_rh = subject.results_dirs.get_dir('longitudinal_segmentation_st_linear') + '_RH'
        #

        # if exists(join(res_dir_lh)):
        #     shutil.move()

        try:
            segmenterLH.label_fusion(subject, force_flag=force_flag)
            segmenterRH.label_fusion(subject, force_flag=force_flag)
        except:
            continue

elif len(subject_list) == 1:
    segmenterLH.label_fusion(subject_list[0])
    segmenterRH.label_fusion(subject_list[0])

else:
    results = Parallel(n_jobs=num_cores)(delayed(segmenterLH.label_fusion)(subject) for subject in subject_list)
    print(results)
    results = Parallel(n_jobs=num_cores)(delayed(segmenterRH.label_fusion)(subject) for subject in subject_list)
    print(results)








