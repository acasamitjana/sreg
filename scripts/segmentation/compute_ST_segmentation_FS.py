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

device = 'cpu' #'cuda:0'#

class FS_LinNonlin(results_utils.LabelFusion):

    def compute_p_label(self, timepoints, image_list, flow_dir, age_list, tp, subject, parameter_dict, regnet_model):

        interp_func = layers.SpatialInterpolation(padding_mode='zeros').to()

        results_dir_sbj = subject.results_dirs.get_dir('linear_st')
        p_data = {t_var: {sp_var: np.zeros(tp.image_shape + (len(timepoints),), dtype='float32') for sp_var in spatial_variance} for t_var in temp_variance}
        p_label = {t_var: {sp_var: np.zeros(tp.image_shape + (len(results_utils.UNIQUE_LABELS),), dtype='float32') for sp_var in spatial_variance} for t_var in temp_variance}
        ref_shape = image_list[tp.id].shape

        ii = np.arange(0, ref_shape[0], dtype='int32')
        jj = np.arange(0, ref_shape[1], dtype='int32')
        kk = np.arange(0, ref_shape[2], dtype='int32')

        II, JJ, KK = np.meshgrid(ii, jj, kk, indexing='ij')

        voxMosaic_orig = np.concatenate((II.reshape(-1, 1), JJ.reshape(-1, 1), KK.reshape(-1, 1), np.ones((int(np.prod(ref_shape[:3])), 1), dtype='int32')), axis=1).T
        rasMosaic_orig = np.dot(tp.vox2ras0, voxMosaic_orig).astype('float32')
        voxMosaic_ref = np.dot(np.linalg.inv(subject.vox2ras0), rasMosaic_orig).astype('float32')

        del voxMosaic_orig, ii, jj, kk

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
                tp_svf = nib.load(join(flow_dir, tp.id + '.svf.nii.gz'))
                tp_flo_svf = nib.load(join(flow_dir, tp_flo.id + '.svf.nii.gz'))

                svf = np.asarray(tp_flo_svf.dataobj) - np.asarray(tp_svf.dataobj)
                svf = svf.astype('float32')
                flow = algorithm_utils.integrate_RegNet(svf, subject.image_shape, parameter_dict, device=device, model=regnet_model)
                flow_resampled = def_utils.interpolate3DChannel(np.transpose(flow, (1,2,3,0)), voxMosaic_ref[:3].T)#def_utils.interpolate3D([flow[i] for i in range(3)], voxMosaic_2[:3].T)

                del svf, flow

                # Image
                voxMosaic_ref_nl = voxMosaic_ref.copy()
                voxMosaic_ref_nl[0] += flow_resampled[..., 0]
                voxMosaic_ref_nl[1] += flow_resampled[..., 1]
                voxMosaic_ref_nl[2] += flow_resampled[..., 2]
                rasMosaic_ref_nl = np.dot(subject.vox2ras0, voxMosaic_ref_nl).astype('float32')

                voxMosaic_targ = np.matmul(np.linalg.inv(tp_flo.vox2ras0), rasMosaic_ref_nl).astype('float32')
                voxMosaic_targ = voxMosaic_targ[:3]

                del voxMosaic_ref_nl, flow_resampled

                voxMosaic_targ = voxMosaic_targ.reshape((3,) + ref_shape)
                voxMosaic_targ_torch = voxMosaic_targ.copy()
                voxMosaic_targ_torch = torch.from_numpy(voxMosaic_targ_torch[np.newaxis]).float().to(device)

                im = torch.from_numpy(image_list[tp_flo.id][np.newaxis, np.newaxis]).float().to(device)
                im_resampled = interp_func(im, voxMosaic_targ_torch)#, mode='nearest')
                im_resampled = im_resampled[0, 0].cpu().detach().numpy()

                del voxMosaic_targ_torch

                # deformation = voxMosaic_targ - np.stack((II, JJ, KK), axis=0)
                # jac = results_utils.compute_jacobians(deformation)
                # d_ker = (1 - np.abs(1 - jac))

                del voxMosaic_targ


                mean_im_2 = (im_resampled - image_list[tp.id]) ** 2
                mean_age_2 = (age_list[tp_flo.id] - age_list[tp.id]) ** 2
                for t_var in temp_variance:
                    t_ker = 1 if t_var == 'inf' else np.exp(-0.5 / t_var * mean_age_2)
                    for s_var in spatial_variance:
                        s_ker = 1 if s_var == 'inf' else np.exp(-0.5 / s_var * mean_im_2)
                        p_data[t_var][s_var][..., it_tp_flo] = s_ker * t_ker #* d_ker

                voxMosaic_targ = np.matmul(np.linalg.inv(vox2ras0_fs), rasMosaic_ref_nl).astype('float32')
                voxMosaic_targ = voxMosaic_targ[:3]

                del rasMosaic_ref_nl

            voxMosaic_targ = voxMosaic_targ.reshape((3,) + ref_shape)
            voxMosaic_targ_torch = voxMosaic_targ.copy()
            voxMosaic_targ_torch = torch.from_numpy(voxMosaic_targ_torch[np.newaxis]).float().to(device)

            del voxMosaic_targ

            seg = np.asarray(proxyseg.dataobj)
            seg_onehot = image_utils.one_hot_encoding(seg, categories=results_utils.UNIQUE_LABELS).astype('float32')

            del seg

            seg_onehot = torch.from_numpy(seg_onehot[np.newaxis]).float().to(device)
            seg_resampled = interp_func(seg_onehot, voxMosaic_targ_torch)

            del seg_onehot, voxMosaic_targ_torch

            seg_resampled = np.transpose(seg_resampled[0].cpu().detach().numpy(), (1, 2, 3, 0))

            for t_var in temp_variance:
                for s_var in spatial_variance:
                    p_label[t_var][s_var] += p_data[t_var][s_var][..., it_tp_flo, np.newaxis] * seg_resampled


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

demo_dict = read_demo_info(demo_fields=['AGE'])
data_loader = DataLoader(sid_list=initial_subject_list, linear=True)
parameter_dict = configFile.get_config_dict(data_loader.image_shape)

subject_list = data_loader.subject_list
subject_list = list(filter(lambda x: x.id in demo_dict.keys(), subject_list))

print('[LONGITUDINAL SEGMENTATION] Start processing.')
spatial_variance = [3**2, 5**2, 7**2] # in grayscale
temp_variance = [1, 'inf'] #in years^2

segmenter = FS_LinNonlin(demo_dict=demo_dict, parameter_dict=parameter_dict, reg_algorithm=reg_algorithm, device=device,
                         long_seg_algo='freesurfer', temp_variance=temp_variance, spatial_variance=spatial_variance)

if num_cores == 1:
    for subject in subject_list:
        segmenter.label_fusion(subject, force_flag=force_flag, suffix='_JAC')

elif len(subject_list) == 1:
    segmenter.label_fusion(subject_list[0])

else:
    results = Parallel(n_jobs=num_cores)(delayed(segmenter.label_fusion)(subject) for subject in subject_list)








