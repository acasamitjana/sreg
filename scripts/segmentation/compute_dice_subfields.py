from os.path import exists
from argparse import ArgumentParser
import time

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, font_manager as fm

# project imports
from setup import *
from database.data_loader import DataLoader
from scripts import config_dev as configFile
from src.algorithm import *
from src.utils import algorithm_utils, deformation_utils as def_utils, io_utils, image_utils, results_utils
from src import metrics, models, layers


fpath = '/usr/share/fonts/truetype/msttcorefonts/Arial_Black.ttf'#'/usr/share/fonts/truetype/msttcorefonts/Courier_New.ttf'
prop_bold = fm.FontProperties(fname=fpath)
fpath = '/usr/share/fonts/truetype/msttcorefonts/arial.ttf'
prop = fm.FontProperties(fname=fpath)
prop_legend = fm.FontProperties(fname=fpath, size=16)


def register_timepoints(flow, ref_image_shape, flo_image, ref_vox2ras0, flo_vox2ras0, subject_vox2ras0, mode='linear'):
    ii = np.arange(0, ref_image_shape[0]),
    jj = np.arange(0, ref_image_shape[1]),
    kk = np.arange(0, ref_image_shape[2])

    II, JJ, KK = np.meshgrid(ii, jj, kk, indexing='ij')

    voxMosaic = np.concatenate((II.reshape(-1, 1), JJ.reshape(-1, 1), KK.reshape(-1, 1), np.ones((int(np.prod(ref_image_shape[:3])), 1))), axis=1).T
    rasMosaic = np.dot(ref_vox2ras0, voxMosaic)
    voxMosaic_2 = np.dot(np.linalg.inv(subject_vox2ras0), rasMosaic)

    flow_resampled = def_utils.interpolate3D([flow[i] for i in range(3)], voxMosaic_2[:3].T)

    voxMosaic_3 = voxMosaic_2
    voxMosaic_3[0] += flow_resampled[0]
    voxMosaic_3[1] += flow_resampled[1]
    voxMosaic_3[2] += flow_resampled[2]
    rasMosaic_3 = np.dot(subject_vox2ras0, voxMosaic_3)
    im_resampled = def_utils.interpolate3D(flo_image, rasMosaic_3, vox2ras0=flo_vox2ras0, resized_shape=ref_image_shape, mode=mode)

    return im_resampled


class ComputeDiceSubfields(object):

    def __init__(self, filekey, device='cpu'):

        self.filekey = filekey
        self.interp_func = layers.SpatialInterpolation(padding_mode='zeros').to(device)
        self.device = device

    def compute_1x1x1_inputs(self, tp, tp_flo, flow, ref_shape):
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

        if flow is not None:
            voxMosaic_lin_ref = np.dot(np.linalg.inv(subject.vox2ras0), rasMosaic_lin_ref).astype('float32')

            voxMosaic_nl_ref = voxMosaic_lin_ref.copy()
            voxMosaic_lin_ref = voxMosaic_lin_ref[:3].reshape((3,) + ref_shape)
            voxMosaic_lin_ref_torch = voxMosaic_lin_ref.copy()
            voxMosaic_lin_ref_torch = torch.from_numpy(voxMosaic_lin_ref_torch[np.newaxis]).float().to('cpu')
            flow_torch = torch.from_numpy(flow[np.newaxis]).float().to('cpu')
            flow_resampled = self.interp_func(flow_torch, voxMosaic_lin_ref_torch)
            flow_resampled = flow_resampled[0].cpu().detach().numpy()

            voxMosaic_nl_ref[0] += flow_resampled[0].reshape((-1,))
            voxMosaic_nl_ref[1] += flow_resampled[1].reshape((-1,))
            voxMosaic_nl_ref[2] += flow_resampled[2].reshape((-1,))

            rasMosaic_nl_flo = np.dot(subject.vox2ras0, voxMosaic_nl_ref).astype('float32')

        else:
            rasMosaic_nl_flo = rasMosaic_lin_ref.copy()

        rasMosaic_flo = np.matmul(affine_matrix_flo, rasMosaic_nl_flo).astype('float32')
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

    def resample_ras_space(self, image, v2r, rasMosaic_ref, mode='bilinear'):

        ref_shape = rasMosaic_ref.shape[1:]
        voxMosaic_targ = np.matmul(np.linalg.inv(v2r), rasMosaic_ref.reshape(4, -1)).astype('float32')
        voxMosaic_targ = voxMosaic_targ[:3]

        voxMosaic_targ = voxMosaic_targ.reshape((3,) + ref_shape)
        voxMosaic_targ_torch = voxMosaic_targ.copy()
        voxMosaic_targ_torch = torch.from_numpy(voxMosaic_targ_torch[np.newaxis]).float().to('cpu')
        im = torch.from_numpy(image[np.newaxis]).float().to('cpu')
        im_resampled = self.interp_func(im, voxMosaic_targ_torch, mode=mode)

        return im_resampled[0]

    def register_timepoints(self, tp, timepoints, flow_dict, ref_shape):

        filepath = tp.init_path[self.filekey]
        proxyseg_ref = nib.load(filepath)
        seg_ref = np.array(proxyseg_ref.dataobj)
        seg_ref_HP = np.zeros_like(seg_ref)
        seg_ref_HP[seg_ref < 1000] = 1
        seg_ref_HP[seg_ref == 0] = 0
        seg_ref_AM = np.zeros_like(seg_ref)
        seg_ref_AM[seg_ref > 1000] = 1

        rasMosaic_ref = self.create_ref_space(proxyseg_ref)

        dice_results = []
        for it_tp_flo, tp_flo in enumerate(timepoints):

            if it_tp_flo == it_tp_ref:
                dice_dict = {k: 1.0 for k in EXTENDED_SUBFIELDS_LABEL_DICT.keys()}

            else:
                filepath = tp_flo.init_path[self.filekey]
                proxyseg_flo = nib.load(filepath)
                seg_flo = np.array(proxyseg_flo.dataobj)

                rasDefField = self.compute_1x1x1_inputs(tp, tp_flo, flow_dict[tp_flo.id], ref_shape)#flow_dict[tp_flo.id], ref_shape)

                ################################
                rasDefField_resampled = self.resample_ras_space(rasDefField, tp.vox2ras0, rasMosaic_ref)
                rasDefField_resampled = rasDefField_resampled.cpu().detach().numpy()

                ################################

                seg_resampled = self.resample_ras_space(seg_flo[np.newaxis], proxyseg_flo.affine, rasDefField_resampled
                                                        , mode='nearest')
                seg_resampled = seg_resampled.cpu().detach().numpy()
                seg_resampled = seg_resampled[0]

                dice = metrics.fast_dice(seg_ref, seg_resampled, labels=list(EXTENDED_SUBFIELDS_LABEL_DICT.values()))
                dice_dict = {k: np.round(dice[it_k], 2) for it_k, k in enumerate(EXTENDED_SUBFIELDS_LABEL_DICT.keys())}

                seg_resampled_HP = np.zeros_like(seg_resampled)
                seg_resampled_HP[seg_resampled<1000] = 1
                seg_resampled_HP[seg_resampled==0] = 0
                seg_resampled_AM = np.zeros_like(seg_resampled)
                seg_resampled_AM[seg_resampled>1000] = 1
                dice = metrics.dice(seg_ref_HP, seg_resampled_HP)
                dice_dict['TotalHP'] = np.round(dice, 2)
                dice = metrics.dice(seg_ref_AM, seg_resampled_AM)
                dice_dict['TotalAM'] = np.round(dice, 2)

            dice_dict['TID'] = tp_flo.id
            dice_results.append(dice_dict)

        return dice_results


print('\n\n\n\n\n')
print('# --------------------- #')
print('# Computing dice scores #')
print('# --------------------- #')
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

##############
# Processing #
##############
data_loader = DataLoader(sid_list=initial_subject_list)
parameter_dict = configFile.get_config_dict(data_loader.image_shape)
subject_list = data_loader.subject_list

regnet_model = models.RegNet(
    nb_unet_features=[parameter_dict['ENC_NF'], parameter_dict['DEC_NF']],
    inshape=parameter_dict['VOLUME_SHAPE'],
    int_steps=10,
    int_downsize=parameter_dict['UPSAMPLE_LEVELS'],
)
fieldnames = ['TID'] + list(EXTENDED_SUBFIELDS_LABEL_DICT.keys())
lh_fn = ComputeDiceSubfields(filekey='subfields.lh')
rh_fn = ComputeDiceSubfields(filekey='subfields.rh')

for it_subject, subject in enumerate(subject_list):
    print('   Subject: ' + str(subject.id))

    subject_shape = subject.image_shape
    timepoints = subject.timepoints
    flow_dir_regnet = subject.results_dirs.get_dir('nonlinear_registration_bidir' + str(parameter_dict['LOSS_REGISTRATION_SMOOTHNESS']['lambda']))
    flow_dir_st_regnet = subject.results_dirs.get_dir('nonlinear_st_bidir' + str(parameter_dict['LOSS_REGISTRATION_SMOOTHNESS']['lambda']))
    flow_dir_nr = subject.results_dirs.get_dir('nonlinear_registration_niftyreg')
    flow_dir_st_nr = subject.results_dirs.get_dir('nonlinear_st_niftyreg')

    print('     o Reading the SVF files')
    svf_list_regnet = []
    svf_list_nr = []
    for tp in timepoints:
        # Flow
        proxyflow = nib.load(join(flow_dir_st_regnet, tp.id + '.svf.nii.gz'))
        svf_list_regnet.append(proxyflow)

        # proxyflow = nib.load(join(flow_dir_st_nr, tp.id + '.svf.nii.gz'))
        # svf_list_nr.append(proxyflow)

    print('     o Computing the segmentation')
    for it_tp_ref, tp_ref in enumerate(timepoints):

        print('        - Timepoint ' + tp_ref.id, end='', flush=True)

        ##################################################################

        long_dir = subject.results_dirs.get_dir('longitudinal_segmentation_st_bidir') + '_LH'
        if not exists(long_dir): os.makedirs(long_dir)
        results_filepath_lh = join(long_dir, tp_ref.id + '_dice_st.txt')
        long_dir = subject.results_dirs.get_dir('longitudinal_segmentation_st_bidir') + '_RH'
        if not exists(long_dir): os.makedirs(long_dir)
        results_filepath_rh = join(long_dir, tp_ref.id + '_dice_st.txt')

        if not exists(results_filepath_lh) or not exists(results_filepath_rh):
            flow_dict = {}
            for it_tp_flo, tp_flo in enumerate(timepoints):
                if tp_flo.id == tp_ref.id:
                    flow_dict[tp_flo.id] = None

                else:
                    svf = np.asarray(svf_list_regnet[it_tp_flo].dataobj)-np.asarray(svf_list_regnet[it_tp_ref].dataobj)
                    flow = algorithm_utils.integrate_RegNet(svf, subject.image_shape_template, parameter_dict,
                                                            device='cpu', model=regnet_model)

                    flow_dict[tp_flo.id] = flow


            dice_results_lh = lh_fn.register_timepoints(tp_ref, timepoints, flow_dict, tp_ref.image_shape)
            dice_results_rh = rh_fn.register_timepoints(tp_ref, timepoints, flow_dict, tp_ref.image_shape)

            results_utils.write_volume_results(dice_results_lh, results_filepath_lh,
                                               fieldnames=fieldnames, attach_overwrite='w')
            results_utils.write_volume_results(dice_results_rh, results_filepath_rh,
                                               fieldnames=fieldnames, attach_overwrite='w')

        ##################################################################

        # long_dir = subject.results_dirs.get_dir('longitudinal_segmentation_st_niftyreg') + '_LH'
        # if not exists(long_dir): os.makedirs(long_dir)
        # results_filepath_lh = join(long_dir, tp_ref.id + '_dice_st.txt')
        # long_dir = subject.results_dirs.get_dir('longitudinal_segmentation_st_niftyreg') + '_RH'
        # if not exists(long_dir): os.makedirs(long_dir)
        # results_filepath_rh = join(long_dir, tp_ref.id + '_dice_st.txt')
        # if not exists(results_filepath_lh) or not exists(results_filepath_rh):
        #     flow_dict = {}
        #     for it_tp_flo, tp_flo in enumerate(timepoints):
        #         svf = np.asarray(svf_list_nr[it_tp_flo].dataobj) - np.asarray(svf_list_nr[it_tp_ref].dataobj)
        #         flow = algorithm_utils.integrate_NR(svf, subject.image_shape_template)
        #
        #         flow_dict[tp_flo.id] = flow
        #
        #     dice_results_lh = lh_fn.register_timepoints(tp_ref, timepoints, flow_dict, tp_ref.image_shape)
        #     dice_results_rh = rh_fn.register_timepoints(tp_ref, timepoints, flow_dict, tp_ref.image_shape)
        #
        #
        #     results_utils.write_volume_results(dice_results_lh, results_filepath_lh,
        #                                        fieldnames=fieldnames, attach_overwrite='w')
        #     results_utils.write_volume_results(dice_results_rh, results_filepath_rh,
        #                                        fieldnames=fieldnames, attach_overwrite='w')

        ##################################################################

        long_dir = subject.results_dirs.get_dir('longitudinal_segmentation_st_bidir') + '_LH'
        if not exists(long_dir): os.makedirs(long_dir)
        results_filepath_lh = join(long_dir, tp_ref.id + '_dice_direct.txt')
        long_dir = subject.results_dirs.get_dir('longitudinal_segmentation_st_bidir') + '_RH'
        if not exists(long_dir): os.makedirs(long_dir)
        results_filepath_rh = join(long_dir, tp_ref.id + '_dice_direct.txt')
        if not exists(results_filepath_lh) or not exists(results_filepath_rh):
            flow_dict = {}
            for it_tp_flo, tp_flo in enumerate(timepoints):

                if tp_flo.id == tp_ref.id:
                    flow_dict[tp_flo.id] = None
                    continue

                if exists(join(flow_dir_regnet, tp_ref.id + '_to_' + tp_flo.id + '.svf.nii.gz')):
                    proxy = nib.load(join(flow_dir_regnet, tp_ref.id + '_to_' + tp_flo.id + '.svf.nii.gz'))
                    svf = np.asarray(proxy.dataobj)
                else:
                    proxy = nib.load(join(flow_dir_regnet, tp_flo.id + '_to_' + tp_ref.id + '.svf.nii.gz'))
                    svf = -np.asarray(proxy.dataobj)

                flow = algorithm_utils.integrate_RegNet(svf, subject.image_shape_template, parameter_dict,
                                                        device='cpu', model=regnet_model)
                flow_dict[tp_flo.id] = flow

            dice_results_lh = lh_fn.register_timepoints(tp_ref, timepoints, flow_dict, tp_ref.image_shape)
            dice_results_rh = rh_fn.register_timepoints(tp_ref, timepoints, flow_dict, tp_ref.image_shape)

            results_utils.write_volume_results(dice_results_lh, results_filepath_lh,
                                               fieldnames=fieldnames, attach_overwrite='w')
            results_utils.write_volume_results(dice_results_rh, results_filepath_rh,
                                               fieldnames=fieldnames, attach_overwrite='w')

        ##################################################################

        # long_dir = subject.results_dirs.get_dir('longitudinal_segmentation_st_niftyreg') + '_LH'
        # if not exists(long_dir): os.makedirs(long_dir)
        # results_filepath_lh = join(long_dir, tp_ref.id + '_dice_direct.txt')
        # long_dir = subject.results_dirs.get_dir('longitudinal_segmentation_st_niftyreg') + '_RH'
        # if not exists(long_dir): os.makedirs(long_dir)
        # results_filepath_rh = join(long_dir, tp_ref.id + '_dice_direct.txt')
        # if not exists(results_filepath_lh) or not exists(results_filepath_rh):
        #     flow_dict = {}
        #     for it_tp_flo, tp_flo in enumerate(timepoints):
        #         if exists(join(flow_dir_nr, tp_ref.id + '_to_' + tp_flo.id + '.svf.nii.gz')):
        #             proxy = nib.load(join(flow_dir_nr, tp_ref.id + '_to_' + tp_flo.id + '.svf.nii.gz'))
        #             svf = np.asarray(proxy.dataobj)
        #         else:
        #             proxy = nib.load(join(flow_dir_nr, tp_flo.id + '_to_' + tp_ref.id + '.svf.nii.gz'))
        #             svf = -np.asarray(proxy.dataobj)
        #
        #         flow = algorithm_utils.integrate_NR(svf, subject.image_shape_template)
        #         flow_dict[tp_flo.id] = flow
        #
        #     dice_results_lh = lh_fn.register_timepoints(tp_ref, timepoints, flow_dict, tp_ref.image_shape)
        #     dice_results_rh = rh_fn.register_timepoints(tp_ref, timepoints, flow_dict, tp_ref.image_shape)
        #
        #     results_utils.write_volume_results(dice_results_lh, results_filepath_lh,
        #                                        fieldnames=fieldnames, attach_overwrite='w')
        #     results_utils.write_volume_results(dice_results_rh, results_filepath_rh,
        #                                        fieldnames=fieldnames, attach_overwrite='w')

    print('\n')
    print('DONE')

        ##################################################################

        # long_dir = subject.results_dirs.get_dir('longitudinal_segmentation_st_bidir') + hemi
        # results_filepath_regnet = join(long_dir, tp_ref.id + '_dice_direct.txt')
        # long_dir = subject.results_dirs.get_dir('longitudinal_segmentation_st_niftyreg') + hemi
        # results_filepath_nr = join(long_dir, tp_ref.id + '_dice_direct.txt')
        # if not exists(results_filepath_regnet) or not exists(results_filepath_nr):
        #     seg_ref = np.array(proxyseg_ref.dataobj)
        #     dice_results_regnet = []
        #     dice_results_nr = []
        #     for it_tp_flo, tp_flo in enumerate(timepoints):
        #         if it_tp_flo == it_tp_ref:
        #             dice_dict_regnet = {k: 1.0 for k in EXTENDED_SUBFIELDS_LABEL_DICT.keys()}
        #             dice_dict_nr = {k: 1.0 for k in EXTENDED_SUBFIELDS_LABEL_DICT.keys()}
        #
        #         else:
        #             filepath = tp_flo.init_path['subfields.' + hemi.lower()]
        #             proxyseg_flo = nib.load(filepath)
        #             seg_flo = np.array(proxyseg_flo.dataobj)
        #
        #             if exists(join(flow_dir_regnet, tp_ref.id + '_to_' + tp_flo.id + '.svf.nii.gz')):
        #                 proxy = nib.load(join(flow_dir_regnet, tp_ref.id + '_to_' + tp_flo.id + '.svf.nii.gz'))
        #                 svf = np.asarray(proxy.dataobj)
        #             else:
        #                 proxy = nib.load(join(flow_dir_regnet, tp_flo.id + '_to_' + tp_ref.id + '.svf.nii.gz'))
        #                 svf = -np.asarray(proxy.dataobj)
        #
        #             flow = algorithm_utils.integrate_RegNet(svf, subject.image_shape_template, parameter_dict,
        #                                                     device='cpu', model=regnet_model)
        #
        #             seg_resampled = register_timepoints_regnet(flow, tp_ref.image_shape, seg_flo, tp_ref.vox2ras0,
        #                                                        tp_flo.vox2ras0, subject.vox2ras0, mode='nearest')
        #
        #             dice = metrics.fast_dice(seg_ref, seg_resampled, labels=list(EXTENDED_SUBFIELDS_LABEL_DICT.values()))
        #             dice_dict_regnet = {k: dice[it_k] for it_k, k in enumerate(EXTENDED_SUBFIELDS_LABEL_DICT.keys())}
        #
        #             ##################################################################
        #
        #             if exists(join(flow_dir_regnet, tp_ref.id + '_to_' + tp_flo.id + '.svf.nii.gz')):
        #                 proxy = nib.load(join(flow_dir_regnet, tp_ref.id + '_to_' + tp_flo.id + '.svf.nii.gz'))
        #                 svf = np.asarray(proxy.dataobj)
        #             else:
        #                 proxy = nib.load(join(flow_dir_regnet, tp_flo.id + '_to_' + tp_ref.id + '.svf.nii.gz'))
        #                 svf = -np.asarray(proxy.dataobj)
        #
        #             flow = algorithm_utils.integrate_NR(svf, subject.image_shape_template)
        #
        #             seg_resampled = register_timepoints_nr(flow, tp_ref.image_shape, seg_flo, tp_ref.vox2ras0,
        #                                                        tp_flo.vox2ras0, subject.vox2ras0, mode='nearest')
        #
        #             dice = metrics.fast_dice(seg_ref, seg_resampled, labels=list(EXTENDED_SUBFIELDS_LABEL_DICT.values()))
        #             dice_dict_nr = {k: dice[it_k] for it_k, k in enumerate(EXTENDED_SUBFIELDS_LABEL_DICT.keys())}
        #
        #             del flow, svf, seg_flo, seg_resampled
        #
        #         dice_dict_regnet['TID'] = tp_flo.id
        #         dice_dict_nr['TID'] = tp_flo.id
        #         dice_results_regnet.append(dice_dict_regnet)
        #         dice_results_nr.append(dice_dict_nr)
        #
        #     results_utils.write_volume_results(dice_results_regnet, results_filepath_regnet,
        #                                        fieldnames=fieldnames, attach_overwrite='w')
        #     results_utils.write_volume_results(dice_results_nr, results_filepath_nr,
        #                                        fieldnames=fieldnames, attach_overwrite='w')



    # print('DONE')

#
# labels_to_write = ['Thalamus', 'Lateral-Ventricle', 'Hippocampus', 'Amygdala', 'Caudate', 'Pallidum', 'Putamen']
#
# for label in labels_to_write:
#     fieldnames = ['Right-' + label, 'Left-' + label]
#     dataframe = {'Dice': [], 'Labels': [], 'Method': []}
#     for it_subject, subject in enumerate(subject_list):
#         print('   Subject: ' + str(subject.id))
#         timepoints = subject.timepoints
#         long_dir = subject.results_dirs.get_dir('longitudinal_segmentation_'+reg_algorithm)
#
#         print('     o Computing the segmentation')
#         for it_tp, tp in enumerate(timepoints):
#             filepath_st = join(long_dir, tp.id + '_dice_st.txt')
#             results_st = read_volume_results(filepath_st, fieldnames=fieldnames)
#             filepath_direct = join(long_dir, tp.id + '_dice_direct.txt')
#             results_r = read_volume_results(filepath_direct, fieldnames=fieldnames)
#             for k, v in results_r.items():
#                 dataframe['Dice'].append(np.mean(v))
#                 dataframe['Labels'].append(k)
#                 dataframe['Method'].append('Reg')
#
#             for k, v in results_st.items():
#                 dataframe['Dice'].append(np.mean(v))
#                 dataframe['Labels'].append(k)
#                 dataframe['Method'].append('ST')
#
#
#     data_frame = pd.DataFrame(dataframe)
#     plt.figure()#figsize=(8.6, 7.0)
#     x = sns.boxplot(x="Labels", y="Dice", hue="Method", data=data_frame, linewidth=2.5, palette=sns.color_palette("bright"))
#     x.grid()
#     x.set_axisbelow(True)
#     x.locator_params(axis='y', tight=True, nbins=6)
#     plt.axes(x)
#     handles, labels = x.get_legend_handles_labels()
#     x.legend(handles, labels, loc=2, ncol=2, prop=prop_legend)#, bbox_to_anchor=(0.5, 1.05))
#     x.set_title('Dice scores',fontproperties=prop, fontsize=20)# y=1.0, pad=42, )
#
#
#     plt.xlabel('ROI', fontproperties=prop_bold, fontsize=18)
#     plt.ylabel('Dice scores', fontproperties=prop_bold, fontsize=18)
#     plt.yticks(rotation=90, fontproperties=prop, fontsize=16)
#     plt.xticks(fontproperties=prop, fontsize=16)
#
#     plt.savefig(join(REGISTRATION_DIR, 'Results', label + '.png'))
#     plt.close()


