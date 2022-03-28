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
from src.utils import algorithm_utils, deformation_utils as def_utils, io_utils, image_utils
from database import read_demo_info
from src import models, layers


UNIQUE_LABELS = np.array([  0,   2,   3,   4,   5,   7,   8,  10,  11,  12,  13,  14,  15,
        16,  17,  18,  24,  26,  28,  30,  31,  41,  42,  43,  44,  46,
        47,  49,  50,  51,  52,  53,  54,  58,  60,  62,  63,  77,  80,
        85, 251, 252, 253, 254, 255], dtype=np.uint8)

def compute_p_label(timepoints, image_list, svf_list, age_list, tp, subject, parameter_dict):

    interp_func = layers.SpatialInterpolation(padding_mode='zeros').to('cuda:0')

    results_dir_sbj = subject.results_dirs.get_dir('linear_st')
    p_data = {t_var: {sp_var: np.zeros(tp.image_shape + (len(timepoints),), dtype='float32') for sp_var in spatial_variance} for t_var in temp_variance}
    p_label = {t_var: {sp_var: np.zeros(tp.image_shape + (len(UNIQUE_LABELS),), dtype='float32') for sp_var in spatial_variance} for t_var in temp_variance}
    ref_shape = image_list[tp.id].shape

    ii = np.arange(0, ref_shape[0])
    jj = np.arange(0, ref_shape[1])
    kk = np.arange(0, ref_shape[2])

    II, JJ, KK = np.meshgrid(ii, jj, kk, indexing='ij')

    voxMosaic_orig = np.concatenate((II.reshape(-1, 1), JJ.reshape(-1, 1), KK.reshape(-1, 1), np.ones((int(np.prod(ref_shape[:3])), 1))), axis=1).T
    rasMosaic_orig = np.dot(tp.vox2ras0, voxMosaic_orig)
    voxMosaic_ref = np.dot(np.linalg.inv(subject.vox2ras0), rasMosaic_orig)

    del II, JJ, KK, voxMosaic_orig, ii, jj, kk

    for it_tp_2, tp_2 in enumerate(timepoints[1:]):
        fileparts = tp_2.id.split('_')
        filepath = join(FS_DIR, fileparts[0] + '_' + fileparts[1] + '_' + str(int(fileparts[4])) + '_' + fileparts[5] + '_' + fileparts[6] + '.cross.aseg.mgz')
        proxyseg = nib.load(filepath)
        seg = np.asarray(proxyseg.dataobj)
        seg_onehot = image_utils.one_hot_encoding(seg, categories=UNIQUE_LABELS).astype('float32')

        cog = tp_2.get_cog()
        affine_matrix = read_affine_matrix(join(results_dir_sbj, tp_2.id + '.aff'), full=True)
        affine_matrix[:3, 3] += cog
        vox2ras0_fs = np.dot(np.linalg.inv(affine_matrix), proxyseg.affine)

        if tp_2.id == tp.id:
            for t_var in temp_variance:
                for s_var in spatial_variance:
                    p_data[t_var][s_var][..., it_tp_2] = 1

            rasMosaic_targ = rasMosaic_orig.copy()
            voxMosaic_targ = np.matmul(np.linalg.inv(vox2ras0_fs), rasMosaic_targ)#np.dot(np.linalg.inv(vox2ras0_fs), np.dotnp.dot(tp.vox2ras0, voxMosaic_targ))
            voxMosaic_targ = voxMosaic_targ[:3]
            im_resampled = image_list[tp_2.id]

        else:
            svf = np.asarray(svf_list[tp_2.id].dataobj) - np.asarray(svf_list[tp.id].dataobj)
            flow = algorithm_utils.integrate_RegNet(svf, subject.image_shape, parameter_dict)
            flow_resampled = def_utils.interpolate3DChannel(np.transpose(flow, (1,2,3,0)), voxMosaic_ref[:3].T)#def_utils.interpolate3D([flow[i] for i in range(3)], voxMosaic_2[:3].T)

            del svf, flow

            # Image
            voxMosaic_ref_nl = voxMosaic_ref.copy()
            voxMosaic_ref_nl[0] += flow_resampled[..., 0]
            voxMosaic_ref_nl[1] += flow_resampled[..., 1]
            voxMosaic_ref_nl[2] += flow_resampled[..., 2]
            rasMosaic_ref_nl = np.dot(subject.vox2ras0, voxMosaic_ref_nl)

            voxMosaic_targ = np.matmul(np.linalg.inv(tp_2.vox2ras0), rasMosaic_ref_nl)
            voxMosaic_targ = voxMosaic_targ[:3]

            del voxMosaic_ref_nl, flow_resampled

            voxMosaic_targ = voxMosaic_targ.reshape((1,3) + ref_shape)
            voxMosaic_targ = torch.from_numpy(voxMosaic_targ).float().to('cuda:0')

            im = torch.from_numpy(image_list[tp_2.id][np.newaxis, np.newaxis]).float().to('cuda:0')
            im_resampled = interp_func(im, voxMosaic_targ)
            im_resampled = im_resampled[0, 0].cpu().detach().numpy()

            del voxMosaic_targ

            mean_im_2 = (im_resampled - image_list[tp.id]) ** 2
            mean_age_2 = (age_list[tp_2.id] - age_list[tp.id]) ** 2
            for t_var in temp_variance:
                for s_var in spatial_variance:
                    p_data[t_var][s_var][..., it_tp_2] = np.exp(-0.5 / s_var * mean_im_2) \
                                                         * np.exp(-0.5 / t_var * mean_age_2)

            # del im_resampled

            voxMosaic_targ = np.matmul(np.linalg.inv(proxyseg.affine), np.matmul(affine_matrix, rasMosaic_ref_nl))
            voxMosaic_targ = voxMosaic_targ[:3]

            del rasMosaic_ref_nl

        voxMosaic_targ = voxMosaic_targ.reshape((1, 3) + ref_shape)
        voxMosaic_targ = torch.from_numpy(voxMosaic_targ).float().to('cuda:0')

        seg_onehot = torch.from_numpy(seg_onehot[np.newaxis]).float().to('cuda:0')
        seg_resampled = interp_func(seg_onehot, voxMosaic_targ)

        del voxMosaic_targ

        seg_resampled = np.transpose(seg_resampled[0].cpu().detach().numpy(), (1, 2, 3, 0))

        for t_var in temp_variance:
            for s_var in spatial_variance:
                p_label[t_var][s_var] += p_data[t_var][s_var][..., it_tp_2, np.newaxis] * seg_resampled

        # if DEBUG:
        #     import pdb
        #     save_dir = subject.results_dirs.get_dir('longitudinal_segmentation_st_freesurfer')
        #     t_var = 1
        #     s_var = 9
        #     proxy = nib.load(tp.init_path['resample'])
        #     img = nib.Nifti1Image(im_resampled, proxy.affine)
        #     nib.save(img, join(save_dir, str(t_var) + '_' + str(s_var), tp.id + '_' + tp_2.id + '.image.nii.gz'))
        #     img = nib.Nifti1Image(np.argmax(seg_resampled, axis=-1).astype('uint16'), proxy.affine)
        #     nib.save(img, join(save_dir, str(t_var) + '_' + str(s_var), tp.id + '_' + tp_2.id + '.seg.nii.gz'))
        #     pdb.set_trace()

        del seg_resampled

    for t_var in temp_variance:
        for s_var in spatial_variance:
            p_label[t_var][s_var] = p_label[t_var][s_var] / np.sum(p_data[t_var][s_var], axis=-1, keepdims=True)

    # for it_tp_2, tp_2 in enumerate(timepoints):
    #     fileparts = tp_2.id.split('_')
    #     filepath = join(FS_DIR, fileparts[0] + '_' + fileparts[1] + '_' + str(int(fileparts[4])) + '_' + fileparts[5] + '_' + fileparts[6] + '.cross.aseg.mgz')
    #     proxyseg = nib.load(filepath)
    #     seg = np.asarray(proxyseg.dataobj)
    #     seg_onehot = image_utils.one_hot_encoding(seg, categories=UNIQUE_LABELS)
    #
    #     cog = tp_2.get_cog()
    #     affine_matrix = read_affine_matrix(join(results_dir_sbj, tp_2.id + '.aff'), full=True)
    #     affine_matrix[:3, 3] += cog
    #     vox2ras0_fs = np.matmul(np.linalg.inv(affine_matrix), proxyseg.affine)
    #
    #     if tp_2.id == tp.id:
    #         voxMosaic_targ = voxMosaic_orig.copy()
    #         voxMosaic_targ = np.dot(np.linalg.inv(vox2ras0_fs), np.dot(tp.vox2ras0, voxMosaic_targ))
    #         voxMosaic_targ = voxMosaic_targ[:3]
    #         voxMosaic_targ = voxMosaic_targ.reshape((1,3) + ref_shape)
    #         voxMosaic_targ = torch.from_numpy(voxMosaic_targ).float().to('cuda:0')
    #
    #     else:
    #         svf = np.asarray(svf_list[tp_2.id].dataobj) - np.asarray(svf_list[tp.id].dataobj)
    #         flow = algorithm_utils.integrate_RegNet(svf, subject.image_shape, parameter_dict)
    #         flow_resampled = def_utils.interpolate3DChannel(np.transpose(flow, (1,2,3,0)), voxMosaic_ref[:3].T)#def_utils.interpolate3D([flow[i] for i in range(3)], voxMosaic_2[:3].T)
    #
    #         del flow, svf
    #
    #         voxMosaic_ref_nl = voxMosaic_ref.copy()
    #         voxMosaic_ref_nl[0] += flow_resampled[..., 0]
    #         voxMosaic_ref_nl[1] += flow_resampled[..., 1]
    #         voxMosaic_ref_nl[2] += flow_resampled[..., 2]
    #         del flow_resampled
    #
    #         rasMosaic_ref_nl = np.dot(subject.vox2ras0, voxMosaic_ref_nl)
    #         del voxMosaic_ref_nl
    #
    #         voxMosaic_targ = np.matmul(np.linalg.inv(vox2ras0_fs), rasMosaic_ref_nl)
    #         del rasMosaic_ref_nl
    #         voxMosaic_targ = voxMosaic_targ[:3]
    #         voxMosaic_targ = voxMosaic_targ.reshape((1,3) + ref_shape)
    #         voxMosaic_targ = torch.from_numpy(voxMosaic_targ).float().to('cuda:0')
    #
    #
    #     seg_onehot = torch.from_numpy(seg_onehot[np.newaxis]).float().to('cuda:0')
    #     seg_resampled = interp_func(seg_onehot, voxMosaic_targ)
    #
    #     del voxMosaic_targ
    #
    #     seg_resampled = np.transpose(seg_resampled[0].cpu().detach().numpy(), (1,2,3,0))
    #
    #     for t_var in temp_variance:
    #         for s_var in spatial_variance:
    #             p_label[t_var][s_var] += p_data[t_var][s_var][..., it_tp_2, np.newaxis] * seg_resampled
    #
    #     del seg_resampled

    return p_label

def write_volume_results(volume_dict, filepath, fieldnames=None):
    if fieldnames is None:
        fieldnames = ['TID'] + list(LABEL_DICT.keys())

    with open(filepath, 'w') as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()
        csvwriter.writerows(volume_dict)


def get_vols_post(post):
    n_labels = post.shape[-1]
    vols = {}
    for l in range(n_labels):
        mask_l = post[..., l]
        mask_l[post[..., l] < 0.05] = 0
        vols[l] = np.sum(mask_l)

    return vols


def label_fusion(subject):
    print('Subject: ' + str(subject.id))
    timepoints = subject.timepoints
    flow_dir = subject.results_dirs.get_dir('nonlinear_st_' + reg_algorithm)
    results_dir_sbj = subject.results_dirs.get_dir('longitudinal_segmentation_st_freesurfer')
    # if exists(join(results_dir_sbj, str(temp_variance[-1]) + '_' + str(spatial_variance[-1]), 'vols_st.txt')):
    #     spatial_variance = [1 ** 2, 3 ** 2, 5 ** 2, 7 ** 2]  # in grayscale
    #     temp_variance = [2]
    #     print('Subject: ' + str(subject.id) + '. DONE')
    #     return None


    timepoints_to_run = timepoints# timepoints_to_run = list(filter(lambda x: not exists(join(results_dir_sbj, str(temp_variance[-1]) + '_' + str(spatial_variance[-1]) , x.id + '.nii.gz')), timepoints))
    if not timepoints_to_run:
        print('Subject: ' + str(subject.id) + '. DONE')
        return

    print('  o Reading the input files')
    image_list = {}
    age_list = {}
    svf_list = {}
    for tp in timepoints:
        # Age
        age_list[tp.id] = float(demo_dict[subject.id][tp.id]['AGE'])

        # Data
        seg = tp.load_seg()
        image = tp.load_data()

        # Normalize image
        wm_mask = (seg == 2) | (seg == 41)
        m = np.mean(image[wm_mask])
        image = 110 * image / m
        image_list[tp.id] = image

        del image, seg, wm_mask

        # Flow
        proxyflow = nib.load(join(flow_dir, tp.id + '.svf.nii.gz'))
        svf_list[tp.id] = proxyflow

    print('  o Computing the segmentation')
    st_vols = {t_var: {sp_var: [] for sp_var in spatial_variance} for t_var in temp_variance}
    for tp in timepoints_to_run:
        print('        - Timepoint ' + tp.id, end=':', flush=True)

        t_0 = time.time()
        p_label_dict = compute_p_label(timepoints, image_list, svf_list, age_list,  tp, subject, parameter_dict)
        t_1 = time.time()
        print(str(t_1-t_0) + ' seconds.')

        proxy = nib.load(tp.init_path['resample'])
        for t_var in temp_variance:
            for s_var in spatial_variance:
                # p = nib.load(join(results_dir_sbj, str(t_var) + '_' + str(s_var), tp.id + '.post.nii.gz'))
                # img = nib.Nifti1Image(np.array(p.dataobj), proxy.affine)
                # nib.save(img, join(results_dir_sbj, str(t_var) + '_' + str(s_var), tp.id + '.post.nii.gz'))
                #
                # p = nib.load(join(results_dir_sbj, str(t_var) + '_' + str(s_var), tp.id + '.nii.gz'))
                # img = nib.Nifti1Image(np.array(p.dataobj), proxy.affine)
                # nib.save(img, join(results_dir_sbj, str(t_var) + '_' + str(s_var), tp.id + '.nii.gz'))

                p_label = p_label_dict[t_var][s_var]
                mask = np.sum(p_label, axis=-1) > 0.5
                fake_vol = np.argmax(p_label, axis=-1).astype('uint16')
                true_vol = np.zeros_like(fake_vol)
                for it_ul, ul in enumerate(UNIQUE_LABELS): true_vol[fake_vol == it_ul] = ul
                true_vol = true_vol * mask

                img = nib.Nifti1Image(p_label, proxy.affine)
                nib.save(img, join(results_dir_sbj, str(t_var) + '_' + str(s_var), tp.id + '.post.nii.gz'))

                img = nib.Nifti1Image(true_vol, proxy.affine)
                nib.save(img, join(results_dir_sbj, str(t_var) + '_' + str(s_var), tp.id + '.nii.gz'))


                del p_label, fake_vol, true_vol, mask

        del p_label_dict

    print('Subject: ' + str(subject.id) + '. DONE')


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
subject_list = list(filter(lambda x:  '_222' not in x.id and '_231' not in x.id and '_199' not in x.id and '_218' not in x.id, subject_list))

print('[LONGITUDINAL SEGMENTATION] Start processing.')
spatial_variance = [3**2]#[1**2, 3**2, 5**2, 7**2] # in grayscale
temp_variance = [1]#[1/4, 1/2, 1, 2] #in years^2

if num_cores == 1:
    for subject in subject_list:
        label_fusion(subject)
        # try:
        #     label_fusion(subject)
        # except:
        #     continue

elif len(subject_list) == 1:
    label_fusion(subject_list[0])

else:
    results = Parallel(n_jobs=num_cores)(delayed(label_fusion)(subject) for subject in subject_list)








