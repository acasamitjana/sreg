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


# labels_to_write = ['Thalamus', 'Lateral-Ventricle', 'Hippocampus', 'Amygdala', 'Caudate', 'Pallidum', 'Putamen', 'Accumbens', 'Inf-Lat-Ventricle']
# keep_labels = ['Right-' + l for l in labels_to_write] + ['Left-' + l for l in labels_to_write]
# UNIQUE_LABELS = np.asarray([0] + [lab for labstr, lab in LABEL_DICT.items() if labstr in keep_labels], dtype=np.uint8)
# UNIQUE_LABELS = np.asarray( [lab for labstr, lab in LABEL_DICT.items()], dtype=np.uint8)
UNIQUE_LABELS = np.array([  0,   2,   3,   4,   5,   7,   8,  10,  11,  12,  13,  14,  15,
        16,  17,  18,  24,  26,  28,  30,  31,  41,  42,  43,  44,  46,
        47,  49,  50,  51,  52,  53,  54,  58,  60,  62,  63,  77,  80,
        85, 251, 252, 253, 254, 255], dtype=np.uint8)

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
    results_dir_sbj = join(subject.results_dirs.get_dir('longitudinal_segmentation_st_freesurfer'), 'template')
    if not exists(results_dir_sbj): os.makedirs(results_dir_sbj)
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
        # # Age
        # age_list[tp.id] = float(demo_dict[subject.id][tp.id]['AGE'])
        #
        # # Data
        # seg = tp.load_seg()
        # image = tp.load_data()
        #
        # # Normalize image
        # wm_mask = (seg == 2) | (seg == 41)
        # m = np.mean(image[wm_mask])
        # image = 110 * image / m
        # image_list[tp.id] = image
        #
        # del image, seg, wm_mask

        # Flow
        proxyflow = nib.load(join(flow_dir, tp.id + '.svf.nii.gz'))
        svf_list[tp.id] = proxyflow

    # Model
    int_steps = parameter_dict['INT_STEPS'] if parameter_dict['FIELD_TYPE'] == 'velocity' else 0
    if int_steps == 0: assert parameter_dict['UPSAMPLE_LEVELS'] == 1
    model = models.RegNet(
        nb_unet_features=[parameter_dict['ENC_NF'], parameter_dict['DEC_NF']],
        inshape=parameter_dict['VOLUME_SHAPE'],
        int_steps=int_steps,
        int_downsize=parameter_dict['UPSAMPLE_LEVELS'],
    )
    model.to("cuda:0")


    print('  o Computing template segmentation')
    if not exists(join(results_dir_sbj, 'template.1nii.gz')):
        t_0 = time.time()
        p_label = compute_p_label_template(timepoints, image_list, svf_list, age_list, subject, parameter_dict, model)
        t_1 = time.time()
        print(str(t_1 - t_0) + ' seconds.')

        mask = np.sum(p_label, axis=-1) > 0.5
        fake_vol = np.argmax(p_label, axis=-1).astype('uint16')
        true_vol = np.zeros_like(fake_vol)
        for it_ul, ul in enumerate(UNIQUE_LABELS): true_vol[fake_vol == it_ul] = ul
        true_vol = true_vol * mask

        img = nib.Nifti1Image(p_label, subject.vox2ras0)
        nib.save(img, join(results_dir_sbj, 'template.post.nii.gz'))

        img = nib.Nifti1Image(true_vol,subject.vox2ras0)
        nib.save(img, join(results_dir_sbj, 'template.nii.gz'))
    else:
        proxy_label = nib.load(join(results_dir_sbj, 'template.post.nii.gz'))
        p_label = np.asarray(proxy_label.dataobj)

    p_label = np.transpose(p_label, axes=(3, 0, 1, 2))

    print('  o Computing timepoints segmentation')
    st_vols = []
    for tp in timepoints_to_run:
        print('        - Timepoint ' + tp.id, flush=True)

        p_label_posterior = compute_p_label_timepoint(p_label, svf_list, parameter_dict, tp, model)

        proxy = nib.load(tp.init_path['resample'])
        mask = np.sum(p_label_posterior, axis=-1) > 0.5
        fake_vol = np.argmax(p_label_posterior, axis=-1).astype('uint16')
        true_vol = np.zeros_like(fake_vol)
        for it_ul, ul in enumerate(UNIQUE_LABELS): true_vol[fake_vol == it_ul] = ul
        true_vol = true_vol * mask

        img = nib.Nifti1Image(p_label_posterior, proxy.affine)
        nib.save(img, join(results_dir_sbj, tp.id + '.post.nii.gz'))

        img = nib.Nifti1Image(true_vol, proxy.affine)
        nib.save(img, join(results_dir_sbj, tp.id + '.nii.gz'))

        vols = get_vols_post(p_label_posterior)
        st_vols_dict = {k: vols[[it_ul for it_ul, ul in enumerate(UNIQUE_LABELS) if ul == val][0]] for k, val in LABEL_DICT.items() if val in UNIQUE_LABELS}
        st_vols_dict['TID'] = tp.id
        st_vols.append(st_vols_dict)

        del p_label_posterior, fake_vol, true_vol, mask

    del p_label
    fieldnames = ['TID'] + list(LABEL_DICT.keys())
    write_volume_results(st_vols, join(results_dir_sbj, 'vols_st.txt'), fieldnames=fieldnames)

    print('Subject: ' + str(subject.id) + '. DONE')


def compute_p_label_template(timepoints, image_list, svf_list, age_list, subject, parameter_dict, regnet_model):

    interp_func = layers.SpatialInterpolation(padding_mode='zeros').cuda()
    affine_dir = subject.results_dirs.get_dir('linear_st')
    subject_shape = subject.image_shape

    # proxy = nib.load(subject.nonlinear_template_orig)
    # template = np.array(proxy.dataobj)
    # p_data = {t_var: {sp_var: np.zeros(subject_shape + (len(timepoints),), dtype='float32') for sp_var in spatial_variance} for t_var in temp_variance}
    # p_label = {t_var: {sp_var: np.zeros(subject_shape + (len(UNIQUE_LABELS),), dtype='float32') for sp_var in spatial_variance} for t_var in temp_variance}
    p_label = np.zeros(subject_shape + (len(UNIQUE_LABELS),), dtype='float32')

    ii = np.arange(0, subject_shape[0], dtype='int32')
    jj = np.arange(0, subject_shape[1], dtype='int32')
    kk = np.arange(0, subject_shape[2], dtype='int32')

    II, JJ, KK = np.meshgrid(ii, jj, kk, indexing='ij')

    del ii, jj, kk

    voxMosaic_orig = np.concatenate((II.reshape(-1, 1), JJ.reshape(-1, 1), KK.reshape(-1, 1), np.ones((int(np.prod(subject_shape[:3])), 1), dtype='int32')), axis=1).T

    del II, JJ, KK,

    for it_tp, tp in enumerate(timepoints):
        fileparts = tp.id.split('_')
        filepath = join(FS_DIR, fileparts[0] + '_' + fileparts[1] + '_' + str(int(fileparts[4])) + '_' + fileparts[5] + '_' + fileparts[6] + '.cross.aseg.mgz')
        proxyseg = nib.load(filepath)
        cog = tp.get_cog()
        affine_matrix = read_affine_matrix(join(affine_dir, tp.id + '.aff'), full=True)
        affine_matrix[:3, 3] += cog
        vox2ras0_fs = np.dot(np.linalg.inv(affine_matrix), proxyseg.affine).astype('float32')

        svf = np.asarray(svf_list[tp.id].dataobj).astype('float32')
        flow = algorithm_utils.integrate_RegNet(svf, subject.image_shape, parameter_dict, device='cuda:0', model=regnet_model)
        flow_resampled = def_utils.interpolate3DChannel(np.transpose(flow, (1,2,3,0)), voxMosaic_orig[:3].T)#def_utils.interpolate3D([flow[i] for i in range(3)], voxMosaic_2[:3].T)

        del svf, flow

        # Image
        voxMosaic_ref_nl = voxMosaic_orig.copy().astype('float32')
        voxMosaic_ref_nl[0] += flow_resampled[..., 0]
        voxMosaic_ref_nl[1] += flow_resampled[..., 1]
        voxMosaic_ref_nl[2] += flow_resampled[..., 2]
        rasMosaic_ref_nl = np.dot(subject.vox2ras0, voxMosaic_ref_nl).astype('float32')

        del voxMosaic_ref_nl

        voxMosaic_targ = np.matmul(np.linalg.inv(vox2ras0_fs), rasMosaic_ref_nl).astype('float32')
        voxMosaic_targ = voxMosaic_targ[:3]

        del rasMosaic_ref_nl

        voxMosaic_targ = voxMosaic_targ.reshape((1, 3) + subject_shape)
        voxMosaic_targ = torch.from_numpy(voxMosaic_targ).float().to('cuda:0')

        seg = np.asarray(proxyseg.dataobj)
        seg_onehot = image_utils.one_hot_encoding(seg, categories=UNIQUE_LABELS).astype('float32')
        seg_onehot = torch.from_numpy(seg_onehot[np.newaxis]).float().to('cuda:0')
        seg_resampled = interp_func(seg_onehot, voxMosaic_targ)

        del voxMosaic_targ

        seg_resampled = np.transpose(seg_resampled[0].cpu().detach().numpy(), (1, 2, 3, 0))


        p_label += seg_resampled / len(timepoints)
        # if DEBUG:
        #     img = nib.Nifti1Image(seg_resampled, subject.vox2ras0)
        #     nib.save(img, join(join(subject.results_dirs.get_dir('longitudinal_segmentation_st_freesurfer'), 'template'), 'template.nii.gz'))

        # for t_var in temp_variance:
        #     for s_var in spatial_variance:
        #         p_label[t_var][s_var] += p_data[t_var][s_var][..., it_tp, np.newaxis] * seg_resampled

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

    # for t_var in temp_variance:
    #     for s_var in spatial_variance:
    #         p_label[t_var][s_var] = p_label[t_var][s_var] / np.sum(p_data[t_var][s_var], axis=-1, keepdims=True)


    return p_label


def compute_p_label_timepoint(p_label_template, svf_list, parameter_dict, tp, regnet_model):

    interp_func = layers.SpatialInterpolation(padding_mode='zeros').to('cuda:0')

    ii = np.arange(0, tp.image_shape[0], dtype='int32')
    jj = np.arange(0, tp.image_shape[1], dtype='int32')
    kk = np.arange(0, tp.image_shape[2], dtype='int32')

    II, JJ, KK = np.meshgrid(ii, jj, kk, indexing='ij')
    voxMosaic_orig = np.concatenate((II.reshape(-1, 1), JJ.reshape(-1, 1), KK.reshape(-1, 1), np.ones((int(np.prod(tp.image_shape[:3])), 1), dtype='int32')), axis=1).T
    rasMosaic_orig = np.dot(tp.vox2ras0, voxMosaic_orig).astype('float32')
    voxMosaic_ref = np.dot(np.linalg.inv(subject.vox2ras0), rasMosaic_orig).astype('float32')

    del II, JJ, KK, ii, jj, kk, voxMosaic_orig, rasMosaic_orig

    svf = -np.asarray(svf_list[tp.id].dataobj).astype('float32')
    flow = algorithm_utils.integrate_RegNet(svf, subject.image_shape, parameter_dict, device='cuda:0', model=regnet_model)
    flow_resampled = def_utils.interpolate3DChannel(np.transpose(flow, (1,2,3,0)), voxMosaic_ref[:3].T)#def_utils.interpolate3D([flow[i] for i in range(3)], voxMosaic_2[:3].T)

    del svf, flow

    # Image
    voxMosaic_targ = voxMosaic_ref.copy()[:3]
    voxMosaic_targ[0] += flow_resampled[..., 0]
    voxMosaic_targ[1] += flow_resampled[..., 1]
    voxMosaic_targ[2] += flow_resampled[..., 2]

    voxMosaic_targ = voxMosaic_targ.reshape((1, 3) + tp.image_shape)
    voxMosaic_targ = torch.from_numpy(voxMosaic_targ).float().to('cuda:0')

    p_label_template = torch.from_numpy(p_label_template[np.newaxis]).float().to('cuda:0')
    seg_resampled = interp_func(p_label_template, voxMosaic_targ)

    del voxMosaic_targ, p_label_template

    seg_resampled = np.transpose(seg_resampled[0].cpu().detach().numpy(), (1, 2, 3, 0))

    return seg_resampled




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
subject_list = list(filter(lambda x: '_222' not in x.id and '_231' not in x.id and '_199' not in x.id and '_218' not in x.id, subject_list))

print('[LONGITUDINAL SEGMENTATION] Start processing.')
spatial_variance = [1**2, 3**2, 5**2, 7**2] # in grayscale
temp_variance = [1/4, 1/2, 1, 2] #in years^2

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








