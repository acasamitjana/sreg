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


def label_fusion(subject):
    print('Subject: ' + str(subject.id))
    timepoints = subject.timepoints
    results_dir_sbj = subject.results_dirs.get_dir('longitudinal_segmentation_st_freesurfer')

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

    print('  o Computing the segmentation')
    for tp in timepoints:
        print('        - Timepoint ' + tp.id, flush=True)

        for t_var in temp_variance:
            for s_var in spatial_variance:
                if not exists(join(results_dir_sbj, str(t_var) + '_' + str(s_var), tp.id + '_to_template.nii.gz')):
                    labelfile = join(results_dir_sbj, str(t_var) + '_' + str(s_var), tp.id + '.post.nii.gz')
                    p_label = register_p_label(subject, tp, labelfile, parameter_dict, model)

                    p_label[np.isnan(p_label)] = 0
                    mask = np.sum(p_label, axis=-1) > 0.5
                    fake_vol = np.argmax(p_label, axis=-1).astype('uint16')
                    true_vol = np.zeros_like(fake_vol)
                    for it_ul, ul in enumerate(UNIQUE_LABELS): true_vol[fake_vol == it_ul] = ul
                    true_vol = true_vol * mask

                    img = nib.Nifti1Image(p_label, subject.vox2ras0)
                    nib.save(img, join(results_dir_sbj, str(t_var) + '_' + str(s_var), tp.id + '_to_template.post.nii.gz'))

                    img = nib.Nifti1Image(true_vol, subject.vox2ras0)
                    nib.save(img, join(results_dir_sbj, str(t_var) + '_' + str(s_var), tp.id + '_to_template.nii.gz'))

                    del p_label

        if not exists(join(results_dir_sbj,  tp.id + '_to_template.fs.post.nii.gz')):
            fileparts = tp.id.split('_')
            labelfile = join(FS_DIR, fileparts[0] + '_' + fileparts[1] + '_' + str(int(fileparts[4])) + '_' + fileparts[5] + '_' + fileparts[6] + '.cross.aseg.mgz')
            p_label = register_p_label_fs(subject, tp, labelfile, parameter_dict, model)

            p_label[np.isnan(p_label)] = 0
            mask = np.sum(p_label, axis=-1) > 0.5
            fake_vol = np.argmax(p_label, axis=-1).astype('uint16')
            true_vol = np.zeros_like(fake_vol)
            for it_ul, ul in enumerate(UNIQUE_LABELS): true_vol[fake_vol == it_ul] = ul
            true_vol = true_vol * mask

            img = nib.Nifti1Image(p_label, subject.vox2ras0)
            nib.save(img, join(results_dir_sbj,  tp.id + '_to_template.fs.post.nii.gz'))

            img = nib.Nifti1Image(true_vol, subject.vox2ras0)
            nib.save(img, join(results_dir_sbj,  tp.id + '_to_template.fs.nii.gz'))

            del p_label

    print('Subject: ' + str(subject.id) + '. DONE')


def register_p_label(subject, tp, labelfile, parameter_dict, regnet_model):
    flow_dir = subject.results_dirs.get_dir('nonlinear_st_' + reg_algorithm)
    proxyseg = nib.load(labelfile)
    proxyflow = nib.load(join(flow_dir, tp.id + '.svf.nii.gz'))

    interp_func = layers.SpatialInterpolation(padding_mode='zeros').to('cuda:0')

    ref_shape = subject.image_shape

    ii = np.arange(0, ref_shape[0], dtype='int32')
    jj = np.arange(0, ref_shape[1], dtype='int32')
    kk = np.arange(0, ref_shape[2], dtype='int32')

    II, JJ, KK = np.meshgrid(ii, jj, kk, indexing='ij')

    voxMosaic_orig = np.concatenate((II.reshape(-1, 1), JJ.reshape(-1, 1), KK.reshape(-1, 1), np.ones((int(np.prod(ref_shape[:3])), 1), dtype='int32')), axis=1).T

    del II, JJ, KK, ii, jj, kk,

    svf = np.asarray(proxyflow.dataobj).astype('float32')
    flow = algorithm_utils.integrate_RegNet(svf, subject.image_shape, parameter_dict, device='cuda:0', model=regnet_model)
    flow_resampled = def_utils.interpolate3DChannel(np.transpose(flow, (1, 2, 3, 0)), voxMosaic_orig[:3].T)  # def_utils.interpolate3D([flow[i] for i in range(3)], voxMosaic_2[:3].T)

    voxMosaic_orig = voxMosaic_orig.astype('float32')
    voxMosaic_orig[0] += flow_resampled[..., 0]
    voxMosaic_orig[1] += flow_resampled[..., 1]
    voxMosaic_orig[2] += flow_resampled[..., 2]

    del flow, flow_resampled, svf

    rasMosaic_targ = np.dot(subject.vox2ras0, voxMosaic_orig).astype('float32')
    voxMosaic_targ = np.matmul(np.linalg.inv(tp.vox2ras0), rasMosaic_targ).astype('float32')
    voxMosaic_targ = voxMosaic_targ[:3]

    voxMosaic_targ = voxMosaic_targ.reshape((1, 3) + ref_shape)
    voxMosaic_targ = torch.from_numpy(voxMosaic_targ).float().to('cuda:0')

    seg = np.asarray(proxyseg.dataobj)
    seg = np.transpose(seg, axes=(3,0,1,2))
    seg = torch.from_numpy(seg[np.newaxis]).float().to('cuda:0')
    seg_resampled = interp_func(seg, voxMosaic_targ)
    seg_resampled = np.transpose(seg_resampled[0].cpu().detach().numpy(), (1, 2, 3, 0))

    return seg_resampled


def register_p_label_fs(subject, tp, labelfile, parameter_dict, regnet_model):
    flow_dir = subject.results_dirs.get_dir('nonlinear_st_' + reg_algorithm)
    proxyseg = nib.load(labelfile)
    proxyflow = nib.load(join(flow_dir, tp.id + '.svf.nii.gz'))
    affine_dir = subject.results_dirs.get_dir('linear_st')

    cog = tp.get_cog()
    affine_matrix = read_affine_matrix(join(affine_dir, tp.id + '.aff'), full=True)
    affine_matrix[:3, 3] += cog
    vox2ras0_fs = np.dot(np.linalg.inv(affine_matrix), proxyseg.affine).astype('float32')

    interp_func = layers.SpatialInterpolation(padding_mode='zeros').to('cuda:0')

    ref_shape = subject.image_shape

    ii = np.arange(0, ref_shape[0], dtype='int32')
    jj = np.arange(0, ref_shape[1], dtype='int32')
    kk = np.arange(0, ref_shape[2], dtype='int32')

    II, JJ, KK = np.meshgrid(ii, jj, kk, indexing='ij')

    voxMosaic_orig = np.concatenate((II.reshape(-1, 1), JJ.reshape(-1, 1), KK.reshape(-1, 1), np.ones((int(np.prod(ref_shape[:3])), 1), dtype='int32')), axis=1).T

    del II, JJ, KK, ii, jj, kk,

    svf = np.asarray(proxyflow.dataobj).astype('float32')
    flow = algorithm_utils.integrate_RegNet(svf, subject.image_shape, parameter_dict, device='cuda:0', model=regnet_model)
    flow_resampled = def_utils.interpolate3DChannel(np.transpose(flow, (1, 2, 3, 0)), voxMosaic_orig[:3].T)  # def_utils.interpolate3D([flow[i] for i in range(3)], voxMosaic_2[:3].T)

    voxMosaic_orig = voxMosaic_orig.astype('float32')
    voxMosaic_orig[0] += flow_resampled[..., 0]
    voxMosaic_orig[1] += flow_resampled[..., 1]
    voxMosaic_orig[2] += flow_resampled[..., 2]

    del flow, flow_resampled, svf

    rasMosaic_targ = np.dot(subject.vox2ras0, voxMosaic_orig).astype('float32')
    voxMosaic_targ = np.matmul(np.linalg.inv(vox2ras0_fs), rasMosaic_targ).astype('float32')
    voxMosaic_targ = voxMosaic_targ[:3]

    voxMosaic_targ = voxMosaic_targ.reshape((1, 3) + ref_shape)
    voxMosaic_targ = torch.from_numpy(voxMosaic_targ).float().to('cuda:0')

    seg = np.asarray(proxyseg.dataobj)
    seg = image_utils.one_hot_encoding(seg, categories=UNIQUE_LABELS).astype('float32')
    seg = torch.from_numpy(seg[np.newaxis]).float().to('cuda:0')
    seg_resampled = interp_func(seg, voxMosaic_targ)
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








