from os.path import exists
from argparse import ArgumentParser
import time

from joblib import delayed, Parallel
import torch
from scipy.ndimage import gaussian_filter

# project imports
from setup import *
from database.data_loader import DataLoader
from scripts import config_dev as configFile
from src.algorithm import *
from src.utils import algorithm_utils, deformation_utils as def_utils, io_utils, image_utils
from database import read_demo_info
from src import models, layers, losses

device = "cpu" # "cuda:0"
ncc_func = losses.NCC_Loss(device=device, kernel_var=[5, 5, 5])

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

def compute_p_label(timepoints, image_list, svf_list, age_list, tp, subject, parameter_dict, unique_labels):

    interp = layers.SpatialInterpolation()

    p_data = np.zeros(tp.image_shape + (len(timepoints),))
    ref_shape = image_list[tp.id].shape
    ref_image = torch.from_numpy(image_list[tp.id][np.newaxis, np.newaxis]).float().to(device)

    ii = np.arange(0, ref_shape[0]),
    jj = np.arange(0, ref_shape[1]),
    kk = np.arange(0, ref_shape[2])

    II, JJ, KK = np.meshgrid(ii, jj, kk, indexing='ij')

    voxMosaic = np.concatenate((II.reshape(-1, 1), JJ.reshape(-1, 1), KK.reshape(-1, 1), np.ones((int(np.prod(ref_shape[:3])), 1))), axis=1).T
    rasMosaic = np.dot(tp.vox2ras0, voxMosaic)
    voxMosaic_2 = np.dot(np.linalg.inv(subject.vox2ras0), rasMosaic)

    for it_tp_2, tp_2 in enumerate(timepoints):
        if tp_2.id == tp.id:
            p_data[..., it_tp_2] = 1

        else:
            svf = np.asarray(svf_list[tp_2.id].dataobj) - np.asarray(svf_list[tp.id].dataobj)
            flow = algorithm_utils.integrate_RegNet(svf, subject.image_shape, parameter_dict)
            flow_resampled = def_utils.interpolate3D([flow[i] for i in range(3)], voxMosaic_2[:3].T)

            voxMosaic_3 = voxMosaic_2.copy()
            voxMosaic_3[0] += flow_resampled[0]
            voxMosaic_3[1] += flow_resampled[1]
            voxMosaic_3[2] += flow_resampled[2]
            rasMosaic_3 = np.dot(subject.vox2ras0, voxMosaic_3)

            voxMosaic_4 = np.matmul(np.linalg.inv(tp_2.vox2ras0), rasMosaic_3)
            voxMosaic_4 = voxMosaic_4[:3]

            del voxMosaic_3, rasMosaic_3

            im = image_list[tp_2.id][np.newaxis, np.newaxis]
            data = torch.from_numpy(im).float().to(device)
            voxMosaic_4_torch = torch.from_numpy(voxMosaic_4.reshape((1,3) + ref_shape)).float().to(device)
            im_resampled = interp(data, voxMosaic_4_torch)
            ncc_resampled = ncc_func.ncc(im_resampled, ref_image)

            # Compute NCC
            p_data[..., it_tp_2] = ncc_resampled[0, 0].cpu().detach().numpy()

            del flow, svf,  im_resampled, ncc_resampled, voxMosaic_4_torch, voxMosaic_4

    # pdb.set_trace()
    # proxy = nib.load(tp.init_path['resample'])
    # img = nib.Nifti1Image(p_data, proxy.affine)
    # nib.save(img, 'pdata_1.nii.gz')

    # Compute ranking images
    r_data = np.argsort(np.argsort(p_data, axis=-1)[..., ::-1]).astype('float')
    # proxy = nib.load(tp.init_path['resample'])
    # img = nib.Nifti1Image(r_data, proxy.affine)
    # nib.save(img, 'rdata_1.nii.gz')

    # Compute weights: gaussian smoothing
    for it_d in range(p_data.shape[-1]): p_data[..., it_d] = gaussian_filter(np.exp(-tp_sigma*r_data[..., it_d]), sigma=spatial_sigma)

    del r_data
    # Normalise weights:
    p_data = p_data / np.sum(p_data, axis=-1, keepdims=True)
    # proxy = nib.load(tp.init_path['resample'])
    # img = nib.Nifti1Image(p_data, proxy.affine)
    # nib.save(img, 'pdata_f_1.nii.gz')

    p_label = np.zeros(tp.image_shape + (len(unique_labels),))
    for it_tp_2, tp_2 in enumerate(timepoints):
        proxyseg = nib.load(join(tp_2.get_filepath('linear_post')))

        if tp_2.id == tp.id:
            seg_resampled = tp.load_posteriors()

        else:
            svf = np.asarray(svf_list[tp_2.id].dataobj) - np.asarray(svf_list[tp.id].dataobj)
            flow = algorithm_utils.integrate_RegNet(svf, subject.image_shape, parameter_dict)
            flow_resampled = def_utils.interpolate3D([flow[i] for i in range(3)], voxMosaic_2[:3].T)

            del svf

            voxMosaic_3 = voxMosaic_2.copy()
            voxMosaic_3[0] += flow_resampled[0]
            voxMosaic_3[1] += flow_resampled[1]
            voxMosaic_3[2] += flow_resampled[2]
            rasMosaic_3 = np.dot(subject.vox2ras0, voxMosaic_3)

            voxMosaic_4 = np.matmul(np.linalg.inv(tp_2.vox2ras0), rasMosaic_3)
            voxMosaic_4 = voxMosaic_4[:3]

            del rasMosaic_3, voxMosaic_3, flow_resampled, flow

            posteriors = np.transpose(np.asarray(proxyseg.dataobj), [3, 0, 1, 2])[np.newaxis]
            data = torch.from_numpy(posteriors).float().to(device)
            voxMosaic_4_torch = torch.from_numpy(voxMosaic_4.reshape((1, 3) + ref_shape)).float().to(device)
            seg_resampled = interp(data, voxMosaic_4_torch)
            seg_resampled = np.transpose(seg_resampled[0].cpu().detach().numpy(), [1,2,3,0])

            # seg_resampled = def_utils.interpolate3DLabel(proxyseg, voxMosaic_4.T, resized_shape=ref_shape)
            # seg_resampled = np.concatenate([s[..., np.newaxis] for s in seg_resampled], axis=-1)

            del proxyseg, posteriors, data, voxMosaic_4_torch

        p_label += p_data[..., it_tp_2, np.newaxis] * seg_resampled

        del seg_resampled

    return p_label

def label_fusion(subject):
    print('Subject: ' + str(subject.id))
    timepoints = subject.timepoints
    flow_dir = subject.results_dirs.get_dir('nonlinear_st_' + reg_algorithm)
    results_dir_sbj = subject.results_dirs.get_dir('longitudinal_segmentation_st_' + reg_algorithm) + '_LNCC'
    if exists(join(results_dir_sbj, 'vols_st.txt')):
        print('Subject: ' + str(subject.id) + '. DONE')
        return None

    if not exists(join(results_dir_sbj, 'regist')): os.makedirs(join(results_dir_sbj, 'regist'))

    timepoints = list(filter(lambda x: not exists(join(results_dir_sbj, x.id + '.nii.gz')), timepoints))

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
    st_vols = []
    for it_tp, tp in enumerate(timepoints):
        print('        - Timepoint ' + tp.id, end='', flush=True)

        unique_labels = np.asarray(list(LABEL_DICT.values()))
        p_label = compute_p_label(timepoints, image_list, svf_list, age_list,  tp, subject,
                                  parameter_dict, unique_labels)

        fake_vol = np.argmax(p_label, axis=-1).astype('uint16')
        true_vol = np.zeros_like(fake_vol)
        for it_ul, ul in enumerate(unique_labels): true_vol[fake_vol == it_ul] = ul

        proxy = nib.load(tp.init_path['resample'])
        img = nib.Nifti1Image(p_label, proxy.affine)
        nib.save(img, join(results_dir_sbj, tp.id + '.post.nii.gz'))

        img = nib.Nifti1Image(true_vol, proxy.affine)
        nib.save(img, join(results_dir_sbj, tp.id + '.nii.gz'))

        vols = get_vols_post(p_label)
        st_vols_dict = {k: vols[it_k] for it_k, k in enumerate(LABEL_DICT.keys())}
        st_vols_dict['TID'] = tp.id
        st_vols.append(st_vols_dict)

    fieldnames = ['TID'] + list(LABEL_DICT.keys())
    write_volume_results(st_vols, join(results_dir_sbj, 'vols_st.txt'), fieldnames=fieldnames)

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

##############
# Processing #
##############
demo_dict = read_demo_info(demo_fields=['AGE'])
# if DB == 'MIRIAD_retest':
#     timepoints_filter = lambda x: '_2.nii.gz' not in x
#     data_loader = DataLoader(sid_list=initial_subject_list, linear=True, timepoints_filter=timepoints_filter)
# else:
#     data_loader = DataLoader(sid_list=initial_subject_list, linear=True)
data_loader = DataLoader(sid_list=initial_subject_list, linear=True)
parameter_dict = configFile.get_config_dict(data_loader.image_shape)

subject_list = data_loader.subject_list
subject_list = list(filter(lambda x: x.id in demo_dict.keys(), subject_list))
subject_list = subject_list
subject_list = list(filter(lambda x: '_222' not in x.id and '_231' not in x.id and '_199' not in x.id and '_218' not in x.id, subject_list))

print('[LONGITUDINAL SEGMENTATION] Start processing.')
spatial_sigma = 2 # in grayscale
tp_sigma = 1/5 #in years^2

# done_sbj = ['miriad_242_AD_F']
# subject_list = list(filter(lambda x: x.id not in done_sbj, subject_list))

if num_cores == 1:
    for subject in subject_list:
        label_fusion(subject)

elif len(subject_list) == 1:
    label_fusion(subject_list[0])

else:
    results = Parallel(n_jobs=num_cores)(delayed(label_fusion)(subject) for subject in subject_list)








