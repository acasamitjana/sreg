from os.path import exists
import time
from argparse import ArgumentParser
from datetime import date, datetime
from joblib import delayed, Parallel


# project imports
from setup import *
from database.data_loader import DataLoader
from scripts import config_dev as configFile
from src.algorithm import *
from src.utils import algorithm_utils, deformation_utils as def_utils, io_utils, image_utils



def register_timepoints(flow, ref_image, flo_image, ref_vox2ras0, flo_vox2ras0, subject_vox2ras0, mode='linear'):
    ii = np.arange(0, ref_image.shape[0]),
    jj = np.arange(0, ref_image.shape[1]),
    kk = np.arange(0, ref_image.shape[2])

    II, JJ, KK = np.meshgrid(ii, jj, kk, indexing='ij')

    voxMosaic = np.concatenate(
        (II.reshape(-1, 1), JJ.reshape(-1, 1), KK.reshape(-1, 1), np.ones((int(np.prod(ref_image.shape[:3])), 1))),
        axis=1).T
    rasMosaic = np.dot(ref_vox2ras0, voxMosaic)
    voxMosaic_2 = np.dot(np.linalg.inv(subject_vox2ras0), rasMosaic)

    flow_resampled = def_utils.interpolate3D([flow[i] for i in range(3)], voxMosaic_2[:3].T)

    voxMosaic_3 = voxMosaic_2
    voxMosaic_3[0] += flow_resampled[0]
    voxMosaic_3[1] += flow_resampled[1]
    voxMosaic_3[2] += flow_resampled[2]
    rasMosaic_3 = np.dot(subject_vox2ras0, voxMosaic_3)
    im_resampled = def_utils.interpolate3D(flo_image, rasMosaic_3, vox2ras0=flo_vox2ras0, resized_shape=ref_image.shape,
                                           mode=mode)

    return im_resampled

def register_timepoints_seg(flow, ref_image, flo_image, ref_vox2ras0, flo_vox2ras0, subject_vox2ras0, mode='linear'):
    ii = np.arange(0, ref_image.shape[0]),
    jj = np.arange(0, ref_image.shape[1]),
    kk = np.arange(0, ref_image.shape[2])

    II, JJ, KK = np.meshgrid(ii, jj, kk, indexing='ij')

    voxMosaic = np.concatenate((II.reshape(-1, 1), JJ.reshape(-1, 1), KK.reshape(-1, 1), np.ones((int(np.prod(ref_image.shape[:3])), 1))), axis=1).T
    rasMosaic = np.dot(ref_vox2ras0, voxMosaic)
    voxMosaic_2 = np.dot(np.linalg.inv(subject_vox2ras0), rasMosaic)

    flow_resampled = def_utils.interpolate3D([flow[i] for i in range(3)], voxMosaic_2[:3].T)

    voxMosaic_3 = voxMosaic_2
    voxMosaic_3[0] += flow_resampled[0]
    voxMosaic_3[1] += flow_resampled[1]
    voxMosaic_3[2] += flow_resampled[2]
    rasMosaic_3 = np.dot(subject_vox2ras0, voxMosaic_3)
    im_resampled = def_utils.interpolate3DLabel(flo_image, rasMosaic_3, vox2ras0=flo_vox2ras0, resized_shape=ref_image.shape, mode=mode)

    return im_resampled


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
        vols[l] = np.sum(mask_l)

    return vols

def label_fusion(subject):
    print('   Subject: ' + str(subject.id))

    subject_shape = subject.image_shape
    timepoints = subject.timepoints
    flow_dir = subject.results_dirs.get_dir('nonlinear_registration')
    results_dir_sbj = subject.results_dirs.get_dir('longitudinal_segmentation_registration')

    # timepoints = list(filter(lambda x: not exists(join(results_dir_sbj, x.id + '.nii.gz')), timepoints))

    print('     o Reading the input files')
    image_list = []
    for tp in timepoints:
        # Distance map
        seg = tp.load_seg()
        image = tp.load_data()

        # Normalize image
        wm_mask = seg == 2
        m = np.mean(image[wm_mask])
        image = 110 * image / m
        image_list.append(image)

        del wm_mask, image, seg

    print('     o Computing the segmentation')
    regnet_vols = []
    for it_tp, tp in enumerate(timepoints):
        print('        - Timepoint ' + tp.id, end='', flush=True)

        if exists(join(results_dir_sbj, tp.id + '.nii.gz')):
            print('')
            continue

        print('(registering images, ', end='', flush=True)
        # Register distance maps
        unique_labels = np.unique(tp.load_seg())
        p_data = np.zeros(tp.image_shape + (len(timepoints),))
        p_label = np.zeros(tp.image_shape + (len(unique_labels),))
        for it_tp_2, tp_2 in enumerate(timepoints):
            if it_tp_2 == it_tp:
                # Label Fusion
                p_data[..., it_tp_2] = np.exp(-0.5 / variance * (image_list[it_tp_2] - image_list[it_tp]) ** 2)

            else:
                if exists(join(flow_dir, tp.id + '_to_' + tp_2.id + '.svf.nii.gz')):
                    proxy = nib.load(join(flow_dir, tp.id + '_to_' + tp_2.id + '.svf.nii.gz'))
                    svf = np.asarray(proxy.dataobj)
                else:
                    proxy = nib.load(join(flow_dir, tp_2.id + '_to_' + tp.id + '.svf.nii.gz'))
                    svf = -np.asarray(proxy.dataobj)

                flow = algorithm_utils.integrate_RegNet(svf, subject_shape, parameter_dict)

                im_resampled = register_timepoints(flow, image_list[it_tp], image_list[it_tp_2], tp.vox2ras0,
                                                   tp_2.vox2ras0, subject.vox2ras0)

                # Label Fusion
                p_data[..., it_tp_2] = np.exp(-0.5 / variance * (im_resampled - image_list[it_tp]) ** 2)

        # Label Fusion
        print('registering distance maps, ', end='', flush=True)
        p_data = p_data / np.sum(p_data, axis=-1, keepdims=True)
        for it_tp_2, tp_2 in enumerate(timepoints):
            if it_tp_2 == it_tp:
                p_label += p_data[..., it_tp_2, np.newaxis] * tp.load_posteriors()

            else:
                if exists(join(flow_dir, tp.id + '_to_' + tp_2.id + '.svf.nii.gz')):
                    proxy = nib.load(join(flow_dir, tp.id + '_to_' + tp_2.id + '.svf.nii.gz'))
                    svf = np.asarray(proxy.dataobj)
                else:
                    proxy = nib.load(join(flow_dir, tp_2.id + '_to_' + tp.id + '.svf.nii.gz'))
                    svf = -np.asarray(proxy.dataobj)
                flow = algorithm_utils.integrate_RegNet(svf, subject_shape, parameter_dict)

                proxyseg = nib.load(join(tp_2.get_filepath('linear_post')))
                seg_resampled = register_timepoints_seg(flow, image_list[it_tp], proxyseg, tp.vox2ras0, tp_2.vox2ras0,
                                                        subject.vox2ras0)
                seg_resampled = np.concatenate([s[..., np.newaxis] for s in seg_resampled], axis=-1)

                # Label Fusion
                p_label += p_data[..., it_tp_2, np.newaxis] * seg_resampled

        print('computing labels)', end='\n', flush=True)

        fake_vol = np.argmax(p_label, axis=-1).astype('uint16')
        true_vol = np.zeros_like(fake_vol)
        for it_ul, ul in enumerate(unique_labels): true_vol[fake_vol == it_ul] = ul

        proxy = nib.load(tp.init_path['resample'])
        img = nib.Nifti1Image(p_label, proxy.affine)
        nib.save(img, join(results_dir_sbj, tp.id + '.post.nii.gz'))

        img = nib.Nifti1Image(true_vol, proxy.affine)
        nib.save(img, join(results_dir_sbj, tp.id + '.nii.gz'))

        vols = get_vols_post(p_label)
        regnet_vols_dict = {k: vols[it_k] for it_k, k in enumerate(LABEL_DICT.keys())}
        regnet_vols_dict['TID'] = tp.id
        regnet_vols.append(regnet_vols_dict)

    fieldnames = ['TID'] + list(LABEL_DICT.keys())
    write_volume_results(regnet_vols, join(results_dir_sbj, 'vols_regnet.txt'), fieldnames=fieldnames)
    print('DONE')

    # Compute volumes
print('\n\n\n\n\n')
print('# ----------------------------------------------------------------- #')
print('# Running the longitudinal segmentation script (direct raw version) #')
print('# ----------------------------------------------------------------- #')
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

arguments = arg_parser.parse_args()
initial_subject_list = arguments.subjects


##############
# Processing #
##############

data_loader = DataLoader(sid_list=initial_subject_list, linear=True)
subject_list = data_loader.subject_list

parameter_dict = configFile.get_config_dict(data_loader.image_shape)

print('[LONGITUDINAL SEGMENTATION] Start processing.')
variance = 10 * 10
temp_variance = 0.5 #in years^^

# done_sbj = ['miriad_205_AD_F', 'miriad_231_HC_M']
# subject_list = list(filter(lambda  x: x not in done_sbj, subject_list))
if num_cores == 1:
    for subject in subject_list:
        label_fusion(subject)

elif len(subject_list) == 1:
    label_fusion(subject_list[0])

else:
    results = Parallel(n_jobs=num_cores)(delayed(label_fusion)(subject) for subject in subject_list)