from os.path import join
import csv

import nibabel as nib
import numpy as np

from setup import *
from database.data_loader import DataLoader


def get_vols_post(post, res=1):
    '''

    :param post: posterior probabilities
    :param res: mm^3 per voxel
    :return:
    '''

    n_labels = post.shape[-1]
    n_dims = len(post.shape[:-1])
    if isinstance(res, int):
        res = [res]*n_dims
    vol_vox = np.prod(res)

    vols = {}
    for l in range(n_labels):
        mask_l = post[..., l]
        mask_l[post[..., l] < 0.05] = 0
        vols[l] = np.sum(mask_l) * vol_vox

    return vols

def write_volume_results(volume_dict, filepath, fieldnames=None, attach_overwrite='w'):
    if fieldnames is None:
        fieldnames = ['TID'] + list(LABEL_DICT.keys())

    with open(filepath, attach_overwrite) as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()
        csvwriter.writerows(volume_dict)

labels_to_write = ['Thalamus', 'Lateral-Ventricle', 'Hippocampus', 'Amygdala', 'Caudate', 'Pallidum', 'Putamen',
                   'Accumbens', 'Inf-Lat-Ventricle']
keep_labels = ['Right-' + l for l in labels_to_write] + ['Left-' + l for l in labels_to_write]

UNIQUE_LABELS = np.asarray([0] + [lab for labstr, lab in LABEL_DICT.items() if labstr in keep_labels], dtype=np.uint8)
initial_subject_list = ['miriad_205_AD_F', 'miriad_225_AD_M', 'miriad_222_AD_F']
data_loader = DataLoader(sid_list=initial_subject_list, linear=True)
subject_list = data_loader.subject_list
spatial_variance = [3**2, 5**2, 7**2] # in grayscale
temp_variance = [1, 'inf'] #in years^2

for subject in subject_list:
    print(subject.id)
    timepoints = subject.timepoints
    results_dir_sbj = subject.results_dirs.get_dir('longitudinal_segmentation_st_freesurfer') + '_JAC'
    st_vols = {t_var: {sp_var: [] for sp_var in spatial_variance} for t_var in temp_variance}
    for tp in timepoints:
        print('  - ' + tp.id)

        for t_var in temp_variance:
            for s_var in spatial_variance:
                proxy = nib.load(join(results_dir_sbj, str(t_var) + '_' + str(s_var), tp.id + '.post.nii.gz'))
                p_label = np.asarray(proxy.dataobj)
                p_label[np.isnan(p_label)] = 0

                vols = get_vols_post(p_label)
                st_vols_dict = {k: vols[[it_ul for it_ul, ul in enumerate(UNIQUE_LABELS) if ul == val][0]] for k, val in LABEL_DICT.items() if val in UNIQUE_LABELS}
                st_vols_dict['TID'] = tp.id
                st_vols[t_var][s_var].append(st_vols_dict)

                del p_label

    fieldnames = ['TID'] + list(LABEL_DICT.keys())
    for t_var in temp_variance:
        for s_var in spatial_variance:
            write_volume_results(st_vols[t_var][s_var], join(results_dir_sbj, str(t_var) + '_' + str(s_var), 'vols_st.txt'), fieldnames=fieldnames)
