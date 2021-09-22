from os.path import join
from argparse import ArgumentParser

import nibabel as nib
import numpy as np

from database.data_loader import DataLoader
from database.databaseConfig import FCIEN_DB
arg_parser = ArgumentParser(description='Computes the video for several subjects')
arg_parser.add_argument('--subject', default=None, nargs='+')

arguments = arg_parser.parse_args()
subject_list = arguments.subject

data_loader = DataLoader(FCIEN_DB, rid_list=subject_list)

for subject in data_loader.subject_list:
    mri_list = []
    vox2ras0 = np.load(join(subject.data_path, 'template_vox2ras0', str(subject.id) + '_lin_template_v2r.npy'))

    for tp in subject.timepoints:
        proxy = nib.load(tp.image_linear_path)
        mri_list.append(np.squeeze(np.asarray(proxy.dataobj)))

    template = np.median(mri_list, axis=0)
    img = nib.Nifti1Image(template, vox2ras0)
    nib.save(img, join(subject.data_path, 'linear_template.nii.gz'))

#
#
#
# PATH = join(config_data.ALGORITHM_DIR, 'ST_NR_lineal', 'l1', subject)
# PATH = join(config_data.ALGORITHM_DIR, 'ST_NR_lineal', 'l1', subject)
#
# if exists(PATH):
#     files = listdir(PATH)
# else:
#     print("No files found for subject " + str(subject))
#     exit()
#
# files = filter(lambda x: '_image_NR.nii.gz' in x, files)
#
# mri_list = []
# vox2ras0 = np.eye(4)
# for f in files:
#     proxy = nib.load(join(PATH, f))
#     vox2ras0 = proxy.affine
#     mri_list.append(np.squeeze(np.asarray(proxy.dataobj)))
#
# template = np.median(mri_list, axis=0)
# img = nib.Nifti1Image(template, vox2ras0)
# nib.save(img, join(PATH, 'linear_template.nii.gz'))

