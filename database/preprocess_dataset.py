from os.path import exists, dirname
from os import makedirs
from argparse import ArgumentParser

import numpy as np
import nibabel as nib
from skimage.morphology import ball
from scipy.ndimage.morphology import binary_dilation

from src.utils.io import create_results_dir
from database.data_loader import DataLoader
from setup import *

arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--subjects', default=None, nargs='+')

arguments = arg_parser.parse_args()
subject_list = arguments.subjects

create_results_dir(REGISTRATION_DIR, subdirs=['resampled', 'segmentations', 'masks', 'images', 'Linear', 'Nonlinear'])


print('\n\n\n\n\n')
print('# ---------------------------------------------------------------------------------------- #')
print('# Preprocessing initial dataset with images and segmentations (e.g., FreeSurfer, Synthseg) #')
print('# ---------------------------------------------------------------------------------------- #')
print('\n\n')

print('\n[PREPROCESSING]: computing masks, dilated masks and centering images')

data_loader = DataLoader(sid_list=subject_list)
subject_list = data_loader.subject_list

for sbj in subject_list:
    print(' Subject: ' + sbj.sid)

    for tp in sbj.timepoints:
        if not exists(dirname(tp.mask_path)):
            makedirs(dirname(tp.mask_path))

        mri = tp.load_data()
        seg = tp.load_seg()
        vox2ras0 = tp.vox2ras0

        # Compute mask and dilated mask
        if not exists( tp.mask_dilated_path) or not exists(tp.mask_dilated_path) or not exists(tp.image_centered_path) \
                or not exists(tp.mask_centered_path) or not exists(tp.mask_dilated_centered_path) \
                or not exists(tp.seg_centered_path):
            mask = seg > 0

            se = ball(3)
            mask_dilated = binary_dilation(mask, se)

            img = nib.Nifti1Image(mask.astype('uint8'), tp.vox2ras0)
            nib.save(img, tp.mask_path)

            img = nib.Nifti1Image(mask_dilated.astype('uint8'), tp.vox2ras0)
            nib.save(img, tp.mask_dilated_path)

            # Compute COG
            idx = np.where(mask_dilated>0)
            cog = np.concatenate((np.mean(idx, axis=1), [1]), axis=0)
            cog_ras = np.matmul(vox2ras0, cog)


            # Apply COG
            vox2ras0_centered = vox2ras0
            vox2ras0_centered[:, 3] -= cog_ras

            img = nib.Nifti1Image(mri, vox2ras0_centered)
            nib.save(img, tp.image_centered_path)

            img = nib.Nifti1Image(mask.astype('uint8'), vox2ras0_centered)
            nib.save(img, tp.mask_centered_path)

            img = nib.Nifti1Image(mask_dilated.astype('uint8'), vox2ras0_centered)
            nib.save(img, tp.mask_dilated_centered_path)

            img = nib.Nifti1Image(seg.astype('uint8'), vox2ras0_centered)
            nib.save(img, tp.seg_centered_path)

print('\n')
print('[PREPROCESSING]: Done\n')




