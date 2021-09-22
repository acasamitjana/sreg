import os

import nibabel as nib
import numpy as np


subject_in = '012'
subject_out = '012_rotated'
sbj_directory_in = '/home/acasamitjana/Data/BrainAging/FCIEN_OK/'
sbj_directory_out = '/home/acasamitjana/Data/BrainAging/FCIEN_OK/'

subdirs = ['images_resampled', 'segmentations', 'masks']


timepoints = ['00', '01', '02', '03', '04', '05', '06', '07', '08']

random_angle = 20
for t in timepoints:
    mripath = os.path.join(sbj_directory_in, 'images_resampled', subject_in, t + '.nii.gz')

    proxy = nib.load(mripath)
    mri = np.asarray(proxy.dataobj)
    vox2ras0 = proxy.affine

    angles = (2 * np.random.rand(3) - 1) * random_angle/180*np.pi
    print(angles /np.pi *180)

    cr = [s//2 for s in proxy.shape]
    T0 = np.eye(4)
    T0_inv = np.eye(4)
    T1 = np.eye(4)
    T2 = np.eye(4)
    T3 = np.eye(4)

    T0[0, 3] = -cr[0]
    T0[1, 3] = -cr[1]
    T0[2, 3] = -cr[2]

    T1[1, 1] = np.cos(angles[0])
    T1[1, 2] = -np.sin(angles[0])
    T1[2, 1] = np.sin(angles[0])
    T1[2, 2] = np.cos(angles[0])

    T2[0, 0] = np.cos(angles[1])
    T2[0, 2] = np.sin(angles[1])
    T2[2, 0] = -np.sin(angles[1])
    T2[2, 2] = np.cos(angles[1])

    T3[0, 0] = np.cos(angles[2])
    T3[0, 1] = -np.sin(angles[2])
    T3[1, 0] = np.sin(angles[2])
    T3[1, 1] = np.cos(angles[2])

    T0_inv[0, 3] = cr[0]
    T0_inv[1, 3] = cr[1]
    T0_inv[2, 3] = cr[2]
    rotation_matrix = T0_inv @ T3 @ T2 @ T1 @ T0


    vox2ras0 = vox2ras0 @ rotation_matrix

    for subd in subdirs:
        in_filepath = os.path.join(sbj_directory_in, subd, subject_in, t + '.nii.gz')
        out_filepath = os.path.join(sbj_directory_out, subd, subject_out, t + '.nii.gz')
        proxy = nib.load(in_filepath)
        data = np.asarray(proxy.dataobj)
        img = nib.Nifti1Image(data, vox2ras0)
        nib.save(img, out_filepath)
