from os import listdir
from os.path import join

import nibabel as nib
import numpy as np
from skimage.transform import resize

from src.utils.algorithm_utils import integrate_NR
from src.utils.image_utils import deform3D
from src.utils.image_transform import ScaleNormalization

PATH = '/home/acasamitjana/Results/Registration/BrainAging/FCIEN/ST/ST_NR/l1//012'
DATA_PATH = '/home/acasamitjana/Data/BrainAging/FCIEN/012'

svf_shape = (94, 94, 94)
template_shape = (189, 189, 189)
files = listdir(PATH)
norm = ScaleNormalization(range=[0, 1])
for it_ref in range(8):
    print(it_ref)
    for it_flo in range(it_ref+1,8):
        print('  - ' + str(it_flo))
        ref_str = "{:02d}".format(it_ref)
        flo_str = "{:02d}".format(it_flo)

        field = np.zeros((3,) + svf_shape)
        proxy = nib.load(join(PATH, ref_str + '.field_x.nii.gz'))
        field[0] = np.squeeze(np.asarray(proxy.dataobj))
        proxy = nib.load(join(PATH, ref_str + '.field_y.nii.gz'))
        field[1] = np.squeeze(np.asarray(proxy.dataobj))
        proxy = nib.load(join(PATH, ref_str + '.field_z.nii.gz'))
        field[2] = np.squeeze(np.asarray(proxy.dataobj))


        field2 = np.zeros((3, ) + svf_shape)
        proxy = nib.load(join(PATH, flo_str + '.field_x.nii.gz'))
        field2[0] = np.squeeze(np.asarray(proxy.dataobj))
        proxy = nib.load(join(PATH, flo_str + '.field_y.nii.gz'))
        field2[1] = np.squeeze(np.asarray(proxy.dataobj))
        proxy = nib.load(join(PATH, flo_str + '.field_z.nii.gz'))
        field2[2] = np.squeeze(np.asarray(proxy.dataobj))

        field_fi = -field + field2
        dims = ['x', 'y', 'z']
        for it_d, d in enumerate(dims):
            f = resize(field_fi[it_d], (189,189,189))
            img = nib.Nifti1Image(f, proxy.affine)
            nib.save(img, join(PATH, ref_str + '_to_' + flo_str + '.field_' + d + '.nii.gz'))

        # ff = integrate_NR(field_fi, (189,189,189))
        #
        # proxy = nib.load(join(DATA_PATH, flo_str, 'image.reoriented.nii.gz'))
        # image = np.asarray(proxy.dataobj)
        # image = norm(image)
        #
        # image_ff = deform3D(image, ff)
        #
        # img = nib.Nifti1Image(image_ff, proxy.affine)
        # nib.save(img, join(PATH, ref_str + '_to_' + flo_str + '.nii.gz'))

