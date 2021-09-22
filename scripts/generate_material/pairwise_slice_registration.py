import subprocess
from os.path import join, exists
import os
import copy

import cv2
import numpy as np
from PIL import Image
import nibabel as nib
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter

from database import databaseConfig
from src.utils.io import create_results_dir
from src.utils.image_utils import bilinear_interpolate
from database.Allen.data_loader import DataLoaderBlock as DL_slices, DataLoader as DL_subject
from scripts.Allen import read_slice_info


def pad_image(im, margin):

    new_shape = (im.shape[0] + 2*margin[0], im.shape[1] + 2*margin[1])
    new_im = np.zeros(new_shape, dtype=im.dtype)
    new_im[margin[0]:-margin[0], margin[1]:-margin[1]] = im
    return new_im


ALGORITHM_DIR = '/home/acasamitjana/Results/Registration/Allen/Pairwise_reg'
DATA_DIR = '/home/acasamitjana/Data/Allen_paper'


HISTO_res = 0.032
HISTO_THICKNESS = 0.05
BLOCK_res = 0.25

HISTO_AFFINE = np.zeros((4,4))
HISTO_AFFINE[0, 1] = BLOCK_res
HISTO_AFFINE[2, 0] = -BLOCK_res
HISTO_AFFINE[1, 2] = -HISTO_THICKNESS
HISTO_AFFINE[0, 3] = -93*BLOCK_res
HISTO_AFFINE[2, 3] = -79*BLOCK_res

OUTPUT_res = 0.25
OUTPUT_THICKNESS = {'IHC': 0.4, 'NISSL': 0.2}

NIFTY_REG_DIR = '/home/acasamitjana/Software_MI/niftyreg-git/build/'
ALADINcmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_aladin'
F3Dcmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_f3d'
TRANSFORMcmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_transform'
REScmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_resample'
scontrol = [-8, -8]
tempdir = '/tmp'

nslices = 641

BT_DB_slices = databaseConfig.Allen_MRI
data_loader_subject = DL_subject(BT_DB_slices)

data_loader_slices = DL_slices(BT_DB_slices)
block_list = data_loader_slices.subject_list

file = 'slice_separation_extended.csv'
slice_num_dict = read_slice_info(file)

for stain in ['NISSL']:
    print(' ' + stain)
    init_sbj = 0
    OUTPUT_PATH = ALGORITHM_DIR
    create_results_dir(OUTPUT_PATH, subdirs=['IHC_gray', 'NISSL_gray',
                                             'IHC_masks', 'NISSL_masks',
                                             'IHC_affine', 'NISSL_affine',
                                             'IHC_gray_affine', 'NISSL_gray_affine',
                                             'IHC_masks_affine', 'NISSL_masks_affine',
                                             ])


    # print('   Slice number:', end=' ', flush=True)
    # slice_info = slice_num_dict[stain].pop('slice_320.png')
    # tree_pos = int(slice_info['tree_pos'])
    # sid = "{:03d}".format(tree_pos)
    #
    # #### Read image and deform
    # filename = slice_info['filename']
    # H_filepath = join(DATA_DIR, stain.lower(), 'images_extended', filename)
    # H_mask_filepath = join(DATA_DIR, stain.lower(), 'masks_extended', filename)
    # H_orig = cv2.imread(H_filepath)
    # H_gray = cv2.cvtColor(H_orig, cv2.COLOR_BGR2GRAY)
    # H_gray = gaussian_filter((H_gray / 255).astype(np.double), sigma=5)
    #
    # M = cv2.imread(H_mask_filepath, flags=0)
    # M = (M / np.max(M)) > 0.5
    # H_gray[~M] = 0
    # M = (255 * M).astype('uint8')
    # H_gray = (255 * H_gray).astype('uint8')
    #
    # resized_shape = tuple([int(i * HISTO_res / BLOCK_res) for i in H_orig.shape])
    # H_gray = cv2.resize(H_gray, (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_LINEAR)
    # M = cv2.resize(M, (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_LINEAR)
    #
    # M = pad_image(M, [120, 40])
    # H_gray = pad_image(H_gray, [120, 40])
    #
    # img = Image.fromarray(M, mode='L')
    # img.save(join(OUTPUT_PATH, stain + '_masks', 'slice_' + sid + '.png'))
    # img = Image.fromarray(H_gray, mode='L')
    # img.save(join(OUTPUT_PATH, stain + '_gray', 'slice_' + sid + '.png'))
    # if stain == 'NISSL':
    #
    #     img = Image.fromarray(M, mode='L')
    #     img.save(join(OUTPUT_PATH, stain + '_masks_affine', 'slice_' + sid + '.png'))
    #
    #     img = Image.fromarray(H_gray, mode='L')
    #     img.save(join(OUTPUT_PATH, stain + '_gray_affine', 'slice_' + sid + '.png'))
    #
    #     matrix = np.eye(4)
    #     with open(join(OUTPUT_PATH, stain + '_affine', 'slice_' + sid + '.aff'), 'w') as writeFile:
    #         for i in range(4):
    #             writeFile.write(' '.join([str(matrix[i, j]) for j in range(4)]))
    #             writeFile.write('\n')
    #
    # else:
    #     refFile = join(OUTPUT_PATH, 'NISSL_gray_affine', 'slice_' + sid + '.png')
    #     refMaskFile = join(OUTPUT_PATH, 'NISSL_masks_affine', 'slice_' + sid + '.png')
    #
    #     floGrayFile = join(OUTPUT_PATH, stain + '_gray', 'slice_' + sid + '.png')
    #     floMaskFile = join(OUTPUT_PATH, stain + '_masks', 'slice_' + sid + '.png')
    #
    #     outputGrayFile = join(OUTPUT_PATH, stain + '_gray_affine', 'slice_' + sid + '.png')
    #     outputMaskFile = join(OUTPUT_PATH, stain + '_masks_affine', 'slice_' + sid + '.png')
    #
    #     affineFile = join(OUTPUT_PATH, stain + '_affine', 'slice_' + sid + '.aff')
    #
    #     subprocess.call(
    #         [ALADINcmd, '-ref', refFile, '-flo', floGrayFile, '-aff', affineFile, '-res', outputGrayFile,
    #          '-ln', '4', '-lp', '3', '-pad', '0'], stdout=subprocess.DEVNULL)
    #     subprocess.call(
    #         [REScmd, '-ref', refFile, '-flo', floGrayFile, '-trans', affineFile, '-res', outputGrayFile,
    #          '-inter', '3', '-voff'])
    #     subprocess.call(
    #         [REScmd, '-ref', refMaskFile, '-flo', floMaskFile, '-trans', affineFile, '-res', outputMaskFile,
    #          '-inter', '0', '-voff'])
    #
    # for slice_num in range(321, 642):
    #     slice_filename = 'slice_' + "{:02d}".format(slice_num) + '.png'
    #     slice_info = slice_num_dict[stain][slice_filename]
    #     print(slice_info['tree_pos'], end=' ', flush=True)
    #     tree_pos = int(slice_info['tree_pos'])
    #     sid =  "{:02d}".format(tree_pos)
    #     #### Read image and deform
    #     filename = slice_info['filename']  # 'slice_' + "{:03d}".format(int(slice_info['it_slice'])) + '.png'
    #     H_filepath = join(DATA_DIR, stain.lower(), 'images_extended', filename)
    #     H_mask_filepath = join(DATA_DIR, stain.lower(), 'masks_extended', filename)
    #     H_orig = cv2.imread(H_filepath)
    #     H_gray = cv2.cvtColor(H_orig, cv2.COLOR_BGR2GRAY)
    #     H_gray = gaussian_filter((H_gray / 255).astype(np.double), sigma=5)
    #
    #     M = cv2.imread(H_mask_filepath, flags=0)
    #     M = (M / np.max(M)) > 0.5
    #     H_gray[~M] = 0
    #     M = (255 * M).astype('uint8')
    #     H_gray = (255 * H_gray).astype('uint8')
    #
    #     resized_shape = tuple([int(i * HISTO_res / BLOCK_res) for i in H_orig.shape])
    #     H_gray = cv2.resize(H_gray, (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_LINEAR)
    #     M = cv2.resize(M, (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_LINEAR)
    #
    #     img = Image.fromarray(M, mode='L')
    #     img.save(join(OUTPUT_PATH, stain + '_masks', 'slice_' + sid + '.png'))
    #     img = Image.fromarray(H_gray, mode='L')
    #     img.save(join(OUTPUT_PATH, stain + '_gray', 'slice_' + sid + '.png'))
    #
    #     sid_pre = "{:03d}".format(tree_pos-1)
    #
    #     refFile = join(OUTPUT_PATH, stain + '_gray_affine', 'slice_' + sid_pre + '.png')
    #     refMaskFile = join(OUTPUT_PATH, stain + '_masks_affine', 'slice_' + sid_pre + '.png')
    #
    #     floGrayFile = join(OUTPUT_PATH, stain + '_gray', 'slice_' + sid + '.png')
    #     floMaskFile = join(OUTPUT_PATH, stain + '_masks', 'slice_' + sid + '.png')
    #
    #     outputGrayFile = join(OUTPUT_PATH, stain + '_gray_affine', 'slice_' + sid + '.png')
    #     outputMaskFile = join(OUTPUT_PATH, stain + '_masks_affine', 'slice_' + sid + '.png')
    #
    #     affineFile = join(OUTPUT_PATH, stain + '_affine', 'slice_' + sid + '.aff')
    #
    #     subprocess.call(
    #         [ALADINcmd, '-ref', refFile, '-flo', floGrayFile, '-aff', affineFile, '-res', outputGrayFile,
    #          '-ln', '4', '-lp', '3', '-pad', '0'], stdout=subprocess.DEVNULL)
    #     subprocess.call(
    #         [REScmd, '-ref', refFile, '-flo', floGrayFile, '-trans', affineFile, '-res', outputGrayFile,
    #          '-inter', '3', '-voff'])
    #     subprocess.call(
    #         [REScmd, '-ref', refMaskFile, '-flo', floMaskFile, '-trans', affineFile, '-res', outputMaskFile,
    #          '-inter', '0', '-voff'])
    #
    for slice_num in np.arange(100,0, -1):
        slice_filename = 'slice_' + "{:03d}".format(slice_num) + '.png'
        slice_info = slice_num_dict[stain][slice_filename]
        print(slice_info['tree_pos'], end=' ', flush=True)
        tree_pos = int(slice_info['tree_pos'])
        sid = "{:02d}".format(tree_pos)
        #### Read image and deform
        filename = slice_info['filename']  # 'slice_' + "{:03d}".format(int(slice_info['it_slice'])) + '.png'
        H_filepath = join(DATA_DIR, stain.lower(), 'images_extended', filename)
        H_mask_filepath = join(DATA_DIR, stain.lower(), 'masks_extended', filename)
        H_orig = cv2.imread(H_filepath)
        H_gray = cv2.cvtColor(H_orig, cv2.COLOR_BGR2GRAY)
        H_gray = gaussian_filter((H_gray / 255).astype(np.double), sigma=5)

        M = cv2.imread(H_mask_filepath, flags=0)
        M = (M / np.max(M)) > 0.5
        H_gray[~M] = 0
        M = (255 * M).astype('uint8')
        H_gray = (255 * H_gray).astype('uint8')

        resized_shape = tuple([int(i * HISTO_res / BLOCK_res) for i in H_orig.shape])
        H_gray = cv2.resize(H_gray, (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_LINEAR)
        M = cv2.resize(M, (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_LINEAR)

        img = Image.fromarray(M, mode='L')
        img.save(join(OUTPUT_PATH, stain + '_masks', 'slice_' + sid + '.png'))
        img = Image.fromarray(H_gray, mode='L')
        img.save(join(OUTPUT_PATH, stain + '_gray', 'slice_' + sid + '.png'))

        sid_pre = "{:03d}".format(tree_pos + 1)

        refFile = join(OUTPUT_PATH, stain + '_gray_affine', 'slice_' + sid_pre + '.png')
        refMaskFile = join(OUTPUT_PATH, stain + '_masks_affine', 'slice_' + sid_pre + '.png')

        floGrayFile = join(OUTPUT_PATH, stain + '_gray', 'slice_' + sid + '.png')
        floMaskFile = join(OUTPUT_PATH, stain + '_masks', 'slice_' + sid + '.png')

        outputGrayFile = join(OUTPUT_PATH, stain + '_gray_affine', 'slice_' + sid + '.png')
        outputMaskFile = join(OUTPUT_PATH, stain + '_masks_affine', 'slice_' + sid + '.png')

        affineFile = join(OUTPUT_PATH, stain + '_affine', 'slice_' + sid + '.aff')

        subprocess.call(
            [ALADINcmd, '-ref', refFile, '-flo', floGrayFile, '-aff', affineFile, '-res', outputGrayFile,
             '-ln', '4', '-lp', '3', '-pad', '0'], stdout=subprocess.DEVNULL)
        subprocess.call(
            [REScmd, '-ref', refFile, '-flo', floGrayFile, '-trans', affineFile, '-res', outputGrayFile,
             '-inter', '3', '-voff'])
        subprocess.call(
            [REScmd, '-ref', refMaskFile, '-flo', floMaskFile, '-trans', affineFile, '-res', outputMaskFile,
             '-inter', '0', '-voff'])



    M = cv2.imread(join(OUTPUT_PATH, stain + '_masks_affine', 'slice_0001.png'))
    volume_shape = M.shape[:2] + (nslices,)
    output_volume = np.zeros(volume_shape)
    output_mask = np.zeros(volume_shape)
    velocity_field = np.zeros((2,) + volume_shape)
    displacement_field = np.zeros((2,) + volume_shape)

    for slice_num in range(1, nslices):
        print(slice_num)
        sid_ref = "{:04d}".format(slice_num)
        sid_flo = "{:04d}".format(slice_num + 1)
        # Nonlinear registration
        refFile = join(OUTPUT_PATH, stain + '_gray_affine', 'slice_' + sid_ref + '.png')
        floFile = join(OUTPUT_PATH, stain + '_gray_affine', 'slice_' + sid_flo + '.png')
        refMaskFile = join(OUTPUT_PATH, stain + '_masks_affine', 'slice_' + sid_ref + '.png')
        floMaskFile = join(OUTPUT_PATH, stain + '_masks_affine', 'slice_' + sid_flo + '.png')

        if slice_num == 1:
            output_volume[..., slice_num] = cv2.imread(refFile)[..., 0]
            output_mask[..., slice_num] = cv2.imread(refMaskFile)[..., 0]

        if not exists(refFile) or not exists(floFile): continue

        outputFile = join(tempdir, 'outputFile.png')
        outputMaskFile = join(tempdir, 'outputMaskFile.png')
        nonlinearField = join(tempdir, 'nonlinearField.nii.gz')
        dummyFileNifti = join(tempdir, 'dummyFileNifti.nii.gz')

        subprocess.call([F3Dcmd, '-ref', refFile, '-flo', floFile, '-res', outputFile, '-cpp', dummyFileNifti,
                         '-rmask', refMaskFile, '-sx', str(scontrol[0]), '-sy', str(scontrol[1]), '-ln', '4', '-lp',
                         '3', '--lncc', '7', '-pad', '0',
                         '-vel', '-voff'], stdout=subprocess.DEVNULL)
        subprocess.call([TRANSFORMcmd, '-ref', refFile, '-flow', dummyFileNifti, nonlinearField],
                        stdout=subprocess.DEVNULL)
        subprocess.call(
            [REScmd, '-ref', refMaskFile, '-flo', floMaskFile, '-trans', nonlinearField, '-res', outputMaskFile,
             '-inter', '0', '-voff'], stdout=subprocess.DEVNULL)

        data = Image.open(outputFile)
        output_volume[..., slice_num] = np.array(data) / 254.0

        data = Image.open(outputMaskFile)
        output_mask[..., slice_num] = np.array(data) / 254.0

        YY, XX = np.meshgrid(np.arange(0, volume_shape[0]), np.arange(0, volume_shape[1]), indexing='ij')

        proxy = nib.load(nonlinearField)
        proxyarray = np.transpose(np.squeeze(np.asarray(proxy.dataobj)), [2, 1, 0])
        proxyarray[np.isnan(proxyarray)] = 0
        finalarray = np.zeros_like(proxyarray)
        finalarray[0] = proxyarray[0] - XX
        finalarray[1] = proxyarray[1] - YY
        velocity_field[..., slice_num] = finalarray

        nstep = 7
        flow_x = finalarray[0] / 2 ** nstep
        flow_y = finalarray[1] / 2 ** nstep
        for it_step in range(nstep):
            x = XX + flow_x
            y = YY + flow_y
            incx = bilinear_interpolate(flow_x, x, y)
            incy = bilinear_interpolate(flow_y, x, y)
            flow_x = flow_x + incx.reshape(volume_shape[:2])
            flow_y = flow_y + incy.reshape(volume_shape[:2])

        flow = np.concatenate((flow_y[np.newaxis], flow_x[np.newaxis]))
        displacement_field[..., slice_num] = flow

    img = nib.Nifti1Image(output_volume, HISTO_AFFINE)
    nib.save(img, join(ALGORITHM_DIR, stain + '.nii.gz'))

    img = nib.Nifti1Image(output_mask, HISTO_AFFINE)
    nib.save(img, join(ALGORITHM_DIR, stain + '.mask.nii.gz'))

    img = nib.Nifti1Image(displacement_field, HISTO_AFFINE)
    nib.save(img, join(ALGORITHM_DIR, stain + '.flow.nii.gz'))

    img = nib.Nifti1Image(velocity_field[1], HISTO_AFFINE)
    nib.save(img, join(ALGORITHM_DIR, stain + '.estimated_field_y.res.nii.gz'))

    img = nib.Nifti1Image(velocity_field[0], HISTO_AFFINE)
    nib.save(img, join(ALGORITHM_DIR, stain + '.estimated_field_y.res.nii.gz'))

