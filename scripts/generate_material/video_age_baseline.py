from os.path import join, exists
from os import makedirs

import nibabel as nib
import numpy as np
from skimage.transform import rotate as imrotate, rescale as imrescale
from PIL import Image
import cv2

# project imports
from scripts import config_data
from database.data_loader import DataLoader
from database.databaseConfig import FCIEN_DB
from src.utils.image_utils import deform3D
from src.utils.algorithm_utils import integrate_NR


def compute_frames(data_list, T, ages, framedir, interp_method):
    def function(v_age_i, it_v_age_i):
        framefile = join(framedir, 'frame_' + "{:03d}".format(it_v_age_i) + '.png')
        print(str(it_v_age_i), end=' ', flush=True)
        if exists(framefile):
            return

        idx = np.where(ages >= v_age_i)
        i_right = np.min(idx[0])
        idx = np.where(ages <= v_age_i)
        i_left = np.max(idx[0])

        if i_left == i_right:
            F_vol = np.asarray(data_list[i_left].tolist())

        else:
            w_right = (v_age_i - ages[i_left]) / (ages[i_right] - ages[i_left])
            w_left = 1 - w_right
            R = -np.squeeze(T[i_left])
            I_left = deform3D(data_list[i_left], integrate_NR(-R, image_shape=image_shape, int_end=w_right), interp_method)
            I_right = deform3D(data_list[i_right], integrate_NR(R, image_shape=image_shape, int_end=w_left), interp_method)
            F_vol = w_left * I_left + w_right * I_right

        order = 1 if interp_method == 'bilinear' else 0
        AX = imrotate(np.flipud(F_vol[..., axial_slice]), 90, order=order).astype('uint8')
        SA = imrotate(F_vol[:, sagittal_slice], 90, order=order).astype('uint8')
        CO = imrotate(F_vol[coronal_slice], 90, order=order).astype('uint8')

        maxsiz = (max([AX.shape[0], CO.shape[0], SA.shape[0]]), max([AX.shape[1], CO.shape[1], SA.shape[1]]))

        F = np.zeros((maxsiz[0], 3 * maxsiz[1]), dtype='uint8')
        F[:AX.shape[0], :AX.shape[1]] = AX
        F[:SA.shape[0], maxsiz[1]: maxsiz[1] + SA.shape[1]] = SA
        F[:CO.shape[0], 2 * maxsiz[1]: 2 * maxsiz[1] + CO.shape[1]] = CO

        FF = imrescale(F, 2, order=order)

        if interp_method == 'bilinear':
            img = Image.fromarray((255 * FF).astype('uint8'), mode='L')
        else:
            raise ValueError("Still not ready to save ASEG or APARC files")

        img.save(framefile)

    return function

SAVE_VIDEO = True
SHOW_DIFFERENCE_MAP = False
OUTPUT_DIR = join(config_data.OBSERVATIONS_DIR_NR)
SUBJECT = '012'
video_path = join(OUTPUT_DIR, SUBJECT, "video_v0.mp4")
data_type = 'image'

####################
# Movie parameters #
####################

# data used
data_loader = DataLoader(FCIEN_DB, rid_list=SUBJECT)
subject = data_loader.subject_list[0]
image_shape = subject.image_shape

# slices for the animation
coronal_slice = 112
sagittal_slice = 106
axial_slice = 103

# video features
age_resolution = 0.05
framerate_fps = 15
show_difference_map = True
font = cv2.FONT_HERSHEY_SIMPLEX# describe the type of font to be used.

#number of parallel workers
n_workers_video = 4

# directories
video_file = join(OUTPUT_DIR, SUBJECT, 'video_c' + str(coronal_slice) + '_s_' + str(sagittal_slice) + '_a_' + str(axial_slice) + '.avi')
framedir = join(OUTPUT_DIR, SUBJECT, 'frames_' + data_type +'_c' + str(coronal_slice) + '_s_' + str(sagittal_slice) + '_a_' + str(axial_slice))
if not exists(framedir): makedirs(framedir)
if exists(video_file):
    print('Video file already there; skipping ...')
    exit()

print('Reading image volumes and transforms')
data_list = []
T = []
ages = []
for timep in subject.slice_list:

    ages.append(timep.demodict['AGE'])

    if data_type == 'image':
        data_list.append(timep.load_data()*1.5) #extra brightness
    elif data_type == 'aseg':
        data_list.append(timep.load_aseg())
    elif data_type == 'aparc+aseg':
        data_list.append(timep.load_aparc_aseg())
    else:
        raise ValueError("Not a valide /*data_type*/")

    if timep == subject.slice_list[-1]:
        continue
    proxy = nib.load(join(OUTPUT_DIR, subject.id, timep.id + '_to_' +  "{:02d}".format(int(timep.id) + 1) + '.field_x.nii.gz'))
    svfx = np.asarray(proxy.dataobj)
    proxy = nib.load(join(OUTPUT_DIR, subject.id, timep.id + '_to_' +  "{:02d}".format(int(timep.id) + 1) + '.field_y.nii.gz'))
    svfy = np.asarray(proxy.dataobj)
    proxy = nib.load(join(OUTPUT_DIR, subject.id, timep.id + '_to_' +  "{:02d}".format(int(timep.id) + 1) + '.field_z.nii.gz'))
    svfz = np.asarray(proxy.dataobj)

    svf = np.concatenate((svfx[np.newaxis], svfy[np.newaxis], svfz[np.newaxis]), axis=0)
    T.append(svf)

N = len(ages)
v_ages = np.unique(ages + np.around(np.arange(np.min(ages), np.max(ages), age_resolution),3).tolist())
ages = np.asarray(ages)

interp_method = 'bilinear' if data_type == 'image' else 'nearest'

print('Working on frame N=(' + str(len(v_ages)) + ')', end=' ', flush=True)
from joblib import delayed, Parallel
compute_frames_function = compute_frames(data_list, T, ages, framedir, interp_method)
r = Parallel(n_jobs=8)(delayed(compute_frames_function)(v_age_i, it_v_age_i) for it_v_age_i, v_age_i in enumerate(v_ages))

if 'maxsiz' not in locals():
    frame = cv2.imread(join(framedir, 'frame_' + "{:03d}".format(1) + '.png'))
    maxsiz = frame.shape

print('\nComposing frames into video')
fourcc = cv2.VideoWriter.fourcc(*"mp4v")
if SHOW_DIFFERENCE_MAP:
    video = cv2.VideoWriter(video_path, fourcc, fps=framerate_fps, frameSize=(maxsiz[1], 2*maxsiz[0]), isColor=True)
else:
    video = cv2.VideoWriter(video_path, fourcc, fps=framerate_fps, frameSize=(maxsiz[1], maxsiz[0]), isColor=True)

if SAVE_VIDEO:
    for it_v_age_i, v_age_i in enumerate(v_ages):
        framefile = join(framedir, 'frame_' + "{:03d}".format(it_v_age_i) + '.png')
        frame = cv2.imread(framefile)
        if len(frame.shape) == 2:
            frame = np.tile(frame[..., np.newaxis], 3)

        if SHOW_DIFFERENCE_MAP:
            framediff = 128 * np.ones(frame.shape, dtype='uint8')
            if it_v_age_i > 0:
                diffmap = 128 + 1 / age_resolution * np.mean(np.double(frame_prev) - np.double(frame), axis=-1)
                for it_c in range(3):
                    framediff[..., it_c] = diffmap.astype('uint8')

            image = np.concatenate((frame, framediff), axis=0)
            frame_prev = frame
        else:
            image = frame
            frame_prev = frame

        text = 'Age: ' + str(np.round(v_age_i, 2))
        text2 = ''
        if v_age_i in ages:
            text2 += 'Timepoint ' + str(np.where(v_age_i == ages)[0][0])

        cv2.putText(
            image,
            text,
            (700, 100), #cols, rows
            font,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_4,
        )

        cv2.putText(
            image,
            text2,
            (675, 450),  # cols, rows
            font,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_4,
        )

        video.write(image)
        if it_v_age_i == 0:
            for it in range(framerate_fps):
                    video.write(image)
        if v_age_i in ages:
            for it in range(framerate_fps//2):
                video.write(image)

    video.release()
    cv2.destroyAllWindows()
    cap = cv2.VideoCapture(video_path)