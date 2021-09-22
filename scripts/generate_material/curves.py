import pdb
from os.path import join, exists
from os import makedirs

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from skimage.transform import resize as imresize

# project imports
from scripts import config_data
from database.data_loader_pre import DataLoader
from database.databaseConfig import TEST_DB
from src.utils.image_utils import deform3D
from src.utils.algorithm_utils import integrate_NR

def compute_jacobian(field):
    jac_matrix = np.zeros(field.shape[1:] + (3,3))
    for it_func in range(3):
        jac_matrix[1:, :, :, it_func, 0] = np.diff(field[it_func], axis=0)
        jac_matrix[:, 1:, :, it_func, 1] = np.diff(field[it_func], axis=1)
        jac_matrix[:, :, 1:, it_func, 2] = np.diff(field[it_func], axis=2)

    return np.linalg.det(jac_matrix)

def compute_curves(flow, t_ages, ages, image_shape, masklist=None):
    curves = np.zeros((3,) + image_shape + (len(t_ages),))
    a0 = ages[0]
    offset = 0
    for it_a1, a1 in enumerate(ages[1:]):
        idx = np.where((t_ages >= a0) & (t_ages < a1))
        m = np.min(idx[0])
        M = np.max(idx[0])
        field = integrate_NR(flow[it_a1] - flow[it_a1 - 1], image_shape=image_shape)
        jacobian_det = compute_jacobian(field)
        for it_i in range(m, M+1):
            curves[..., it_i+1] = jacobian_det*(t_ages[it_i+1]-a0) + offset
        offset = curves[..., M]
        a0 = a1

    return curves

OUTPUT_DIR = join(config_data.ALGORITHM_DIR, 'ST_NR', 'l1')
SUBJECT = '024_S_2239'
curves_path_x = join(OUTPUT_DIR, SUBJECT, "curves_x.png")
curves_path_y = join(OUTPUT_DIR, SUBJECT, "curves_y.png")
curves_path_z = join(OUTPUT_DIR, SUBJECT, "curves_z.png")
curves_path = {0: curves_path_x, 1: curves_path_y, 2: curves_path_z}
##############
# Parameters #
##############

# data used
data_loader = DataLoader(TEST_DB, rid_list=SUBJECT)
subject = data_loader.subject_list[0]
image_shape = subject.image_shape

print('Reading image volumes and transforms')
mask_list = []
T = []
ages = []
interesting_labels = [17, 53] # None
for timep in subject.slice_list:

    ages.append(timep.demodict['AGE'])

    proxy = nib.load(join(OUTPUT_DIR, subject.id, timep.id + '.field_x.nii.gz'))
    svfx = np.asarray(proxy.dataobj)
    proxy = nib.load(join(OUTPUT_DIR, subject.id, timep.id + '.field_y.nii.gz'))
    svfy = np.asarray(proxy.dataobj)
    proxy = nib.load(join(OUTPUT_DIR, subject.id, timep.id + '.field_z.nii.gz'))
    svfz = np.asarray(proxy.dataobj)

    svf = np.concatenate((svfx[np.newaxis], svfy[np.newaxis], svfz[np.newaxis]), axis=0)
    T.append(svf)

    if interesting_labels is not None:
        mask = timep.load_labels()
        mask = np.logical_or.reduce([mask == l for l in interesting_labels])
        # mask = imresize(mask, output_shape=svfz.shape, order=1) > 0.5
        mask_list.append(mask)

# age_resolution=0.1
# t_ages = np.unique(ages + np.around(np.arange(np.min(ages), np.max(ages), age_resolution),3).tolist())
ages = np.asarray(ages)
t_ages = ages
N = len(ages)
T_shape = T[0].shape
curves_full = compute_curves(T, t_ages, ages, image_shape, mask_list)
curves_full = np.transpose(curves_full.reshape((3, -1, len(t_ages))), axes=[0,2,1])
mask = mask_list[0] if interesting_labels is not None else np.sum(np.sum(curves_full, axis=0), axis=0) > 0
mask = mask.reshape(-1)
curves = curves_full[..., mask]
pdb.set_trace()
NC=8
features = np.sqrt(np.diff(curves[0]) ** 2 + np.diff(curves[1]) ** 2 + np.diff(curves[2]) ** 2)
cluster = KMeans(n_clusters=NC)
labels = cluster.fit_predict(features.T)
for it_dim in range(T_shape[0]):
    print(it_dim)
    c_i = curves[it_dim]

    fig, ax = plt.subplots(2, 4)
    for it_c in range(NC):
        col = np.mod(it_c, 4)
        row = it_c // 4

        idx = np.where(labels == it_c)
        ax[row, col].plot(t_ages, c_i[..., idx[0]], 'r')
        ax[row, col].set_ylim(np.min(c_i), np.max(c_i))

    plt.savefig(curves_path[it_dim])
    # y = curves[it_dim].reshape(-1, len(t_ages)).T
    # plt.figure()
    # # for it_y in range(y.shape[0]):
    # plt.plot(t_ages, y[:,mask])
    # plt.savefig(curves_path[it_dim])