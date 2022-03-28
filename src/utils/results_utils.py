import csv
import pdb
from os.path import exists, join
import time

import numpy as np
import nibabel as nib

from setup import *
from src import models, layers

LABELS_TO_WRITE = ['Thalamus', 'Lateral-Ventricle', 'Hippocampus', 'Amygdala', 'Caudate', 'Pallidum', 'Putamen',
                   'Accumbens', 'Inf-Lat-Ventricle']
KEEP_LABELS_STR = ['Background'] + ['Right-' + l for l in LABELS_TO_WRITE] + ['Left-' + l for l in LABELS_TO_WRITE]
UNIQUE_LABELS = np.asarray([lab for labstr, lab in LABEL_DICT.items() if labstr in KEEP_LABELS_STR], dtype=np.uint8)
KEEP_LABELS_IDX = [SYNTHSEG_LUT[ul] for ul in UNIQUE_LABELS]
LABELS_TO_ANALYSIS = ['Lateral-Ventricle', 'Caudate', 'Pallidum', 'Putamen', 'Thalamus', 'Amygdala', 'Hippocampus']

SUBFIELDS_LABELS = [
    0.,  203.,  211.,  212.,  215.,  226.,  233.,  234.,  235.,
    236.,  237.,  238.,  239.,  240.,  241.,  242.,  243.,  244.,
    245.,  246., 7001., 7003., 7005., 7006., 7007., 7008., 7009., 7010., 7015.
]
SUBFIELDS_LABELS = [int(l) for l in SUBFIELDS_LABELS]

def read_FS_volumes(file, labs=None):
    etiv = 0
    fs_vols = {}
    start = False
    with open(file, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ')
        for row in csvreader:
            row_cool = list(filter(lambda x: x != '', row))
            if start is True:
                fs_vols[row_cool[4]] = row_cool[3]

            if 'ColHeaders' in row_cool and start is False:
                start = True
            elif 'EstimatedTotalIntraCranialVol,' in row_cool:
                etiv = float(row_cool[-2].split(',')[0])

    vols = {**{it_l: float(fs_vols[l]) for l, it_l in labs.items() if 'Thalamus' not in l},
            **{it_l: float(fs_vols[l + '-Proper']) for l, it_l in labs.items() if 'Thalamus' in l}}

    return vols, etiv


def write_volume_results(volume_dict, filepath, fieldnames=None, attach_overwrite='a'):
    if fieldnames is None:
        fieldnames = ['TID'] + list(LABEL_DICT.keys())

    write_header = True if (not exists(filepath) or attach_overwrite == 'w') else False
    with open(filepath, attach_overwrite) as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            csvwriter.writeheader()
        if isinstance(volume_dict, list):
            csvwriter.writerows(volume_dict)
        else:
            csvwriter.writerow(volume_dict)


def read_volume_results(filepath, fieldnames=None, label_dict=None):
    if fieldnames is None:
        fieldnames = KEEP_LABELS_STR

    if label_dict is None:
        label_dict=LABEL_DICT

    with open(filepath, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)  # , fieldnames=fieldnames)
        results = {}
        for row in csvreader:
            results[row['TID']] = {}
            for f in fieldnames:
                results[row['TID']][label_dict[f]] = float(row[f])

        return results


def get_vols(seg, res=1, labels=None):
    if labels is None:
        labels = np.unique(seg)

    n_dims = len(seg.shape)
    if isinstance(res, int):
        res = [res] * n_dims
    vol_vox = np.prod(res)

    vols = {}
    for l in labels:
        mask_l = seg == l
        vols[int(l)] = np.round(np.sum(mask_l) * vol_vox, 2)

    return vols


def get_vols_post(post, res=1):
    '''

    :param post: posterior probabilities
    :param res: mm^3 per voxel
    :return:
    '''

    n_labels = post.shape[-1]
    n_dims = len(post.shape[:-1])
    if isinstance(res, int):
        res = [res] * n_dims
    vol_vox = np.prod(res)

    vols = {}
    for l in range(n_labels):
        mask_l = post[..., l]
        mask_l[post[..., l] < 0.05] = 0
        vols[l] = np.round(np.sum(mask_l) * vol_vox, 2)

    return vols


def compute_jacobians(flow):
    jacobian_maps = np.zeros(flow.shape[1:] + (3, 3))

    for it_dim in range(3):
        fmap = flow[it_dim]

        dx = fmap[1:, :, :] - fmap[:-1, :, :]
        dy = fmap[:, 1:, :] - fmap[:, :-1, :]
        dz = fmap[:, :, 1:] - fmap[:, :, :-1]

        jacobian_maps[1:, :, :, it_dim, 0] = dx
        jacobian_maps[:, 1:, :, it_dim, 1] = dy
        jacobian_maps[:, :, 1:, it_dim, 2] = dz
        jacobian_maps[0, :, :, it_dim, 0] = fmap[0]
        jacobian_maps[:, 0, :, it_dim, 1] = fmap[:, 0]
        jacobian_maps[:, :, 0, it_dim, 2] = fmap[:, :, 0]

    jacobian_maps[..., 0, 0] += 1
    jacobian_maps[..., 1, 1] += 1
    jacobian_maps[..., 2, 2] += 1
    return np.linalg.det(jacobian_maps)


class LabelFusion(object):

    def __init__(self, demo_dict, parameter_dict, reg_algorithm='bidir', long_seg_algo='bidir',
                 temp_variance=None, spatial_variance=None, device='cpu'):

        self.demo_dict = demo_dict
        self.p_dict = parameter_dict
        self.reg_algorithm = reg_algorithm
        self.long_seg_algo = long_seg_algo
        self.temp_variance = temp_variance if temp_variance is not None else ['inf']
        self.spatial_variance = spatial_variance if spatial_variance is not None else ['inf']
        self.device = device

    @NotImplementedError
    def compute_label(self, timepoints, image_list, svf_list, age_list, tp, subject, p_dict, model):
        pass


    def prepare_data(self, subject, force_flag, results_dir_sbj):
        timepoints = subject.timepoints
        flow_dir = subject.results_dirs.get_dir('nonlinear_st_' + self.reg_algorithm)

        for tp in timepoints:
            if not exists(join(flow_dir, tp.id + '.svf.nii.gz')):
                print('Subject: ' + str(subject.id) + ' has not SVF maps available. Please, run the ST algorithm before')
                return None, None, None, None, None

        if force_flag:
            timepoints_to_run = timepoints
            for t_var in self.temp_variance:
                for s_var in self.spatial_variance:
                    last_dir = join(results_dir_sbj, str(t_var) + '_' + str(s_var))
                    if exists(join(last_dir, 'vols_st.txt')):
                        os.remove(join(last_dir, 'vols_st.txt'))
        else:
            last_dir = join(results_dir_sbj, str(self.temp_variance[-1]) + '_' + str(self.spatial_variance[-1]))
            timepoints_to_run = list(
                filter(lambda x: not exists(join(last_dir, 'vols_st_' + str(x.id) + '.txt')), timepoints))

        if DEBUG and force_flag: timepoints_to_run = timepoints  # [0:1]

        if not timepoints_to_run:
            vf = join(results_dir_sbj, str(self.temp_variance[-1]) + '_' + str(self.spatial_variance[-1]), 'vols_st.txt')
            if not exists(vf):
                self.write_vols(timepoints, results_dir_sbj)

            print('Subject: ' + str(subject.id) + '. DONE')
            return None, None, None, None, None

        # timepoints_to_run = timepoints[-1:]
        print('  o Reading the input files')
        image_list = {}
        age_list = {}
        # svf_list = {}
        for tp in timepoints:
            # Age
            age_list[tp.id] = float(self.demo_dict[subject.id][tp.id]['AGE'])

            # Data
            seg = tp.load_seg()
            image = tp.load_data()

            # Normalize image
            wm_mask = (seg == 2) | (seg == 41)
            m = np.mean(image[wm_mask])
            image = 110 * image / m
            image_list[tp.id] = image

            del image, seg, wm_mask
            # proxyflow = nib.load(join(flow_dir, tp.id + '.svf.nii.gz'))
            # svf_list[tp.id] = proxyflow

        # Model
        int_steps = self.p_dict['INT_STEPS'] if self.p_dict['FIELD_TYPE'] == 'velocity' else 0
        if int_steps == 0: assert self.p_dict['UPSAMPLE_LEVELS'] == 1
        model = models.RegNet(
            nb_unet_features=[self.p_dict['ENC_NF'], self.p_dict['DEC_NF']],
            inshape=self.p_dict['VOLUME_SHAPE'],
            int_steps=int_steps,
            int_downsize=self.p_dict['UPSAMPLE_LEVELS'],
        )
        model.to(self.device)

        return timepoints_to_run, image_list, flow_dir, age_list, model

    def write_vols(self, timepoints, results_dir_sbj):
        st_vols = {t_var: {sp_var: [] for sp_var in self.spatial_variance} for t_var in self.temp_variance}
        for tp in timepoints:
            for t_var in self.temp_variance:
                for s_var in self.spatial_variance:
                    last_dir = join(results_dir_sbj, str(t_var) + '_' + str(s_var))
                    st_vols_dict = read_volume_results(join(last_dir, 'vols_st_' + str(tp.id) + '.txt'))
                    st_vols_dict = {LABEL_DICT_REVERSE[k]: np.round(v, 2) for k, v in st_vols_dict[tp.id].items()}
                    st_vols_dict['TID'] = tp.id
                    st_vols[t_var][s_var].append(st_vols_dict)

        fieldnames = ['TID'] + list(LABEL_DICT.keys())
        for t_var in self.temp_variance:
            for s_var in self.spatial_variance:
                write_volume_results(st_vols[t_var][s_var],
                                     join(results_dir_sbj, str(t_var) + '_' + str(s_var), 'vols_st.txt'),
                                     fieldnames=fieldnames, attach_overwrite='w')

    def label_fusion(self, subject, suffix='', force_flag=False):
        print('Subject: ' + str(subject.id))
        timepoints = subject.timepoints
        results_dir_sbj = subject.results_dirs.get_dir('longitudinal_segmentation_st_' + self.long_seg_algo) + suffix
        if not exists(results_dir_sbj): os.makedirs(results_dir_sbj)

        timepoints_to_run, image_list, flow_dir, age_list, model = self.prepare_data(subject, force_flag, results_dir_sbj)
        if timepoints_to_run is None:
            return

        print('  o Computing the segmentation')
        st_vols = {t_var: {sp_var: [] for sp_var in self.spatial_variance} for t_var in self.temp_variance}
        for tp in timepoints_to_run:

            # -------- #
            t_0 = time.time()
            print('        - Timepoint ' + tp.id, end=':', flush=True)
            p_label_dict = self.compute_p_label(timepoints, image_list, flow_dir, age_list, tp, subject, self.p_dict,
                                                model)

            if DEBUG:
                unique_labels = np.asarray(list(LABEL_DICT.values()))
            else:
                unique_labels = UNIQUE_LABELS

            proxy = nib.load(tp.init_path['resample'])
            aff = proxy.affine
            pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
            for t_var in self.temp_variance:
                for s_var in self.spatial_variance:
                    if not exists(join(results_dir_sbj, str(t_var) + '_' + str(s_var))):
                        os.makedirs(join(results_dir_sbj, str(t_var) + '_' + str(s_var)))

                    p_label = p_label_dict[t_var][s_var]
                    p_label[np.isnan(p_label)] = 0
                    mask = np.sum(p_label, axis=-1) > 0.5
                    fake_vol = np.argmax(p_label, axis=-1).astype('uint16')
                    true_vol = np.zeros_like(fake_vol)
                    for it_ul, ul in enumerate(unique_labels): true_vol[fake_vol == it_ul] = ul
                    true_vol = true_vol * mask

                    if DEBUG:
                        img = nib.Nifti1Image(p_label, proxy.affine)
                        nib.save(img, join(results_dir_sbj, str(t_var) + '_' + str(s_var), tp.id + '.post.nii.gz'))

                        img = nib.Nifti1Image(true_vol, proxy.affine)
                        nib.save(img, join(results_dir_sbj, str(t_var) + '_' + str(s_var), tp.id + '.nii.gz'))

                    vols = get_vols_post(p_label, res=pixdim)
                    st_vols_dict = {k: vols[[it_ul for it_ul, ul in enumerate(unique_labels) if ul == val][0]] for
                                    k, val in LABEL_DICT.items() if val in unique_labels}
                    st_vols_dict['TID'] = tp.id
                    st_vols[t_var][s_var] = st_vols_dict

                    del p_label, fake_vol, true_vol, mask

            fieldnames = ['TID'] + list(LABEL_DICT.keys())
            for t_var in self.temp_variance:
                for s_var in self.spatial_variance:
                    vols_dir = join(results_dir_sbj, str(t_var) + '_' + str(s_var), 'vols_st_' + str(tp.id) + '.txt')
                    write_volume_results(st_vols[t_var][s_var], vols_dir, fieldnames=fieldnames, attach_overwrite='w')

            del p_label_dict

            t_1 = time.time()

            print(str(t_1 - t_0) + ' seconds.')
            # -------- #

        self.write_vols(timepoints, results_dir_sbj)

        print('Subject: ' + str(subject.id) + '. DONE')


class LabelFusionSubfields(LabelFusion):

    def __init__(self, filekey, demo_dict, parameter_dict, reg_algorithm='bidir', long_seg_algo='bidir',
                 temp_variance=None, spatial_variance=None, device='cpu', suffix=''):
        super().__init__(demo_dict=demo_dict,
                         parameter_dict=parameter_dict,
                         reg_algorithm=reg_algorithm,
                         long_seg_algo=long_seg_algo,
                         temp_variance=temp_variance,
                         spatial_variance=spatial_variance,
                         device=device
                         )

        self.suffix = suffix
        self.filekey = filekey
        self.interp_func = layers.SpatialInterpolation(padding_mode='zeros').to(device)

    def write_vols(self, timepoints, results_dir_sbj):
        st_vols = {t_var: {sp_var: [] for sp_var in self.spatial_variance} for t_var in self.temp_variance}
        st_vols_hard = {t_var: {sp_var: [] for sp_var in self.spatial_variance} for t_var in self.temp_variance}
        for tp in timepoints:
            for t_var in self.temp_variance:
                for s_var in self.spatial_variance:
                    last_dir = join(results_dir_sbj, str(t_var) + '_' + str(s_var))
                    st_vols_dict = read_volume_results(join(last_dir, 'vols_st_' + str(tp.id) + '.txt'),
                                                       fieldnames=list(EXTENDED_SUBFIELDS_LABEL_DICT.keys()),
                                                       label_dict=EXTENDED_SUBFIELDS_LABEL_DICT)
                    st_vols_dict = {EXTENDED_SUBFIELDS_LABEL_DICT_REVERSE[k]: np.round(v, 2) for k, v in st_vols_dict[tp.id].items()}
                    st_vols_dict['TID'] = tp.id
                    st_vols[t_var][s_var].append(st_vols_dict)

                    st_vols_dict = read_volume_results(join(last_dir, 'vols_st_' + str(tp.id) + '.hard.txt'),
                                                       fieldnames=list(EXTENDED_SUBFIELDS_LABEL_DICT.keys()),
                                                       label_dict=EXTENDED_SUBFIELDS_LABEL_DICT)
                    st_vols_dict = {EXTENDED_SUBFIELDS_LABEL_DICT_REVERSE[k]: np.round(v, 2) for k, v in
                                    st_vols_dict[tp.id].items()}
                    st_vols_dict['TID'] = tp.id
                    st_vols_hard[t_var][s_var].append(st_vols_dict)

        fieldnames = ['TID'] + list(EXTENDED_SUBFIELDS_LABEL_DICT.keys())
        for t_var in self.temp_variance:
            for s_var in self.spatial_variance:
                write_volume_results(st_vols[t_var][s_var],
                                     join(results_dir_sbj, str(t_var) + '_' + str(s_var), 'vols_st.txt'),
                                     fieldnames=fieldnames, attach_overwrite='w')

                write_volume_results(st_vols_hard[t_var][s_var],
                                     join(results_dir_sbj, str(t_var) + '_' + str(s_var), 'vols_st.hard.txt'),
                                     fieldnames=fieldnames, attach_overwrite='w')

    def label_fusion(self, subject,  force_flag=False):
        print('Subject: ' + str(subject.id))
        timepoints = subject.timepoints
        results_dir_sbj = subject.results_dirs.get_dir('longitudinal_segmentation_st_' + self.long_seg_algo) + self.suffix
        if not exists(results_dir_sbj): os.makedirs(results_dir_sbj)

        timepoints_to_run, image_list, flow_dir, age_list, model = self.prepare_data(subject, force_flag, results_dir_sbj)
        if timepoints_to_run is None:
            return

        print('  o Computing the segmentation')
        st_vols = {t_var: {sp_var: [] for sp_var in self.spatial_variance} for t_var in self.temp_variance}
        st_vols_hard = {t_var: {sp_var: [] for sp_var in self.spatial_variance} for t_var in self.temp_variance}
        for tp in timepoints_to_run:

            # -------- #
            t_0 = time.time()
            print('        - Timepoint ' + tp.id, end=':', flush=True)
            p_label_dict = self.compute_p_label(timepoints, image_list, flow_dir, age_list, tp, subject, self.p_dict,
                                                model)

            unique_labels = SUBFIELDS_LABELS

            proxy = nib.load(tp.init_path[self.filekey])
            aff = proxy.affine
            pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
            for t_var in self.temp_variance:
                for s_var in self.spatial_variance:
                    if not exists(join(results_dir_sbj, str(t_var) + '_' + str(s_var))):
                        os.makedirs(join(results_dir_sbj, str(t_var) + '_' + str(s_var)))

                    p_label = p_label_dict[t_var][s_var]
                    p_label[np.isnan(p_label)] = 0
                    mask = np.sum(p_label, axis=-1) > 0.5
                    fake_vol = np.argmax(p_label, axis=-1).astype('uint16')
                    true_vol = np.zeros_like(fake_vol)
                    for it_ul, ul in enumerate(unique_labels): true_vol[fake_vol == it_ul] = ul
                    true_vol = true_vol * mask

                    if DEBUG:
                        img = nib.Nifti1Image(p_label, proxy.affine)
                        nib.save(img, join(results_dir_sbj, str(t_var) + '_' + str(s_var), tp.id + '.post.nii.gz'))

                        img = nib.Nifti1Image(true_vol, proxy.affine)
                        nib.save(img, join(results_dir_sbj, str(t_var) + '_' + str(s_var), tp.id + '.nii.gz'))

                    vols = get_vols_post(p_label, res=pixdim)
                    st_vols_dict = {k: vols[[it_ul for it_ul, ul in enumerate(unique_labels) if ul == val][0]] for
                                    k, val in SUBFIELDS_LABEL_DICT.items() if val in unique_labels}
                    st_vols_dict['TotalAM'] = sum([v for k, v in st_vols_dict.items() if EXTENDED_SUBFIELDS_LABEL_DICT[k] in AM_LABEL_DICT.values()])
                    st_vols_dict['TotalHP'] = sum([v for k, v in st_vols_dict.items() if EXTENDED_SUBFIELDS_LABEL_DICT[k] in HP_LABEL_DICT.values()])
                    st_vols_dict['TID'] = tp.id
                    st_vols[t_var][s_var] = st_vols_dict

                    vols = get_vols(true_vol, res=pixdim, labels=unique_labels)
                    st_vols_dict = {k: vols[val] for k, val in SUBFIELDS_LABEL_DICT.items() if val in unique_labels}
                    st_vols_dict['TotalAM'] = sum([v for k, v in st_vols_dict.items() if
                                                   EXTENDED_SUBFIELDS_LABEL_DICT[k] in AM_LABEL_DICT.values()])
                    st_vols_dict['TotalHP'] = sum([v for k, v in st_vols_dict.items() if
                                                   EXTENDED_SUBFIELDS_LABEL_DICT[k] in HP_LABEL_DICT.values()])
                    st_vols_dict['TID'] = tp.id
                    st_vols_hard[t_var][s_var] = st_vols_dict

                    del p_label, fake_vol, true_vol, mask

            fieldnames = ['TID'] + list(EXTENDED_SUBFIELDS_LABEL_DICT.keys())
            for t_var in self.temp_variance:
                for s_var in self.spatial_variance:
                    vols_dir = join(results_dir_sbj, str(t_var) + '_' + str(s_var), 'vols_st_' + str(tp.id) + '.txt')
                    write_volume_results(st_vols[t_var][s_var], vols_dir, fieldnames=fieldnames, attach_overwrite='w')

                    vols_dir = join(results_dir_sbj, str(t_var) + '_' + str(s_var), 'vols_st_' + str(tp.id) + '.hard.txt')
                    write_volume_results(st_vols_hard[t_var][s_var], vols_dir, fieldnames=fieldnames, attach_overwrite='w')

            del p_label_dict

            t_1 = time.time()

            print(str(t_1 - t_0) + ' seconds.')
            # -------- #

        self.write_vols(timepoints, results_dir_sbj)

        print('Subject: ' + str(subject.id) + '. DONE')
