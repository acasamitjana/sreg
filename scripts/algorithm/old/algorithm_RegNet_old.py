from os.path import exists
from os import makedirs
import time
from argparse import ArgumentParser

# project imports
from database.data_loader import DataLoader
from database.databaseConfig import FCIEN_DB
from src.utils import algorithm_utils
from src.utils.image_utils import deform3D
from scripts import config_dev as configFile, config_data
from src.algorithm import *

parameter_dict = configFile.CONFIG_DICT['BASE']
SUBJECT_LIST = ['012', '001', '002', '005', '006', '010', '011', '013', '014', '015', '024', '031', '033', '034',
                '041', '042','043', '045', '046', '052', '053', '057', '063', '065', '071', '075', '076', '090',
                '091', '095', '098', '103', '105', '106', '107', '108', '114', '116', '119', '121', '122', '135',
                '143', '145', '149' ,'152', '161', '168', '169', '170', '173', '176', '181', '187',  '191', '200',
                '204', '233', '235', '241', '246', '249', '309', '354', '361', '371', '439']

if __name__ == '__main__':

    # Parameters
    arg_parser = ArgumentParser(description='Computes the prediction of certain models')
    arg_parser.add_argument('--cost', type=str, default='l1', choices=['l1', 'l2'], help='Likelihood cost function')
    arg_parser.add_argument('--mdil', type=int, default=7, help='Mask dilation factor')
    arg_parser.add_argument('--subject', default=None, choices=SUBJECT_LIST, nargs='+')

    arguments = arg_parser.parse_args()
    cost = arguments.cost
    mdil = arguments.mdil
    subject_list = arguments.subject if arguments.subject is not None else SUBJECT_LIST

    observations_dir = config_data.OBSERVATIONS_DIR_REGNET
    algorithm_dir = config_data.ALGORITHM_DIR
    results_dir = join(algorithm_dir, 'ST_RegNet', cost)
    if not exists(results_dir):
        makedirs(results_dir)

    data_loader = DataLoader(FCIEN_DB, rid_list=subject_list)
    subject_list = data_loader.subject_list

    print('[START] Processing')
    for it_subject, subject in enumerate(subject_list):
        print('   - Subject: ' + str(subject.id))

        if not exists(subject.template):
            continue

        subject_shape = subject.image_shape
        slice_list = subject.slice_list
        nslices = len(subject.slice_list)

        if nslices == 1:
            continue



        input_dir = join(observations_dir, subject.id)
        subject_dir = join(parameter_dict['DB_CONFIG']['DATA_PATH'], subject.id)

        if not exists(join(input_dir, slice_list[-2].id + '_to_' + slice_list[-1].id + '.field_x.nii.gz')):
            print('[WARNING] No observations found for subject ' + subject.id + ' and NiftyReg ')
            continue
        else:
            proxy = nib.load(join(input_dir, slice_list[-2].id + '_to_' + slice_list[-1].id + '.field_x.nii.gz'))
            cp_shape = proxy.shape[:3]

        results_dir_sbj = join(results_dir, subject.id)
        if not exists(join(results_dir_sbj)):
            makedirs(results_dir_sbj)
        elif exists(join(results_dir_sbj, slice_list[-1].id + '.nii.gz')):
            print('[DONE] Subject ' + subject.id + ' has already been processed')
            continue

        ####################################################################################################
        ####################################################################################################

        print('[' + str(subject.id) + ' - Init Graph] Reading SVFs ...')
        t_init = time.time()

        graph_structure = init_st2(slice_list, subject_dir, input_dir, cp_shape, se=np.ones((mdil, mdil, mdil)))
        R, M, W, NK = graph_structure
        print('[' + str(subject.id) + ' - Init Graph] Total Elapsed time: ' + str(time.time() - t_init))

        print('[' + str(subject.id) + ' - ALGORITHM] Running the algorithm ...')
        t_init = time.time()
        if cost == 'l2':
            Tres = st2_L2_global(R, W, nslices)

        else:
            Tres = st2_L1(R, M, W, nslices)

        for it_sl, sl in enumerate(slice_list):
            img = nib.Nifti1Image(Tres[0, ..., it_sl], subject.vox2ras0)
            nib.save(img, join(results_dir_sbj, sl.id + '.field_x.nii.gz'))
            img = nib.Nifti1Image(Tres[1, ..., it_sl], subject.vox2ras0)
            nib.save(img, join(results_dir_sbj, sl.id + '.field_y.nii.gz'))
            img = nib.Nifti1Image(Tres[2, ..., it_sl], subject.vox2ras0)
            nib.save(img, join(results_dir_sbj, sl.id + '.field_z.nii.gz'))

        print('[' + str(subject.id) + ' - ALGORITHM] Total Elapsed time: ' + str(time.time() - t_init))

        ####################################################################################################
        ####################################################################################################

        print('[' + str(subject.id) + ' - INTEGRATION] Computing deformation field ... ')
        t_init = time.time()
        for it_sl, sl in enumerate(slice_list):

            flow = algorithm_utils.integrate_RegNet(Tres[..., it_sl], subject_shape, parameter_dict)

            # proxy = nib.load(join(results_dir_sbj, sl.id + '.field_x.nii.gz'))
            # Tx = np.asarray(proxy.dataobj)
            #
            # proxy = nib.load(join(results_dir_sbj, sl.id + '.field_y.nii.gz'))
            # Ty = np.asarray(proxy.dataobj)
            #
            # proxy = nib.load(join(results_dir_sbj, sl.id + '.field_z.nii.gz'))
            # Tz = np.asarray(proxy.dataobj)
            #
            # flow = algorithm_utils.integrate_RegNet(np.concatenate((Tx[np.newaxis],
            #                                                         Ty[np.newaxis],
            #                                                         Tz[np.newaxis]),axis=0), subject_shape, parameter_dict)

            img = nib.Nifti1Image(flow, subject.vox2ras0)
            nib.save(img, join(results_dir_sbj, sl.id + '.flow.nii.gz'))
            del flow

        print('[' + str(subject.id) + ' - INTEGRATION] Total Elapsed time: ' + str(time.time() - t_init))

        ####################################################################################################
        ####################################################################################################
        t_init = time.time()
        print('[' + str(subject.id) + ' - DEFORM] Deforming images ... ')

        for it_sl, sl in enumerate(slice_list):
            proxy = nib.load(join(results_dir_sbj, sl.id + '.flow.nii.gz'))
            flow = np.asarray(proxy.dataobj)

            mri = sl.load_data()
            image_deformed = deform3D(mri, flow)
            img = nib.Nifti1Image(image_deformed, subject.vox2ras0)
            nib.save(img, join(results_dir_sbj, sl.id + '_image.nii.gz'))
            del mri, image_deformed

            mask = sl.load_mask()
            mask_deformed = deform3D(mask, flow)
            img = nib.Nifti1Image(mask_deformed, subject.vox2ras0)
            nib.save(img, join(results_dir_sbj, sl.id + '_mask.nii.gz'))
            del mask, mask_deformed

            labels = sl.load_aseg()
            labels_deformed = deform3D(labels, flow)
            img = nib.Nifti1Image(labels_deformed, subject.vox2ras0)
            nib.save(img, join(results_dir_sbj, sl.id + '_aseg.nii.gz'))
            del labels, labels_deformed, flow

        print('[' + str(subject.id) + ' - DEFORM] Total Elapsed time: ' + str(time.time() - t_init))

