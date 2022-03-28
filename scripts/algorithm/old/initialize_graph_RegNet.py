# imports
from os.path import join, exists
from os import makedirs
import time
from argparse import ArgumentParser
import itertools

# third party imports
import numpy as np
import torch
import nibabel as nib

# project imports
from database.data_loader import DataLoader
from database.databaseConfig import FCIEN_DB
from src import datasets, models
from scripts import config_dev as configFile, config_data
from src.utils.algorithm_utils import initialize_graph_RegNet


#####################
# Global parameters #
#####################

SCONTROL = [-8, -8, -8]
results_dir = config_data.OBSERVATIONS_DIR_REGNET
tempdir = join(results_dir, 'tmp')
if not exists(tempdir):
    makedirs(tempdir)
parameter_dict = configFile.CONFIG_REGISTRATION


SUBJECT_LIST = ['014', '012', '001','002', '005', '006', '010', '011', '013', '015', '024', '031', '033', '034',
                '041', '042','043', '045', '046', '052', '053', '057', '063', '065', '071', '075', '076', '090',
                '091', '095', '098', '103', '105', '106', '107', '108', '114', '116', '119', '121', '122', '135',
                '143', '145', '149' ,'152', '161', '168', '169', '170', '173', '176', '181', '187',  '191', '200',
                '204', '233', '235', '241', '246', '249', '309', '354', '361', '371', '439']


arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--model', default='standard', choices=['standard', 'bidir'])
arg_parser.add_argument('--subject', default=None, choices=SUBJECT_LIST, nargs='+')

arguments = arg_parser.parse_args()
model_type = arguments.model
subject_list = arguments.subject if arguments.subject is not None else SUBJECT_LIST

parameter_dict['RESULTS_DIR'] = parameter_dict['RESULTS_DIR'] + '_' + model_type
use_gpu = parameter_dict['USE_GPU'] and torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

kwargs_testing = {}
kwargs_generator = {'num_workers': 1, 'pin_memory': True} if use_gpu else {}

###################
# Tree parameters #
###################

print('Loading dataset ...\n')
data_loader = DataLoader(FCIEN_DB, rid_list=subject_list)
subject_list = [data_loader.subject_dict[S] for S in subject_list]#data_loader.subject_list


########################################
########### Run registration ###########
########################################
missing_subjects = []
for it_subject, subject in enumerate(subject_list):
    slice_list = subject.sid_list
    if len(slice_list) == 1:
        print(subject.id)
        continue

    if not exists(subject.get_timepoint('00').image_path):
        missing_subjects.append(subject.id)
        print(subject.get_timepoint('00').image_path)
        continue

    for sid_ref, sid_flo in itertools.combinations(slice_list, 2):

        print('Registering subject: ' + subject.rid + '. From T=' + str(sid_ref) + ' to T=' + str(sid_flo) + '.',
              end = ' ', flush=True)


        dataset = datasets.RegistrationDataset3D(
            [[subject.get_timepoint(sid_ref), subject.get_timepoint(sid_flo)]],
            tf_params=parameter_dict['TRANSFORM'],
            norm_params=parameter_dict['NORMALIZATION'],
        )

        generator = torch.utils.data.DataLoader(
            dataset,
            batch_size=parameter_dict['BATCH_SIZE'],
            shuffle=False,
            **kwargs_generator
        )
        ################
        # Registration #
        ################
        filename = str(sid_ref) + '_to_' + str(sid_flo)
        results_dir_sbj = join(results_dir, subject.rid)
        if not exists(results_dir_sbj): makedirs(results_dir_sbj)
        # if exists(join(results_dir_sbj, filename + '.field_z.nii.gz')):
        #     continue

        image_shape = dataset.image_shape
        int_steps = 7
        model = models.RegNet(
            nb_unet_features=[parameter_dict['ENC_NF'], parameter_dict['DEC_NF']],
            inshape=parameter_dict['VOLUME_SHAPE'],
            int_steps=int_steps,
            int_downsize=parameter_dict['UPSAMPLE_LEVELS'],
        )

        model = model.to(device)
        weightsfile = 'model_checkpoint.181.pth'
        checkpoint = torch.load(join(parameter_dict['RESULTS_DIR'], 'checkpoints', weightsfile), map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        t_init = time.time()
        output_results = initialize_graph_RegNet(model, generator, image_shape, device)

        registered_image, registered_mask, velocity_field, displacement_field = output_results

        print('Elapsed time: ' + str(np.round(time.time() - t_init, 2)))

        # Update affine (due to padding/cropping)

        vox2ras0 = subject.vox2ras0
        diff_shape = [image_shape[it_d] - subject.image_shape[it_d] for it_d in range(3)]
        if sum(np.abs(diff_shape)) > 0:
            tx = np.eye(4)
            tx[0, -1] = -(diff_shape[0] // 2)
            tx[1, -1] = -(diff_shape[1] // 2)
            tx[2, -1] = -(diff_shape[2] // 2)
            vox2ras0 = vox2ras0 @ tx

        # Save output forward tree
        img = nib.Nifti1Image(registered_image, vox2ras0)
        nib.save(img, join(results_dir_sbj, filename + '.nii.gz'))

        img = nib.Nifti1Image(registered_mask, vox2ras0)
        nib.save(img, join(results_dir_sbj, filename + '.mask.nii.gz'))

        img = nib.Nifti1Image(velocity_field[0], vox2ras0)
        nib.save(img, join(results_dir_sbj, filename + '.field_x.nii.gz'))

        img = nib.Nifti1Image(velocity_field[1], vox2ras0)
        nib.save(img, join(results_dir_sbj, filename + '.field_y.nii.gz'))

        img = nib.Nifti1Image(velocity_field[2], vox2ras0)
        nib.save(img, join(results_dir_sbj, filename + '.field_z.nii.gz'))

print(missing_subjects)
