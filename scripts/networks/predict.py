#py
from argparse import ArgumentParser
from os.path import join, exists
from os import makedirs

#external imports
import torch
import nibabel as nib
import numpy as np

#project imports
from src import models, datasets
from src.utils.io_utils import  create_results_dir, worker_init_fn
from src.test import predict_registration
from src.utils.visualization import plot_results
from scripts import config_dev as configFile
from database.data_loader import DataLoader

if __name__ == '__main__':

    ####################################
    ############ PARAMETERS ############
    ####################################
    """ PARSE ARGUMENTS FROM CLI """
    arg_parser = ArgumentParser(description='Computes the prediction of certain models')
    arg_parser.add_argument('--epoch_number', default='FI', help='Load model from the epoch specified')
    arg_parser.add_argument('--model', default='bidir', choices=['standard', 'bidir'])
    arg_parser.add_argument('--subjects', default=None, nargs='+')
    arg_parser.add_argument('--num_pred', default=None, type=int)


    arguments = arg_parser.parse_args()
    epoch_weights = str(arguments.epoch_number)
    model_type = arguments.model
    initial_subject_list = arguments.subjects
    NUM_PRED = arguments.num_pred
    parameter_dict = configFile.CONFIG_REGISTRATION

    use_gpu = False#torch.cuda.is_available() and parameter_dict['USE_GPU']
    device = torch.device("cuda:0" if use_gpu else "cpu")
    kwargs_generator = {'num_workers': 1, 'pin_memory': use_gpu, 'worker_init_fn': worker_init_fn}

    parameter_dict = configFile.get_config_dict((100,100,100))
    parameter_dict['RESULTS_DIR'] = parameter_dict['RESULTS_DIR'] + '_' + model_type
    results_file = join(parameter_dict['RESULTS_DIR'], 'results', 'training_results.csv')
    log_keys = ['loss_registration', 'loss_registration_smoothness',
                'w_loss_registration', 'w_loss_registration_smoothness',
                'loss', 'time_duration (s)']
    plot_results(results_file, keys=log_keys)
    exit()

    ###################################
    ########### DATA LOADER ###########
    ###################################
    print('Loading dataset ...\n')
    timepoints_filter = lambda x: x in ['miriad_257_AD_F_10_MR_1.nii.gz', 'miriad_257_AD_F_01_MR_1.nii.gz']
    data_loader = DataLoader(linear=True, sid_list=initial_subject_list, timepoints_filter=timepoints_filter)
    subject_list = data_loader.subject_list
    number_of_subjects = len(subject_list) if NUM_PRED is None else NUM_PRED
    parameter_dict = configFile.get_config_dict(data_loader.image_shape)
    parameter_dict['RESULTS_DIR'] = parameter_dict['RESULTS_DIR'] + '_' + model_type


    dataset = datasets.RegistrationDataset3D(
        subject_list * 5,
        affine_params=parameter_dict['AFFINE'],
        nonlinear_params=parameter_dict['NONLINEAR'],
        tf_params=parameter_dict['TRANSFORM'],
        da_params=parameter_dict['DATA_AUGMENTATION'],
        norm_params=parameter_dict['NORMALIZATION'],
        mask_dilation=True,
    )

    generator_test = torch.utils.data.DataLoader(
        dataset,
        batch_size=parameter_dict['BATCH_SIZE'],
        shuffle=False,
        **kwargs_generator
    )

    #################################
    ############# MDOEL #############
    #################################
    image_shape = dataset.image_shape

    # Registration Network
    int_steps = parameter_dict['INT_STEPS'] if parameter_dict['FIELD_TYPE'] == 'velocity' else 0
    if int_steps == 0: assert parameter_dict['UPSAMPLE_LEVELS'] == 1

    registration = models.RegNet(
        nb_unet_features=[parameter_dict['ENC_NF'], parameter_dict['DEC_NF']],
        inshape=parameter_dict['VOLUME_SHAPE'],
        int_steps=int_steps,
        int_downsize=parameter_dict['UPSAMPLE_LEVELS'],
    )
    registration = registration.to(device)

    da_model = None#tensor_utils.TensorDeformation(image_shape, parameter_dict['NONLINEAR'].lowres_size, device)

    epoch_results_dir = 'model_checkpoint.' + epoch_weights
    weightsfile = 'model_checkpoint.' + epoch_weights + '.pth'
    checkpoint = torch.load(join(parameter_dict['RESULTS_DIR'], 'checkpoints', weightsfile), map_location=device)
    registration.load_state_dict(checkpoint['state_dict'])
    registration.eval()

    results_dir = join(parameter_dict['RESULTS_DIR'], 'results', epoch_results_dir)
    for bid in data_loader.subject_dict.keys():
        if not exists(join(results_dir, bid)):
            makedirs(join(results_dir, bid))

    ###################################
    ############# RESULTS #############
    ###################################
    print('Writing results')

    create_results_dir(parameter_dict['RESULTS_DIR'])
    plot_results(join(parameter_dict['RESULTS_DIR'], 'results', 'training_results.csv'),
                 keys=['loss_registration', 'loss_registration_smoothness', 'loss'])

    for batch_idx, data_dict in enumerate(generator_test):
        if batch_idx * parameter_dict['BATCH_SIZE'] >= number_of_subjects and batch_idx > 0:
            break

        print(str(batch_idx * parameter_dict['BATCH_SIZE']) + '/' + str(int(number_of_subjects)))

        ref_rid_list = data_dict['rid']
        output_results = predict_registration(data_dict, registration, device, da_model)

        print('   - ', end=' ', flush=True)
        for it_image, rid_image in enumerate(ref_rid_list):
            reg_rid = str(rid_image)
            sbj_rid = reg_rid.split('.')[0]
            slice_rid = reg_rid.split('.')[1]
            ref_rid = slice_rid.split('_to_')[0]
            flo_rid = slice_rid.split('_to_')[1]

            print(reg_rid + ' ', end=' ', flush=True)

            ref, flo, reg_r, reg_f = output_results[:4]
            img = nib.Nifti1Image(ref[0], data_loader.subject_dict[sbj_rid].vox2ras0)
            nib.save(img, join(results_dir, sbj_rid, ref_rid + '.nii.gz'))

            img = nib.Nifti1Image(flo[0], data_loader.subject_dict[sbj_rid].vox2ras0)
            nib.save(img, join(results_dir, sbj_rid, flo_rid + '.nii.gz'))

            img = nib.Nifti1Image(reg_r[0], data_loader.subject_dict[sbj_rid].vox2ras0)
            nib.save(img, join(results_dir, sbj_rid, slice_rid + '.nii.gz'))

            img = nib.Nifti1Image(reg_f[0], data_loader.subject_dict[sbj_rid].vox2ras0)
            nib.save(img, join(results_dir, sbj_rid, flo_rid + '_to_' + ref_rid + '.nii.gz'))

            img = nib.Nifti1Image(np.transpose(np.squeeze(output_results[-1]), axes=[1, 2, 3, 0]),
                                  data_loader.subject_dict[sbj_rid].vox2ras0)
            nib.save(img, join(results_dir, sbj_rid, slice_rid + '.flow.nii.gz'))

        print('\n')

    print('Predicting done.')
