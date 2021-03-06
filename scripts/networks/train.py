#py
from argparse import ArgumentParser
from os.path import join
import time

#external imports
import torch

#project imports
from src import losses, models, datasets
from src.utils.io_utils import ResultsWriter, create_results_dir, ExperimentWriter, worker_init_fn
from src.callbacks import History, ModelCheckpoint, PrinterCallback, ToCSVCallback, LRDecay
from src.training import train, train_bidirectional
from src.utils.visualization import plot_results
from scripts import config_dev as configFile
from database.data_loader import DataLoader
from setup import *

##############
# Parameters #
##############

if __name__ == '__main__':

    arg_parser = ArgumentParser(description='Computes the prediction of certain models')
    arg_parser.add_argument('--model', default='standard', choices=['standard', 'bidir'])
    arg_parser.add_argument('--mask', action='store_true')
    arg_parser.add_argument('--subjects', default=None, nargs='+')

    arguments = arg_parser.parse_args()
    mask_flag = arguments.mask
    model_type = arguments.model
    initial_subject_list = arguments.subjects

    parameter_dict = configFile.CONFIG_REGISTRATION

    use_gpu = torch.cuda.is_available() and parameter_dict['USE_GPU']
    device = torch.device("cuda:0" if use_gpu else "cpu")

    kwargs_training = {'mask_flag': mask_flag}
    kwargs_generator = {'num_workers': 4, 'pin_memory': use_gpu, 'worker_init_fn': worker_init_fn}

    attach = True if parameter_dict['STARTING_EPOCH'] > 0 else False

    ###################################
    ########### DATA LOADER ###########
    ###################################
    data_loader = DataLoader(linear=True, sid_list=initial_subject_list, reg_algorithm=model_type + str(parameter_dict['LOSS_REGISTRATION_SMOOTHNESS']['lambda']))
    final_subject_list = list(filter(lambda s: len(s.timepoints) > 4, data_loader.subject_list))
    data_loader = DataLoader(linear=True, sid_list=[s.id for s in final_subject_list], reg_algorithm=model_type + str(parameter_dict['LOSS_REGISTRATION_SMOOTHNESS']['lambda']))
    subject_list = data_loader.subject_list
    if DB == 'MIRIAD_retest':
        subject_list = list(filter(lambda x: '231' not in x.id, subject_list))

    parameter_dict = configFile.get_config_dict(data_loader.image_shape)

    parameter_dict['RESULTS_DIR'] = parameter_dict['RESULTS_DIR'] + '_' + model_type
    create_results_dir(parameter_dict['RESULTS_DIR'])
    resultsWriter = ResultsWriter(join(parameter_dict['RESULTS_DIR'], 'experiment_parameters.txt'), attach=attach)
    experimentWriter = ExperimentWriter(join(parameter_dict['RESULTS_DIR'], 'experiment.txt'), attach=attach)
    experimentWriter.write('Loading dataset ...\n')

    resultsWriter.write('Experiment parameters\n')
    for key, value in parameter_dict.items():
        resultsWriter.write(key + ': ' + str(value))
        resultsWriter.write('\n')
    resultsWriter.write('\n')

    dataset = datasets.RegistrationDataset3D(
        subject_list,
        affine_params=parameter_dict['AFFINE'],
        nonlinear_params=parameter_dict['NONLINEAR'],
        tf_params=parameter_dict['TRANSFORM'],
        da_params=parameter_dict['DATA_AUGMENTATION'],
        norm_params=parameter_dict['NORMALIZATION'],
        mask_dilation=True,
    )

    sampler = datasets.IntraModalTrainingSampler(
        subject_list
    )
    dataset.N = sampler.N

    generator_train = torch.utils.data.DataLoader(
        dataset,
        batch_size=parameter_dict['BATCH_SIZE'],
        sampler=sampler,
        **kwargs_generator
    )
    # import pdb
    # for idx in sampler:
    #     data_dict = dataset[idx]
    #     pdb.set_trace()

    #################################
    ############# MODEL #############
    #################################
    experimentWriter.write('Loading model ...\n')
    image_shape = dataset.image_shape
    da_model = models.TensorDeformation(image_shape, parameter_dict['NONLINEAR'].get_lowres_size(image_shape), device)

    int_steps = parameter_dict['INT_STEPS'] if parameter_dict['FIELD_TYPE'] == 'velocity' else 0
    if int_steps == 0: assert parameter_dict['UPSAMPLE_LEVELS'] == 1

    registration = models.RegNet(
        nb_unet_features=[parameter_dict['ENC_NF'], parameter_dict['DEC_NF']],
        inshape=parameter_dict['VOLUME_SHAPE'],
        int_steps=int_steps,
        int_downsize=parameter_dict['UPSAMPLE_LEVELS'],
    )

    registration = registration.to(device)
    optimizer = torch.optim.Adam(registration.parameters(), lr=parameter_dict['LEARNING_RATE'])

    if parameter_dict['STARTING_EPOCH'] > 0:
        weightsfile = 'model_checkpoint.' + str(parameter_dict['STARTING_EPOCH'] - 1) + '.pth'
        checkpoint = torch.load(join(parameter_dict['RESULTS_DIR'], 'checkpoints', weightsfile))
        optimizer.load_state_dict(checkpoint['optimizer'])
        registration.load_state_dict(checkpoint['state_dict'])

    # Losses
    reg_loss = losses.DICT_LOSSES[parameter_dict['LOSS_REGISTRATION']['name']]
    reg_loss = reg_loss(name='registration', device=device, **parameter_dict['LOSS_REGISTRATION']['params'])

    reg_smooth_loss = losses.DICT_LOSSES[parameter_dict['LOSS_REGISTRATION_SMOOTHNESS']['name']]
    reg_smooth_loss = reg_smooth_loss(name='registration_smoothness', loss_mult=parameter_dict['UPSAMPLE_LEVELS'],
                                      **parameter_dict['LOSS_REGISTRATION_SMOOTHNESS']['params'])

    loss_function_dict = {
        'registration': reg_loss,
        'registration_smoothness': reg_smooth_loss,
    }
    loss_weights_dict = {
        'registration': parameter_dict['LOSS_REGISTRATION']['lambda'],
        'registration_smoothness': parameter_dict['LOSS_REGISTRATION_SMOOTHNESS']['lambda'],
    }

    experimentWriter.write('Model ...\n')
    for name, param in registration.named_parameters():
        if param.requires_grad:
            experimentWriter.write(name + '. Shape:' + str(torch.tensor(param.data.size()).numpy()))
            experimentWriter.write('\n')

    ####################################
    ############# TRAINING #############
    ####################################
    experimentWriter.write('Training ...\n')
    experimentWriter.write('Number of images = ' + str(len(data_loader)))

    # Callbacks
    checkpoint_dir = join(parameter_dict['RESULTS_DIR'], 'checkpoints')
    results_file = join(parameter_dict['RESULTS_DIR'], 'results', 'training_results.csv')
    log_keys = ['loss_' + lossname for lossname in loss_function_dict.keys()] + ['w_loss_' + l for l in loss_function_dict.keys()] + ['loss', 'time_duration (s)']

    logger = History(log_keys)
    training_printer = PrinterCallback()
    lrdecay = LRDecay(optimizer, n_iter_start=0, n_iter_finish=parameter_dict['N_EPOCHS'])
    model_checkpoint = ModelCheckpoint(checkpoint_dir, parameter_dict['SAVE_MODEL_FREQUENCY'])
    training_tocsv = ToCSVCallback(filepath=results_file, keys=log_keys)

    callback_list = [logger, model_checkpoint, training_printer, training_tocsv]#, lrdecay]

    for cb in callback_list:
        cb.on_train_init(registration, starting_epoch=parameter_dict['STARTING_EPOCH'])

    for epoch in range(parameter_dict['STARTING_EPOCH'], parameter_dict['N_EPOCHS']):
        epoch_start_time = time.time()

        logs_dict = {}
        for cb in callback_list:
            cb.on_epoch_init(registration, epoch)

        registration.train()
        if model_type == 'standard':
            train(registration, optimizer, device, generator_train, epoch, loss_function_dict,
                  loss_weights_dict, callback_list, da_model, **kwargs_training)

        elif model_type == 'bidir':
            train_bidirectional(registration, optimizer, device, generator_train, epoch, loss_function_dict,
                                loss_weights_dict, callback_list, da_model, **kwargs_training)

        else:
            raise ValueError("Please, specify a valid model_type")

        epoch_end_time = time.time()
        logs_dict['time_duration (s)'] = epoch_end_time - epoch_start_time

        for cb in callback_list:
            cb.on_epoch_fi(logs_dict, registration, epoch, optimizer=optimizer)

    for cb in callback_list:
        cb.on_train_fi(registration)

    plot_results(results_file, keys=log_keys)

    print('Done.')
