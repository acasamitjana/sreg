# imports
from os.path import join, exists
from os import makedirs, rmdir
import time
from argparse import ArgumentParser
import shutil

# third party imports
import numpy as np
import nibabel as nib
import itertools

# project imports
from database.data_loader import DataLoader
from scripts import config_dev as configFile
from src.utils.algorithm_utils import initialize_graph_NR, initialize_graph_RegNet
from src.utils.io_utils import query_yes_no
from setup import *

print('\n\n\n\n\n')
print('# ------------------------------------------------- #')
print('# Initialize Non-linear registration using NiftyReg #')
print('# ------------------------------------------------- #')
print('\n\n')


#####################
# Global parameters #
#####################

parameter_dict = configFile.CONFIG_REGISTRATION

arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--subjects', default=None, nargs='+')
arg_parser.add_argument('--reg_algorithm', default='bidir', choices=['standard', 'bidir', 'niftyreg'])

arguments = arg_parser.parse_args()
initial_subject_list = arguments.subjects
reg_algorithm = arguments.reg_algorithm

###################
# Tree parameters #
###################

print('Loading dataset ...\n')
reg_name = reg_algorithm
if reg_algorithm == 'bidir':
    reg_name += str(parameter_dict['LOSS_REGISTRATION_SMOOTHNESS']['lambda'])

data_loader = DataLoader(linear=True, sid_list=initial_subject_list, reg_algorithm=reg_name)
subject_list = data_loader.subject_list
parameter_dict = configFile.get_config_dict(data_loader.image_shape)
parameter_dict['RESULTS_DIR'] = parameter_dict['RESULTS_DIR'] + '_' + reg_algorithm

####################
# Run registration #
####################
missing_subjects = []
for it_subject, subject in enumerate(subject_list):
    PROCESS_REPEATED_FLAG = False
    timepoints = subject.timepoints


    if not exists(subject.get_timepoint().get_filepath('linear_resampled_image')):
        missing_subjects.append(subject.id)
        print('[NO DATA AVAILABLE] Subject: ' + subject.get_timepoint().get_filepath('linear_resampled_image') + '.')
        continue

    if len(timepoints) == 1:
        print(' Subject: ' + subject.id + ' has only 1 timepoint. No registration is made.')
        shutil.copy(subject.get_timepoint().get_filepath('linear_resampled_image'), subject.nonlinear_template)
        continue

    nonlin_dir = 'nonlinear_registration_' + reg_algorithm
    if reg_algorithm in ['standard', 'bidir']:  nonlin_dir += str(parameter_dict['LOSS_REGISTRATION_SMOOTHNESS']['lambda'])
    results_dir_sbj = subject.results_dirs.get_dir(nonlin_dir)
    tempdir = join(results_dir_sbj, 'tmp')
    if not exists(tempdir):
        makedirs(tempdir)

    first_repeated = 0
    print('[INITIALIZE GRAPH] Registering subject: ' + subject.sid )
    for tp_ref, tp_flo in itertools.combinations(timepoints, 2):

        ################
        # Registration #
        ################
        filename = str(tp_ref.tid) + '_to_' + str(tp_flo.tid)
        # if exists(join(results_dir_sbj, filename + '.svf.nii.gz')) and first_repeated == 0 and PROCESS_REPEATED_FLAG:
        #     question = ' Subject: ' + subject.sid + ' has already some computed registrations in ' + \
        #                results_dir_sbj + '.\n Do you want to proceed and overwrite or cancel?'
        #     PROCESS_REPEATED_FLAG = query_yes_no(question=question)
        #     first_repeated += 1

        if exists(join(results_dir_sbj, filename + '.svf.nii.gz')) and not PROCESS_REPEATED_FLAG:
            break

        print('  o From T=' + str(tp_ref.tid) + ' to T=' + str(tp_flo.tid) + '.', end=' ', flush=True)
        t_init = time.time()
        if reg_algorithm in ['standard', 'bidir']:
            initialize_graph_RegNet([tp_ref, tp_flo],  parameter_dict, results_dir=results_dir_sbj,
                                    filename=filename, vox2ras0=subject.vox2ras0, subject_shape=subject.image_shape,
                                    epoch='LAST', use_gpu=False)
        else:
            initialize_graph_NR([tp_ref, tp_flo], results_dir=results_dir_sbj, filename=filename,
                                vox2ras=subject.vox2ras0, tempdir=tempdir)


        print('Elapsed time: ' + str(np.round(time.time() - t_init, 2)))

    print('[INITIALIZE GRAPH] -- DONE -- Subject ' + subject.id + ' has been registered.')
    print('\n')

    # if not DEBUG:
    #     rmdir(subject.results_dirs.get_dir('linear_resampled'))
