# imports
from os.path import join, exists
from os import makedirs, rmdir
import time
from argparse import ArgumentParser
import shutil

# third party imports
import numpy as np
import itertools

# project imports
from database.data_loader import DataLoader
from scripts import config_dev as configFile
from src.utils.algorithm_utils import initialize_graph_NR_lineal
from src.utils.io_utils import query_yes_no
from setup import *

print('\n\n\n\n\n')
print('# --------------------------------------------- #')
print('# Initialize Linear registration using NiftyReg #')
print('# --------------------------------------------- #')
print('\n\n')

#####################
# Global parameters #
#####################
parameter_dict = configFile.CONFIG_REGISTRATION

arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--subjects', default=None, nargs='+')

arguments = arg_parser.parse_args()
subject_list = arguments.subjects


###################
# Tree parameters #
###################

print('Loading dataset ...\n')
data_loader = DataLoader(sid_list=subject_list)
subject_list = data_loader.subject_list

####################
# Run registration #
####################

missing_subjects = []
for it_subject, subject in enumerate(subject_list):

    PROCESS_REPEATED_FLAG = False

    results_dir_sbj = subject.results_dirs.get_dir('linear_registration')
    tempdir = join(results_dir_sbj, 'tmp')
    if not exists(tempdir):
        makedirs(tempdir)

    timepoints = subject.timepoints


    if not exists(subject.get_timepoint().get_filepath('preprocessing_resample_centered')):
        missing_subjects.append(subject.id)
        print('[NO DATA AVAILABLE] Subject: ' + subject.id + '.')
        print('\n')
        continue

    if len(timepoints) == 1:
        print(' Subject: ' + subject.id + ' has only 1 timepoint. No registration is made.')
        shutil.copy(subject.get_timepoint().init_path['resample'], subject.linear_template)
        print('\n')
        continue

    first_repeated = 0
    for tp_ref, tp_flo in itertools.combinations(timepoints, 2):

        filename = str(tp_ref.id) + '_to_' + str(tp_flo.id)

        if exists(join(results_dir_sbj, filename + '.aff')) and first_repeated == 0 and PROCESS_REPEATED_FLAG:
            question = ' Subject: ' + subject.id + ' has already some computed registrations in ' + \
                       results_dir_sbj + '.\n Do you want to proceed and overwrite or cancel?'

            PROCESS_REPEATED_FLAG = query_yes_no(question=question)
            first_repeated += 1

        if exists(join(results_dir_sbj, filename + '.aff')) and not PROCESS_REPEATED_FLAG:
            print(' Subject: ' + subject.id + ' has been already processed.')
            continue

        t_init = time.time()
        print('Registering subject: ' + subject.sid + '. From T=' + str(tp_ref.id) + ' to T=' + str(tp_flo.id) + '.')
        initialize_graph_NR_lineal([tp_ref, tp_flo], results_dir=results_dir_sbj, filename=filename, tempdir=tempdir)
        print('NR elapsed time: ' + str(np.round(time.time() - t_init, 2)))

    if not DEBUG:
        if exists(subject.results_dirs.get_dir('preprocessing_resample_centered')):
            shutil.rmtree(subject.results_dirs.get_dir('preprocessing_resample_centered'))
        if exists(subject.results_dirs.get_dir('preprocessing_mask_centered')):
            shutil.rmtree(subject.results_dirs.get_dir('preprocessing_mask_centered'))
        if exists(subject.results_dirs.get_dir('preprocessing_mask_centered_dilated')):
            shutil.rmtree(subject.results_dirs.get_dir('preprocessing_mask_centered_dilated'))
