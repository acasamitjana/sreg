# imports
from os.path import join, exists
from os import makedirs
import time
from argparse import ArgumentParser
import shutil

# third party imports
import numpy as np
import itertools

# project imports
from setup import *
from database.data_loader import DataLoader
from scripts import config_dev as configFile
from src.utils.algorithm_utils import initialize_graph_NR_lineal
from src.utils.io import query_yes_no


print('\n\n\n\n\n')
print('# --------------------------------------------- #')
print('# Initialize Linear registration using NiftyReg #')
print('# --------------------------------------------- #')
print('\n\n')

#####################
# Global parameters #
#####################
parameter_dict = configFile.CONFIG_REGISTRATION
results_dir = OBSERVATIONS_DIR_LINEAL


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

    PROCESS_REPEATED_FLAG = True

    tempdir = join(results_dir, subject.sid, 'tmp')
    if not exists(tempdir):
        makedirs(tempdir)

    timepoints = subject.timepoints

    if len(timepoints) == 1:
        print(' Subject: ' + subject.id + ' has only 1 timepoint. No registration is made.')
        shutil.copy(subject.get_timepoint().image_path, subject.linear_template)
        continue

    if not exists(subject.get_timepoint().image_path):
        missing_subjects.append(subject.id)
        print(' !! Warning !! : Subject: ' + subject.id + ' has missing timepoints.')
        continue

    first_repeated = 0
    for tp_ref, tp_flo in itertools.combinations(timepoints, 2):
        print('Registering subject: ' + subject.sid + '. From T=' + str(tp_ref.id) + ' to T=' + str(tp_flo.id) + '.')


        filename = str(tp_ref.id) + '_to_' + str(tp_flo.id)
        results_dir_sbj = join(results_dir, subject.sid)
        if not exists(results_dir_sbj): makedirs(results_dir_sbj)
        if exists(join(results_dir_sbj, filename + '.aff')) and first_repeated == 0:
            question = ' Subject: ' + subject.id + ' has already some computed registrations in ' + \
                       results_dir_sbj + '.\n Do you want to proceed and overwrite or cancel?'
            PROCESS_REPEATED_FLAG = query_yes_no(question=question)
            first_repeated += 1

        if exists(join(results_dir_sbj, filename + '.svf.nii.gz')) and not PROCESS_REPEATED_FLAG:
            break

        t_init = time.time()
        initialize_graph_NR_lineal([tp_ref, tp_flo], results_dir=results_dir_sbj, filename=filename, tempdir=tempdir)

        print('NR elapsed time: ' + str(np.round(time.time() - t_init, 2)))