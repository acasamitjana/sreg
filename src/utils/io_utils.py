from datetime import datetime, date
from os.path import join, exists
from os import makedirs, listdir
import gc
import subprocess
import sys

import torch
import numpy as np
import csv

def worker_init_fn(wid):
    np.random.seed(np.mod(torch.utils.data.get_worker_info().seed, 2**32-1))

def create_results_dir(results_dir, subdirs=None):
    if subdirs is None:
        subdirs = ['checkpoints', 'results']
    if not exists(results_dir):
        for sd in subdirs:
            makedirs(join(results_dir, sd))
    else:
        for sd in subdirs:
            if not exists(join(results_dir, sd)):
                makedirs(join(results_dir, sd))

def get_memory_used():
    import sys
    local_vars = list(locals().items())
    for var, obj in local_vars: print(var, sys.getsizeof(obj) / 1000000000)


def convert_nifti_directory(directory, extension='.nii.gz'):

    files = listdir(directory)
    for f in files:
        if extension in f:
            continue
        elif '.nii.gz' in f:
            new_f = f[:-6] + extension

        elif '.nii' in f:
            new_f = f[:-4] + extension

        elif '.mgz' in f:
            new_f = f[:-4] + extension

        else:
            continue

        subprocess.call(['mri_convert', join(directory, f), join(directory, new_f)])


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")

def write_affine_matrix(path, affine_matrix):
    with open(path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ')
        for it_row in range(4):
            csvwriter.writerow(affine_matrix[it_row])

def read_lta(file):
    lta = np.zeros((4,4))
    with open(file, 'r') as txtfile:
        lines = txtfile.readlines()
        for it_row, l in enumerate(lines[5:9]):
            aff_row = l.split(' ')[:-1]
            lta[it_row] = [float(ar) for ar in aff_row]

    return lta

def read_affine_matrix(path, full=False):
    with open(path, 'r') as csvfile:
        rotation_matrix = np.zeros((3, 3))
        translation_vector = np.zeros((3,))
        csvreader = csv.reader(csvfile, delimiter=' ')
        for it_row, row in enumerate(csvreader):
            rotation_matrix[it_row, 0] = float(row[0])
            rotation_matrix[it_row, 1] = float(row[1])
            rotation_matrix[it_row, 2] = float(row[2])
            translation_vector[it_row] = float(row[3])
            if it_row == 2:
                break

    if full:
        affine_matrix = np.zeros((4,4))
        affine_matrix[:3, :3] = rotation_matrix
        affine_matrix[:3, 3] = translation_vector
        affine_matrix[3, 3] = 1
        return affine_matrix

    else:
        return rotation_matrix, translation_vector

class DebugWriter(object):

    def __init__(self, debug_flag, filename = None, attach = False):
        self.filename = filename
        self.debug_flag = debug_flag
        if filename is not None:
            date_start = date.today().strftime("%d/%m/%Y")
            time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            if not attach:
                with open(self.filename, 'w') as writeFile:
                    writeFile.write('############################\n')
                    writeFile.write('###### NEW EXPERIMENT ######\n')
                    writeFile.write('############################\n')
                    writeFile.write('Experiment date and time: ' + date_start + '   ' + time_start)
                    writeFile.write('\n')
            else:
                with open(self.filename, 'a') as writeFile:
                    for i in range(4):
                        writeFile.write('\n')
                    writeFile.write('############################\n')
                    writeFile.write('###### NEW EXPERIMENT ######\n')
                    writeFile.write('############################\n')
                    writeFile.write('Experiment date and time: ' + date_start + '   ' + time_start)
                    writeFile.write('\n')

    def write(self, to_write):
        if self.debug_flag:
            if self.filename is None:
                print(to_write, end=' ')
            else:
                with open(self.filename, 'a') as writeFile:
                    writeFile.write(to_write)

class ResultsWriter(object):

    def __init__(self, filename = None, attach = False):
        self.filename = filename
        if filename is not None:
            date_start = date.today().strftime("%d/%m/%Y")
            time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            if not attach:
                with open(self.filename, 'w') as writeFile:
                    writeFile.write('############################\n')
                    writeFile.write('###### NEW EXPERIMENT ######\n')
                    writeFile.write('############################\n')
                    writeFile.write('Experiment date and time: ' + date_start + '   ' + time_start)
                    writeFile.write('\n')
            else:
                with open(self.filename, 'a') as writeFile:
                    for i in range(4):
                        writeFile.write('\n')
                    writeFile.write('############################\n')
                    writeFile.write('###### NEW EXPERIMENT ######\n')
                    writeFile.write('############################\n')
                    writeFile.write('Experiment date and time: ' + date_start + '   ' + time_start)
                    writeFile.write('\n')

    def write(self, to_write):
        if self.filename is None:
            print(to_write, end=' ')
        else:
            with open(self.filename, 'a') as writeFile:
                writeFile.write(to_write)

class ExperimentWriter(object):
    def __init__(self, filename = None, attach = False):
        self.filename = filename
        date_start = date.today().strftime("%d/%m/%Y")
        time_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        if filename is not None:
            method = 'a' if attach else 'w'
            with open(filename, method) as writeFile:
                writeFile.write('Experiment date and time: ' + date_start + '   ' + time_start)
                writeFile.write('\n')

    def write(self, to_write):
        if self.filename is None:
            print(to_write, end=' ')
        else:
            with open(self.filename, 'a') as writeFile:
                writeFile.write(to_write)


def check_gc_torch():
    it_obj = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.dtype)
                it_obj += 1
        except:
            pass

    print('Total number of tensors ' + str(it_obj))
