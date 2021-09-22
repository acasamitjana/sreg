from os.path import join
from os import listdir
import csv
import pdb
from datetime import datetime

import numpy as np
import openpyxl

Date_DICT = {
    'jan': 1,
    'feb': 2,
    'mar': 3,
    'apr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'aug': 8,
    'sept': 9,
    'oct': 10,
    'nov': 11,
    'dec': 12,
}

MRI_DICT = {
    'BRAVO': 1,
    '3DSAG': 2,
}

def get_date(date):
    if isinstance(date, datetime):
        return date
    else:
        try:
            return datetime.strptime(date, "%d/%m/%Y")
        except:
            if date is None:
                return datetime.strptime("1/1/1000", "%d/%m/%Y")
            d, m, y = date.split('-')
            m = Date_DICT[m.lower()]
            return datetime.strptime(str(int(d)) + '/' + str(int(m)) + '/' + str(int(y)), "%d/%m/%Y")

def get_int(value):
    try:
        return int(value)
    except:
        return -4

def read_scan_data(file):
    scan_dict = {}
    with open(file, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            rid = row['SUBJECT']
            t = row['TIMEPOINT']
            scan_date = datetime.strptime(row['SCAN_DATE'], "%d/%m/%Y")
            scan_dict[rid + '_' + t] = scan_date

    return scan_dict

HEADER = ['SUBJECT', 'TIMEPOINT', 'SEX','AGE', 'AGE_ADMISSION', 'AGE_DEATH', 'TIME_IN_RESIDENCE', 'EAO', 'THAL', 'BRAAK_TAU',
          'DX', 'MRI_TYPE']

datadir = '/home/acasamitjana/Data/FCIEN'
input_file = '/home/acasamitjana/Data/FCIEN/SubjectsData_Madrid.xlsx'
scan_file = '/home/acasamitjana/Data/FCIEN/scan_date.csv'
scan_dict = read_scan_data(scan_file)

sbj_rid = []
sbj_info = []
wb = openpyxl.load_workbook(input_file)
ws = wb['Clinical-Neuropath Variables']
is_title = True
max_label = 0
date_birth_dict = {}
for row in ws.iter_rows(values_only=True):
    if is_title:
        is_title = False
        continue

    umacode = get_int(row[1])
    sex = get_int(row[4])
    date_birth = get_date(row[5])#datetime.strptime(row[5], "%Y-%m-%d")
    date_admission = get_date(row[6])#datetime.strptime(, "%Y-%m-%d")
    date_death = get_date(row[7])#datetime.strptime(, "%Y-%m-%d")
    age_admission = np.round(np.abs((date_admission - date_birth).days/365.25), 2)
    age_death = np.round(np.abs((date_death - date_birth).days/365.25), 2)
    time_residence = get_int(row[8])
    eao = get_int(row[11])
    dx = get_int(row[18])
    thal = get_int(row[29])
    braak_tau = get_int(row[31])
    try:
        num_timepoints = get_int(row[-1])
    except:
        continue

    umacode = "{:03d}".format(umacode)
    date_birth_dict[umacode] = date_birth

    for it_t in range(num_timepoints):
        tp = "{:02d}".format(it_t)
        if umacode + '_' + tp in scan_dict.keys():
            age = np.round(np.abs((scan_dict[umacode + '_' + tp] - date_birth).days/365.25), 2)
        else:
            age = -4

        sbj_info.append({
        'SUBJECT': umacode, 'TIMEPOINT': tp, 'SEX': sex,'AGE': age, 'AGE_ADMISSION': age_admission,
        'AGE_DEATH': age_death, 'TIME_IN_RESIDENCE': time_residence, 'EAO': eao, 'THAL': thal, 'BRAAK_TAU': braak_tau,
        'DX': dx, 'MRI_TYPE': 0
        })
        sbj_rid.append(umacode + '_' + tp)

    # sbj_rid.extend([str(umacode) + '_' + str(it_t) for it_t in range(num_timepoints)])
    # sbj_info.extend([{
    #     'SUBJECT': umacode, 'TIMEPOINT': "{:02d}".format(it_t), 'AGE': age[it_t], 'AGE_ADMISSION': age_admission,
    #     'AGE_DEATH': age_death, 'TIME_IN_RESIDENCE': time_residence, 'EAO': eao, 'THAL': thal, 'BRAAK_TAU': braak_tau,
    #     'DX': dx, 'MRI_TYPE': 0
    # } for it_t in range(num_timepoints)])


sbj_rid_2 = []
total_files = listdir(datadir)
total_files = filter(lambda x: '.nii' in x, total_files)
for file in total_files:
    f = file.split('_')
    print(f)
    if 'Cd' in f:
        if len(f) == 7:
            uma, uipa, rnd, visit, rev, numvisit, typemri = f
        elif len(f) == 5:
            uma, uipa, rnd, numvisit, typemri = f
            visit=0
    else:
        if len(f) == 6:
            uma, uipa, visit, rev, numvisit, typemri = f
            if uipa != 'Uipa' and uipa != 'Uipa.':
                visit = 0 # case 11_0
        elif len(f) == 5:
            uma, uipa, visit_rev, numvisit, typemri = f
            visit = visit_rev.split('Rev')[0]
            if uipa != 'Uipa':
                visit = 1 # case 90_1
        else:
            uma, uipa, numvisit, typemri = f
            visit = 0

    umacode = int(uma[3:])
    umacode = "{:03d}".format(umacode)
    visit = "{:02d}".format(int(visit))
    if umacode + '_' + visit not in sbj_rid:
        sbj_rid.append(str(umacode) + '_' + str(visit))
        if umacode + '_' + tp in scan_dict.keys():
            age = np.round(np.abs((scan_dict[umacode + '_' + visit] - date_birth_dict[umacode]).days / 365.25), 2)
        else:
            age = -4

        sbj_info.extend([{
            'SUBJECT': umacode, 'TIMEPOINT': visit, 'SEX': -4, 'AGE': age, 'AGE_ADMISSION': -4, 'AGE_DEATH': 4,
            'TIME_IN_RESIDENCE': -4, 'EAO': -4, 'THAL': -4, 'BRAAK_TAU': -4,
            'DX': -4, 'MRI_TYPE': MRI_DICT[typemri.split('.')[0]]
        }])

    else:
        idx = [it_sbj for it_sbj, sbj in enumerate(sbj_info) if str(sbj['SUBJECT']) + '_' + str(sbj['TIMEPOINT']) == str(umacode) + '_' + str(visit)][0]
        sbj_info[idx]['MRI_TYPE'] += MRI_DICT[typemri.split('.')[0]]


with open(join(datadir, 'subject_info.csv'), 'w') as csvfile:
    csvwriter = csv.DictWriter(csvfile, fieldnames=HEADER, delimiter=',')
    csvwriter.writeheader()
    for sbj in sbj_info:
        csvwriter.writerow(sbj)

