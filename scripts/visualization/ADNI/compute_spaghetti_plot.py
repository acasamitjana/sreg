from os.path import exists
from os import makedirs
from argparse import ArgumentParser
import shutil


from matplotlib import pyplot as plt, font_manager as fm
import seaborn as sns
import pandas as pd


# project imports
from setup import *
from database.data_loader import DataLoader
from scripts import config_dev as configFile
from src.algorithm import *
from database import read_demo_info

def read_volume_results(filepath, fieldnames=list(LABEL_DICT.keys())):

    with open(filepath, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)#, fieldnames=fieldnames)
        results = {f: [] for f in fieldnames}
        for row in csvreader:
            results[row['TID']] = {}
            for f in fieldnames:
                results[row['TID']][LABEL_DICT[f]] = float(row[f])

        return results

def read_FS_volumes(file, labs=LABEL_DICT):
    etiv=0
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

    return {**{it_l: float(fs_vols[l]) for l, it_l in labs.items() if 'Thalamus' not in l},
            **{it_l: float(fs_vols[l + '-Proper']) for l, it_l in labs.items() if 'Thalamus' in l}}, etiv


def get_vols_post(post):

    n_labels = post.shape[-1]
    vols = {}
    for l in range(n_labels):
        mask_l = post[..., l]
        vols[l] = np.sum(mask_l)

    return vols

def write_volume_results(volume_dict, filepath, fieldnames = ['TID'] + list(LABEL_DICT.keys())):
    with open(filepath, 'w') as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()
        csvwriter.writerows(volume_dict)

print('\n\n\n\n\n')
print('# ----------------- #')
print('# Computing volumes #')
print('# ----------------- #')
print('\n\n')

#####################
# Global parameters #
#####################
fpath = '/usr/share/fonts/truetype/msttcorefonts/Arial_Black.ttf'
prop_bold = fm.FontProperties(fname=fpath)
fpath = '/usr/share/fonts/truetype/msttcorefonts/arial.ttf'
prop = fm.FontProperties(fname=fpath)
prop_legend = fm.FontProperties(fname=fpath, size=12)

# Parameters
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--subjects', default=None, nargs='+')

arguments = arg_parser.parse_args()
initial_subject_list = arguments.subjects
labels = {
    'Right-Hippocampus': 53,
    'Left-Hippocampus': 17,

    'Right-Lateral-Ventricle': 43,
    'Left-Lateral-Ventricle': 4,

    'Right-Thalamus': 49,
    'Left-Thalamus': 10,

    'Right-Amygdala': 54,
    'Left-Amygdala': 18,

    'Right-Putamen': 51,
    'Left-Putamen': 12,

    'Right-Pallidum': 52,
    'Left-Pallidum': 13,

    'Right-Caudate': 50,
    'Left-Caudate': 11,

}

FS_DIR = '/home/acasamitjana/Results/Registration/BrainAging/miriad_Eugenio/longitudinal' if 'MIRIAD' in DB else None
demo_dict = read_demo_info(demo_fields=['AGE', 'DX'])
##############
# Processing #
##############

if DB == 'MIRIAD_retest':
    timepoints_filter = lambda x: '_2.nii.gz' not in x
    data_loader = DataLoader(sid_list=initial_subject_list, linear=True, timepoints_filter=timepoints_filter)
else:
    data_loader = DataLoader(sid_list=initial_subject_list, linear=True)
subject_list = data_loader.subject_list
subject_list = list(filter(lambda x: '_255' not in x.id and '_222' not in x.id and '_231' not in x.id and '_199' not in x.id and '_218' not in x.id, subject_list))[2:]

freesurfer_dict = {}
etiv_dict = {}
synthseg_dict = {}
st_dict = {}
direct_dict = {}

print('[COMPUTE VOLUMES] Start.')
output_subject_list = []
sbj_st = 0
sbj_d = 0
for it_subject, subject in enumerate(subject_list):
    print('   Subject: ' + str(subject.id))

    subject_shape = subject.image_shape
    timepoints = subject.timepoints
    long_st_dir = subject.results_dirs.get_dir('longitudinal_segmentation_st')
    long_regnet_dir = subject.results_dirs.get_dir('longitudinal_segmentation_registration')

    freesurfer_dict[subject.id] = {}
    etiv_dict[subject.id] = {}
    synthseg_dict[subject.id] = {}
    st_dict[subject.id] = {}
    direct_dict[subject.id] = {}
    filepath_synthseg = join(subject.results_dirs.get_dir('preprocessing'), 'vols_st.txt')
    filepath_st = join(long_st_dir, 'vols_st.txt')
    filepath_regnet = join(long_regnet_dir, 'vols_direct.txt')

    synthseg_vols = []
    if exists(filepath_st):
        results = read_volume_results(filepath_st)
        for tid, vol_dict in results.items():
            st_dict[subject.id][tid] = vol_dict

        if not np.isnan(st_dict[subject.id][tid][0]):
            output_subject_list.append(subject)

    if not exists(filepath_synthseg):
        fileparts = filepath_synthseg.split('MIRIAD_retest')
        if exists(join(fileparts[0], 'MIRIAD', fileparts[1][1:])):
            shutil.copy(join(fileparts[0], 'MIRIAD', fileparts[1][1:]), filepath_synthseg)
            continue
        for tp in timepoints:
            fake_vols = get_vols_post(tp.load_posteriors())
            synthseg_dict[subject.id][tp.id] = {k: fake_vols[it_k] for it_k, k in enumerate(LABEL_DICT.values())}
            synthseg_vols_dict = {k: synthseg_dict[subject.id][tp.id][it_k] for k, it_k in LABEL_DICT.items()}
            synthseg_vols_dict['TID'] = tp.id
            synthseg_vols.append(synthseg_vols_dict)

        write_volume_results(synthseg_vols, filepath_synthseg)

    else:
        results = read_volume_results(filepath_synthseg)
        for tid, vol_dict in results.items():
            synthseg_dict[subject.id][tid] = vol_dict

    if 'MIRIAD' in DB:
        for tp in timepoints:
            tid_split = tp.id.split('_')
            fs_file = join(FS_DIR, 'miriad_' + tid_split[1] + '_' + str(int(tid_split[4])) + '_MR_1.long.stats.txt')
            if not exists(fs_file):
                freesurfer_dict[subject.id][tp.id] = {k: np.nan for k in st_dict[subject.id][tp.id].keys()}
            else:
                freesurfer_dict[subject.id][tp.id], etiv_dict[subject.id][tp.id] = read_FS_volumes(fs_file, labs=labels)
    else:
        for tp in timepoints:
            freesurfer_dict[subject.id][tp.id] = synthseg_dict[subject.id][tp.id]
            etiv_dict[subject.id][tp.id] = 100


print('[COMPUTE VOLUMES] Saving images.')
if not exists(join(REGISTRATION_DIR, 'Results', 'Spaghetti')):
    makedirs(join(REGISTRATION_DIR, 'Results', 'Spaghetti'))

subject_list = output_subject_list
results_list = [synthseg_dict, st_dict, freesurfer_dict]#, direct_dict]
results_list_id = ['SynthSeg', 'SReg', 'FSDict']#, 'Registration']
for method, method_dict in zip(results_list_id, results_list):
    print('Method: ' + method)
    for lab_str, lab in labels.items():
        print('   Label: ' + lab_str)
        plt.figure(figsize=(8.6, 7.0))  # figsize=(8.6, 7.0)
        for it_subject, subject in enumerate(subject_list):

            timepoints = subject.timepoints
            timepoints.sort(key=lambda x: demo_dict[subject.id][x.id]['AGE'])

            vol_baseline = method_dict[subject.id][timepoints[0].id][lab]
            age_baseline = float(demo_dict[subject.id][timepoints[0].id]['AGE'])

            dataframe = {'Age': [], 'Volume': []}
            dataframe['Age'] = [float(demo_dict[subject.id][tp.id]['AGE']) - age_baseline for tp in timepoints]
            dataframe['Volume'] = [method_dict[subject.id][tp.id][lab]/vol_baseline for tp in timepoints]
            color_dx = 0 if demo_dict[subject.id][timepoints[0].id]['DX'] == 'HC' else 1

            dataframe_pd = pd.DataFrame(dataframe)
            x = sns.lineplot(x="Age", y="Volume", color=sns.color_palette("bright")[color_dx], ci=None, markers=True,
                              data=dataframe_pd, markersize=8, label=demo_dict[subject.id][timepoints[0].id]['DX'])

        # x.grid()
        # x.set_axisbelow(True)
        # plt.axes(x)
        # handles, labels_legend = x.get_legend_handles_labels()
        # x.legend(handles, labels_legend, loc=2, ncol=2, prop=prop_legend)  # , bbox_to_anchor=(0.5, 1.05))
        #
        # x.set_title(lab_str, fontproperties=prop, fontsize=20)  # y=1.0, pad=42, )
        # plt.legend()
        plt.title(lab_str, fontproperties=prop, fontsize=20)
        plt.ylabel('Volume / Volume_baseline', fontproperties=prop_bold, fontsize=17)
        plt.xlabel('Age - Age baseline', fontproperties=prop_bold, fontsize=17)
        plt.yticks(fontproperties=prop, fontsize=15)#,rotation=90)
        ax = plt.gca()
        axhandles, axlabels = ax.get_legend_handles_labels()
        # ax.set_xticks([float(demo_dict[subject.id][tp.id]['AGE']) for tp in timepoints])
        # ax.set_xticklabels([demo_dict[subject.id][tp.id]['AGE'] for tp in timepoints], fontproperties=prop, fontsize=15)#,rotation=25)

        plt.grid()
        plt.legend([], [], frameon=False)
        plt.savefig(join(REGISTRATION_DIR, 'Results', 'Spaghetti', lab_str + '_' + method + '.png'))
        plt.close()

        plt.figure()
        ax = plt.gca()
        ax.legend(axhandles, axlabels, loc=2, ncol=1, prop=prop_legend)
        plt.savefig(join(REGISTRATION_DIR, 'Results', 'Spaghetti', 'legend.png'))

print('[COMPUTE VOLUMES] Done.')
