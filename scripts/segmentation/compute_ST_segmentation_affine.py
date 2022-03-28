from argparse import ArgumentParser
from joblib import delayed, Parallel

# project imports
from setup import *
from database.data_loader import DataLoader
from scripts import config_dev as configFile
from src.algorithm import *
from src.utils import results_utils
from database import read_demo_info
from src import layers

device = 'cpu'#'cuda:0'


class FS_Lin(results_utils.LabelFusion):

    def compute_p_label(self, timepoints, image_list, flow_dir, age_list, tp, subject, parameter_dict, regnet_model):

        interp_func = layers.SpatialInterpolation(padding_mode='zeros').to(device)

        p_data = {t_var: {sp_var: np.zeros(tp.image_shape + (len(timepoints),), dtype='float32') for sp_var in spatial_variance} for t_var in temp_variance}
        p_label = {t_var: {sp_var: np.zeros(tp.image_shape + (len(results_utils.UNIQUE_LABELS),), dtype='float32') for sp_var in spatial_variance} for t_var in temp_variance}
        if DEBUG:
            p_label = {
                t_var: {sp_var: np.zeros(tp.image_shape + (len(list(LABEL_DICT.keys())),), dtype='float32') for sp_var
                        in spatial_variance} for t_var in temp_variance}
        ref_shape = image_list[tp.id].shape

        ii = np.arange(0, ref_shape[0], dtype='int32')
        jj = np.arange(0, ref_shape[1], dtype='int32')
        kk = np.arange(0, ref_shape[2], dtype='int32')

        II, JJ, KK = np.meshgrid(ii, jj, kk, indexing='ij')

        voxMosaic_orig = np.concatenate((II.reshape(-1, 1), JJ.reshape(-1, 1), KK.reshape(-1, 1), np.ones((int(np.prod(ref_shape[:3])), 1), dtype='int32')), axis=1).T
        rasMosaic_orig = np.dot(tp.vox2ras0, voxMosaic_orig).astype('float32')

        del voxMosaic_orig, ii, jj, kk, II, JJ, KK

        for it_tp_flo, tp_flo in enumerate(timepoints):
            if tp_flo.id == tp.id:
                for t_var in temp_variance:
                    for s_var in spatial_variance:
                        p_data[t_var][s_var][..., it_tp_flo] = 1
                        if DEBUG:
                            p_label[t_var][s_var] += tp.load_posteriors()

                        else:
                            p_label[t_var][s_var] += tp.load_posteriors(labels=results_utils.KEEP_LABELS_IDX)

            else:
                rasMosaic_targ = rasMosaic_orig.copy()
                voxMosaic_targ = np.matmul(np.linalg.inv(tp_flo.vox2ras0), rasMosaic_targ).astype('float32')
                voxMosaic_targ = voxMosaic_targ[:3]

                voxMosaic_targ_torch = voxMosaic_targ.copy()
                voxMosaic_targ_torch = voxMosaic_targ_torch.reshape((3,) + ref_shape)
                voxMosaic_targ_torch = torch.from_numpy(voxMosaic_targ_torch[np.newaxis]).float().to(device)

                if DEBUG:
                    post_list = np.transpose(tp_flo.load_posteriors(), axes=[3, 0, 1, 2])

                else:
                    post_list = np.transpose(tp_flo.load_posteriors(labels=results_utils.KEEP_LABELS_IDX),
                                             axes=[3, 0, 1, 2])
                data = np.concatenate((image_list[tp_flo.id][np.newaxis], post_list), axis=0)
                data = torch.from_numpy(data[np.newaxis]).float().to(device)

                data_resampled = interp_func(data, voxMosaic_targ_torch)
                im_resampled = data_resampled[0, 0].cpu().detach().numpy()
                seg_resampled = np.transpose(data_resampled[0, 1:].cpu().detach().numpy(), axes=[1, 2, 3, 0])

                del voxMosaic_targ, data_resampled

                mean_im_2 = (im_resampled - image_list[tp.id]) ** 2

                if DEBUG:
                    results_dir_sbj = subject.results_dirs.get_dir('longitudinal_segmentation_st_' + self.long_seg_algo)
                    if not os.path.exists(join(results_dir_sbj, 'debug')): os.makedirs(join(results_dir_sbj, 'debug'))
                    proxy = nib.load(tp.init_path['resample'])
                    img = nib.Nifti1Image(seg_resampled, proxy.affine)
                    nib.save(img, join(results_dir_sbj, 'debug', tp.id + '_to_' + tp_flo.id + '.post.nii.gz'))

                    img = nib.Nifti1Image(im_resampled, proxy.affine)
                    nib.save(img, join(results_dir_sbj, 'debug', tp.id + '_to_' + tp_flo.id + '.image.nii.gz'))

                del im_resampled

                mean_age_2 = (age_list[tp_flo.id] - age_list[tp.id]) ** 2
                for t_var in temp_variance:
                    t_ker = 1 if t_var == 'inf' else np.exp(-0.5 / t_var * mean_age_2)
                    for s_var in spatial_variance:
                        s_ker = 1 if s_var == 'inf' else np.exp(-0.5 / s_var * mean_im_2)
                        p_data[t_var][s_var][..., it_tp_flo] = s_ker * t_ker
                        p_label[t_var][s_var] += p_data[t_var][s_var][..., it_tp_flo, np.newaxis] * seg_resampled

        for t_var in temp_variance:
            for s_var in spatial_variance:
                p_label[t_var][s_var] = p_label[t_var][s_var] / np.sum(p_data[t_var][s_var], axis=-1, keepdims=True)

        return p_label


print('\n\n\n\n\n')
print('# -------------------------------------------- #')
print('# Running the longitudinal segmentation script #')
print('# -------------------------------------------- #')
print('\n\n')

#####################
# Global parameters #
#####################

parameter_dict_MRI = configFile.REGISTRATION_DIR
REScmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_resample'
TRANSFORMcmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_transform'

# Parameters
arg_parser = ArgumentParser(description='Computes the prediction of certain models')
arg_parser.add_argument('--subjects', default=None, nargs='+')
arg_parser.add_argument('--num_cores', default=1, type=int)
arg_parser.add_argument('--reg_algorithm', default='bidir', choices=['standard', 'bidir', 'niftyreg'])
arg_parser.add_argument('--force', action='store_true')


arguments = arg_parser.parse_args()
initial_subject_list = arguments.subjects
num_cores = arguments.num_cores
reg_algorithm = arguments.reg_algorithm
force_flag = arguments.force

##############
# Processing #
##############
reg_name = reg_algorithm
if reg_algorithm == 'bidir':
    parameter_dict = configFile.CONFIG_REGISTRATION
    reg_name += str(parameter_dict['LOSS_REGISTRATION_SMOOTHNESS']['lambda'])
demo_dict = read_demo_info(demo_fields=['AGE'])
data_loader = DataLoader(sid_list=initial_subject_list, linear=True, reg_algorithm=reg_name)
parameter_dict = configFile.get_config_dict(data_loader.image_shape)

subject_list = data_loader.subject_list
subject_list = list(filter(lambda x: x.id in demo_dict.keys(), subject_list))

print('[LONGITUDINAL SEGMENTATION] Start processing.')
spatial_variance = [1]#[3**2, 7**2, 11**2] # in grayscale
temp_variance = ['inf']#[1/4, 1/2, 1, 2] #in years^2

segmenter = FS_Lin(demo_dict=demo_dict, parameter_dict=parameter_dict, reg_algorithm=reg_name, device=device,
                   long_seg_algo='linear', temp_variance=temp_variance, spatial_variance=spatial_variance)

if num_cores == 1:
    for it_subject, subject in enumerate(subject_list):
        print(str(it_subject+1) + '/' + str(len(subject_list)) + ' -- ', end='', flush=True)
        segmenter.label_fusion(subject, force_flag=force_flag)

elif len(subject_list) == 1:
    segmenter.label_fusion(subject_list[0], force_flag=force_flag)

else:
    results = Parallel(n_jobs=num_cores)(delayed(segmenter.label_fusion)(subject) for subject in subject_list)








