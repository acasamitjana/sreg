from os.path import join
import itertools
import pdb
import csv

import torch
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import binary_dilation
from scipy.optimize import linprog
import gurobipy as gp
import nibabel as nib
from skimage.transform import resize

from src.utils.io_utils import read_affine_matrix
from src.callbacks import History, ModelCheckpoint, PrinterCallback
from src.models import InstanceRigidModel

# Read linear st2 graph
# Formulas extracted from: https://math.stackexchange.com/questions/3031999/proof-of-logarithm-map-formulae-from-so3-to-mathfrak-so3
def init_st2_lineal(timepoints, input_dir, eps = 1e-6):

    timepoints_dict = {
       t.tid: it_t for it_t, t in enumerate(timepoints)
    } # needed for non consecutive timepoints (if we'd like to skip one for whatever reason)

    nk = 0

    N = len(timepoints)
    K = int(N*(N-1)/2)

    phi_log = np.zeros((6, K))

    for tp_ref, tp_flo in itertools.combinations(timepoints, 2):

        tid_ref, tid_flo = tp_ref.id, tp_flo.id

        filename = str(tid_ref) + '_to_' + str(tid_flo)

        rotation_matrix, translation_vector = read_affine_matrix(join(input_dir, filename + '.aff'))

        # Log(R) and Log(T)
        t_norm = np.arccos(np.clip((np.trace(rotation_matrix) - 1) / 2, -1 + eps, 1 - eps)) + eps
        W = 1 / (2 * np.sin(t_norm)) * (rotation_matrix - rotation_matrix.T) * t_norm
        Vinv = np.eye(3) - 0.5 * W + ((1 - (t_norm * np.cos(t_norm / 2)) / (2 * np.sin(t_norm / 2))) / t_norm ** 2) * np.matmul(W, W)


        phi_log[0, nk] = 1 / (2 * np.sin(t_norm)) * (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * t_norm
        phi_log[1, nk] = 1 / (2 * np.sin(t_norm)) * (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * t_norm
        phi_log[2, nk] = 1 / (2 * np.sin(t_norm)) * (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * t_norm

        phi_log[3:, nk] = np.matmul(Vinv, translation_vector)

        nk += 1

    return phi_log

# Read st2 graph
def init_st2(timepoints, input_dir, image_shape, se = None):

    timepoints_dict = {
       t.tid: it_t for it_t, t in enumerate(timepoints)
    } # needed for non consecutive timepoints (if we'd like to skip one for whatever reason)

    nk = 0

    N = len(timepoints)
    K = int(N*(N-1)/2) +1

    w = np.zeros((K, N), dtype='int')
    obs_mask = np.zeros(image_shape + (K,))

    phi = np.zeros((3,) + image_shape + (K, ))
    for sl_ref, sl_flo in itertools.combinations(timepoints, 2):

        sid_ref, sid_flo = sl_ref.id, sl_flo.id

        t0 = timepoints_dict[sl_ref.tid]
        t1 = timepoints_dict[sl_flo.tid]
        filename = str(sid_ref) + '_to_' + str(sid_flo)

        proxy = nib.load(join(input_dir, filename + '.svf.nii.gz'))
        field = np.asarray(proxy.dataobj)
        if field.shape[1:] != image_shape:
            phi[0, ..., nk] = resize(field[0], image_shape)
            phi[1, ..., nk] = resize(field[1], image_shape)
            phi[2, ..., nk] = resize(field[2], image_shape)
        else:
            phi[0, ..., nk] = field[0]
            phi[1, ..., nk] = field[1]
            phi[2, ..., nk] = field[2]

        # Masks
        proxy = nib.load(join(input_dir, filename + '.mask.nii.gz'))
        mask_mov = np.squeeze(np.asarray(proxy.dataobj) > 0).astype('uint8')

        # proxy = nib.load(join(subject_dir, sid_ref,  'mask.reoriented.nii.gz'))
        # mask_ref = (np.asarray(proxy.dataobj) > 0).astype('uint8')
        mask = mask_mov #* mask_ref
        # del mask_mov, mask_ref
        mask = (resize(np.double(mask), image_shape, anti_aliasing=True) > 0).astype('uint8')
        if se is not None:
            mask = binary_dilation(mask, se)

        obs_mask[..., nk] = mask
        w[nk, t0] = -1
        w[nk, t1] = 1
        nk += 1

    obs_mask[..., nk] = (np.sum(obs_mask[..., :nk-1]) > 0).astype('uint8')
    w[nk, :] = 1
    nk += 1
    return phi, obs_mask, w, nk


# Optimization of rigid transforms using pytorch
def st2_lineal_pytorch(logR, timepoints, n_epochs, subject_shape, cost, lr, results_dir_sbj, max_iter=20, patience=5,
                       verbose=True):
    #(R_log, model, optimizer, callbacks, n_epochs, timepoints):

    log_keys = ['loss', 'time_duration (s)']
    logger = History(log_keys)
    model_checkpoint = ModelCheckpoint(join(results_dir_sbj, 'checkpoints'), -1)
    callbacks = [logger, model_checkpoint]
    if verbose: callbacks += [PrinterCallback()]

    model = InstanceRigidModel(subject_shape, timepoints, cost=cost, device='cpu')
    optimizer = torch.optim.LBFGS(params=model.parameters(), lr=lr, max_iter=max_iter, line_search_fn='strong_wolfe')

    min_loss = 1000
    iter_break = 0
    log_dict = {}
    logR = torch.FloatTensor(logR)
    for cb in callbacks:
        cb.on_train_init(model)

    for epoch in range(n_epochs):
        for cb in callbacks:
            cb.on_epoch_init(model, epoch)

        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()

            loss = model(logR, timepoints)
            loss.backward()

            return loss

        optimizer.step(closure=closure)

        loss = model(logR, timepoints)

        if loss < min_loss + 1e-6:
            iter_break = 1
        else:
            iter_break += 1

        if iter_break > patience:
            break

        log_dict['loss'] = loss.item()

        for cb in callbacks:
            cb.on_step_fi(log_dict, model, epoch, iteration=1, N=1)

    T = model._compute_matrix()

    return T.cpu().detach().numpy()


# Gaussian: l2-loss no masks
def st2_L2_global(phi, W, N):

    #Initialize transforms
    # Tres = np.zeros(phi.shape[:4] + (N,))
    print('    Computing weights and updating the transforms')
    precision = 1e-6
    lambda_control = np.linalg.inv((W.T @ W) + precision * np.eye(N)) @ W.T
    Tres = lambda_control @ np.transpose(phi,[1,2,3,4,0])
    Tres = np.transpose(Tres, [4,0,1,2,3])

    return Tres

# Gaussian: l2-loss
def st2_L2(phi, obs_mask, w, N):

    image_shape = obs_mask.shape[:-1]

    #Initialize transforms
    Tres = np.zeros(phi.shape[:4] + (N,))

    print('    Computing weights and updating the transforms')
    precision = 1e-6
    for it_control_row in range(image_shape[0]):
        if np.mod(it_control_row, 10) == 0:
            print('       Row ' + str(it_control_row) + '/' + str(image_shape[0]))

        for it_control_col in range(image_shape[1]):
            # if np.mod(it_control_col, 10) == 0:
            #     print('           Col ' + str(it_control_col) + '/' + str(image_shape[1]))

            for it_control_depth in range(image_shape[2]):

                index_obs = np.where(obs_mask[it_control_row, it_control_col, it_control_depth, :] == 1)[0]
                if index_obs.shape[0] == 0:
                    Tres[:, it_control_row, it_control_col, it_control_depth] = 0
                else:
                    w_control = w[index_obs]
                    phi_control = phi[:,it_control_row, it_control_col, it_control_depth, index_obs]
                    lambda_control = np.linalg.inv(w_control.T @ (w_control + precision*np.eye(N))) @ w_control.T

                    for it_tf in range(N):
                        Tres[0, it_control_row, it_control_col, it_control_depth, it_tf] = lambda_control[it_tf] @ phi_control[0].T
                        Tres[1, it_control_row, it_control_col, it_control_depth, it_tf] = lambda_control[it_tf] @ phi_control[1].T
                        Tres[2, it_control_row, it_control_col, it_control_depth, it_tf] = lambda_control[it_tf] @ phi_control[2].T

    return Tres


# Laplacian: l1-loss
def st2_L1(phi, obs_mask, w, N):

    image_shape = obs_mask.shape[:3]
    Tres = np.zeros(phi.shape[:4] + (N,))
    for it_control_row in range(image_shape[0]):
        if np.mod(it_control_row, 10) == 0:
            print('    Row ' + str(it_control_row) + '/' + str(image_shape[0]))
        for it_control_col in range(image_shape[1]):
            for it_control_depth in range(image_shape[2]):
                index_obs = np.where(obs_mask[it_control_row, it_control_col, it_control_depth, :] == 1)[0]

                if index_obs.shape[0] > 0:
                    w_control = w[index_obs]
                    phi_control = phi[:, it_control_row, it_control_col, it_control_depth, index_obs]
                    n_control = len(index_obs)

                    for it_dim in range(3):

                        # Set objective
                        c_lp = np.concatenate((np.ones((n_control,)), np.zeros((N,))), axis=0)

                        # Set the inequality
                        A_lp = np.zeros((2 * n_control, n_control + N))
                        A_lp[:n_control, :n_control] = -np.eye(n_control)
                        A_lp[:n_control, n_control:] = -w_control
                        A_lp[n_control:, :n_control] = -np.eye(n_control)
                        A_lp[n_control:, n_control:] = w_control
                        A_lp = sp.csr_matrix(A_lp)

                        reg = np.reshape(phi_control[it_dim], (n_control,))
                        b_lp = np.concatenate((-reg, reg), axis=0)

                        result = linprog(c_lp, A_lp, b_lp, bounds=(None, None), method='highs-ds')

                        Tres[it_dim, it_control_row, it_control_col, it_control_depth] = result.x[n_control:]

                        # model = gp.Model('LP')
                        # model.setParam('OutputFlag', False)
                        # model.setParam('Method', 1)
                        #
                        # # Set the parameters
                        # params = model.addMVar(shape=n_control + N, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name='x')
                        #
                        # # Set objective
                        # c_lp = np.concatenate((np.ones((n_control,)), np.zeros((N,))), axis=0)
                        # model.setObjective(c_lp @ params, gp.GRB.MINIMIZE)
                        #
                        # # Set the inequality
                        # A_lp = np.zeros((2 * n_control, n_control + N))
                        # A_lp[:n_control, :n_control] = -np.eye(n_control)
                        # A_lp[:n_control, n_control:] = -w_control
                        # A_lp[n_control:, :n_control] = -np.eye(n_control)
                        # A_lp[n_control:, n_control:] = w_control
                        # A_lp = sp.csr_matrix(A_lp)
                        #
                        # reg = np.reshape(phi_control[it_dim], (n_control,))
                        # b_lp = np.concatenate((-reg, reg), axis=0)
                        #
                        # model.addConstr(A_lp @ params <= b_lp, name="c")
                        #
                        # model.optimize()
                        #
                        # Tres[it_dim, it_control_row, it_control_col, it_control_depth] = params.X[n_control:]

    return Tres

