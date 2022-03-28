import time

def train(model,
          optimizer,
          device,
          generator_train,
          epoch,
          loss_function_dict,
          loss_weight_dict,
          callbacks,
          da_model,
          mask_flag=False
          ):

    total_iter = 0
    N = len(generator_train.dataset)
    model = model.to(device)

    rid_epoch_list = []
    for batch_idx, data_list in enumerate(generator_train):

        ref_image, flo_image = data_list['x_ref'].to(device), data_list['x_flo'].to(device)
        ref_mask, flo_mask = data_list['x_ref_mask'].to(device), data_list['x_flo_mask'].to(device)

        nonlinear_field = [nlf.to(device) for nlf in data_list['nonlinear']] if 'nonlinear' in data_list.keys() else  [None] * 2
        affine_field = [nlf.to(device) for nlf in data_list['affine']] if 'affine' in data_list.keys() else [None] * 2

        rid_epoch_list.extend(data_list['rid'])

        total_iter += len(ref_image)
        model.zero_grad()

        if da_model is not None:
            ref_image = da_model.transform(ref_image, affine_field[0], nonlinear_field[0])
            flo_image = da_model.transform(flo_image, affine_field[1], nonlinear_field[1])
            ref_mask = da_model.transform(ref_mask, affine_field[0], nonlinear_field[0])

        # import pdb
        # import nibabel as nib
        # import numpy as np
        # pdb.set_trace()
        #
        # img = nib.Nifti1Image(ref_image[0, 0].detach().cpu().numpy(), np.eye(4))
        # nib.save(img, 'ref_' + str(batch_idx) + '.nii.gz')
        # img = nib.Nifti1Image(flo_image[0, 0].detach().cpu().numpy(), np.eye(4))
        # nib.save(img, 'flo_' + str(batch_idx) + '.nii.gz')

        reg_image, flow_image, v_image = model(flo_image, ref_image)

        loss_dict = {}
        if mask_flag:
            loss_dict['registration'] = loss_function_dict['registration'](reg_image, ref_image, ref_mask)
        else:
            loss_dict['registration'] = loss_function_dict['registration'](reg_image, ref_image)

        loss_dict['registration_smoothness'] = loss_function_dict['registration_smoothness'](v_image)

        log_dict = {'loss_' + loss_name: loss_value.item() for loss_name, loss_value in loss_dict.items()}

        for k,v in loss_dict.items():
            loss_dict[k] = loss_weight_dict[k] * v

        total_loss = sum([l for l in loss_dict.values()])

        total_loss.backward()
        # plot_grad_flow(model.named_parameters(), save_dir='model_reg' + str(epoch))
        optimizer.step()

        log_dict = {**log_dict,
                    **{'w_loss_' + loss_name: loss_value.item() for loss_name, loss_value in loss_dict.items()},
                    **{'loss': total_loss.item()}}

        for cb in callbacks:
            cb.on_step_fi(log_dict, model, epoch, iteration=total_iter, N=N)


    return callbacks

def train_bidirectional(model,
                        optimizer,
                        device,
                        generator_train,
                        epoch,
                        loss_function_dict,
                        loss_weight_dict,
                        callbacks,
                        da_model=None,
                        mask_flag=True
                        ):

    total_iter = 0
    N = len(generator_train.dataset)
    rid_epoch_list = []

    t_0 = time.time()
    for batch_idx, data_list in enumerate(generator_train):
        t_1 = time.time()
        ref_image, flo_image = data_list['x_ref'].to(device), data_list['x_flo'].to(device)
        ref_mask, flo_mask = data_list['x_ref_mask'].to(device), data_list['x_flo_mask'].to(device)

        nonlinear_field = [nlf.to(device) for nlf in data_list['nonlinear']] if 'nonlinear' in data_list.keys() else [None]*2
        affine_field = [nlf.to(device) for nlf in data_list['affine']] if 'affine' in data_list.keys() else [None]*2
        rid_epoch_list.extend(data_list['rid'])

        total_iter += len(ref_image)
        model.zero_grad()

        if da_model is not None:
            ref_image = da_model.transform(ref_image, affine_field[0], nonlinear_field[0])
            flo_image = da_model.transform(flo_image, affine_field[1], nonlinear_field[1])
            ref_mask = da_model.transform(ref_mask, affine_field[0], nonlinear_field[0])
            flo_mask = da_model.transform(flo_mask, affine_field[1], nonlinear_field[1])

        t_2 = time.time()
        import pdb
        import nibabel as nib
        import numpy as np

        # pdb.set_trace()
        # img = nib.Nifti1Image(ref_image[0, 0].detach().cpu().numpy(), np.eye(4))
        # nib.save(img, 'ref_' + str(batch_idx) + '.nii.gz')
        # img = nib.Nifti1Image(flo_image[0, 0].detach().cpu().numpy(), np.eye(4))
        # nib.save(img, 'flo_' + str(batch_idx) + '.nii.gz')
        # deformation = da_model.get_nonlin_field(nonlinear_field[0])
        # img = nib.Nifti1Image(np.transpose(deformation[0].detach().cpu().numpy(), axes=[1,2,3,0]), np.eye(4))
        # nib.save(img, 'def_' + str(batch_idx) + '.nii.gz')

        reg_flo_image, flow_image, v_image = model(flo_image, ref_image)
        reg_ref_image = model.predict(ref_image, -v_image, svf=True)

        t_3 = time.time()

        loss_dict = {}
        if mask_flag:
            loss_dict['registration'] = loss_function_dict['registration'](reg_ref_image, flo_image, flo_mask)
            loss_dict['registration'] += loss_function_dict['registration'](reg_flo_image, ref_image, ref_mask)
            loss_dict['registration'] = 0.5 * loss_dict['registration']

        else:
            loss_dict['registration'] = loss_function_dict['registration'](reg_ref_image, flo_image)
            loss_dict['registration'] += loss_function_dict['registration'](reg_flo_image, ref_image)
            loss_dict['registration'] = 0.5 * loss_dict['registration']

        loss_dict['registration_smoothness'] = loss_function_dict['registration_smoothness'](v_image)

        log_dict = {'loss_' + loss_name: loss_value.item() for loss_name, loss_value in loss_dict.items()}
        for k, v in loss_dict.items():
            loss_dict[k] = loss_weight_dict[k] * v

        wlog_dict = {'w_loss_' + loss_name: loss_value.item() for loss_name, loss_value in loss_dict.items()}
        total_loss = sum([l for l in loss_dict.values()])
        tlog_dict = {'loss': total_loss.item()}

        t_4 = time.time()

        total_loss.backward()
        optimizer.step()

        log_dict = {**log_dict, **wlog_dict, **tlog_dict}


        for cb in callbacks:
            cb.on_step_fi(log_dict, model, epoch, iteration=total_iter, N=N)

        # print('Loading time: ' + str(t_1 - t_0))
        # print('Load and deform ' + str(t_2 - t_1))
        # print('Forward pass ' + str(t_3 - t_2))
        # print('Compute loss: ' + str(t_4 - t_3))
        # t_0 = time.time()
        # print('Backward and optimiser: ' + str(t_0 - t_4))

    return callbacks


