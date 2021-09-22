
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
        nonlinear_field = [nlf.to(device) for nlf in data_list['nonlinear']]
        affine_field = [nlf.to(device) for nlf in data_list['affine']]
        rid_epoch_list.extend(data_list['rid'])

        total_iter += len(ref_image)
        model.zero_grad()

        if da_model is not None:
            ref_image = da_model.transform(ref_image, affine_field[0], nonlinear_field[0])
            flo_image = da_model.transform(flo_image, affine_field[1], nonlinear_field[1])
            ref_mask = da_model.transform(ref_mask, affine_field[0], nonlinear_field[0])

        reg_image, flow_image, v_image = model(flo_image, ref_image)

        # import pdb
        # pdb.set_trace()
        # # if epoch == 25:
        #
        # import nibabel as nib
        # import numpy as np
        #
        # img = nib.Nifti1Image(ref_image[0, 0].detach().cpu().numpy(), np.eye(4))
        # nib.save(img, 'prova1.nii.gz')
        # img = nib.Nifti1Image(flo_image[0, 0].detach().cpu().numpy(), np.eye(4))
        # nib.save(img, 'prova2.nii.gz')
        # img = nib.Nifti1Image(ref_mask[0, 0].detach().cpu().numpy(), np.eye(4))
        # nib.save(img, 'prova3.nii.gz')
        # img = nib.Nifti1Image(flo_mask[0, 0].detach().cpu().numpy(), np.eye(4))
        # nib.save(img, 'prova4.nii.gz')
        #
        # img = nib.Nifti1Image(v_image[0].detach().cpu().numpy(), np.eye(4))
        # nib.save(img, 'prova3.nii.gz')
        # img = nib.Nifti1Image(np.transpose(flow_image[0].detach().cpu().numpy(),[1,2,3,0]), np.eye(4))
        # nib.save(img, 'prova4.nii.gz')

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
                        da_model,
                        mask_flag=True
                        ):

    total_iter = 0
    N = len(generator_train.dataset)
    rid_epoch_list = []
    for batch_idx, data_list in enumerate(generator_train):

        ref_image, flo_image = data_list['x_ref'].to(device), data_list['x_flo'].to(device)
        ref_mask, flo_mask = data_list['x_ref_mask'].to(device), data_list['x_flo_mask'].to(device)

        nonlinear_field = [nlf.to(device) for nlf in data_list['nonlinear']]
        affine_field = [nlf.to(device) for nlf in data_list['affine']]
        rid_epoch_list.extend(data_list['rid'])

        total_iter += len(ref_image)
        model.zero_grad()

        if da_model is not None:
            ref_image = da_model.transform(ref_image, affine_field[0], nonlinear_field[0])
            flo_image = da_model.transform(flo_image, affine_field[1], nonlinear_field[1])
            ref_mask = da_model.transform(ref_mask, affine_field[0], nonlinear_field[0])
            flo_mask = da_model.transform(flo_mask, affine_field[1], nonlinear_field[1])

        # import pdb
        # pdb.set_trace()
        # import nibabel as nib
        # img = nib.Nifti1Image(ref_image[0,0].detach().cpu().numpy(), data_list['ref_vox2ras0'][0])
        # nib.save(img, 'prova.nii.gz')
        #
        # img = nib.Nifti1Image(flo_image[0, 0].detach().cpu().numpy(), data_list['flo_vox2ras0'][0])
        # nib.save(img, 'prova2.nii.gz')

        reg_flo_image, flow_image, v_image = model(flo_image, ref_image)
        reg_ref_image = model.predict(ref_image, -v_image, svf=True)

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

        total_loss = sum([l for l in loss_dict.values()])


        total_loss.backward()
        optimizer.step()

        log_dict = {**log_dict,
                    **{'w_loss_' + loss_name: loss_value.item() for loss_name, loss_value in loss_dict.items()},
                    **{'loss': total_loss.item()}}

        for cb in callbacks:
            cb.on_step_fi(log_dict, model, epoch, iteration=total_iter, N=N)

    return callbacks


