import numpy as np
import torch


def predict_registration(data_dict, model, device, da_model=None):

    with torch.no_grad():
        ref_image, flo_image = data_dict['x_ref'].to(device), data_dict['x_flo'].to(device)
        ref_mask, flo_mask = data_dict['x_ref_mask'].to(device), data_dict['x_flo_mask'].to(device)
        ref_labels, flo_labels = data_dict['x_ref_labels'].to(device), data_dict['x_flo_labels'].to(device)
        nonlinear_field = [nlf.to(device) for nlf in data_dict['nonlinear']] if 'nonlinear' in data_dict.keys() else [None]*2
        affine_field = [nlf.to(device) for nlf in data_dict['affine']] if 'affine' in data_dict.keys() else [None]*2

        if da_model is not None:
            flo_image_fake = da_model.transform(flo_image, affine_field[0], nonlinear_field[0])
            flo_mask_fake = da_model.transform(flo_mask, affine_field[0], nonlinear_field[0])
            flo_labels_fake = da_model.transform(flo_labels, affine_field[0], nonlinear_field[0])
        else:
            flo_image_fake = flo_image
            flo_mask_fake = flo_mask
            flo_labels_fake = flo_labels

        r, f, v = model(flo_image_fake, ref_image)
        f_rev = model.get_flow_field(-v)

        r_mask = model.predict(flo_mask_fake, f, svf=False)
        r_labels = model.predict(flo_labels_fake, f, svf=False)

        r_flo = model.predict(ref_image, f_rev, svf=False)
        r_flo_mask = model.predict(ref_mask, f_rev, svf=False)
        flow = f.cpu().detach().numpy()

        ref_image = ref_image[:,0].cpu().detach().numpy()
        flo_image = flo_image_fake[:,0].cpu().detach().numpy()
        reg_image_ref = r[:,0].cpu().detach().numpy()
        reg_image_flo = r_flo[:,0].cpu().detach().numpy()

        ref_mask = ref_mask[:,0].cpu().detach().numpy()
        flo_mask = flo_mask_fake[:,0].cpu().detach().numpy()
        reg_mask_ref = r_mask[:,0].cpu().detach().numpy()
        reg_mask_flo = r_flo_mask[:,0].cpu().detach().numpy()
        reg_labels_ref = np.argmax(r_labels.cpu().detach().numpy(), axis=1)

    return ref_image, flo_image, reg_image_ref, reg_image_flo, \
           ref_mask, flo_mask, reg_mask_ref, reg_mask_flo, reg_labels_ref, flow