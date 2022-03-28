import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn



class _Loss(nn.Module):
    def __init__(self, name = None):
        super().__init__()
        self.name = name

class ID_Loss(_Loss):
    def forward(self, x, *args, **kwargs):
        return x

class SSIM(_Loss):
    def __init__(self, name = None, reduction='mean', *args, **kwargs):
        if name is None:
            name='SSIM'
        super().__init__(name=name)
        self.reduction = reduction

    @NotImplementedError
    def _ssim_loss(self, prediction, target, reduction='mean'):
        pass

    def forward(self, prediction, target, mask=None, weight=None, *args, **kwargs):
        ndims = len(prediction.shape)
        if mask is None and weight is None:
            return self._ssim_loss(prediction, target, reduction=self.reduction)
        else:
            res = self._ssim_loss(prediction, target, reduction='none')
            if mask is not None:
                res = res * mask
            if weight is not None:
                res = res * weight

        if self.reduction == 'mean':
            return 1/torch.sum(mask)*torch.sum(res)
            # norm_factor = torch.sum(mask, dim=[it for it in range(1,ndims)], keepdim=True)
            # wk = 1 / norm_factor * torch.sum(mask, dim=[it for it in range(2,ndims)])
            # res = 1 / norm_factor * torch.sum(res, dim=[it for it in range(2,ndims)])
            # res = torch.sum(wk*res, dim=1)
            # res = torch.mean(res)

        elif self.reduction == 'sum':
            return torch.sum(res)
        else:
            return res

class L2_Loss(SSIM):
    def _ssim_loss(self, prediction, target, reduction='mean'):
        return F.mse_loss(prediction, target, reduction=reduction)

class L1_Loss(SSIM):
    def _ssim_loss(self, prediction, target, reduction='mean'):
        return F.l1_loss(prediction, target, reduction=reduction)


class NCC_Loss(_Loss):

    def __init__(self, device, kernel_var=None, name=None, kernel_type='mean', eps=1e-5, *args, **kwargs):
        if name is None:
            name = 'ncc'
        super().__init__(name=name)
        self.device = device
        self.kernel_var = kernel_var
        self.kernel_type = kernel_type
        self.eps = eps

        assert kernel_type in ['mean', 'gaussian', 'linear']

    def _get_kernel(self, kernel_type, kernel_sigma):

        if kernel_type == 'mean':
            kernel = torch.ones([1, 1, *kernel_sigma]).to(self.device)

        elif kernel_type == 'linear':
            raise NotImplementedError("Linear kernel for NCC still not implemented")

        elif kernel_type == 'gaussian':
            kernel_size = kernel_sigma[0] * 3
            kernel_size += np.mod(kernel_size + 1, 2)

            # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
            x_cord = torch.arange(kernel_size)
            x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
            y_grid = x_grid.t()
            xy_grid = torch.stack([x_grid, y_grid], dim=-1)

            mean = (kernel_size - 1) / 2.
            variance = kernel_sigma[0] ** 2.

            # Calculate the 2-dimensional gaussian kernel which is
            # the product of two gaussian distributions for two different
            # variables (in this case called x and y)
            # 2.506628274631 = sqrt(2 * pi)

            kernel = (1. / (2.506628274631 * kernel_sigma[0])) * \
                     torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

            # Make sure sum of values in gaussian kernel equals 1.
            # gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

            # Reshape to 2d depthwise convolutional weight
            kernel = kernel.view(1, 1, kernel_size, kernel_size)
            kernel = kernel.to(self.device)

        return kernel

    def _compute_local_sums(self, I, J, filt, stride, padding):

        ndims = len(list(I.size())) - 2

        I2 = I * I
        J2 = J * J
        IJ = I * J

        conv_fn = getattr(F, 'conv%dd' % ndims)

        I_sum = conv_fn(I, filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, filt, stride=stride, padding=padding)

        win_size = torch.sum(filt)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        return I_var, J_var, cross

    def ncc(self, prediction, target):
        """
            calculate the normalize cross correlation between I and J
            assumes I, J are sized [batch_size, nb_feats, *vol_shape]
            """

        ndims = len(list(prediction.size())) - 2

        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        if self.kernel_var is None:
            if self.kernel_type == 'gaussian':
                kernel_var = [3] * ndims  # sigma=3, radius = 9
            else:
                kernel_var = [9] * ndims  # sigma=radius=9 for mean and linear filter

        else:
            kernel_var = self.kernel_var

        sum_filt = self._get_kernel(self.kernel_type, kernel_var)
        radius = sum_filt.shape[-1]
        pad_no = int(np.floor(radius / 2))

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # Eugenio: bug fixed where cross was not squared when computing cc
        I_var, J_var, cross = self._compute_local_sums(prediction, target, sum_filt, stride, padding)
        cc = cross * cross / (I_var * J_var + self.eps)

        return cc

    def forward(self, prediction, target, mask=None, *args, **kwargs):

        # if mask is not None:
        #     prediction = prediction * mask
        #     target = target * mask

        cc = self.ncc(prediction, target)
        if mask is None:
            return -1.0 * torch.sqrt(torch.mean(cc))
        elif torch.sum(mask) == 0:
            return torch.tensor(0)
        else:
            norm_factor = 1 / (torch.sum(mask))
            return -1.0 * torch.sqrt(norm_factor * torch.sum(cc * mask))

class NMI_Loss(_Loss):

    # Eugenio: what is vol_size needed for?
    def __init__(self, bin_centers, # vol_size,
                 device='cpu', sigma_ratio=0.5, max_clip=1,
                 crop_background=False, patch_size=1, name='nmi'):
        """
        Adapted from:

        Mutual information loss for image-image pairs.
        Author: Courtney Guo
        If you use this loss function, please cite the following:
        Guo, Courtney K. Multi-modal image registration with unsupervised deep learning. MEng. Thesis
        Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces
        Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
        MedIA: Medial Image Analysis. 2019. eprint arXiv:1903.03545
        """
        super().__init__(name=name)
        # self.vol_size = vol_size
        self.max_clip = max_clip
        self.patch_size = patch_size
        self.crop_background = crop_background
        self.mi = self.global_mi
        self.vol_bin_centers = torch.tensor(bin_centers, requires_grad=True, device=device, dtype=torch.float32)
        self.num_bins = len(bin_centers)
        self.sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        self.preterm = 1 / (2 * np.square(self.sigma))

    def global_mi(self, y_true, y_pred, padding_size=15, mask=None):

        ndims = len(y_true.shape) - 2
        conv_fn = getattr(F, 'conv%dd' % ndims)

        if self.crop_background:
            # does not support variable batch size
            thresh = 0.0001
            if mask is None:
                if ndims == 2:
                    filt = torch.ones((1, 1, padding_size, padding_size))
                    stride = [1, 1, 1, 1]
                elif ndims == 3:
                    filt = torch.ones((1, 1, padding_size, padding_size, padding_size))
                    stride = [1, 1, 1, 1, 1]
                else:
                    raise ValueError('Not valid image size')
                smooth = conv_fn(y_true, filt, stride=stride, padding=int((padding_size-1)/2))
                mask = smooth > thresh
            else:
                mask = mask > thresh

            y_pred = torch.masked_select(y_pred, mask).view((1,1,-1))
            y_true = torch.masked_select(y_true, mask).view((1,1,-1))

        else:
            # reshape: flatten images into shape (batch_size, 1, heightxwidthxdepthxchan)
            y_true = y_true.view((1,1,-1))
            y_pred = y_pred.view((1,1,-1))

        nb_voxels = y_pred.shape[2]

        # reshape bin centers to be (1, 1, B)
        o = [1, self.vol_bin_centers.numel(), 1]
        vbc = torch.reshape(self.vol_bin_centers, o)

        # compute image terms
        I_a = torch.exp(- self.preterm * (y_true  - vbc)**2)
        I_a_norm = I_a / torch.sum(I_a, dim=1, keepdim=True)

        I_b = torch.exp(- self.preterm * (y_pred  - vbc)**2)
        I_b_norm = I_b / torch.sum(I_b, dim=1, keepdim=True)

        # compute probabilities
        I_a_norm_permute = torch.transpose(I_a_norm, dim0=1, dim1=2)
        pab = torch.bmm(I_b_norm, I_a_norm_permute)  # should be the right size now, nb_labels x nb_bins
        pab /= nb_voxels

        pa = torch.mean(I_a_norm, dim=-1, keepdim=True)
        pb = torch.mean(I_b_norm, dim=-1, keepdim=True)
        papb = torch.bmm(pb, torch.transpose(pa, dim0=1, dim1=2)) + 1e-5

        return torch.sum(torch.sum(pab * torch.log(pab/papb + 1e-5), dim=1), dim=1)

    def __call__(self, y_true, y_pred, **kwargs):
        y_pred = torch.clamp(y_pred, 0, self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)
        return -self.mi(y_true, y_pred, **kwargs)


###############################
######## Deformation ##########
###############################
class Grad_Loss(_Loss):
    def __init__(self, dim=2, penalty='l2', name=None, loss_mult=None, *args, **kwargs):
        if name is None:
            name='gradient'
        super().__init__(name=name, *args, **kwargs)

        assert dim in [2, 3]
        self.dim = dim
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _grad3d(self, prediction):
        dy = torch.abs(prediction[:, :, 1:, :, :] - prediction[:, :, :-1, :, :])
        dx = torch.abs(prediction[:, :, :, 1:, :] - prediction[:, :, :, :-1, :])
        dz = torch.abs(prediction[:, :, :, :, 1:] - prediction[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)

        return d / 3.0

    def _grad2d(self, prediction):
        dy = torch.abs(prediction[:, :, 1:, :] - prediction[:, :, :-1, :])
        dx = torch.abs(prediction[:, :, :, 1:] - prediction[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)

        return d / 2.0


    def forward(self, prediction, *args, **kwargs):
        if 'mask' in kwargs:
            prediction = prediction * kwargs['mask']

        if self.dim == 2:
            loss = self._grad2d(prediction)
        else:
            loss = self._grad3d(prediction)

        if self.loss_mult is not None:
            loss *= self.loss_mult

        return loss



###############################
###### Semi-supervised ########
###############################
class BCE_Loss(_Loss):
    def __init__(self, weight=None, reduction='mean', name=None, *args, **kwargs):
        if name is None:
            name = 'bce'

        super().__init__(name=name)
        self.weight = weight
        self.reduction = reduction

    def forward(self, prediction, target, *args, **kwargs):
        return F.binary_cross_entropy(prediction, target, weight=self.weight, reduction=self.reduction)

class CrossEntropy_Loss(_Loss):
    def __init__(self, name=None, class_weights=None, ignore_index=-100, reduction='mean'):
        if name is None:
            name = 'cross_entropy'
        super().__init__(name=name)
        self.loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index, reduction=reduction)

    def forward(self, prediction, target, **kwargs):
        """
        Compute the cross-entropy loss between predictions and targets
        Arguments:
            - prediction: torch variable of size (batch_size, num_classes, d1, d2, ..., dN) representing log probabilities for each class
            - target: torch variable of size (batch_size, num_classes, d1, d2, ..., dN) representing  the ground-truth values
        """
        _, labels = target.max(dim=1)
        return self.loss(prediction, labels)

class Dice_Loss(_Loss):
    def __init__(self, name=None, *args, **kwargs):
        if name is None:
            name = 'dice'
        super().__init__(name=name)

    def forward(self, prediction, target, eps=0.0000001):
        """Dice loss.
        Compute the dice similarity loss (approximation of the DSC). The foreground

        Parameters
        ----------
        prediction : torch variable of size (batch_size, num_classes, d1, d2, ..., dN) representing log probabilities for
            each class

        target : torch variable of ssize (batch_size, num_classes, d1, d2, ..., dN) representing a 1-hot encoding of the
            target values

        Returns
        -------
        dice_total :

        """

        smooth = eps  # 1.

        pflat = prediction.view(-1)
        tflat = target.view(-1)
        intersection = (pflat * tflat).sum()

        return 1 - ((2. * intersection + smooth) / (pflat.sum() + tflat.sum() + smooth))



DICT_LOSSES = {
    'ID': ID_Loss,

    'L1': L1_Loss,
    'L2': L2_Loss,
    'NCC': NCC_Loss,
    'NMI': NMI_Loss,

    'BCE': BCE_Loss,
    'Dice': Dice_Loss,
    'CrossEntropy': CrossEntropy_Loss,

    'Grad': Grad_Loss
}

