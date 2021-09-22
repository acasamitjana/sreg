import itertools
import pdb

import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal

from src.layers import  ConvBlock2D, SpatialTransformer, VecInt, RescaleTransform, ConvBlock3D, SpatialTransformerAffine
from src.utils.tensor_utils import Init_net


class BaseModel(nn.Module):
    pass

########################################
##   UNet like
########################################

class Unet(BaseModel):
    """
    Voxelmorph Unet. For more information see voxelmorph.net
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1, activation='lrelu',
                 cpoints_level=1):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = self._default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        nlayers_uparm = len(self.enc_nf) - int(np.log2(cpoints_level))
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            if ndims == 2:
                self.downarm.append(ConvBlock2D(prev_nf, nf, stride=2, activation=activation, norm_layer='none', padding=1))
            elif ndims == 3:
                self.downarm.append(ConvBlock3D(prev_nf, nf, stride=2, activation=activation, norm_layer='none', padding=1))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:nlayers_uparm]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            if ndims == 2:
                self.uparm.append(ConvBlock2D(channels, nf, stride=1, activation=activation, norm_layer='none', padding=1))
            elif ndims == 3:
                self.uparm.append(ConvBlock3D(channels, nf, stride=1, activation=activation, norm_layer='none', padding=1))

            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf = prev_nf + 2 if cpoints_level == 1 else prev_nf + enc_history[nlayers_uparm]
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            if ndims == 2:
                self.extras.append(ConvBlock2D(prev_nf, nf, stride=1, activation=activation, norm_layer='none', padding=1))
            elif ndims == 3:
                self.extras.append(ConvBlock3D(prev_nf, nf, stride=1, activation=activation, norm_layer='none', padding=1))
            prev_nf = nf

    def _default_unet_features(self):
        nb_features = [
            [16, 32, 32, 32],  # encoder
            [32, 32, 32, 32, 32, 16, 16]  # decoder
        ]
        return nb_features

    def forward(self, x):

        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x

########################################
##   Dense Registration/Deformation
########################################

class RegNet(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
    ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            cpoints_level=int_downsize
        )
        # init_net = Init_net()
        # self.unet_model = init_net.init_net(unet_model)

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape), requires_grad=True)
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape), requires_grad=True)

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        # self.resize = RescaleTransform(inshape, factor=1 / int_downsize, gaussian_filter_flag=gaussian_filter_flag) if resize else None
        self.resize = None
        self.fullsize = RescaleTransform(inshape, factor=int_downsize) if resize else None

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        if self.resize:
            flow_field = self.resize(flow_field)

        preint_flow = flow_field

        # integrate to produce diffeomorphic warp
        if self.integrate:
            flow_field = self.integrate(flow_field)

            # resize to final resolution
            if self.fullsize:
                flow_field = self.fullsize(flow_field)

        # warp image with flow field
        y_source = self.transformer(source, flow_field)

        # return non-integrated flow field if training
        if not registration:
            return y_source, flow_field, preint_flow
        else:
            return y_source, flow_field

    def predict(self, image, flow, svf=True, **kwargs):

        if svf:
            flow = self.integrate(flow)

            if self.fullsize:
                flow = self.fullsize(flow)

        return self.transformer(image, flow, **kwargs)

    def get_flow_field(self, flow_field):
        if self.integrate:
            flow_field = self.integrate(flow_field)

            # resize to final resolution
            if self.fullsize:
                flow_field = self.fullsize(flow_field)

        return flow_field


########################################
##   Linear Registration/Deformation
########################################

class InstanceRigidModel(nn.Module):

    def __init__(self, vol_shape, timepoints, reg_weight=0.001, cost='l1', device='cpu', torch_dtype=torch.float):
        super().__init__()

        self.vol_shape = vol_shape
        self.device = device
        self.cost = cost
        self.reg_weight = reg_weight

        self.timepoints = timepoints
        self.N = len(timepoints)
        self.K = int(self.N * (self.N-1) / 2)

        # Parameters
        self.angle = torch.nn.Parameter(torch.zeros(3, self.N))
        self.translation = torch.nn.Parameter(torch.zeros(3, self.N))
        self.angle.requires_grad = True
        self.translation.requires_grad = True


    def _compute_matrix(self):

        angles = self.angle / 180 * np.pi

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        T = torch.zeros((4, 4, self.N))
        T[0, 0] = cos[2]*cos[1]
        T[1, 0] = sin[2]*cos[1]
        T[2, 0] = -sin[1]

        T[0, 1] = cos[2] * sin[1] * sin[0] - sin[2] * cos[0]
        T[1, 1] = sin[2] * sin[1] * sin[0] + cos[2] * cos[0]
        T[2, 1] = cos[1] * sin[0]

        T[0, 2] = cos[2] * sin[1] * cos[0] + sin[2] * sin[0]
        T[1, 2] = sin[2] * sin[1] * cos[0] - cos[2] * sin[0]
        T[2, 2] = cos[1] * cos[0]

        T[0, 3] = self.translation[0]# + self.tr0[0]
        T[1, 3] = self.translation[1]# + self.tr0[1]
        T[2, 3] = self.translation[2]# + self.tr0[2]
        T[3, 3] = 1

        #
        # for n in range(self.N):
        #
        #     T[..., n] = torch.chain_matmul(self.T0inv, T[..., n], self.T0)

        return T


    def _build_combinations(self, timepoints, latent_matrix):

        K = self.K
        timepoints_dict = {
            t.tid: it_t for it_t, t in enumerate(timepoints)
        }  # needed for non consecutive timepoints (if we'd like to skip one for whatever reason)


        Tij = torch.zeros((4, 4, K))

        k = 0
        for tp_ref, tp_flo in itertools.combinations(timepoints, 2):

            t0 = timepoints_dict[tp_ref.tid]
            t1 = timepoints_dict[tp_flo.tid]

            T0k = latent_matrix[..., t0]
            T1k = latent_matrix[..., t1]

            Tij[..., k] = torch.matmul(T1k, torch.inverse(T0k))

            k += 1

        return Tij


    def _compute_log(self, Tij):

        K = Tij.shape[-1]
        R = Tij[:3, :3]
        Tr = Tij[:3, 3]

        logTij = torch.zeros((6, K))

        eps = 1e-6
        for k in range(K):
            t_norm = torch.arccos(torch.clamp((torch.trace(R[..., k]) - 1) / 2, min=-1+eps, max=1-eps)) + eps
            W = 1 / (2 * torch.sin(t_norm)) * (R[..., k] - R[..., k].T) * t_norm
            Vinv = torch.eye(3) - 0.5 * W + ((1 - (t_norm * torch.cos(t_norm / 2)) / (2 * torch.sin(t_norm / 2))) / t_norm ** 2) * torch.matmul(W, W)

            # pdb.set_trace()

            logTij[0, k] = 1 / (2 * torch.sin(t_norm)) * (R[..., k][2, 1] - R[..., k][1, 2]) * t_norm
            logTij[1, k] = 1 / (2 * torch.sin(t_norm)) * (R[..., k][0, 2] - R[..., k][2, 0]) * t_norm
            logTij[2, k] = 1 / (2 * torch.sin(t_norm)) * (R[..., k][1, 0] - R[..., k][0, 1]) * t_norm

            logTij[3:,k] = torch.matmul(Vinv, Tr[..., k])

        return logTij


    def forward(self, logRobs, timepoints):
        Ti = self._compute_matrix()
        Tij = self._build_combinations(timepoints, Ti)
        logTij = self._compute_log(Tij)
        logTi = self._compute_log(Ti)

        if self.cost == 'l1':
            loss = torch.sum(torch.sqrt((logTij - logRobs) ** 2 + 1e-6)) / self.K
        elif self.cost == 'l2':
            loss = torch.sum((logTij - logRobs) ** 2 + 1e-6) / self.K
        else:
            raise ValueError('Cost ' + self.cost + ' not valid. Choose \'l1\' of \'l2\'.' )
        loss += self.reg_weight * torch.sum(logTi**2) / self.K
        return loss



# class InstanceBlockModel(nn.Module):
#

#
#     #
#     # def compute_affine_matrix(self):
#     #     raise NotImplementedError
#     #
#     # def _get_dense_field(self, mesh_matrix, affine_matrix):
#     #     ndims = len(self.vol_shape)
#     #     vol_shape = self.vol_shape
#     #
#     #     # compute locations
#     #     loc_matrix = torch.matmul(affine_matrix, mesh_matrix)  # N x nb_voxels
#     #     loc = torch.reshape(loc_matrix, [ndims] + list(vol_shape))  # *volshape x N
#     #
#     #     # get shifts and return
#     #     shift = loc - mesh_matrix[:ndims].view([ndims] + list(vol_shape))
#     #     return shift.float()
#     #
#     # def forward(self, header, image_shape, mode='bilinear', *args, **kwargs):
#     #
#     #     # Reshape params to matrix and add identity to learn only the shift
#     #     ndims = len(self.vol_shape)
#     #     params_affine = self.compute_affine_matrix()
#     #     new_header = torch.matmul(params_affine, header)
#     #     affine = torch.matmul(torch.inverse(new_header), self.vol_affine)
#     #     affine = torch.unsqueeze(affine[:ndims], dim=0)
#     #
#     #     return affine[:ndims], new_header
#
# class InstanceIndividualRigid(InstanceBlockModel):
#     def __init__(self, vol_shape, vol_affine, num_images=1, device='cpu', torch_dtype=torch.float):
#         super().__init__(vol_shape, vol_affine, device, torch_dtype)
#
#         self.num_images = num_images
#
#         self.slice_translation = torch.nn.Parameter(torch.zeros(self.num_images, 3, 1))
#         self.slice_angle = torch.nn.Parameter(torch.zeros(self.num_images, 3, 1))
#         self.slice_translation.requires_grad = True
#         self.slice_angle.requires_grad = True
#
#
#     def warp(self, image, affine_matrix, **kwargs):
#         image = torch.permute(image[0], [3, 0, 1, 2])
#         out_image = self.transform(image, affine_matrix, **kwargs)
#         return torch.unsqueeze(torch.permute(out_image, [1, 2, 3, 0]), dim=0)
#
#
#     def forward(self, header, image_shape, mode='bilinear', *args, **kwargs):
#
#         affine, new_header = super().forward(header, image_shape, mode, *args, **kwargs)
#
#         # Container for all affine matrices, grouped in "batch size"
#         total_affine = torch.zeros(self.num_images, 3, 4).to(self.device)
#
#         # Reshape params to matrix and add identity to learn only the shift
#         ndims = len(self.vol_shape)
#
#         for it_s in range(self.num_images):
#             params_affine = self.compute_params(self.slice_angle[it_s], self.slice_angle[it_s])
#             total_affine[it_s] = params_affine[:ndims]
#
#         return affine, new_header, total_affine
#
#
#     def compute_params(self, translation, angle):
#
#         T1 = torch.eye(4).to(self.device)
#         T2 = torch.eye(4).to(self.device)
#         T3 = torch.eye(4).to(self.device)
#         T4 = torch.eye(4).to(self.device)
#         T5 = torch.eye(4).to(self.device)
#         T6 = torch.eye(4).to(self.device)
#
#         angles = angle / 180 * np.pi
#
#         T1[0,3] = -self.cr[0]
#         T1[1,3] = -self.cr[1]
#         T1[2,3] = -self.cr[2]
#
#         T2[1, 1] = torch.cos(angles[0])
#         T2[1, 2] = -torch.sin(angles[0])
#         T2[2, 1] = torch.sin(angles[0])
#         T2[2, 2] = torch.cos(angles[0])
#
#         T3[0, 0] = torch.cos(angles[1])
#         T3[0, 2] = torch.sin(angles[1])
#         T3[2, 0] = -torch.sin(angles[1])
#         T3[2, 2] = torch.cos(angles[1])
#
#         T4[0, 0] = torch.cos(angles[2])
#         T4[0, 1] = -torch.sin(angles[2])
#         T4[1, 0] = torch.sin(angles[2])
#         T4[1, 1] = torch.cos(angles[2])
#
#         T5[0, 3] = self.cr[0]
#         T5[1, 3] = self.cr[1]
#         T5[2, 3] = self.cr[2]
#
#         T6[0, 3] = translation[0]
#         T6[1, 3] = translation[1]
#         T6[2, 3] = translation[2]
#
#         T = torch.chain_matmul(T6, T5, T4, T3, T2, T1)
#
#         return T
