import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
eps = 1e-10


class WCE(nn.Module):
    def __init__(self):
        super(WCE, self).__init__()
    
    def forward(self, pred, targets):
        '''
        pred.size: [8, 2, 256, 256]
        labels.size: [8, 256, 256]
        # '''

        loss = -torch.mean(0.7*targets * torch.log(pred[:,1,:,:]+1e-5)+0.3*(1-targets)*torch.log(pred[:,0,:,:]+1e-5))
        return loss


class ModelLossSemsegGatedCRF(torch.nn.Module):
    """
    This module provides an implementation of the Gated CRF Loss for Weakly Supervised Semantic Image Segmentation.
    This loss function promotes consistent label assignment guided by input features, such as RGBXY.
    Please consider using the following bibtex for citation:
    @article{obukhov2019gated,
        author={Anton Obukhov and Stamatios Georgoulis and Dengxin Dai and Luc {Van Gool}},
        title={Gated {CRF} Loss for Weakly Supervised Semantic Image Segmentation},
        journal={CoRR},
        volume={abs/1906.04651},
        year={2019},
        url={http://arxiv.org/abs/1906.04651},
    }
    """
    def forward(
            self, y_hat_softmax, kernels_desc, sample, kernels_radius=11, height_input=256, width_input=256,
            mask_src=None, mask_dst=None, compatibility=None, custom_modality_downsamplers=None, out_kernels_vis=False
    ):
        """
        Performs the forward pass of the loss.
        :param y_hat_softmax: A tensor of predicted per-pixel class probabilities of size NxCxHxW
        :param kernels_desc: A list of dictionaries, each describing one Gaussian kernel composition from modalities.
            The final kernel is a weighted sum of individual kernels. Following example is a composition of
            RGBXY and XY kernels:
            kernels_desc: [{
                'weight': 0.9,          # Weight of RGBXY kernel
                'xy': 6,                # Sigma for XY
                'rgb': 0.1,             # Sigma for RGB
            },{
                'weight': 0.1,          # Weight of XY kernel
                'xy': 6,                # Sigma for XY
            }]
        :param kernels_radius: Defines size of bounding box region around each pixel in which the kernel is constructed.
        :param sample: A dictionary with modalities (except 'xy') used in kernels_desc parameter. Each of the provided
            modalities is allowed to be larger than the shape of y_hat_softmax, in such case downsampling will be
            invoked. Default downsampling method is area resize; this can be overriden by setting.
            custom_modality_downsamplers parameter.
        :param width_input, height_input: Dimensions of the full scale resolution of modalities
        :param mask_src: (optional) Source mask.
        :param mask_dst: (optional) Destination mask.
        :param compatibility: (optional) Classes compatibility matrix, defaults to Potts model.
        :param custom_modality_downsamplers: A dictionary of modality downsampling functions.
        :param out_kernels_vis: Whether to return a tensor with kernels visualized with some step.
        :return: Loss function value.
        """
        assert y_hat_softmax.dim() == 4, 'Prediction must be a NCHW batch'
        N, C, height_pred, width_pred = y_hat_softmax.shape
        device = y_hat_softmax.device


        kernels = self._create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device, custom_modality_downsamplers
        )

        denom = N * height_pred * width_pred

        y_hat_unfolded = self._unfold(y_hat_softmax, kernels_radius)

        product_kernel_x_y_hat = (kernels * y_hat_unfolded) \
            .view(N, C, (kernels_radius * 2 + 1) ** 2, height_pred, width_pred) \
            .sum(dim=2, keepdim=False)

        # Using shortcut for Pott's class compatibility model
        loss = -(product_kernel_x_y_hat * y_hat_softmax).sum()
        loss = kernels.sum() + loss  # comment out to save computation, total loss may go below 0

        return loss / denom

    @staticmethod
    def _downsample(img, modality, height_dst, width_dst, custom_modality_downsamplers):
        if custom_modality_downsamplers is not None and modality in custom_modality_downsamplers:
            f_down = custom_modality_downsamplers[modality]
        else:
            f_down = F.adaptive_avg_pool2d
        return f_down(img, (height_dst, width_dst))

    @staticmethod
    def _create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device, custom_modality_downsamplers
    ):
        kernels = None
        for i, desc in enumerate(kernels_desc):
            weight = desc['weight']
            features = []
            for modality, sigma in desc.items():
                if modality == 'weight':
                    continue
                if modality == 'xy':
                    feature = ModelLossSemsegGatedCRF._get_mesh(N, height_pred, width_pred, device)
                else:
                    feature = sample
                    feature = ModelLossSemsegGatedCRF._downsample(
                        feature, modality, height_pred, width_pred, custom_modality_downsamplers
                    )
                feature /= sigma
                features.append(feature)
            features = torch.cat(features, dim=1)
            kernel = weight * ModelLossSemsegGatedCRF._create_kernels_from_features(features, kernels_radius)
            kernels = kernel if kernels is None else kernel + kernels
        return kernels

    @staticmethod
    def _create_kernels_from_features(features, radius):
        assert features.dim() == 4, 'Features must be a NCHW batch'
        N, C, H, W = features.shape
        kernels = ModelLossSemsegGatedCRF._unfold(features, radius)
        kernels = kernels - kernels[:, :, radius, radius, :, :].view(N, C, 1, 1, H, W)
        kernels = (-0.5 * kernels ** 2).sum(dim=1, keepdim=True).exp()
        kernels[:, :, radius, radius, :, :] = 0
        return kernels

    @staticmethod
    def _get_mesh(N, H, W, device):
        return torch.cat((
            torch.arange(0, W, 1, dtype=torch.float32, device=device).view(1, 1, 1, W).repeat(N, 1, H, 1),
            torch.arange(0, H, 1, dtype=torch.float32, device=device).view(1, 1, H, 1).repeat(N, 1, 1, W)
        ), 1)

    @staticmethod
    def _unfold(img, radius):
        assert img.dim() == 4, 'Unfolding requires NCHW batch'
        N, C, H, W = img.shape
        diameter = 2 * radius + 1
        return F.unfold(img, diameter, 1, radius).view(N, C, diameter, diameter, H, W)

    @staticmethod
    def _visualize_kernels(kernels, radius, height_input, width_input, height_pred, width_pred):
        diameter = 2 * radius + 1
        vis = kernels[:, :, :, :, radius::diameter, radius::diameter]
        vis_nh, vis_nw = vis.shape[-2:]
        vis = vis.permute(0, 1, 4, 2, 5, 3).contiguous().view(kernels.shape[0], 1, diameter * vis_nh, diameter * vis_nw)
        if vis.shape[2] > height_pred:
            vis = vis[:, :, :height_pred, :]
        if vis.shape[3] > width_pred:
            vis = vis[:, :, :, :width_pred]
        if vis.shape[2:] != (height_pred, width_pred):
            vis = F.pad(vis, [0, width_pred-vis.shape[3], 0, height_pred-vis.shape[2]])
        vis = F.interpolate(vis, (height_input, width_input), mode='nearest')
        return vis

class Trans_loss(nn.Module):
    def __init__(self):
        super(Trans_loss, self).__init__()
    
    def forward(self, cam_tran, tran_cam):
        '''
        pred.size: [8, 2, 256, 256]
        labels.size: [8, 256, 256]
        # '''

        #pred = nn.functional.softmax(pred,dim=1)
        #pred=torch.clamp(pred,1e-5,1-1e-5)
        loss =torch.mean((cam_tran-tran_cam)**2)
        return loss