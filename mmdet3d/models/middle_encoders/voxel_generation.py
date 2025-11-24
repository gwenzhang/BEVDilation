import torch
from torch import nn as nn
import torch.nn.functional as F
from mmdet3d.models.backbones.bev_backbone_ded import DEDBackbone
from mmdet3d.models.necks import Mamba_Auto_Regression
import math
from functools import partial
try:
    import spconv.pytorch as spconv
except:
    import spconv as spconv
from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE
if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential
from mmcv.ops import points_in_polygons
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from mmcv.runner import force_fp32

def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class Voxel_Generation(nn.Module):
    def __init__(self,
                img_channels=32,
                lidar_channels=128,
                curve_rank=8,
                curve_template_path=None,
                img_encoder_num=None,
                img_downstrides=None,
                fg_thr=0.4,
                expand_temp='/home/guowen_zhang/code/iccv_tune/BEVDiffuse/data/curve/curve_template_3d_rank_8_continue_z10.pth',
                reverse_expand_temp='/home/guowen_zhang/code/iccv_tune/BEVDiffuse/data/curve/curve_template_3d_rank_8_continue_z10_reverse.pth',
                bev_shape=(180, 180),
                voxel_size=None,
                pc_range=None,
                downstride=8,
                img_feat_downsample=8
                ):
        super().__init__()

        self.fg_thr = fg_thr
        self.img_channels = img_channels
        self.img_feat_downsample = img_feat_downsample
        self.mamba_autoRegression = Mamba_Auto_Regression(lidar_channels, ssm_cfg=None, norm_epsilon=0.00001, 
                    rms_norm=True, template_path=curve_template_path)
        self.img_layer = DEDBackbone(img_encoder_num, img_downstrides, img_channels, img_channels)
        self.fg_pred = nn.Sequential(
            nn.Conv2d(img_channels + lidar_channels, (img_channels + lidar_channels)//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d((img_channels + lidar_channels)//2),
            nn.ReLU(inplace=True),
            nn.Conv2d((img_channels + lidar_channels)//2, 1, 1, bias=True)
        )
        # self.img2lidar_dim = nn.Sequential(
        #     nn.Linear(256, lidar_channels),
        #     nn.LayerNorm(lidar_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(lidar_channels, lidar_channels),
        # )

        self.init_mask = self.init_bev_mask(bev_shape)
        self.downstride = downstride
        self.pc_range = torch.Tensor(pc_range)
        self.voxel_size = torch.Tensor(voxel_size)
        self.add_features = nn.Parameter(torch.randn(1, lidar_channels), requires_grad=True)

        self.visual = dict()

        self.curve_template = dict()
        self.hilbert_spatial_size = dict()
        self.load_template(curve_template_path, curve_rank)
        self.load_template_with_reverse(expand_temp, reverse_expand_temp, curve_rank)

        self.init_weights()
        self._reset_parameters()
        initializer_cfg = None
        self.apply(
            partial(
                _init_weights,
                n_layer=1,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )


    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, spconv.SubMConv3d, spconv.SubMConv2d)):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name:
                nn.init.xavier_uniform_(p)

    def load_template_with_reverse(self, path, path_reverse, rank):
        
        template = torch.load(path)
        template_reverse = torch.load(path_reverse)
        if isinstance(template, dict):
            self.curve_template[f'expand_curve_template_rank{rank}'] = template['data'].reshape(-1)
            self.hilbert_spatial_size[f'expand_curve_template_rank{rank}'] = template['size'] 
        else:
            spatial_size = 2 ** rank
            self.curve_template[f'expand_curve_template_rank{rank}'] = template.reshape(-1)
            self.curve_template[f'expand_curve_template_rank{rank}_reverse'] = template_reverse.reshape(-1)
            self.hilbert_spatial_size[f'expand_curve_template_rank{rank}'] = (1, spatial_size, spatial_size) #[z, y, x]
    
    def load_template(self, path, rank):
        
        template = torch.load(path)
        spatial_size = 2 ** rank
        self.curve_template[f'curve_template_rank{rank}'] = template.reshape(-1)
        self.hilbert_spatial_size[f'curve_template_rank{rank}'] = (1, spatial_size, spatial_size) #[z, y, x]
    
    def forward(self, voxel_feats, img_feats, batch_size, img_input_list=None):
        
        voxel_feats_bev = voxel_feats.dense().detach().mean(dim=2)
        img_feats = self.img_layer(img_feats)
        bev_fg_mask = self.fg_pred(torch.cat([voxel_feats_bev, img_feats], dim=1))
        x, rank = self.expand_indices(voxel_feats, bev_fg_mask, img_input_list)
        x = self.mamba_autoRegression(x, batch_size, self.curve_template[f'curve_template_rank{rank}'], self.hilbert_spatial_size[f'curve_template_rank{rank}'])

        return x, bev_fg_mask
    
    def expand_indices(self, voxel_feat, bev_fg_mask, img_input_list=None):

        device = voxel_feat.features.device
        voxel_indices = voxel_feat.indices
        zshape, yshape, xshape = voxel_feat.spatial_shape
        bsz, _, ysize, xsize = bev_fg_mask.shape
        assert xshape == xsize, yshape == ysize

        fg_mask = (bev_fg_mask.sigmoid() > self.fg_thr)

        # b_inds = voxel_indices[:, 0]
        # y_inds = voxel_indices[:, 2]
        # x_inds = voxel_indices[:, 3]
        # voxel_fg_mask = fg_mask[b_inds, 0, y_inds, x_inds] # decide which axis is x and y by points_in_polygons(bev_pos_xy, bev_corners) in Generate GT Mask
        # voxel_fg_indices = voxel_indices[voxel_fg_mask]
        # rank = math.ceil(math.log2(xshape))
        # zero_indices = self.obtain_curve_indices(voxel_fg_indices, rank, device)

        rank = math.ceil(math.log2(xshape))
        zero_indices = torch.nonzero(fg_mask)
        zero_indices[:, 1] = 0

        zero_mask = (zero_indices[:, 2] >= 0) & (zero_indices[:, 2] < yshape) & \
                (zero_indices[:, 3] >= 0) & (zero_indices[:, 3] < xshape) &\
                    (zero_indices[:, 1] >= 0) & (zero_indices[:, 1] < (zshape))
        zero_indices = zero_indices[zero_mask]
        if self.training:
            self.visual['add_indices'] = zero_indices
            self.visual['gt_indices'] = voxel_indices
        # zero_pos = (zero_indices[:, 1:].flip(1) * (self.downstride * self.voxel_size.to(device)[:3]).unsqueeze(0)) + self.pc_range.to(device)[:3].unsqueeze(0)
        # zero_features = self.extract_img_feat_from_3dpoints(zero_pos, img_input_list, zero_indices[:, 0], bsz)
        # zero_features = self.img2lidar_dim(zero_features)

        # non-overlap indices
        N = voxel_feat.indices.shape[0]
        cat_indices = torch.cat([voxel_feat.indices, zero_indices], dim=0)
        _, _inv, _counts = torch.unique(cat_indices, dim=0, return_counts=True, return_inverse=True)
        non_overlap_mask = _counts[_inv][len(voxel_feat.indices):] == 1
        zero_indices = zero_indices[non_overlap_mask]
        indices_unique = torch.cat([voxel_feat.indices, zero_indices], dim=0)
        inpaint_mask = voxel_feat.features.new_zeros(len(indices_unique), dtype=torch.bool)
        inpaint_mask[N:] = True
        # zero_features = voxel_feat.features.new_zeros((len(zero_indices), voxel_feat.features.shape[1]))
        zero_features = self.add_features.repeat(len(zero_indices), 1)
        features_unique = torch.cat([voxel_feat.features, zero_features], dim=0)
        
        # zero_pos = (zero_indices[:, 1:].flip(1) * (self.downstride * self.voxel_size.to(device)[:3]).unsqueeze(0)) + self.pc_range.to(device)[:3].unsqueeze(0)
        # zero_features = self.extract_img_feat_from_3dpoints(zero_pos, img_input_list, zero_indices[:, 0], bsz)
        # zero_features = self.img2lidar_dim(zero_features)

        # zero_features = voxel_feat.features.new_zeros((len(zero_indices), voxel_feat.features.shape[1]))
        # cat_indices = torch.cat([voxel_feat.indices, zero_indices], dim=0)
        # cat_features = torch.cat([voxel_feat.features, zero_features], dim=0)
        # indices_unique, _inv = torch.unique(cat_indices, dim=0, return_inverse=True)
        # features_unique = voxel_feat.features.new_zeros((indices_unique.shape[0], voxel_feat.features.shape[1]))
        # features_unique.index_add_(0, _inv, cat_features)

        x = SparseConvTensor(
            features=features_unique,
            indices=indices_unique.int(),
            spatial_shape=voxel_feat.spatial_shape,
            batch_size=voxel_feat.batch_size
        )

        return x, rank
    
    # def obtain_curve_indices(self, coors, rank, device):
        
    #     template = self.curve_template[f'expand_curve_template_rank{rank}'].to(device)
    #     reverse_template = self.curve_template[f'expand_curve_template_rank{rank}_reverse'].to(device)
    #     hil_size_z, hil_size_y, hil_size_x = self.hilbert_spatial_size[f'expand_curve_template_rank{rank}']

    #     x = coors[:, 3]
    #     y = coors[:, 2]
    #     z = coors[:, 1]

    #     flat_coors = (z * hil_size_y * hil_size_x + y * hil_size_x + x).long()
    #     hil_inds = template[flat_coors].long()

    #     binds = torch.cat([coors[:, 0], coors[:, 0]])
    #     new_hil_inds = torch.cat([hil_inds-1, hil_inds+1])
    #     flat_coors_add = reverse_template[new_hil_inds]

    #     znew = flat_coors_add // (hil_size_x * hil_size_y)
    #     remainer = flat_coors_add % (hil_size_x * hil_size_y)
    #     ynew = remainer // hil_size_x
    #     xnew = remainer % hil_size_x
    #     bnew = binds
    #     coors_new = torch.stack([bnew, znew, ynew, xnew], dim=1)
    #     coors_new, _inv = torch.unique(coors_new, dim=0, return_inverse=True)

    #     return coors_new

    def obtain_curve_indices(self, coors, rank, device):
        
        template = self.curve_template[f'expand_curve_template_rank{rank}'].to(device)
        reverse_template = self.curve_template[f'expand_curve_template_rank{rank}_reverse'].to(device)
        hil_size_z, hil_size_y, hil_size_x = self.hilbert_spatial_size[f'expand_curve_template_rank{rank}']

        x = coors[:, 3]
        y = coors[:, 2]
        z = coors[:, 1]

        flat_coors = (z * hil_size_y * hil_size_x + y * hil_size_x + x).long()
        hil_inds = template[flat_coors].long()

        binds = coors[:, 0]
        new_hil_inds = hil_inds+1
        flat_coors_add = reverse_template[new_hil_inds]

        znew = flat_coors_add // (hil_size_x * hil_size_y)
        remainer = flat_coors_add % (hil_size_x * hil_size_y)
        ynew = remainer // hil_size_x
        xnew = remainer % hil_size_x
        bnew = binds
        coors_new = torch.stack([bnew, znew, ynew, xnew], dim=1)
        coors_new, _inv = torch.unique(coors_new, dim=0, return_inverse=True)

        return coors_new

    # def obtain_bev_mask_gt(self, gt_bboxes_3d, gt_bboxes_ignore):
        
    #     device = gt_bboxes_ignore[0].device
    #     yshape, xshape, _ = self.init_mask.shape
    #     bev_indices = self.init_mask.to(device).reshape(-1, 2)
    #     voxel_size = self.voxel_size.to(device)
    #     pc_range = self.pc_range.to(device)
    #     bev_pos = (bev_indices * (self.downstride * voxel_size[:2]).unsqueeze(0) + pc_range[:2].unsqueeze(0))
    #     bev_pos_xy = bev_pos.flip(1)

    #     gt_bev_mask = []
    #     for indiv_bboxes, indiv_ignore in zip(gt_bboxes_3d, gt_bboxes_ignore):
    #         indiv_bboxes_bev = indiv_bboxes.corners.to(device)[indiv_ignore][:, [0, 2, 4, 6], :2]
    #         indiv_bboxes_bev = indiv_bboxes_bev.reshape(indiv_bboxes_bev.shape[0], -1)
    #         inpoly = points_in_polygons(bev_pos_xy, indiv_bboxes_bev).any(dim=-1).view(yshape, xshape)
    #         gt_bev_mask(inpoly)

    #     return None

    def obtain_bev_mask_gt(self, gt_bboxes_3d, gt_bboxes_ignore=None, device=None, pred_bev_mask=None):
        
        H, W, _ = self.init_mask.shape
        if not hasattr(self, '_bev_cache'):
            bev_indices = self.init_mask.to(device).view(-1, 2)
            voxel_size = self.voxel_size.to(device)
            pc_range = self.pc_range.to(device)
            self._bev_cache = (
                (bev_indices * (self.downstride * voxel_size[:2])) + pc_range[:2]
            ).flip(1)

        bev_pos_xy = self._bev_cache
        batch_size = len(gt_bboxes_3d)
        gt_bev_mask = torch.zeros((batch_size, H, W), dtype=torch.float, device=device)

        for batch_idx in range(batch_size):
        
            corners = gt_bboxes_3d[batch_idx].corners.to(device)
            # if gt_bboxes_ignore is not None:
            #     corners = corners[gt_bboxes_ignore[batch_idx]]
            if corners.shape[0] == 0:
                continue
            bev_corners = corners[:, [0, 2, 6, 4], :2]
            original_bev_corners = bev_corners.clone()
            bev_corners = bev_corners.reshape(bev_corners.shape[0], -1)
            in_box = points_in_polygons(bev_pos_xy, bev_corners)
            gt_bev_mask[batch_idx] = in_box.any(dim=-1).view(H, W).float()

            # visual to verification
            # append_indices = self.visual['add_indices'][self.visual['add_indices'][:, 0]==batch_idx]
            # bev_mask_np = gt_bev_mask[batch_idx].cpu().numpy()
            # voxel_size = self.voxel_size.to(device)
            # pc_range = self.pc_range.to(device)
            # original_bev_corners = (original_bev_corners - pc_range[:2]) / (voxel_size[:2] * self.downstride)
            # boxes = original_bev_corners.int().cpu().numpy() + 0.5
            # plt.figure()
            # plt.imshow(bev_mask_np, origin='lower', cmap='gray', extent=[0, W, 0, H])
            # for box in boxes:
            #     pts = np.vstack((box, box[0]))
            #     plt.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=0.5)
            # foreground_y, foreground_x = np.nonzero(bev_mask_np)
            # plt.scatter(foreground_x+0.5, foreground_y+0.5, s=5, c='blue', marker='o', label='Foreground Points')

            # # if append_indices.shape[0] > 0:
            # #     append_indices_np = append_indices[:, 1:].cpu().numpy()
            # #     plt.scatter(append_indices_np[:, 2] + 0.5, append_indices_np[:, 1] + 0.5,
            # #                 s=5, c='yellow', marker='x', label='Append Indices')
            
            # plt.title('BEV Map with Box Boundaries and Foreground Points')
            # plt.xlabel('X axis')
            # plt.ylabel('Y axis')
            # plt.legend()
            # plt.savefig(f'/home/guowen_zhang/code/iccv_tune/BEVDiffuse/visual/bev_mask{batch_idx}.png', bbox_inches='tight', pad_inches=0)

            # if pred_bev_mask is not None:
            #     pred_sample = torch.sigmoid(pred_bev_mask[batch_idx].squeeze())
            #     pred_sample = (pred_sample > 0.5).float()
            #     # pred_sample = ((pred_sample > 0.4) & (pred_sample < 0.6)).float()
            #     pred_mask_np = pred_sample.detach().cpu().numpy()
            #     plt.figure()
            #     # plt.imshow(pred_mask_np, origin='lower', cmap='gray', extent=[0, W, 0, H])
            #     custom_cmap = ListedColormap(['white', 'green'])
            #     plt.imshow(pred_mask_np, origin='lower', cmap=custom_cmap, extent=[0, W, 0, H])
            #     # 与 GT 相同，绘制 box（gt_bboxes_3d 投影结果）以对比
            #     for box in boxes:
            #         pts = np.vstack((box, box[0]))
            #         plt.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=0.5)
            #     plt.title('Pred BEV Map with GT Box Boundaries')
            #     plt.xlabel('X axis')
            #     plt.ylabel('Y axis')
            #     plt.savefig(f'/home/guowen_zhang/code/iccv_tune/BEVDiffuse/visual/bev_mask_pred{batch_idx}.png', bbox_inches='tight', pad_inches=0)
            #     plt.close()

            # debug = True

        return gt_bev_mask.long()

    def init_bev_mask(self, bev_shape):
        y_coords, x_coords = torch.meshgrid(torch.arange(bev_shape[0]), torch.arange(bev_shape[1]),indexing='ij')
        bev_indices = torch.stack((y_coords, x_coords), dim=-1).float() + 0.5
        return bev_indices

