from mamba_ssm.models.mixer_seq_simple import create_block
import torch.nn as nn
from functools import partial
import torch
from .ar_input_layer import get_hilbert_index_3d_mamba_lite, get_hilbert_index_2d_mamba_lite
from mmdet3d.models.model_utils.bevdiffuse_util import SparseConvBlock, post_act_block_sparse_3d
try:
    import spconv.pytorch as spconv
except:
    import spconv as spconv

def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out

class Mamba_Auto_Regression(nn.Module):

    def __init__(self, 
                 d_model, 
                 ssm_cfg, 
                 norm_epsilon, 
                 rms_norm, 
                 residual_in_fp32=True, 
                 fused_add_norm=True,
                 device=None,
                 dtype=None,
                 template_path=None):
        super().__init__()

        factory_kwargs = {'device': device, 'dtype':dtype}

        encoder_1 = create_block(
            d_model=d_model,
            ssm_cfg=ssm_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            layer_idx=0,
            **factory_kwargs,
        )

        encoder_2 = create_block(
            d_model=d_model,
            ssm_cfg=ssm_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            layer_idx=1,
            **factory_kwargs,
        )

        self.mamba_encoder_list = nn.ModuleList([encoder_1, encoder_2])
        norm_cls = partial(
            nn.LayerNorm, eps=norm_epsilon, **factory_kwargs
        )

        self.norm = norm_cls(d_model)
        self.norm_back = norm_cls(d_model)
        
        self.curve_template = dict()
        self.hilbert_spatial_size = dict()
        # self.load_template(template_path, 8)

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.norm_output = norm_fn(d_model)

        self.pos_embed =  nn.Sequential(
                nn.Linear(8, d_model),
                nn.BatchNorm1d(d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, d_model),
                )

        # SparseConvLayer
        self.conv_encoder = SparseConvBlock(d_model, 3, 1, 1, norm_fn, f"encoder_subm")

        conv_decoder = []
        conv_decoder.append(post_act_block_sparse_3d(d_model, d_model, 3, norm_fn=norm_fn, conv_type='subm',
                        indice_key=f'decoder_subm'))
        conv_decoder.append(norm_fn(d_model))
        self.conv_decoder = spconv.SparseSequential(*conv_decoder)

    # def load_template(self, path, rank):
    #     template = torch.load(path)
    #     if isinstance(template, dict):
    #         self.curve_template[f'curve_template_rank{rank}'] = template['data'].reshape(-1)
    #         self.hilbert_spatial_size[f'curve_template_rank{rank}'] = template['size'] 
    #     else:
    #         self.curve_template[f'curve_template_rank{rank}'] = template.reshape(-1)
    #         spatial_size = 2 ** rank
    #         self.hilbert_spatial_size[f'curve_template_rank{rank}'] = (1, spatial_size, spatial_size) #[z, y, x]

    def forward(
        self,
        x,
        batch_size,
        clvl_cruve_template,
        clvl_hilbert_spatial_size,
        ):
        # num_shifts = len(pos_dict_list)
        # assert num_shifts in (1, 2)

        mamba_layer1 = self.mamba_encoder_list[0]
        mamba_layer2 = self.mamba_encoder_list[1]

        x_merge = self.conv_encoder(x)
        feats = x_merge.features
        coords = x_merge.indices

        clvl_cruve_template = clvl_cruve_template.to(coords.device)
        # clvl_hilbert_spatial_size = self.hilbert_spatial_size['curve_template_rank8']
        # index_info = get_hilbert_index_3d_mamba_lite(clvl_cruve_template, coords, batch_size, x_merge.spatial_shape[0], \
        #                                                 clvl_hilbert_spatial_size, shift=(0, 0, 0))
        index_info = get_hilbert_index_2d_mamba_lite(clvl_cruve_template, coords, batch_size, \
                                                        clvl_hilbert_spatial_size, shift=(0, 0))
        
        inds_curt_to_next = index_info['inds_curt_to_next']
        inds_next_to_curt = index_info['inds_next_to_curt']

        out_feat_3d_s2 = torch.zeros_like(feats)
        out_feat_3d_s1 = torch.zeros_like(feats)

        pos_embed_coords = torch.zeros([coords.shape[0], 8], device=coords.device, dtype=coords.dtype)
        # pos_embed_coords[:, 0] = coords[:, 1]
        pos_embed_coords[:, 0:2] = (coords[:, 2:] // 12)
        pos_embed_coords[:, 2:4] = (coords[:, 2:] % 12)
        pos_embed_coords[:, 4:6] = ((coords[:, 2:] + 6) // 12)
        pos_embed_coords[:, 6:8] = ((coords[:, 2:] + 6) % 12)
        pos_embed = self.pos_embed(pos_embed_coords.float())

        feats = feats + pos_embed
        for i in range(batch_size):
            b_mask_m2 = coords[:, 0] == i
            feat_m2 = feats[b_mask_m2][inds_curt_to_next[i]][None]
            out_feat_m2 = mamba_layer1(feat_m2, None)
            out_feat_3d_s2[b_mask_m2] = (out_feat_m2[0]).squeeze(0)[inds_next_to_curt[i]]

            # Fackward SSMs
            b_mask_m1 = coords[:, 0] == i
            feat_m1 = feats[b_mask_m1][inds_curt_to_next[i]][None]
            feat_back = feat_m1.flip(1)
            out_feat_back = mamba_layer2(feat_back, None)
            out_feat_3d_s1[b_mask_m1] = (out_feat_back[0]).squeeze(0).flip(0)[inds_next_to_curt[i]]

        out_feat_3d_s2 = self.norm(out_feat_3d_s2)
        out_feat_3d_s1 = self.norm_back(out_feat_3d_s1)

        x_merge = replace_feature(x_merge, out_feat_3d_s1 + out_feat_3d_s2)
        x_merge = self.conv_decoder(x_merge)
        x_merge = replace_feature(x_merge, self.norm_output(x.features + x_merge.features))

        return x_merge
