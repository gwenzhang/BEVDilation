import torch.nn as nn
from .base_bev_backbone import BasicBlock
from .intern_image import BasicBlockDCN
from ..builder import BACKBONES
from ops_dcnv3 import modules as opsm
import torch

class DEDBackbone(nn.Module):

    def __init__(self, 
                num_SBB,
                down_strides,
                feature_dim, 
                input_channels):
        super().__init__()

        # num_SBB = model_cfg.NUM_SBB
        # down_strides = model_cfg.DOWN_STRIDES
        dim = feature_dim
        assert len(num_SBB) == len(down_strides)

        num_levels = len(down_strides)

        first_block = []
        if input_channels != dim:
            first_block.append(BasicBlock(input_channels, dim, down_strides[0], 1, True))
        first_block += [BasicBlock(dim, dim) for _ in range(num_SBB[0])]
        self.encoder = nn.ModuleList([nn.Sequential(*first_block)])

        for idx in range(1, num_levels):
            cur_layers = [BasicBlock(dim, dim, down_strides[idx], 1, True)]
            cur_layers.extend([BasicBlock(dim, dim) for _ in range(num_SBB[idx])])
            self.encoder.append(nn.Sequential(*cur_layers))

        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        for idx in range(num_levels - 1, 0, -1):
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(dim, dim, down_strides[idx], down_strides[idx], bias=False),
                    nn.BatchNorm2d(dim, eps=1e-3, momentum=0.01),
                    nn.ReLU()
                )
            )
            self.decoder_norm.append(nn.BatchNorm2d(dim, eps=1e-3, momentum=0.01))

        self.num_bev_features = dim
        self.init_weights()

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = data_dict['spatial_features']
        x = self.encoder[0](x)

        feats = [x]
        for conv in self.encoder[1:]:
            x = conv(x)
            feats.append(x)

        for deconv, norm, up_x in zip(self.decoder, self.decoder_norm, feats[:-1][::-1]):
            x = norm(deconv(x) + up_x)

        # data_dict['spatial_features_2d'] = x
        # data_dict['spatial_features'] = x
        return x
    
class DEDBackboneDCN(nn.Module):

    def __init__(self, 
                num_SBB,
                down_strides,
                feature_dim, 
                input_channels,
                drop_path=None):
        super().__init__()

        # num_SBB = model_cfg.NUM_SBB
        # down_strides = model_cfg.DOWN_STRIDES
        dim = feature_dim
        assert len(num_SBB) == len(down_strides)

        num_levels = len(down_strides)

        first_block = []
        if input_channels != dim:
            first_block.append(BasicBlock(input_channels, dim, down_strides[0], 1, True))
        first_block += [BasicBlock(dim, dim) for _ in range(num_SBB[0])]
        self.encoder = nn.ModuleList([nn.Sequential(*first_block)])

        self.core_op = 'DCNv3'
        for idx in range(1, num_levels):
            cur_layers = [BasicBlock(dim, dim, down_strides[idx], 1, True)]
            # cur_layers.extend([BasicBlockDCN(dim, dim) for _ in range(num_SBB[idx])])
            cur_layers.extend([BasicBlockDCN(core_op=getattr(opsm, self.core_op),
                    channels=dim, groups=16, mlp_ratio=4, drop=0.0,\
                    drop_path=drop_path[sum(num_SBB[1:idx])+i] if isinstance(drop_path, list) else drop_path, act_layer='GELU',
                    norm_layer='LN', post_norm=True, layer_scale=1e-5, offset_scale=1.0,
                    dw_kernel_size=None, res_post_norm=False, center_feature_scale=False
                    ) for i in range(num_SBB[idx])])
            self.encoder.append(nn.Sequential(*cur_layers))

        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        for idx in range(num_levels - 1, 0, -1):
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(dim, dim, down_strides[idx], down_strides[idx], bias=False),
                    nn.BatchNorm2d(dim, eps=1e-3, momentum=0.01),
                    nn.ReLU()
                )
            )
            self.decoder_norm.append(nn.BatchNorm2d(dim, eps=1e-3, momentum=0.01))

        self.num_bev_features = dim
        self.init_weights()

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, getattr(opsm, self.core_op)):
                m._reset_parameters()

    def forward(self, x):
        # x = data_dict['spatial_features']
        x = self.encoder[0](x)

        feats = [x]
        for conv in self.encoder[1:]:
            x = conv(x)
            feats.append(x)

        for deconv, norm, up_x in zip(self.decoder, self.decoder_norm, feats[:-1][::-1]):
            x = norm(deconv(x) + up_x)

        # data_dict['spatial_features_2d'] = x
        # data_dict['spatial_features'] = x
        return x

@BACKBONES.register_module()
class CascadeDEDBackbone(nn.Module):

    def __init__(self, 
                num_layers,
                num_SBB,
                down_strides,
                feature_dim,
                input_channels):
        super().__init__()

        self.layers = nn.ModuleList()

        start_layer = 0
        drop_path = [
            x.item() for x in torch.linspace(0, 0.5, sum(num_SBB[1:]) * (num_layers - start_layer))
        ]

        for idx in range(num_layers):
            input_dim = input_channels if idx == 0 else feature_dim
            if idx >= start_layer:
                self.layers.append(DEDBackboneDCN(num_SBB, down_strides, feature_dim, input_dim,\
                                    drop_path[sum(num_SBB[1:])*(idx-start_layer):sum(num_SBB[1:])*(idx-start_layer+1)]))
            else:
                self.layers.append(DEDBackbone(num_SBB, down_strides, feature_dim, input_dim))

        self.num_bev_features = feature_dim

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        # data_dict['spatial_features_2d'] = data_dict['spatial_features']
        return [x]
    
class DEDBackboneDCNV1(nn.Module):

    def __init__(self, 
                num_SBB,
                down_strides,
                feature_dim, 
                input_channels,
                groups=8,
                drop_path=None):
        super().__init__()

        # num_SBB = model_cfg.NUM_SBB
        # down_strides = model_cfg.DOWN_STRIDES
        dim = feature_dim
        assert len(num_SBB) == len(down_strides)
        self.core_op = 'DCNv3'
        num_levels = len(down_strides)

        first_block = []
        if input_channels != dim:
            first_block.append(BasicBlock(input_channels, dim, down_strides[0], 1, True))
        # first_block += [BasicBlock(dim, dim) for _ in range(num_SBB[0])]
        first_block += [BasicBlockDCN(core_op=getattr(opsm, self.core_op),
                    channels=dim, groups=groups, mlp_ratio=4, drop=0.0,\
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, act_layer='GELU',
                    norm_layer='LN', post_norm=True, layer_scale=1e-5, offset_scale=1.0,
                    dw_kernel_size=None, res_post_norm=False, center_feature_scale=False
                    ) for i in range(num_SBB[0])]
        self.encoder = nn.ModuleList([nn.Sequential(*first_block)])

        for idx in range(1, num_levels):
            cur_layers = [BasicBlock(dim, dim, down_strides[idx], 1, True)]
            # cur_layers.extend([BasicBlockDCN(dim, dim) for _ in range(num_SBB[idx])])
            cur_layers.extend([BasicBlockDCN(core_op=getattr(opsm, self.core_op),
                    channels=dim, groups=groups, mlp_ratio=4, drop=0.0,\
                    drop_path=drop_path[sum(num_SBB[:idx])+i] if isinstance(drop_path, list) else drop_path, act_layer='GELU',
                    norm_layer='LN', post_norm=True, layer_scale=1e-5, offset_scale=1.0,
                    dw_kernel_size=None, res_post_norm=False, center_feature_scale=False
                    ) for i in range(num_SBB[idx])])
            self.encoder.append(nn.Sequential(*cur_layers))

        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        for idx in range(num_levels - 1, 0, -1):
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(dim, dim, down_strides[idx], down_strides[idx], bias=False),
                    nn.BatchNorm2d(dim, eps=1e-3, momentum=0.01),
                    nn.ReLU()
                )
            )
            self.decoder_norm.append(nn.BatchNorm2d(dim, eps=1e-3, momentum=0.01))

        self.num_bev_features = dim
        self.init_weights()

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, getattr(opsm, self.core_op)):
                m._reset_parameters()

    def forward(self, x):
        # x = data_dict['spatial_features']
        x = self.encoder[0](x)

        feats = [x]
        for conv in self.encoder[1:]:
            x = conv(x)
            feats.append(x)

        for deconv, norm, up_x in zip(self.decoder, self.decoder_norm, feats[:-1][::-1]):
            x = norm(deconv(x) + up_x)

        # data_dict['spatial_features_2d'] = x
        # data_dict['spatial_features'] = x
        return x

class DEDBackboneV1(nn.Module):

    def __init__(self, 
                num_SBB,
                down_strides,
                feature_dim, 
                input_channels):
        super().__init__()

        # num_SBB = model_cfg.NUM_SBB
        # down_strides = model_cfg.DOWN_STRIDES
        dim = feature_dim
        assert len(num_SBB) == len(down_strides)

        num_levels = len(down_strides)

        first_block = []
        if input_channels != dim:
            first_block.append(BasicBlock(input_channels, dim, down_strides[0], 1, True))
        first_block += [BasicBlock(dim, dim) for _ in range(num_SBB[0])]
        self.encoder = nn.ModuleList([nn.Sequential(*first_block)])

        for idx in range(1, num_levels):
            cur_layers = [BasicBlock(dim, dim, down_strides[idx], 1, True)]
            cur_layers.extend([BasicBlock(dim, dim) for _ in range(num_SBB[idx])])
            self.encoder.append(nn.Sequential(*cur_layers))

        self.num_bev_features = dim
        self.init_weights()

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.encoder[0](x)

        feats = [x]
        for conv in self.encoder[1:]:
            x = conv(x)
            feats.append(x)

        return feats

class DEDBackboneDCNV2(nn.Module):

    def __init__(self, 
                num_SBB,
                down_strides,
                feature_dim, 
                input_channels,
                groups=8,
                drop_path=None,
                use_img=False):
        super().__init__()

        # num_SBB = model_cfg.NUM_SBB
        # down_strides = model_cfg.DOWN_STRIDES
        dim = feature_dim
        assert len(num_SBB) == len(down_strides)
        self.core_op = 'DCNv3'
        num_levels = len(down_strides)

        ############### for lidar branch #############
        first_block = []
        if input_channels != dim:
            first_block.append(BasicBlock(input_channels, dim, down_strides[0], 1, True))
        first_block += [BasicBlock(dim, dim) for _ in range(num_SBB[0])]
        # first_block += [BasicBlockDCN(core_op=getattr(opsm, self.core_op),
        #             channels=dim, groups=groups, mlp_ratio=4, drop=0.0,\
        #             drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, act_layer='GELU',
        #             norm_layer='LN', post_norm=True, layer_scale=1e-5, offset_scale=1.0,
        #             dw_kernel_size=None, res_post_norm=False, center_feature_scale=False
        #             ) for i in range(num_SBB[0])]
        # self.encoder = nn.ModuleList([nn.Sequential(*first_block)])
        self.lidar_first_encoder = nn.ModuleList([*first_block])

        self.lidar_second_encoder = nn.ModuleList()
        for idx in range(1, num_levels):
            cur_layers = nn.ModuleList()
            cur_layers.append(BasicBlock(dim, dim, down_strides[idx], 1, True))
            for i in range(num_SBB[idx]):
                cur_layers.append(BasicBlockDCN(core_op=getattr(opsm, self.core_op),
                    channels=dim, groups=groups, mlp_ratio=4, drop=0.0,\
                    drop_path=drop_path[sum(num_SBB[1:idx])+i] if isinstance(drop_path, list) else drop_path, act_layer='GELU',
                    norm_layer='LN', post_norm=True, layer_scale=1e-5, offset_scale=1.0,
                    dw_kernel_size=None, res_post_norm=False, center_feature_scale=False, use_img=use_img))
            self.lidar_second_encoder.append(cur_layers)

        self.lidar_decoder = nn.ModuleList()
        self.lidar_decoder_norm = nn.ModuleList()
        for idx in range(num_levels - 1, 0, -1):
            self.lidar_decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(dim, dim, down_strides[idx], down_strides[idx], bias=False),
                    nn.BatchNorm2d(dim, eps=1e-3, momentum=0.01),
                    nn.ReLU()
                )
            )
            self.lidar_decoder_norm.append(nn.BatchNorm2d(dim, eps=1e-3, momentum=0.01))

        self.num_bev_features = dim
        self.init_weights()

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, getattr(opsm, self.core_op)):
                m._reset_parameters()

    def forward(self, x, ximg=None):

        for layer in self.lidar_first_encoder:
            if isinstance(layer, BasicBlockDCN):
                x = layer(x, ximg[0])
            else:
                x = layer(x)

        feats = [x]
        for index, layer in enumerate(self.lidar_second_encoder):
            for conv in layer:
                if isinstance(conv, BasicBlockDCN):
                    x = conv(x, ximg[index+1])
                else:
                    x = conv(x)
            feats.append(x)

        for deconv, norm, up_x in zip(self.lidar_decoder, self.lidar_decoder_norm, feats[:-1][::-1]):
            x = norm(deconv(x) + up_x)

        return x

############################################################################

@BACKBONES.register_module()
class CascadeDEDBackboneV1(nn.Module):

    def __init__(self, 
                num_layers,
                num_SBB,
                down_strides,
                feature_dim,
                input_channels,
                groups=8,
                groups_img=8):
        super().__init__()

        self.layers = nn.ModuleList()
        self.img_layers = nn.ModuleList()

        start_layer = 0
        use_img_start = 2
        drop_path = [
            x.item() for x in torch.linspace(0, 0.5, sum(num_SBB[1:]) * (num_layers - start_layer))
        ]

        for idx in range(num_layers):
            input_dim = input_channels if idx == 0 else feature_dim
            if idx >= start_layer:
                self.layers.append(DEDBackboneDCNV2(num_SBB, down_strides, feature_dim, input_dim, groups,\
                                    drop_path[sum(num_SBB[1:])*(idx-start_layer):sum(num_SBB[1:])*(idx-start_layer+1)], use_img=True))
            else:
                self.layers.append(DEDBackbone(num_SBB, down_strides, feature_dim, input_dim))

        self.num_bev_features = feature_dim

        img_feature_dim = 32
        img_input_channels = 32
        num_SBB[0] = 1
        input_dim = img_input_channels if idx == 0 else img_feature_dim
        self.img_layers.append(DEDBackboneV1(num_SBB, down_strides, img_feature_dim, input_dim))

    def forward(self, x, img_feats=None):

        for layer in self.img_layers:
            img_feats = layer(img_feats)

        for index, layer in enumerate(self.layers):
            x = layer(x, img_feats)

        return [x]
