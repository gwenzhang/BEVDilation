import torch
import torch.nn as nn
from functools import partial

try:
    import spconv.pytorch as spconv
except:
    import spconv as spconv

from mmdet.models.backbones.resnet import BasicBlock
from mmdet3d.ops.sparse_block import replace_feature

norm_fn_1d = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
norm_fn_2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)


def post_act_block_sparse_2d(input_dim, output_dim, kernel_size, stride=1, padding=0, norm_fn=norm_fn_1d, conv_type='subm', indice_key=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv2d(input_dim, output_dim, kernel_size, bias=False, indice_key=indice_key)

    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(input_dim, output_dim, kernel_size, stride=stride, padding=padding, bias=False, indice_key=indice_key)

    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv2d(input_dim, output_dim, kernel_size, indice_key=indice_key, bias=False)

    else:
        raise NotImplementedError

    return spconv.SparseSequential(conv, norm_fn(output_dim), nn.ReLU())



def post_act_block_sparse_3d(input_dim, output_dim, kernel_size, stride=1, padding=0, norm_fn=norm_fn_1d, conv_type='subm', indice_key=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(input_dim, output_dim, kernel_size, bias=False, indice_key=indice_key)

    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(input_dim, output_dim, kernel_size, stride=stride, padding=padding, bias=False, indice_key=indice_key)

    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(input_dim, output_dim, kernel_size, indice_key=indice_key, bias=False)

    else:
        raise NotImplementedError

    return spconv.SparseSequential(conv, norm_fn(output_dim), nn.ReLU())

class DepthEncoderResNet(nn.Module):
    def __init__(self, input_channel, input_channel_img, hidden_channel, depth_layers):
        super().__init__()

        self.depth_layers = depth_layers

        # self.conv_depth = nn.Sequential(
        #     nn.Conv2d(input_channel, hidden_channel, kernel_size=3, padding=1, bias=True),
        #     nn.BatchNorm2d(hidden_channel),
        #     nn.ReLU(inplace=True)
        # )
        self.lidar_input_net = nn.Sequential(
                nn.Conv2d(1, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(8, 32, 5, stride=4, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, hidden_channel, 5, stride=int(2 * 4 / 8),
                          padding=2),
                nn.BatchNorm2d(hidden_channel),
                nn.ReLU(True))

        self.inplanes = hidden_channel
        self._norm_layer = nn.BatchNorm2d

        self.layers = nn.ModuleList()
        self.fuse_layers = nn.ModuleList()
        self.output_layers = nn.ModuleList()
        for i in range(len(depth_layers)):
            if i == 0:
                stride = 1
            else:
                stride = 2

            self.layers.append(self._make_layer(BasicBlock, hidden_channel, depth_layers[i], stride=stride))
            self.fuse_layers.append(nn.Conv2d(input_channel_img+hidden_channel, hidden_channel, kernel_size=3, padding=1))


    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, sparse_depth, img_inputs):
        # depth = self.conv_depth(sparse_depth)
        depth = self.lidar_input_net(sparse_depth)

        img_outputs = []
        for i in range(len(img_inputs)):
            depth = self.layers[i](depth)
            depth = torch.cat([depth, img_inputs[i]], dim=1)
            depth = self.fuse_layers[i](depth)
            img_outputs.append(depth.clone())

        return img_outputs
    
class SparseConvBlock(spconv.SparseModule):

    def __init__(self, dim, kernel_size, stride, num_layer, norm_fn, indice_key):
        super(SparseConvBlock, self).__init__()

        first_block = post_act_block_sparse_3d(
            dim, dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
            norm_fn=norm_fn, indice_key=f'spconv_{indice_key}', conv_type='spconv')

        block_list = [first_block if stride > 1 else nn.Identity()]
        for _ in range(num_layer):
            block_list.append(OneSpraseConv(dim, dim, norm_fn=norm_fn, indice_key=indice_key))

        self.blocks = spconv.SparseSequential(*block_list)

    def forward(self, x):
        return self.blocks(x)
    
class OneSpraseConv(spconv.SparseModule):
    
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, bias=None, norm_fn=None, downsample=None, indice_key=None):
        super(OneSpraseConv, self).__init__()

        assert norm_fn is not None
        if bias is None:
            bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out

def check_loss_dict(loss_dict):

    for loss_name, loss_value in loss_dict.items():
        if torch.isnan(loss_value).any():
            raise ValueError(f"NaN detected in loss: '{loss_name}'. Training stopped.")
        elif torch.isinf(loss_value).any():
            raise ValueError(f"Inf detected in loss: '{loss_name}'. Training stopped.")