import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


__all__ = [
    'semi_DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161'
]



class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, kernel_size, stride, padding, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1',
                        nn.Conv3d(
                            num_input_features,
                            bn_size * growth_rate,
                            kernel_size=1,
                            stride=1,
                            bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2',
                        nn.Conv3d(
                            bn_size * growth_rate,
                            growth_rate,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, kernel_size, stride, padding, growth_rate,
                 drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, kernel_size, stride, padding, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features,pool_kernel,pool_stride,padding=0):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv',
                        nn.Conv3d(
                            num_input_features,
                            num_output_features,
                            kernel_size=1,
                            stride=1,
                            bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_stride,padding=padding))


class semi_DenseNet(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self,
                 growth_rate=32,
                 spatial_config=(6,12),
                 temp_config=(6, ),
                 spatialtemp_config=(12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=101):

        super(semi_DenseNet, self).__init__()


        # First convolution
        self.spatfeatures = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     3,
                     num_init_features,
                     kernel_size=(1, 7, 7),
                     stride=(1, 2, 2),
                     padding=(0, 3, 3),
                     bias=False)),
                ('norm0', nn.BatchNorm3d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))),
            ]))
                
        self.spattrans = nn.Sequential()
                
        
        # Each spatial denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(spatial_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                kernel_size=(1,3,3),
                stride=(1,1,1),
                padding=(0,1,1),
                growth_rate=growth_rate,
                drop_rate=drop_rate)
            self.spatfeatures.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(spatial_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                    pool_kernel=(1,2,2),
                    pool_stride=(1,2,2))
                self.spatfeatures.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            if i == len(spatial_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features,
                    pool_kernel=(1,2,2),
                    pool_stride=(1,2,2))
                
                self.spattrans.add_module('spatial_transition%d' % (i + 1), trans)
                num_spatfeatures = num_features

        
        # Each temporal denseblock
        self.tempfeatures = nn.Sequential()
        self.temptrans = nn.Sequential()
        for i, num_layers in enumerate(temp_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                kernel_size=(3,1,1),
                stride=(1,1,1),
                padding=(1,0,0),
                growth_rate=growth_rate,
                drop_rate=drop_rate)
            self.tempfeatures.add_module('temp_denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(temp_config):
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                    pool_kernel=(3,1,1),
                    pool_stride=(2,1,1),
                    padding=(1,0,0))
                self.temptrans.add_module('temp_transition%d' % (i + 1), trans)
                num_tempfeatures = num_features
                num_features = num_features // 2
                

        # Each spatial-temporal denseblock
        self.spattempfeatures = nn.Sequential()
        for i, num_layers in enumerate(spatialtemp_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                kernel_size=3,
                stride=1,
                padding=1,
                growth_rate=growth_rate,
                drop_rate=drop_rate)
            self.spattempfeatures.add_module('spatialtemp_denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i == 0: #first block
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features,
                    pool_kernel=2,
                    pool_stride=2)
                self.spattempfeatures.add_module('spatialtemp_transition%d' % (i + 1), trans)
            elif i != len(spatialtemp_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                    pool_kernel=2,
                    pool_stride=2)
                self.spattempfeatures.add_module('spatialtemp_transition%d' % (i + 1), trans)
                num_features = num_features // 2


        # Final batch norm
        self.spattempfeatures.add_module('st_norm5', nn.BatchNorm3d(num_features))
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        # Linear layer
        self.st_classifier = nn.Linear(num_spatfeatures + num_tempfeatures + num_features, num_classes)

    def forward(self, x):

        features = self.spatfeatures(x)
        spatfeatures = F.relu(features, inplace=True)
        spatfeatures = self.avgpool(spatfeatures)
        
        features = self.spattrans(features)
        features = self.tempfeatures(features) 
        tempfeatures = F.relu(features, inplace=True)
        tempfeatures = self.avgpool(tempfeatures)
        
        features = self.temptrans(features)
        features = self.spattempfeatures(features)
        spattempfeatures = F.relu(features, inplace=True)
        spattempfeatures = self.avgpool(spattempfeatures)
        
        out = torch.cat((spatfeatures,tempfeatures,spattempfeatures),1)
        out = out.view(out.size(0), -1)
        out = self.st_classifier(out)
        
        return out


def densenet121(**kwargs):
    model = semi_DenseNet(
        num_init_features=64,
        growth_rate=32,
        spatial_config = (6, 6),
        temp_config=(6, ),
        spatialtemp_config=(12, 12, 16),
        **kwargs)
    return model


def densenet169(**kwargs):
    model = semi_DenseNet(
        num_init_features=64,
        growth_rate=32,
        spatial_config = (6, 6),
        temp_config=(6, ),
        spatialtemp_config=(16, 16, 32),
        **kwargs)
    return model


def densenet201(**kwargs):
    model = semi_DenseNet(
        num_init_features=64,
        growth_rate=32,
        spatial_config = (6, 6),
        temp_config=(6, ),
        spatialtemp_config=(24, 24, 32),
        **kwargs)
    return model


def densenet161(**kwargs):
    model = semi_DenseNet(
        num_init_features=96,
        growth_rate=48,
        spatial_config = (6, 6),
        temp_config=(6, ),
        spatialtemp_config=(18, 18, 24),
        **kwargs)
    return model
