import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


__all__ = ['semi_ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']



def conv3x3x3(in_planes, out_planes, stride=1, groups=1, st_type='spatial'):
    """3x3x3 convolution with padding"""
    if(st_type == 'spatial'):
        kernel_size = (1,3,3)
        stride_size = (1,stride,stride)
        padding = (0,1,1)
    elif(st_type == 'temporal'):
        kernel_size = (3,1,1)
        stride_size = (stride,1,1)
        padding = (1,0,0)
    else:
        kernel_size = 3
        stride_size = stride
        padding=1
        
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride_size,
                     padding=padding, groups=groups, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1, st_type='spatialtemp'):
    """1x1x1 convolution"""
    if(st_type == 'spatial'):
        stride_size = (1,stride,stride)
    else:
        stride_size = stride
    
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride_size, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, st_type='spatial', norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3x3(inplanes, planes, stride, st_type=st_type)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, st_type=st_type)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, st_type='spatial', norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3x3(planes, planes, stride, groups, st_type)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class semi_ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=101, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(semi_ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        planes = [int(width_per_group * groups * 2 ** i) for i in range(4)]

        self.inplanes = planes[0]
        self.conv1 = nn.Conv3d(3, planes[0], kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3),
                               bias=False)
        self.bn1 = norm_layer(planes[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.layer1 = self._make_layer(block, planes[0], layers[0][0], groups=groups, st_type='spatial', norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, planes[1], int(layers[0][1]), stride=2, groups=groups, st_type='spatial', norm_layer=norm_layer)
        
        self.templayer = self._make_layer(block, planes[1], int(layers[1]), stride=1, groups=groups, st_type='temporal', norm_layer=norm_layer)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        
        self.inplanes = planes[1]* block.expansion
        self.layer3a = self._make_layer(block, planes[2], int(math.ceil(layers[2][0])), stride=1, groups=groups, st_type='spatialtemp', norm_layer=norm_layer)
        self.layer3b = self._make_layer(block, planes[2], int(math.floor(layers[2][1])), stride=2, groups=groups, st_type='spatialtemp', norm_layer=norm_layer)
        self.layer4x = self._make_layer(block, planes[3], layers[2][2], stride=2, groups=groups, st_type='spatialtemp', norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))

        self.fc1 = nn.Linear(planes[3] * block.expansion, num_classes)
        self.fc2 = nn.Linear((planes[1]+planes[1]+planes[3]) * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, groups=1, st_type='spatial', norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride, st_type),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups, st_type, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups, st_type=st_type, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)       
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.maxpool1(x)
        spatpool = self.avgpool(x)

        x = self.templayer(x)
        x = self.maxpool2(x)
        temppool = self.avgpool(x)

        x = self.layer3a(x)
        x = self.layer3b(x)
        x = self.layer4x(x)
        x = self.avgpool(x)
        spattemppool = x
        
        x = torch.cat((spatpool, temppool, spattemppool),1)
        x = x.view(x.size(0), -1)
        x = self.fc2(x)
        
        return x


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = semi_ResNet(BasicBlock, [(2, 1), 1, (1, 1, 2)], **kwargs) #[2, 2, 2, 2]

    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = semi_ResNet(BasicBlock, [(3, 2), 2, (3, 3, 3)], **kwargs) #[3, 4, 6, 3]

    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = semi_ResNet(Bottleneck, [(3, 2), 2, (3, 3, 3)], **kwargs) #[3, 4, 6, 3]

    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = semi_ResNet(Bottleneck, [(3, 2), 2, (12, 11, 3)], **kwargs) #[3, 4, 23, 3]

    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = semi_ResNet(Bottleneck, [(3, 4), 4, (18, 18, 3)], **kwargs) #[3, 8, 36, 3]

    return model
