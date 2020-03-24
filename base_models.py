import torch
from torch import nn
import torchvision

import re
import semi_c3d, semi_resnet, semi_densenet 
import torch.utils.model_zoo as model_zoo


model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',

    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

def base_model(model, model_depth=None, n_classes=101,pretrained=True):
    assert model in [
        'semi_c3d', 'semi_densenet', 'semi_resnet'
    ]

    if model == 'semi_c3d':       
        model = semi_c3d.semi_C3D(num_classes=n_classes)
        
        if pretrained:
            model = load_vgg16_weights(model, 'vgg16')
        
    if model == 'semi_resnet':
        assert model_depth in [18, 34, 50, 101, 152]
        
        if model_depth == 18:
            model = semi_resnet.resnet18(
                num_classes=n_classes)
            arch = 'resnet18'
        elif model_depth == 34:
            model = semi_resnet.resnet34(
                num_classes=n_classes)
            arch = 'resnet34'
        elif model_depth == 50:
            model = semi_resnet.resnet50(
                num_classes=n_classes)
            arch = 'resnet50'
        elif model_depth == 101:
            model = semi_resnet.resnet101(
                num_classes=n_classes)
            arch = 'resnet101'
        elif model_depth == 152:
            model = semi_resnet.resnet152(
                num_classes=n_classes)
            arch = 'resnet152'
            
        if pretrained:
            model = load_resnet_weights(model, arch)
            
            
    elif model == 'semi_densenet':
        assert model_depth in [121, 169, 201, 161]
        
        if model_depth == 121:
            model = semi_densenet.densenet121(
                num_classes=n_classes)
            arch = 'densenet121'
            
        elif model_depth == 169:
            model = semi_densenet.densenet169(
                num_classes=n_classes)
            arch = 'densenet169'
            
        elif model_depth == 201:
            model = semi_densenet.densenet201(
                num_classes=n_classes)
            arch = 'densenet201'
            
        elif model_depth == 161:
            model = semi_densenet.densenet161(
                num_classes=n_classes)
            arch = 'densenet161'
            
        if pretrained:
            model = load_densenet_weights(model, arch)

    return model



def load_vgg16_weights(model, arch):

    print("=> using pre-trained model '{}'".format(model_urls[arch])) 
    pretrained_dict = model_zoo.load_url(model_urls[arch])
    model_dict = model.state_dict()

    new_dict={}
    for k, v in pretrained_dict.items():
        
        if k in model_dict:
            if 'weight' in k:
                v = v.unsqueeze(2)
            new_dict[k] = v 
        
    model_dict.update(new_dict) 
    model.load_state_dict(model_dict)

    for name,param in model.named_parameters():
        if name in pretrained_dict:
            param.requires_grad = False
    
    return model


def load_resnet_weights(model, arch):
    
    print("=> using pre-trained model '{}'".format(model_urls[arch])) 
    pretrained_dict = model_zoo.load_url(model_urls[arch])
    model_dict = model.state_dict()

    new_dict={}
    list_weights=['conv','downsample.0.weight']
    for k, v in pretrained_dict.items():
        if k in model_dict:
            if any(weights in k for weights in list_weights):
                v = v.unsqueeze(2)
            new_dict[k] = v 
        
    model_dict.update(new_dict) 
    model.load_state_dict(model_dict)

    for name,param in model.named_parameters():
        if name in pretrained_dict:
            param.requires_grad = False
    
    return model


def load_densenet_weights(model, arch):

    print("=> using pre-trained model '{}'".format(model_urls[arch]))    
    pretrained_dict = model_zoo.load_url(model_urls[arch])
    pretrained_dict = {k.replace('features','spatfeatures') : v for k,v in pretrained_dict.items()}

    pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    for key in list(pretrained_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            pretrained_dict[new_key] = pretrained_dict[key]
            del pretrained_dict[key]

    model_dict = model.state_dict()

    new_dict={}
    for k, v in pretrained_dict.items():
        if k in model_dict:
            if 'conv' in k:
                v = v.unsqueeze(2)
            new_dict[k] = v 
        
    model_dict.update(new_dict) 
    model.load_state_dict(model_dict)

    for name,param in model.named_parameters():
        if name in pretrained_dict:
            param.requires_grad = False
        
    return model




# test
if __name__=='__main__':

    #model = base_model('semi_c3d',pretrained=True)
    #model = base_model('semi_resnet',34,101,pretrained=True)
    model = base_model('semi_densenet',121,101,pretrained=True)

    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_pretrained_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad==False)
    print(f'{total_pretrained_params:,} pretrained parameters.')

