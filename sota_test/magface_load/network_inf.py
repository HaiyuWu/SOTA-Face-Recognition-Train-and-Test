import sys
sys.path.append("..")
from backbones import magface_resnet as iresnet
from collections import OrderedDict
from tqdm import tqdm
import os
import torch.nn.functional as F
import torch.nn as nn
import torch


def load_features(args):
    if f"{args.net_mode}_{args.depth}" == 'ir_34':
        features = iresnet.iresnet34(
            pretrained=False,
            num_classes=512,
        )
    elif f"{args.net_mode}_{args.depth}" == 'ir_18':
        features = iresnet.iresnet18(
            pretrained=False,
            num_classes=512,
        )
    elif f"{args.net_mode}_{args.depth}" == 'ir_50':
        features = iresnet.iresnet50(
            pretrained=False,
            num_classes=512,
        )
    elif f"{args.net_mode}_{args.depth}" == 'ir_100':
        features = iresnet.iresnet100(
            pretrained=False,
            num_classes=512,
        )
    else:
        raise ValueError()
    return features


class NetworkBuilder_inf(nn.Module):
    def __init__(self, args):
        super(NetworkBuilder_inf, self).__init__()
        self.features = load_features(args)

    def forward(self, input):
        # add Fp, a pose feature
        x = self.features(input)
        return x


def load_dict_inf(model_path, model):
    if os.path.isfile(model_path):
        if torch.cuda.device_count() < 1:
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        else:
            checkpoint = torch.load(model_path)
        _state_dict = clean_dict_inf(model, checkpoint['state_dict'])
        model_dict = model.state_dict()
        model_dict.update(_state_dict)
        model.load_state_dict(model_dict)
        # delete to release more space
        del checkpoint
        del _state_dict
    else:
        sys.exit("=> No checkpoint found at '{}'".format(model_path))
    return model


def clean_dict_inf(model, state_dict):
    _state_dict = OrderedDict()
    for k, v in state_dict.items():
        # # assert k[0:1] == 'features.module.'
        new_k = 'features.'+'.'.join(k.split('.')[2:])
        if new_k in model.state_dict().keys() and \
           v.size() == model.state_dict()[new_k].size():
            _state_dict[new_k] = v
        # assert k[0:1] == 'module.features.'
        new_kk = '.'.join(k.split('.')[1:])
        if new_kk in model.state_dict().keys() and \
           v.size() == model.state_dict()[new_kk].size():
            _state_dict[new_kk] = v
    num_model = len(model.state_dict().keys())
    num_ckpt = len(_state_dict.keys())
    if num_model != num_ckpt:
        sys.exit("=> Not all weights loaded, model params: {}, loaded params: {}".format(
            num_model, num_ckpt))
    return _state_dict


def builder_inf(args):
    model = NetworkBuilder_inf(args)
    # Used to run inference
    model = load_dict_inf(args.model_path, model)
    return model
