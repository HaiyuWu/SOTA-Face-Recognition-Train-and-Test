from datetime import datetime


def get_time():
    return (str(datetime.now())[:-10]).replace(" ", "-").replace(":", "-")


def separate_bn_param(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]

    paras_only_bn = []
    paras_wo_bn = []

    for layer in modules:
        if "model" in str(layer.__class__):
            continue
        if "container" in str(layer.__class__):
            continue
        else:
            if "batchnorm" in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn
