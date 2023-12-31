from datetime import datetime
import logging
import os
import sys


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


class AverageMeter(object):
    """Computes and stores the average and current value
    """

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# for faster output
def init_logging(models_root):
    log_root = logging.getLogger()
    log_root.setLevel(logging.INFO)
    formatter = logging.Formatter("Training: %(asctime)s-%(message)s")
    handler_file = logging.FileHandler(os.path.join(models_root, "training.log"))
    handler_stream = logging.StreamHandler(sys.stdout)
    handler_file.setFormatter(formatter)
    handler_stream.setFormatter(formatter)
    log_root.addHandler(handler_file)
    log_root.addHandler(handler_stream)
