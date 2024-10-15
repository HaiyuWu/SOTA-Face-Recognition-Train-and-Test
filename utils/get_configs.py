import importlib
from os import path, makedirs
from losses import *
import torch


RECOGNITION_HEAD = {
        "arcface": ArcFace,
        "combined": CombinedMarginLoss,
        "cosface": CosFace,
        "adaface": AdaFace,
        "sphereface": SphereFace,
        "magface": MagFace,
        "curricularface": CurricularFace,
        "uniface": UniFace,
        # "circleloss": CircleLoss
    }


def get_config(config_file):
    temp_module_name = []
    for part in config_file.split("/")[::-1]:
        cur_part = path.splitext(part)[0]
        if cur_part != "configs":
            temp_module_name.append(cur_part)
        else:
            break
    config = importlib.import_module("configs.base")
    cfg = config.config
    config = importlib.import_module("configs." + ".".join(temp_module_name[::-1]))
    job_cfg = config.config
    cfg.update(job_cfg)
    # head initialization
    cfg["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg.head == "combined":
        m1, m2, m3 = cfg.margin
        cfg["recognition_head"] = RECOGNITION_HEAD[cfg.head](m1=m1, m2=m2, m3=m3)
    elif cfg.head == "magface":
        l_a, u_a, l_margin, u_margin = cfg.margin
        cfg["recognition_head"] = RECOGNITION_HEAD[cfg.head](l_a=l_a, u_a=u_a, l_margin=l_margin, u_margin=u_margin)
    elif cfg.head == "uniface":
        margin, bias_init = cfg.margin
        cfg["recognition_head"] = RECOGNITION_HEAD[cfg.head](margin=margin, bias_init=bias_init)
    else:
        cfg["recognition_head"] = RECOGNITION_HEAD[cfg.head](margin=cfg.margin)

    cfg["work_path"] = path.join("./workspace/", cfg.prefix)
    cfg["model_path"] = path.join(cfg.work_path, "models")
    cfg["log_path"] = path.join(cfg.work_path, "log")
    return cfg


def create_path(file_path):
    if not path.exists(file_path):
        makedirs(file_path)
