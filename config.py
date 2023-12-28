from os import makedirs, path
import torch
from easydict import EasyDict
# from torch.nn.functional import softplus
from losses_dist import *


class Config(EasyDict):
    RECOGNITION_HEAD = {
        "arcface": ArcFace,
        "combined": CombinedMarginLoss,
        "cosface": CosFace,
        "adaface": AdaFace,
        "sphereface": SphereFace,
        "magface": MagFace,
        # "circleloss": CircleLoss
    }

    def __init__(self, args):
        self.prefix = args.prefix
        self.head = args.head.lower()
        self.input_size = [112, 112]
        self.embedding_size = 512
        self.depth = args.depth
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = args.batch_size
        self.weight_decay = 5e-4
        self.lr = args.lr
        self.momentum = 0.9
        self.pin_memory = True
        self.frequency_log = 20
        self.epochs = args.epochs
        self.reduce_lr = [8, 12, 14]
        self.workers = args.workers
        self.train_list = args.train_list
        self.train_source = args.train_source
        self.val_list = args.val_list
        self.val_source = args.val_source
        self.output_type = torch.long
        self.recognition_head = self.RECOGNITION_HEAD[self.head]
        self.margin = args.margin
        self.fp16 = args.fp16
        self.sample_rate = args.sample_rate
        self.mask = args.mask
        self.local_rank = args.local_rank
        self.work_path = path.join("./workspace/", self.prefix)
        self.model_path = path.join(self.work_path, "models")
        self.log_path = path.join(self.work_path, "log")
        if args.local_rank == -1 or args.local_rank == 0:
            self.create_path(self.model_path)
            self.create_path(self.log_path)

    def create_path(self, file_path):
        if not path.exists(file_path):
            makedirs(file_path)
