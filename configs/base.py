from easydict import EasyDict

config = EasyDict()

config.prefix = "arcface-r50-ms1mv2"
config.head = "arcface"
config.input_size = [112, 112]
config.embedding_size = 512
config.model = "iresnet"
config.depth = "50"
config.batch_size = 512
config.weight_decay = 5e-4
config.lr = 0.1
config.momentum = 0.9
config.pin_memory = True
config.frequency_log = 20
config.epochs = 20
config.reduce_lr = [8, 12, 14]
config.workers = 2
config.num_ims = 5822653
config.train_source = "./datasets/ms1mv2.lmdb"
config.val_list = ["lfw", "cfp_fp", "agedb_30"]
config.val_source = "./test_sets"
config.margin = 0.5
config.fp16 = True
config.sample_rate = 1.0
config.mask = None
