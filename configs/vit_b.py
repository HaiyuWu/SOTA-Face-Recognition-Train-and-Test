from easydict import EasyDict

config = EasyDict()

config.prefix = "arcface-vits-ms1mv2"
config.head = "arcface"
config.input_size = [112, 112]
config.embedding_size = 512
config.model = "vit"
config.depth = "b"
config.batch_size = 128
config.optimizer = "adamw"
config.lr = 0.1
config.epochs = 40
config.reduce_lr = [14, 21, 27]
config.train_source = "./datasets/ms1mv2.lmdb"
config.val_list = ["lfw", "cfp_fp", "agedb_30"]
config.val_source = "./test_sets"
config.margin = 0.5
config.mask = None
config.warmup_epoch = config.epochs // 10
config.augment = True
config.scheduler = True
