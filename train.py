import argparse
import os
import logging
import numpy as np
import torch
from time import time
from torch import optim, distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data import LMDBDataLoader,WebDataLoader, get_val_pair, setup_seed
from lr_scheduler import PolyScheduler
from model import iresnet, PartialFC_V2, get_vit
import verification
from utils import *
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook


try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:13584",
        rank=rank,
        world_size=world_size,
    )


class Train:
    def __init__(self, config):
        self.config = config

        if local_rank == 0:
            create_path(self.config.model_path)
            create_path(self.config.log_path)
            init_logging(self.config.work_path)

        torch.cuda.set_device(local_rank)

        self.dataset = LMDBDataLoader(
            config=self.config,
            train=True
        )

        # self.dataset = WebDataLoader(
        #     config=self.config,
        #     train=True
        # )
        self.train_loader = self.dataset.get_loader()

        class_num = self.dataset.class_num()
        img_num = self.dataset.get_length()

        if self.config.model == "iresnet":
            self.model = iresnet(self.config.depth, fp16=self.config.fp16, mode=self.config.mode).to(local_rank)
        elif self.config.model == "vit":
            self.model = get_vit(self.config.depth).to(local_rank)
            # self.model =  torch.compile(self.model)

        self.head = self.config.recognition_head

        paras_only_bn, paras_wo_bn = separate_bn_param(self.model)

        # only write at main process
        if rank == 0:
            self.writer = SummaryWriter(config.log_path)
            dummy_input = torch.zeros(1, 3, 112, 112).to(local_rank)
            self.writer.add_graph(self.model, dummy_input)

            for key, value in self.config.items():
                logging.info("%-25s %s", key, value)
        else:
            self.writer = None

        self.model = torch.nn.parallel.DistributedDataParallel(
            module=self.model, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16
        )
        self.model.register_comm_hook(None, fp16_compress_hook)
        # for using checkpoint
        self.model._set_static_graph()

        self.head = PartialFC_V2(
            self.head, self.config.embedding_size, class_num, self.config.sample_rate, self.config.fp16
        ).to(local_rank)

        if self.config.optimizer == "sgd":
            self.optimizer = optim.SGD(
                [
                    {"params": paras_wo_bn, "weight_decay": self.config.weight_decay},
                    {
                        "params": self.head.parameters(),
                        "weight_decay": self.config.weight_decay,
                    },
                    {"params": paras_only_bn},
                ],
                lr=self.config.lr,
                momentum=self.config.momentum,
            )
        elif self.config.optimizer == "adamw":
            self.optimizer = optim.AdamW(
                params=[
                    {"params": self.model.parameters()},
                    {
                        "params": self.head.parameters(),
                    },
                ],
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise

        total_batch = self.config.batch_size * world_size
        if self.config.scheduler:
            print("PolyScheduler is used!")
            warmup_step = img_num // total_batch * self.config.warmup_epoch
            total_step = img_num // total_batch * self.config.epochs

            self.lr_scheduler = PolyScheduler(
                optimizer=self.optimizer,
                base_lr=self.config.lr,
                max_steps=total_step,
                warmup_steps=warmup_step,
                last_epoch=-1
            )

        self.validation_list = []
        for val_name in config.val_list:
            if local_rank == 0:
                print(f"Loading {val_name}...")
            dataset, issame = get_val_pair(self.config.val_source, val_name)
            self.validation_list.append([dataset, issame, val_name])

        self.train_logger = TrainLogger(
            total_batch,
            self.config.frequency_log,
            self.dataset.get_length() // total_batch * self.config.epochs,
            self.config.epochs,
            self.writer
        )

        self.save_file(self.config, "config.txt")
        self.save_file(self.optimizer, "optimizer.txt")
        self.tensorboard_loss_every = 1000
        self.best_acc = -1
        self.best_step = 0

    def run(self):
        self.model.train()
        self.head.train()
        loss_am = AverageMeter()
        amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)
        step = 1
        for epoch in range(self.config.epochs):
            if isinstance(self.train_loader, DataLoader):
                self.train_loader.sampler.set_epoch(epoch)
            if not self.config.scheduler and epoch + 1 in self.config.reduce_lr:
                self.reduce_lr()

            for idx, data in enumerate(self.train_loader):
                imgs, labels = data
                embeddings = self.model(imgs)
                loss = self.head(embeddings, labels)

                if self.config.fp16:
                    amp.scale(loss).backward()
                    amp.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    amp.step(self.optimizer)
                    amp.update()
                    self.optimizer.zero_grad()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                if self.config.scheduler:
                    self.lr_scheduler.step()
                else:
                    self.optimizer.step()
                loss_am.update(loss.item(), 1)

                # lrs_of_this_epoch = [x['lr'] for x in self.optimizer.param_groups]
                # lr = self.lr_scheduler.get_last_lr()
                # print(lr)
                # self.train_logger(step, epoch,lr[0], loss_am, local_rank)
                self.train_logger(step, epoch, loss_am, local_rank)

                step += 1

            self.save_model(step)

    def save_model(self, step):
        if local_rank == 0:
            val_acc, _ = self.evaluate(step)
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_step = step
            save_state(self.model, self.optimizer, self.config, val_acc, step, head=self.head)
            logging.info(f"Best accuracy: {self.best_acc:.5f} at step {self.best_step}")

    def reduce_lr(self):
        for params in self.optimizer.param_groups:
            params["lr"] /= 10

    def evaluate(self, step):
        if local_rank == 0:
            self.model.eval()
            val_loss = 0
            val_acc = 0
            logging.info(f"Validating...")
            for idx, validation in enumerate(self.validation_list):
                dataset, issame, val_name = validation
                acc, std = self.evaluate_recognition(dataset, issame)
                self.writer.add_scalar("{} acc".format(val_name), acc, step)
                logging.info(f"{val_name}: {acc:.5f}+-{std:.5f}")
                val_acc += acc

            val_acc /= idx + 1
            self.writer.add_scalar("Mean acc", val_acc, step)
            logging.info(f"Mean accuracy: {val_acc:.5f}")
            self.model.train()

            return val_acc, val_loss

    def l2_norm(self, input: torch.Tensor, axis=1):
        norm = torch.norm(input, 2, axis, True)
        output = torch.div(input, norm)
        return output, norm

    def evaluate_recognition(self, samples, issame, nrof_folds=10):
        embedding_length = len(samples) // 2
        embeddings = np.zeros([embedding_length, self.config.embedding_size])

        with torch.no_grad():
            for idx in range(0, embedding_length, self.config.batch_size):
                batch_flip = torch.tensor(samples[embedding_length + idx: embedding_length + idx + self.config.batch_size])
                batch_or = torch.tensor(samples[idx: idx + batch_flip.shape[0]])
                if self.config.add_flip:
                    embeddings[idx: idx + self.config.batch_size] = self.model(batch_or.to(local_rank)).cpu() + \
                                                                    self.model(batch_flip.to(local_rank)).cpu()
                elif self.config.add_norm:
                    embeddings_flip, norms_flip = self.l2_norm(self.model(batch_flip.to(local_rank)), axis=1)
                    embeddings_or, norms_or = self.l2_norm(self.model(batch_or.to(local_rank)), axis=1)
                    embeddings[idx:idx + self.config.batch_size] = (embeddings_flip * norms_flip +
                                                                embeddings_or * norms_or).cpu()
                else:
                    embeddings[idx: idx + self.config.batch_size] = self.model(batch_or.to(local_rank)).cpu()
                idx += self.config.batch_size
        normalized_embedding = np.divide(embeddings, np.linalg.norm(embeddings, 2, 1, True))

        tpr, fpr, accuracy = verification.evaluate(
            normalized_embedding, issame, nrof_folds
        )

        return round(accuracy.mean(), 5), round(accuracy.std(), 5)

    def save_file(self, string, file_name):
        file = open(os.path.join(self.config.work_path, file_name), "w")
        file.write(str(string))
        file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a recognition model."
    )
    parser.add_argument("--config_file", "-config", help="path of config file.", default="./configs/base.py", type=str)
    parser.add_argument("--device", default='0', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    config = get_config(args.config_file)
    setup_seed(seed=42, cuda_deterministic=False)
    train = Train(config)
    train.run()
