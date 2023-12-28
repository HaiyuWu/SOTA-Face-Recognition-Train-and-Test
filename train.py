import argparse
import os
import numpy as np
import torch
from torch import optim, distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config import Config
from data import LMDBDataLoader, get_val_pair, setup_seed
from model import iresnet, PartialFC_V2
import verification
from utils import save_state, TrainLogger, separate_bn_param
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
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


class Train:
    def __init__(self, config):
        self.config = config
        assert self.config.head != "circleloss", "Please use 'train_with_circlelosss.py'."

        torch.cuda.set_device(local_rank)

        self.dataset = LMDBDataLoader(
            config=self.config,
            train=True
        )
        self.train_loader = self.dataset.get_loader()

        class_num = self.dataset.class_num()

        self.model = iresnet(self.config.depth, fp16=self.config.fp16).to(local_rank)

        # add m=self.config.margin to change margin
        self.head = self.config.recognition_head()

        paras_only_bn, paras_wo_bn = separate_bn_param(self.model)

        # only write at main process
        if rank == 0:
            self.writer = SummaryWriter(config.log_path)
            dummy_input = torch.zeros(1, 3, 112, 112).to(local_rank)
            self.writer.add_graph(self.model, dummy_input)
        else:
            self.writer = None

        self.model = torch.nn.parallel.DistributedDataParallel(
            module=self.model, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
            find_unused_parameters=False
        )
        self.model.register_comm_hook(None, fp16_compress_hook)

        self.head = PartialFC_V2(
            self.head, self.config.embedding_size, class_num, self.config.sample_rate, self.config.fp16
        ).to(local_rank)

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

        self.validation_list = []
        for val_name in config.val_list:
            if local_rank == 0:
                print(f"Loading {val_name}...")
            dataset, issame = get_val_pair(self.config.val_source, val_name)
            self.validation_list.append([dataset, issame, val_name])

        self.save_file(self.config, "config.txt")

        self.save_file(self.optimizer, "optimizer.txt")
        self.tensorboard_loss_every = 1000

    def run(self):
        self.model.train()
        self.head.train()
        amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)
        running_loss = 0.0
        step = 0
        best_step = 0
        best_acc = -1

        for epoch in range(self.config.epochs):
            if isinstance(self.train_loader, DataLoader):
                self.train_loader.sampler.set_epoch(epoch)
            train_logger = TrainLogger(
                self.config.batch_size, self.config.frequency_log, world_size
            )

            if epoch + 1 in self.config.reduce_lr:
                self.reduce_lr()

            for idx, data in enumerate(self.train_loader):
                imgs, labels = data
                self.optimizer.zero_grad()
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
                self.optimizer.step()

                running_loss += loss.item()

                self.optimizer.step()

                if local_rank == 0:
                    if step % self.tensorboard_loss_every == 0:
                        loss_board = running_loss / self.tensorboard_loss_every
                        self.writer.add_scalar("train_loss", loss_board, step)
                        running_loss = 0.0
                    train_logger(
                        epoch, self.config.epochs, idx, len(self.train_loader), loss.item()
                    )
                step += 1
            if local_rank == 0:
                val_acc, _ = self.evaluate(step)
                self.model.train()
                best_acc, best_step = self.save_model(
                    val_acc, best_acc, step, best_step
                )
                print(f"Best accuracy: {best_acc:.5f} at step {best_step}")

        val_acc, val_loss = self.evaluate(step)
        best_acc = self.save_model(val_acc, best_acc, step, best_step)
        print(f"Best accuracy: {best_acc} at step {best_step}")

    def save_model(self, val_acc, best_acc, step, best_step):
        if val_acc > best_acc:
            best_acc = val_acc
            best_step = step
        save_state(self.model, self.optimizer, self.config, val_acc, step, head=self.head)

        return best_acc, best_step

    def reduce_lr(self):
        for params in self.optimizer.param_groups:
            params["lr"] /= 10

        print(self.optimizer)

    def tensorboard_val(self, accuracy, step, loss=0, dataset=""):
        if local_rank == 0:
            self.writer.add_scalar("{}val_acc".format(dataset), accuracy, step)

    def evaluate(self, step):
        val_loss = 0
        val_acc = 0
        print("Validating...")
        for idx, validation in enumerate(self.validation_list):
            dataset, issame, val_name = validation
            acc, std = self.evaluate_recognition(dataset, issame)
            self.tensorboard_val(acc, step, dataset=f"{val_name}_")
            print(f"{val_name}: {acc:.5f}+-{std:.5f}")
            val_acc += acc

        val_acc /= idx + 1
        self.tensorboard_val(val_acc, step)
        print(f"Mean accuracy: {val_acc:.5f}")

        return val_acc, val_loss

    def evaluate_recognition(self, samples, issame, nrof_folds=10):
        self.model.eval()
        embeddings = np.zeros([len(samples), self.config.embedding_size])
        with torch.no_grad():
            for idx in range(0, len(samples), self.config.batch_size):
                batch = torch.tensor(samples[idx: idx + self.config.batch_size]).to(local_rank)
                embeddings[
                    idx : idx + self.config.batch_size
                ] = self.model.module(batch).cpu()
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

    # network and training parameters
    parser.add_argument(
        "--epochs", "-e", help="Number of epochs.", default=20, type=int
    )
    parser.add_argument(
        "--depth", "-d", help="Number of layers [50, 100, 152].", default="100", type=str
    )
    parser.add_argument("--lr", "-lr", help="Learning rate.", default=0.1, type=float)
    parser.add_argument("--batch_size", "-b", help="Batch size.", default=512, type=int)
    parser.add_argument("--workers", "-w", help="Workers number.", default=2, type=int)
    parser.add_argument(
        "--num_classes", "-nc", help="Number of classes.", default=85742, type=int
    )

    # training/validation configuration
    parser.add_argument("--train_list", "-t", help="List of images to train.")
    parser.add_argument(
        "--val_list",
        "-v",
        help="List of images to validate, or datasets to validate (recognition).",
        default=["lfw", "cfp_fp", "cplfw", "agedb_30", "calfw", "eclipse", "hadrian", "mlfw", "sllfw"],
        nargs="+"
    )
    parser.add_argument(
        "--train_source", "-ts", help="Path to the train images, or dataset LMDB file."
    )
    parser.add_argument(
        "--val_source", "-vs", help="Path to the val images.", default="./test_sets"
    )
    parser.add_argument(
        "--head",
        "-hd",
        help="If recognition, which head to use [arcface, cosface, adaface, sphereface].",
        type=str,
    )
    parser.add_argument("--margin", "-margin", help="Margin", default=0.5, type=float)
    parser.add_argument("--sample_rate", "-rate", help="sample rate", default=1.0, type=float)
    parser.add_argument("--prefix", "-p", help="Prefix to save the model.", type=str)
    parser.add_argument(
        "--mask", "-mask", help="selected image mask.", default=None
    )
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument("--fp16", "-fp16", help="using partial fc", action="store_true")
    args = parser.parse_args()

    config = Config(args)

    setup_seed(seed=42, cuda_deterministic=False)

    train = Train(config)
    train.run()
