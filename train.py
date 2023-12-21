import argparse
from os import path
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from torch import autograd, optim
from torch.nn import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from config import Config
from data.data_loader_train_lmdb import LMDBDataLoader
from data.load_test_sets_recognition import get_val_pair
from model.model_wrapper import ModelWrapper
from model.iresnet import iresnet
import verification
from utils.model_loader import save_state
from utils.train_logger import TrainLogger
from utils.utils import separate_bn_param


class Train:
    def __init__(self, config):
        self.config = config
        assert self.config.head != "circleloss", "Please use 'train_with_circlelosss.py'."
        self.writer = SummaryWriter(config.log_path)

        if path.isfile(self.config.train_source):
            self.train_loader = LMDBDataLoader(
                config=self.config,
                lmdb_path=self.config.train_source,
                train=True,
                mask=self.config.mask,
            )

        class_num = self.train_loader.class_num()
        print(len(self.train_loader.dataset))
        print(f"Classes: {class_num}")

        self.model = iresnet(self.config.depth)

        # add  to change margin
        self.head = self.config.recognition_head(
            classnum=class_num, m=self.config.margin
        )

        self.full_model = ModelWrapper(self.model, self.head).to(self.config.device)
        # print(self.full_model)
        paras_only_bn, paras_wo_bn = separate_bn_param(self.model)

        dummy_input = torch.zeros(1, 3, 112, 112).to(self.config.device)
        self.writer.add_graph(self.full_model, dummy_input)

        if torch.cuda.device_count() > 1:
            print(f"Model will use {torch.cuda.device_count()} GPUs!")
            self.full_model = DataParallel(self.full_model)

        if self.config.val_source is not None:
            self.validation_list = []
            for val_name in config.val_list:
                dataset, issame = get_val_pair(self.config.val_source, val_name)
                self.validation_list.append([dataset, issame, val_name])

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

        print(self.config)
        self.save_file(self.config, "config.txt")

        print(self.optimizer)
        self.save_file(self.optimizer, "optimizer.txt")

        self.tensorboard_loss_every = max(len(self.train_loader) // 100, 1)

    def run(self):
        self.full_model.train()
        running_loss = 0.0
        step = 0
        best_step = 0
        best_acc = -1

        for epoch in range(self.config.epochs):
            train_logger = TrainLogger(
                self.config.batch_size, self.config.frequency_log
            )

            if epoch + 1 in self.config.reduce_lr:
                self.reduce_lr()

            for idx, data in enumerate(self.train_loader):
                imgs, labels = data
                imgs = imgs.to(self.config.device)
                labels = labels.to(self.config.device)

                self.optimizer.zero_grad()

                outputs = self.full_model(imgs, labels)
                loss = self.config.loss(outputs, labels)

                loss.backward()
                running_loss += loss.item()

                self.optimizer.step()

                if loss == torch.nan:
                    for name, param in self.full_model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm()
                            print(f'Parameter {name} has gradient norm {grad_norm:.2f}')

                if step % self.tensorboard_loss_every == 0:
                    loss_board = running_loss / self.tensorboard_loss_every
                    self.writer.add_scalar("train_loss", loss_board, step)
                    running_loss = 0.0
                train_logger(
                    epoch, self.config.epochs, idx, len(self.train_loader), loss.item()
                )
                step += 1
            val_acc, _ = self.evaluate(step)
            self.full_model.train()
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
        save_state(self.full_model, self.optimizer, self.config, val_acc, step)

        return best_acc, best_step

    def reduce_lr(self):
        for params in self.optimizer.param_groups:
            params["lr"] /= 10

        print(self.optimizer)

    def tensorboard_val(self, accuracy, step, loss=0, dataset=""):
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
        self.full_model.eval()
        embeddings = np.zeros([len(samples), self.config.embedding_size])

        with torch.no_grad():
            for idx in range(0, len(samples), self.config.batch_size):
                batch = torch.tensor(samples[idx : idx + self.config.batch_size])
                embeddings[
                    idx : idx + self.config.batch_size
                ] = self.full_model.module.model(batch.to(self.config.device)).cpu()
                idx += self.config.batch_size
        normalized_embedding = np.divide(embeddings, np.linalg.norm(embeddings, 2, 1, True))
        tpr, fpr, accuracy = verification.evaluate(
            normalized_embedding, issame, nrof_folds
        )

        return round(accuracy.mean(), 5), round(accuracy.std(), 5)

    def save_file(self, string, file_name):
        file = open(path.join(self.config.work_path, file_name), "w")
        file.write(str(string))
        file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a race, gender, age or recognition models."
    )

    # network and training parameters
    parser.add_argument(
        "--epochs", "-e", help="Number of epochs.", default=20, type=int
    )
    parser.add_argument(
        "--depth", "-d", help="Number of layers [50, 100, 152].", default=100, type=str
    )
    parser.add_argument("--lr", "-lr", help="Learning rate.", default=0.1, type=float)
    parser.add_argument("--batch_size", "-b", help="Batch size.", default=512, type=int)
    parser.add_argument("--workers", "-w", help="Workers number.", default=16, type=int)
    parser.add_argument(
        "--num_classes", "-nc", help="Number of classes.", default=85742, type=int
    )

    # training/validation configuration
    parser.add_argument("--train_list", "-t", help="List of images to train.")
    parser.add_argument(
        "--val_list",
        "-v",
        help="List of images to validate, or datasets to validate (recognition).",
        default=["lfw", "cfp_fp", "cplfw", "agedb_30", "calfw", "cfh", "cb"],
    )
    parser.add_argument(
        "--train_source", "-ts", help="Path to the train images, or dataset LMDB file."
    )
    parser.add_argument(
        "--val_source", "-vs", help="Path to the val images, or dataset LMDB file."
    )
    parser.add_argument(
        "--head",
        "-hd",
        help="If recognition, which head to use [arcface, cosface, adaface, sphereface].",
        type=str,
    )
    parser.add_argument("--margin", "-margin", help="Margin", default=0.5, type=float)
    parser.add_argument("--prefix", "-p", help="Prefix to save the model.", type=str)
    parser.add_argument(
        "--mask", "-mask", help="selected image mask.", default=None
    )

    args = parser.parse_args()

    config = Config(args)

    torch.manual_seed(42)
    np.random.seed(42)

    train = Train(config)
    train.run()
