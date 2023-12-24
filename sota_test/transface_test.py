import argparse
from os import path
import numpy as np
import torch
from tqdm import tqdm
from torch.nn import DataParallel
from backbones.transface_vit import get_vit
import sys
sys.path.append("..")
from data.load_test_sets_recognition import get_val_pair
import verification


class Test:
    def __init__(self, args):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.create_model(args)

        if torch.cuda.device_count() > 0:
            print(f"Model will use {torch.cuda.device_count()} GPUs!")
            self.model = DataParallel(self.model)
            self.model = self.model.to(self.device)
        self.validation_list = []
        for val_name in args.val_list:
            dataset, issame = get_val_pair(args.val_source, val_name)
            self.validation_list.append([dataset, issame, val_name])

    def create_model(self, args):
        model = get_vit(args)
        model.load_state_dict(torch.load(args.model_path))
        return model

    def evaluate(self):
        val_acc = 0
        print("Validating...")
        for idx, validation in enumerate(self.validation_list):
            dataset, issame, val_name = validation
            acc, std = self.evaluate_recognition(dataset, issame)
            print(f"{val_name}: {acc:.5f}+-{std:.5f}")
            val_acc += acc
        val_acc /= idx + 1

        print(f"Mean accuracy: {val_acc:.5f}\n")

    def evaluate_recognition(self, samples, issame, nrof_folds=10):
        self.model.eval()
        embeddings = np.zeros([len(samples), 512])
        with torch.no_grad():
            for idx in range(0, len(samples), args.batch_size):
                batch = torch.tensor(samples[idx: idx + args.batch_size])
                embeddings[
                    idx: idx + args.batch_size
                ] = self.model(batch.to(self.device))[0].cpu()
                idx += args.batch_size
        normalized_embedding = np.divide(embeddings, np.linalg.norm(embeddings, 2, 1, True))
        tpr, fpr, accuracy = verification.evaluate(
            normalized_embedding, issame, nrof_folds
        )

        return round(accuracy.mean(), 5), round(accuracy.std(), 5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="test a recognition model."
    )
    parser.add_argument(
        "--model_path", "-model", help="model path.", type=str
    )
    parser.add_argument(
        "--vit_mode", "-mode", help="ViT mode [s, b, l].", type=str, default="l"
    )
    parser.add_argument("--batch_size", "-b", help="Batch size.", default=512, type=int)
    parser.add_argument(
        "--val_list",
        "-v",
        help="List of images to validate, or datasets to validate (recognition).",
        nargs="+",
        default=["lfw", "cfp_fp", "cplfw", "agedb_30", "calfw", "hadrian", "eclipse"],
    )
    parser.add_argument(
        "--val_source", "-vs", help="Path to the val images, or dataset LMDB file.", default="../test_sets"
    )

    args = parser.parse_args()

    test = Test(args)
    test.evaluate()
