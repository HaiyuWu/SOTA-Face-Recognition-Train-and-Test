import argparse
from os import path
import numpy as np
import torch
from tqdm import tqdm
import onnxruntime as ort
import sys
sys.path.append("..")
from data.load_test_sets_recognition import get_val_pair
import verification
ort.set_default_logger_severity(3)


class Test:
    def __init__(self, args):
        if torch.cuda.is_available():
            self.model = ort.InferenceSession(args.model_path, None, providers=["CUDAExecutionProvider"])
        else:
            self.model = ort.InferenceSession(args.model_path, None)
        self.input_name = self.model.get_inputs()[0].name

        self.validation_list = []
        for val_name in args.val_list:
            # load one dataset
            dataset, issame = get_val_pair(args.val_source, val_name)
            # add it to the validation list
            self.validation_list.append([dataset, issame, val_name])

    def evaluate(self):
        val_acc = 0
        print("Validating...")
        for idx, validation in enumerate(self.validation_list):
            # images and labels for one dataset
            dataset, issame, val_name = validation
            acc, std = self.evaluate_recognition(dataset, issame)
            print(f"{val_name}: {acc:.5f}+-{std:.5f}")
            val_acc += acc
        val_acc /= idx + 1

        print(f"Mean accuracy: {val_acc:.5f}\n")

    def evaluate_recognition(self, samples, issame, nrof_folds=10):
        # initialize the embeddings
        embeddings = np.zeros([len(samples), 512])
        with torch.no_grad():
            # extract the features
            for idx in range(0, len(samples), args.batch_size):
                batch = torch.tensor(samples[idx: idx + args.batch_size])
                embeddings[
                    idx: idx + args.batch_size
                ] = np.array(self.model.run(None, {'{}'.format(self.input_name): batch.numpy()})).squeeze(0)
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
    parser.add_argument("--batch_size", "-b", help="Batch size.", default=512, type=int)
    parser.add_argument("--workers", "-w", help="Workers number.", default=16, type=int)

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
