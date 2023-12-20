"""
Uses weights and models implementation' from
https://github.com/deepinsight/insightface
"""
import argparse
import numpy as np
from tqdm import tqdm
import sys
sys.path.append("..")
from data.load_test_sets_recognition import get_val_pair
import verification
sys.path.insert(0, "/afs/crc.nd.edu/user/h/hwu6/Private/insightface-master/deploy/")
import face_model


class Test:
    def __init__(self, model, args):
        self.model = model

        self.validation_list = []
        for val_name in args.val_list:
            dataset, issame = get_val_pair(args.val_source, val_name)
            # use BGR
            dataset = ((dataset[:, ::-1, :, :].transpose(0, 2, 3, 1).copy() * 0.5 + 0.5) * 255).astype(np.uint8)
            self.validation_list.append([dataset, issame, val_name])

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
        embeddings = np.zeros([len(samples), 512])

        for idx in range(len(samples)):
            img = samples[idx]
            img = self.model.get_input_aligned(img)
            feature = self.model.get_feature(img)
            embeddings[idx] = feature
        tpr, fpr, accuracy = verification.evaluate(
            embeddings, issame, nrof_folds
        )

        return round(accuracy.mean(), 5), round(accuracy.std(), 5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="test a recognition model."
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

    # ArcFace params
    parser.add_argument("--image-size", default="112,112", help="")
    parser.add_argument(
        "--model",
        help="path to model.",
        default="/afs/crc.nd.edu/user/h/hwu6/Private/Pre-trained_model/model-r100-ii/model,0",
    )
    parser.add_argument("--ga-model", default="", help="path to load model.")
    parser.add_argument("--gender_model", default="", help="path to load model.")
    parser.add_argument("--gpu", default=0, type=int, help="gpu id")
    parser.add_argument(
        "--det",
        default=1,
        type=int,
        help="mtcnn: 1 means using R+O, 0 means detect from begining",
    )
    parser.add_argument("--flip", default=0, type=int, help="whether do lr flip aug")
    parser.add_argument(
        "--threshold", default=1.24, type=float, help="ver dist threshold"
    )

    args = parser.parse_args()
    model = face_model.FaceModel(args)

    test = Test(model, args)
    test.evaluate()
