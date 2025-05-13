import argparse
import numpy as np
import torch
from os import path, makedirs
from torch.nn import DataParallel
from model import iresnet, PartialFC_V2, get_vit
from data import TestDataLoader
from tqdm import tqdm


class Extractor(object):
    def __init__(self, args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.create_model(args)

        if torch.cuda.device_count() > 0:
            print(f"Model will use {torch.cuda.device_count()} GPUs!")
            self.model = DataParallel(self.model)
            self.model = self.model.to(self.device)
        self.model.eval()

    def create_model(self, args):
        if args.model == "iresnet":
            model = iresnet(args.depth)
        elif args.model == "vit":
            model = get_vit(args.depth)
        model.load_state_dict(torch.load(args.model_path))
        return model

    def get_im_id(self, im_path):
        sep = im_path.split("/")
        return f"{sep[-2]}/{sep[-1][:-3]}"

    def extract(self, args):
        test_loader = TestDataLoader(args.image_paths, args.batch_size, args.workers)
        for im, im_path in tqdm(test_loader):
            features = (self.model(im)).cpu().detach().numpy()
            for i in range(len(features)):
                im_id = self.get_im_id(im_path[i])
                save_folder = path.join(args.destination, path.split(im_id)[0])
                if not path.exists(save_folder):
                    makedirs(save_folder)
                np.save(path.join(args.destination, im_id + "npy"), features[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image feature extraction."
    )
    parser.add_argument(
        "--model_path", "-model_path", help="model path.", type=str
    )
    parser.add_argument(
        "--model", "-model", help="iresnet/vit.", type=str, default="iresnet"
    )
    parser.add_argument(
        "--mode", "-mode", help="using SE attention [normal/se].", type=str, default="normal"
    )
    parser.add_argument(
        "--depth", "-d",
        help="layers size: resnet [18, 34, 50, 100, 152, 200] / vit [s, b, l].",
        default="100",
        type=str
    )
    parser.add_argument("--batch_size", "-b", help="Batch size.", default=512, type=int)
    parser.add_argument("--workers", "-w", help="workers.", default=2, type=int)
    parser.add_argument("--image_paths", "-i", help="A file contains image paths.", type=str)
    parser.add_argument("--destination", "-dest", help="destination.", type=str, default="./features")

    args = parser.parse_args()

    extractor = Extractor(args)
    extractor.extract(args)
