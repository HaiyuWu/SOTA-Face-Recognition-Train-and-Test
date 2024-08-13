import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImageList(ImageFolder):
    def __init__(self, image_list):
        self.samples = np.asarray(pd.read_csv(image_list, header=None)).squeeze(1)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((112, 112)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img = Image.open(self.samples[index]).convert("RGB")
        return self.transform(img), self.samples[index]


class TestDataLoader(DataLoader):
    def __init__(
            self, image_list, batch_size, workers
    ):
        self._dataset = ImageList(image_list)

        super(TestDataLoader, self).__init__(
            self._dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=workers,
            drop_last=False,
        )
