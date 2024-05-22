import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import warnings

from src.xplane_autoland.controllers.glideslope_controller import GlideSlopeController

class AutolandImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, load_png=False,
                 glideslope_controller=None):
        """
        glideslope_controller: the glideslope controller you plan to use with a trained network
                               uses default parameters if not supplied
        """
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.load_png = load_png # if True loads an image, otherwise assumes it's a tensor

        if glideslope_controller is None:
            glideslope_controller = GlideSlopeController(gamma=3) # default 3 degree glideslope
        self.glideslope_controller = glideslope_controller

        # normalize with expected maximum values in meters
        # note: in practice can be larger, it will just map to a value > 1
        self.orient_norm_divisor = torch.FloatTensor([180., 180., 180., 2000.])
        # Note: saw issues where it gave preference to accuracy in x
        # with this unequal normalization
        # self.norm_divisor = torch.FloatTensor([12464., 500.])
        self.norm_divisor = 150.

        if target_transform is not None:
            warnings.warn("Using non-default target transform removes label normalization. Remember to normalize in your provided transformation")
            self.target_transform = target_transform
        else:
            self.target_transform = lambda x: x / self.norm_divisor

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, -1].split("/")[-1])
        if self.load_png:
            image = read_image(img_path)
        else:
            image = torch.load(img_path)
        label = self.img_labels.iloc[idx, 4:6].to_numpy(dtype=np.float32)
        # label is y (crosstrack error) and h (height)
        # y is already an error
        # we want to turn h into an error
        x = self.img_labels.iloc[idx, 3]
        label[1] = self.glideslope_controller.get_glideslope_height_at(x) - label[1]
        label = torch.from_numpy(label)
        orient_alt = np.zeros((4,), dtype=np.float32)
        orient_alt[:-1] = self.img_labels.iloc[idx, :3].to_numpy(dtype=np.float32)
        orient_alt[-1] = self.img_labels.iloc[idx, 5]
        orient_alt /= self.orient_norm_divisor
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        #print("I am here")
        return image, orient_alt, label


if __name__ == "__main__":
    dataset = AutolandImageDataset("./data-small/states.csv", "./data-small/images")
    img, orient_alt, label = dataset[0]

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    dataloaders = {
        "train": torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=1),
        "val":  torch.utils.data.DataLoader(test_dataset, batch_size=5, shuffle=True, num_workers=1)
    }

    assert train_size >= 5

    for img, orient_alt, label in dataloaders["train"]:
        print("img.shape", img.shape)
        print("orient_alt.shape", orient_alt.shape)
        print("label.shape", label.shape)
        break
    print("done")