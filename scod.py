# Source: https://github.com/StanfordASL/scod-module/blob/main/demos/1d_regression.ipynb

#%load_ext autoreload
#%autoreload 2

import scod
from scod.distributions import Normal
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import trange

from matplotlib import pyplot as plt
from src.xplane_autoland.vision.perception import AutolandPerceptionModel
from src.xplane_autoland.vision.xplane_data import AutolandImageDataset


# THIS SCRIPT IS NOT MEANT TO BE RUN AS IS! - PLEASE MAKE ADJUSTMENTS BELOW TO MAKE IT COMPATIBLE FOR OUR CASE BY COMPLETING "TODO" TAGS


## Load dist layer - I don't know which one to choose yet
# dist_layer = 6 # TODO: Select dist layer
# dist_layer = scod.distributions.NormalMeanParamLayer() 
# OR dist_layer = scod.distributions.NormalMeanDiagVarParamLayer()

## Attach dist layer to model and train it with our dataset
model = AutolandPerceptionModel()
model.load("/home/achadbo/XPlaneAutolandScenario/models/2024-4-1/best_model_params.pt")
dist_layer = scod.distributions.NormalMeanParamLayer()

scod_model = scod.SCOD(model)

data_dir = "/media/storage_drive/ULI Datasets/OOD Data/dataWPI_50-10"
full_dataset = AutolandImageDataset(f"{data_dir}/states.csv", f"{data_dir}/images")
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
# might need to experiment with different values for "batch_size" and "lr"
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
rwy_img, orient_alt, labels  = next(iter(train_dataloader))
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


scod_model.process_dataset(train_dataset, dist_layer)

# ood_detector = scod.OodDetector(scod_model, dist_layer)
# print(rwy_img.size())
# print(orient_alt.size())
# ood_signal = ood_detector(rwy_img, orient_alt)

