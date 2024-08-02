import scod
from scod.distributions import Normal
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import trange
import csv
from pathlib import Path
import itertools
import numpy

from matplotlib import pyplot as plt
from src.xplane_autoland.vision.perception import AutolandPerceptionModel
from src.xplane_autoland.vision.xplane_data import AutolandImageDataset



model = AutolandPerceptionModel()
model.load("/home/achadbo/Desktop/safe_cone_model.pt")
dist_layer = scod.distributions.NormalMeanDiagVarParamLayer()

scod_model = scod.SCOD(model)

data_dir = "/media/storage_drive/ULI Datasets/dataWPI_1500_50_10/safe"
full_dataset = AutolandImageDataset(f"{data_dir}/states.csv", f"{data_dir}/images")
train_size = int(0.80 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
# might need to experiment with different values for "batch_size" and "lr"
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

scod_model.process_dataset(train_dataset, dist_layer)
torch.save(scod_model, "/home/achadbo/XPlaneAutolandScenario/models/scod/2024-7-29/scod_safeset_normmean_80percent.pt")
# scod_torchscript = torch.jit.script(scod_model)
# scod_torchscript.save("/home/achadbo/XPlaneAutolandScenario/models/scod/2024-6-24/scod_model_17000_80percent.pt")