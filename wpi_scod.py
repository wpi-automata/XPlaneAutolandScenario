# Source: https://github.com/StanfordASL/scod-module/blob/main/demos/1d_regression.ipynb

#%load_ext autoreload
#%autoreload 2

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
import math

from matplotlib import pyplot as plt
from src.xplane_autoland.vision.perception import AutolandPerceptionModel
from src.xplane_autoland.vision.xplane_data import AutolandImageDataset

# img_1 = imgs[i, :, :, :]
# if False not in torch.eq(img_1, img):
#     print("imgs the same")
# alt_1 = alts[i, :]
# if False not in torch.eq(alt_1, alt):
#     print("alts the same")

# def record_signal (imgs, alts, detector):
#     signals = []
#     for i in range(len(imgs)): # TODO: find a better way to do this. pairs and zip didn't work
#         img = imgs[i]
#         alt = alts[i]
#         signal = detector(img, alt)
#         signals.append(signal)
#     return signals

## Load dist layer - I don't know which one to choose yet
# dist_layer = 6 # TODO: Select dist layer
# dist_layer = scod.distributions.NormalMeanParamLayer() 
# OR dist_layer = scod.distributions.NormalMeanDiagVarParamLayer()

## Attach dist layer to model and train it with our dataset
dist_layer = scod.distributions.NormalMeanParamLayer()



# torchscript = torch.jit.load("/home/achadbo/XPlaneAutolandScenario/models/scod/2024-6-18/scod_model_200_80percent.pt")
# torchscript.eval()
scod_model = torch.load("models/scod/2024-7-29/scod_model_safeset_80percent.pt")
scod_model.eval()
# scod_model = scod.SCOD(model)




data_dir = "/media/storage_drive/ULI Datasets/dataWPI_1500_50_10/unsafe"
full_dataset = AutolandImageDataset(f"{data_dir}/states.csv", f"{data_dir}/images")
train_size = int(0.0 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
# might need to experiment with different values for "batch_size" and "lr"
batch = 600
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch, shuffle=False)
indices = val_dataset.indices
print(type(train_dataset.dataset.img_labels))

ood_signals = []
ood_detector = scod.OodDetector(scod_model, dist_layer)
print("Size of dataset: " + str(val_size))
print("Batch size: " + str(batch))
total_batches = math.ceil(val_size / batch)
print("Total number of batches: " + str(total_batches))
for i in range(total_batches):
    rwy_imgs, orient_alts, labels  = next(iter(val_dataloader))
    ood_signal = ood_detector(rwy_imgs, orient_alts)
    temp = ood_signal.detach().numpy()
    ood_signals = numpy.append(ood_signals, temp)
    print("Batches completed: " + str(i + 1))
print("Signals calculated")



statepath = Path(f"/home/achadbo/XPlaneAutolandScenario/Subsets/scod_unsafe")
if not statepath.is_file():
    with open(str(statepath), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['phi', 'theta', 'psi', 'x', 'y', 'h', 'img source', 'signal'])

f = open(str(statepath), 'a')
writer = csv.writer(f)
with open(f"{data_dir}/states.csv") as states_file:
    csv_reader = csv.reader(states_file, delimiter=',')
    i = 0
    for index in indices:
        states_file.seek(0)
        data = next(itertools.islice(csv.reader(states_file), index, None))
        signal = float(ood_signals[i])
        data.append(signal)
        writer.writerow(data)
        i += 1
        





