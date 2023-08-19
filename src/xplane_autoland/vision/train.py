# Adapted from example: https://pytorch.org/vision/stable/models.html
# and this tutorial: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html


import argparse
from datetime import date
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import random
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from xplane_autoland.vision.perception import AutolandPerceptionModel
from xplane_autoland.vision.xplane_data import AutolandImageDataset

cudnn.benchmark = True
plt.ion()   # interactive mode


def set_all_seeds(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True


def get_logger(save_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger()
    file_handler = logging.FileHandler(f"{save_dir}/output.log")
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def train_model(model, criterion, optimizer, dataloaders, dataset_sizes,
                logger, save_dir, num_epochs=25, scheduler=None, report_interval=100):

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)

    # saves to ./runs/ by default
    writer = SummaryWriter(log_dir=f"{save_dir}/tensorboard_dir/")

    logger.info(f"Starting training on {device}")

    lr = optimizer.param_groups[0]["lr"]
    logger.info(f"LR: {lr}")
    writer.add_scalar(f"LR", lr, -1)

    # Create a temporary directory to save training checkpoints
    best_model_params_path = f'{save_dir}/best_model_params.pt'

    torch.save(model.state_dict(), best_model_params_path)
    best_loss = float("inf")

    since = time.time()
    for epoch in range(num_epochs):
        lr = optimizer.param_groups[0]["lr"]
        if lr < 1e-9:
            logger.info(f"Learning Stopped: learning rate has been dropped too low by scheduler (lr={lr})")
            break

        logger.info(f'Epoch {epoch}/{num_epochs - 1}')
        logger.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for bidx, (rwy_img, orient_alt, labels) in enumerate(dataloaders[phase]):
                rwy_img = rwy_img.to(device)
                orient_alt = orient_alt.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(rwy_img, orient_alt)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * rwy_img.size(0)

                if bidx and bidx % report_interval == 0:
                    loss_report = running_loss / (bidx * dataloaders[phase].batch_size)
                    logger.info(f'Mean Running Loss/{phase}: {loss_report:.4f}')
                    writer.add_scalar(f"Mean Running Loss/{phase}", loss_report, bidx/report_interval + epoch * math.ceil(dataset_sizes[phase]/report_interval))

            epoch_loss = running_loss / dataset_sizes[phase]

            logger.info(f'#### Epoch Loss/{phase}: {epoch_loss:.4f} ####')
            writer.add_scalar(f"Epoch Loss/{phase}", epoch_loss, epoch)

            if phase == 'val' and scheduler is not None:
                scheduler.step(epoch_loss)
                lr = optimizer.param_groups[0]["lr"]
                logger.info(f"LR: {lr}")
                writer.add_scalar(f"LR", lr, epoch)

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), best_model_params_path)

    time_elapsed = time.time() - since
    logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Best val loss: {best_loss:4f}')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a state estimator for the autoland scenario")
    parser.add_argument("--seed", type=int, default=0, help="The random seed")
    parser.add_argument("--save-dir", help="The directory to save everything in", default=None)
    parser.add_argument("--data-dir", help="The directory with image data should contain a states.csv and images directory", default=None)
    parser.add_argument("--resnet-version", help="Choose which resnet to use", default="50", choices=["18", "50"])
    parser.add_argument("--fixed-lr", action="store_true", help="Don't vary the learning rate")
    parser.add_argument("--num-epochs", type=int, help="The number of epochs to run", default=300)
    parser.add_argument("--unfreeze", action="store_true", help="Don't freeze the backbone")
    parser.add_argument("--start-model", help="The model to load in first", default=None)
    args = parser.parse_args()

    save_dir = args.save_dir
    if save_dir is None:
        today = date.today()
        save_dir = f"./models/{today.year}-{today.month}-{today.day}"

    save_dir = Path(save_dir)
    if save_dir.exists():
        raise ValueError(f"Save directory already exists: {save_dir}\n" 
                         f"Please specify a different directory or move the previous one. Will not overwrite.")
    else:
        save_dir.mkdir()

    logger = get_logger(save_dir)
    logger.info(f"Full command-line arguments: {sys.argv}")
    logger.info(f"Setting seed to: {args.seed}")
    set_all_seeds(args.seed)

    logger.info(f"Saving files to: {save_dir}")
    logger.info(f"Using ResNet{args.resnet_version}")

    # Step 1: Initialize model with the best available weights
    model = AutolandPerceptionModel(resnet_version=args.resnet_version,
                                    freeze=not args.unfreeze)
    if args.start_model is not None:
        logger.info(f"Loading model from: {args.start_model}")
        model.load(args.start_model)

    # Collect the dataset
    data_dir = args.data_dir
    if data_dir is None:
        data_dir="./data-initial"
    logger.info(f"Using data from: {data_dir}")
    full_dataset = AutolandImageDataset(f"{data_dir}/processed-states.csv", f"{data_dir}/images", transform=model.preprocess)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    logger.info(f"Dataset Train size: {train_size}, Val size: {val_size}")
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    dataloaders = {
        "train": torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4),
        "val":  torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=4)
    }
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

    criterion = nn.MSELoss(reduction="mean")
    lr = 1e-6 if args.fixed_lr else 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = None if args.fixed_lr else lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    train_model(model, criterion, optimizer, dataloaders, dataset_sizes, logger, save_dir, 
                num_epochs=args.num_epochs, scheduler=scheduler)
