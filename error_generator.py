import argparse
import csv
from datetime import date
from pathlib import Path
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

from src.xplane_autoland.xplane_connect.vision_driver import XPlaneVisionDriver
from src.xplane_autoland.vision.perception import AutolandPerceptionModel
from src.xplane_autoland.vision.perception import AutolandPerceptionModel
from src.xplane_autoland.vision.xplane_data import AutolandImageDataset
from src.xplane_autoland.controllers.glideslope_controller import GlideSlopeController


def generate_data(plane, dataloader, save_dir):
    gsc = GlideSlopeController(gamma=3)
    print(f"Attempting to save data...")
    statepath = Path(f"{save_dir}/generated_states.csv")
    if not statepath.is_file():
        with open(str(statepath), "w") as f:
            writer = csv.writer(f)
            writer.writerow(["y_err", "h_err_NN", "h_err_true"])

    f = open(str(statepath), "a")
    writer = csv.writer(f)

    # Iterate over the dataset
    # processed states values for reference: phi,theta,psi,x,y,h,imagename- you'll need to enter the states.csv file and make sure the strings are deleted
    print(f"Iterating over the data")
    with open(
        f"/media/storage_drive/ULI Datasets/OOD Data/dataWPI_50-10/states.csv"
    ) as states_file:  # Another line to make sure consistent
        csv_reader = csv.reader(states_file, delimiter=",")
        for row, (rwy_img, orient_alt, _) in zip(csv_reader, dataloader):

            # iterate over the data and pass it into est_state
            y_err, h_err = plane.est_pos_state(rwy_img, orient_alt)
            h_err_true = gsc.get_glideslope_height_at(float(row[3])) - float(
                row[5]
            )  # get the true height error for reference

            # return the est_state values in the x and y and save them to a csv file
            writer.writerow([y_err, h_err, h_err_true])

    print(f"Done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run pre-generated data through the state estimator"
    )
    parser.add_argument(
        "--save-dir", help="The directory to save everything in", default=None
    )
    parser.add_argument(
        "--data-dir",
        help="The directory with image data should contain a states.csv and images directory",
        default=None,
    )
    parser.add_argument(
        "--resnet-version",
        help="Choose which resnet to use",
        default="50",
        choices=["18", "50"],
    )
    parser.add_argument(
        "--model",
        help="The path to model parameters (*.pt) for a vision network. Note must have XPlane fullscreen for screenshots",
        default=None,
    )
    args = parser.parse_args()

    this_dir = Path(__file__).parent
    repo_dir = this_dir.resolve()

    today = date.today()
    save_dir = None
    if args.save_dir:
        save_dir = Path(f"{args.save_dir}")
    else:
        save_dir = Path(
            f"{repo_dir}/errors/{today.year}-{today.month}-{today.day}/450"
        )  # Can also rewrite to use arg parsing

    if not save_dir.exists():
        save_dir.mkdir()

    model = AutolandPerceptionModel(resnet_version=args.resnet_version)
    if args.model:
        model.load(args.model)
    else:
        model.load(
            "models/2024-4-1/best_model_params.pt"
        )  # Load in model manually (can also re-write to use the --model param if preferred)
    model.eval()
    plane = XPlaneVisionDriver(model)

    # 1: Collect the Dataset
    data_dir = args.data_dir
    if not args.data_dir:
        data_dir = "/media/storage_drive/ULI Datasets/OOD Data/dataWPI_50-10"  # Make sure this is the right one
    dataset = AutolandImageDataset(f"{data_dir}/states.csv", f"{data_dir}/images")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1
    )

    # 2: Generate Data
    generate_data(plane, dataloader, save_dir)
