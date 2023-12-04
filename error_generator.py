import argparse
import csv
from datetime import date
from pathlib import Path
import torch 

from xplane_autoland.xplane_connect.vision_driver import XPlaneVisionDriver
from xplane_autoland.vision.perception import AutolandPerceptionModel
from xplane_autoland.vision.perception import AutolandPerceptionModel
from xplane_autoland.vision.xplane_data import AutolandImageDataset

def generate_data(plane, dataloader, save_dir):

    print(f"Attempting to save data...")
    statepath = Path(f"{save_dir}/generated_states.csv")
    if not statepath.is_file():
        with open(str(statepath), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])

    f = open(str(statepath), 'a')
    writer = csv.writer(f)

    #Iterate over the dataset
    print(f"Iterating over the data")
    for rwy_img, orient_alt, _ in dataloader:
   
        #iterate over the data and pass it into est_state
        x, y = plane.est_pos_state(rwy_img, orient_alt)
        #return the est_state values in the x and y and save them to a csv file 
        writer.writerow([x, y])
        
    print(f"Done")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Run pre-generated data through the state estimator")
    parser.add_argument("--save-dir", help="The directory to save everything in", default=None)
    parser.add_argument("--data-dir", help="The directory with image data should contain a states.csv and images directory", default=None)
    parser.add_argument("--resnet-version", help="Choose which resnet to use", default="50", choices=["18", "50"])
    parser.add_argument("--model", help="The path to model parameters (*.pt) for a vision network. Note must have XPlane fullscreen for screenshots", default=None)
    args = parser.parse_args()

    this_dir = Path(__file__).parent
    repo_dir = this_dir.resolve()

    today = date.today()
    save_dir = Path(f"{repo_dir}/errors/{today.year}-{today.month}-{today.day}")
    if not save_dir.exists():
        save_dir.mkdir()

    model = AutolandPerceptionModel(resnet_version="50")
    model.load("/home/colette/XPlaneAutolandScenario_ColetteCopy/models/vision/2023-8-10/best_model_params.pt")
    model.eval()
    plane = XPlaneVisionDriver(model)
    
    #1: Collect the Dataset 
    data_dir = args.data_dir
    if data_dir is None:
        data_dir=str(repo_dir/"data")
    dataset = AutolandImageDataset(f"{data_dir}/processed-states.csv", f"{data_dir}/images", transform=model.preprocess)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    #2: Generate Data
    generate_data(plane, dataloader,save_dir)
