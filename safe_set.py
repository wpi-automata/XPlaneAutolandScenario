import argparse
import sys
import csv
import time
import torch 
from datetime import date
from pathlib import Path

# Order of bounds: 'phi', 'theta', 'psi', 'x', 'y', 'h'
upper_bounds = [0, 0, 0, 0, 0, 0]
lower_bounds = [0, 0, 0, 0, 0, 0]

def get_safe(save_dir, data_dir):
    statepath = Path(f"{save_dir}/safe_set.csv")
    if not statepath.is_file():
        with open(str(statepath), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['phi', 'theta', 'psi', 'x', 'y', 'h'])

    f = open(str(statepath), 'a')
    writer = csv.writer(f)

    with open(f"{data_dir}/states.csv") as states_file: #Another line to make sure consistent 
        csv_reader = csv.reader(states_file, delimiter=',')
        for row in csv_reader:
            safe = True
            for i in range(len(row)):
                item = row[i]
                if (item > upper_bounds[i]) and (item < lower_bounds[i]): #if item is outside bounds, reject
                    safe = False
                    break
            if safe:
                writer.writerow(row)
                    

            #return the est_state values in the x and y and save them to a csv file 
            #writer.writerow([y_err, h_err, h_err_true])

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Get the subset of NN data that is considered 'safe'")
    # parser.add_argument("--data-dir", help="The path to the dataset", default="/media/storage_drive/ULI Datasets/OOD Data/dataWPI_50-10")
    # args = parser.parse_args()

    this_dir = Path(__file__).parent
    repo_dir = this_dir.resolve()

    today = date.today()
    save_dir = Path(f"{repo_dir}/safe_sets/{today.year}-{today.month}-{today.day}")
    if not save_dir.exists():
        save_dir.mkdir()

    # data_dir = args.data_dir
    # if not args.data_dir:
    #     data_dir="" #TODO: Find a default for this
    data_dir = "/media/storage_drive/ULI Datasets/OOD Data/dataWPI_50-10"

    get_safe(save_dir, data_dir)