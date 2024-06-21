import argparse
import sys
import csv
import time
import torch 
import math
from datetime import date
from pathlib import Path

# Order of bounds: 'phi', 'theta', 'psi', 'x', 'y', 'h'
rads = math.radians(10)
tan = math.tan(rads) #tangent of 10 degrees (in radians)
runway_dist = 17000

def get_safe(save_dir, data_dir):
    statepath = Path(f"{save_dir}/safe_set.csv")
    if not statepath.is_file():
        with open(str(statepath), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['phi', 'theta', 'psi', 'x', 'y', 'h', 'img source'])

    f = open(str(statepath), 'a')
    writer = csv.writer(f)

    with open(f"{data_dir}/states.csv") as states_file: #Another line to make sure consistent 
        csv_reader = csv.reader(states_file, delimiter=',')
        h_initial = float(next(csv_reader)[5])
        slope = h_initial / runway_dist
        for row in csv_reader:
            safe = True
            x = float(row[3]) #Distance from runway
            r = tan * math.sqrt(pow(x, 2) + pow((x * slope), 2)) #Radius of safe set
            upper_bounds = [r, r, (x * slope) + r]
            lower_bounds = [r * -1, r * -1, (x * slope) - r]
            for i in range(3):
                item = float(row[3 + i]) #Skip phi, theta, and psi
                print(i)
                if (item > upper_bounds[i]) or (item < lower_bounds[i]): #if item is outside bounds, reject
                    safe = False
                    break
            if safe:
                writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get the subset of NN data that is considered 'safe'")
    parser.add_argument("--data-dir", help="The path to the dataset", default="/media/storage_drive/ULI Datasets/OOD Data/dataWPI_50-10")
    args = parser.parse_args()

    this_dir = Path(__file__).parent
    repo_dir = this_dir.resolve()

    today = date.today()
    save_dir = Path(f"{repo_dir}/safe_sets/{today.year}-{today.month}-{today.day}")
    if not save_dir.exists():
        save_dir.mkdir()

    data_dir = args.data_dir

    get_safe(save_dir, data_dir)