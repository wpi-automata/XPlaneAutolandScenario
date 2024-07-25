import argparse
import sys
import csv
import time
import torch 
import math
import re
import shutil
from datetime import date
from pathlib import Path

# Order of bounds: 'phi', 'theta', 'psi', 'x', 'y', 'h'
rads = math.radians(10)
tan = math.tan(rads) #tangent of 10 degrees (in radians)
runway_dist = 17000
runway_elev = 361 #height of runway above sea level
safe_slope = math.tan(math.radians(3.5))

def prep_csv(save_dir, name):
    statepath = Path(f"{save_dir}/{name}")
    if not statepath.is_file():
        with open(str(statepath), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['phi', 'theta', 'psi', 'x', 'y', 'h', 'img source'])
    f = open(str(statepath), 'a')
    writer = csv.writer(f)
    return writer

def get_safe(save_dir, data_dir, unsafe_dir):
    safe_img_dir = f"{save_dir}/images"
    unsafe_img_dir = f"{unsafe_dir}/images"
    writer = prep_csv(save_dir, "states.csv") #All data that falls within the safe cone is saved here
    unsafe_writer = prep_csv(unsafe_dir, "states.csv")

    with open(f"{data_dir}/states.csv") as states_file:
        csv_reader = csv.reader(states_file, delimiter=',')
        next(csv_reader) #Skip label/header line
        slope = safe_slope
        rows = 0
        for row in csv_reader:
            safe = True
            img_name = re.findall("image_\S+", row[6])
            img_path = f"{data_dir}/images/{img_name[0]}" #Grab the image path to copy over to save_dir
            x = float(row[3]) #Distance from runway
            r = tan * math.sqrt(pow(x, 2) + pow((x * slope), 2)) #Radius of safe set
            upper_bounds = [r, (x * slope) + r] #Upper bound of y and h
            lower_bounds = [r * -1, (x * slope) - r] #Lower bound of y and h
            for i in range(2): #Iterate through y and h
                item = float(row[4 + i]) #Skip phi, theta, and psi.
                if(i == 1): item -= runway_elev 
                if (item > upper_bounds[i]) or (item < lower_bounds[i]): #if item is outside bounds, reject
                    rows += 1
                    safe = False
                    unsafe_writer.writerow(row) #Add item to 'unsafe' csv
                    shutil.copy(img_path, unsafe_img_dir)
                    break

            if safe:
                writer.writerow(row)
                shutil.copy(img_path, safe_img_dir)
                
        print("Rows removed from original file: " + str(rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get the subset of NN data that is considered 'safe'")
    parser.add_argument("--data-dir", help="The path to the dataset", default="/media/storage_drive/ULI Datasets/dataWPI_1500_50_10")
    args = parser.parse_args()

    this_dir = Path(__file__).parent
    repo_dir = this_dir.resolve()
    data_dir = args.data_dir

    today = date.today()
    save_dir = Path(f"{data_dir}/safe")
    if not save_dir.exists():
        save_dir.mkdir()

    unsafe_dir = Path(f"{data_dir}/unsafe")
    if not unsafe_dir.exists():
        unsafe_dir.mkdir()

    

    get_safe(save_dir, data_dir, unsafe_dir)