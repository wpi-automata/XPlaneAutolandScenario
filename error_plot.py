import matplotlib.pyplot as plt
from pathlib import Path
import csv
from datetime import date
import pandas as pd
import argparse

def combine_data(save_dir, states, gen_states):

    statepath = Path(f"{save_dir}/combined_states.csv")
    if not statepath.is_file():
        with open(str(statepath), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['error_h', 'error_y','phi', 'theta', 'psi', 'x', 'y', 'h'])

    f = open(str(statepath), 'a')
    writer = csv.writer(f)

    #Iterate through csv's: 
    with open(f"{states}") as states_file: #make sure all paths for dataset and generated_states are consistent 
        with open(f"{gen_states}") as gen_states_file:
            csv_reader1 = csv.reader(states_file, delimiter=',')
            csv_reader2 = csv.reader(gen_states_file, delimiter=',')
            line_count = 0
            for row1, row2 in zip(csv_reader1, csv_reader2):
                if line_count == 0:
                    line_count += 1
                else:
                    #Get errors of the NN 
                    error_h = float(row2[1]) - float(row2[2])
                    error_y = float(row1[4]) - float(row2[0]) #y - y_err = error of NN = ehat - e = e' in crosstrack (Note: put on y axis)
                    #Record everything else (looking back could also add in the predicted values to this set- took care of this in MATLAB script but kind of annoying)
                    phi = row1[0]
                    theta = row1[1]
                    psi = row1[2]
                    x = row1[3]
                    y = row1[4]
                    h = row1[5]
                    writer.writerow([error_h, error_y, phi, theta, psi, x, y, h])
                    line_count += 1
            print(f'Processed {line_count} lines.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run pre-generated data through the state estimator")
    parser.add_argument("--states-file", help="The states.csv file of dataset", default=None)
    parser.add_argument("--gen-states", help="The generated_states.csv generated from error_generator.py", default=None)
    parser.add_argument("--offset", help="The number of meters in distance offset", default="150")
    args = parser.parse_args()

    this_dir = Path(__file__).parent
    repo_dir = this_dir.resolve()
    states_file = args.states_file
    if not states_file:
        states_file = "{repo_dir}/dataWPI_12464/states.csv" #Backup if parameter not given, will likely error. Change later
    
    gen_states = args.gen_states
    if not gen_states:
        gen_states = "{repo_dir}/errors/2024-4-23/150/generated_states.csv" #Backup if parameter not given, will likely error. Change later

    today = date.today()
    save_dir = Path(f"{repo_dir}/plots/{today.year}-{today.month}-{today.day}/{args.offset}") #Change based on needs 
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    
    combine_data(save_dir, states_file, gen_states)