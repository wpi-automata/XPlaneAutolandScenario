import matplotlib.pyplot as plt
from pathlib import Path
import csv
from datetime import date
import pandas as pd

def combine_data(save_dir, repo_dir):

    statepath = Path(f"{save_dir}/combined_states.csv")
    if not statepath.is_file():
        with open(str(statepath), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['error_x', 'error_y','phi', 'theta', 'psi', 'x', 'y', 'h'])

    f = open(str(statepath), 'a')
    writer = csv.writer(f)

    with open(f"{repo_dir}/data/processed-states.csv") as states_file:
        with open(f"{repo_dir}/errors/2023-11-27/generated_states.csv") as gen_states_file:
            csv_reader1 = csv.reader(states_file, delimiter=',')
            csv_reader2 = csv.reader(gen_states_file, delimiter=',')
            line_count = 0
            for row1, row2 in zip(csv_reader1, csv_reader2):
                if line_count == 0:
                    line_count += 1
                else:
                    error_x = float(row1[3]) - float(row2[0])
                    error_y = float(row1[4]) - float(row2[1])
                    phi = row1[0]
                    theta = row1[1]
                    psi = row1[2]
                    x = row1[3]
                    y = row1[4]
                    h = row1[5]
                    writer.writerow([error_x, error_y, phi, theta, psi, x, y, h])
                    line_count += 1
            print(f'Processed {line_count} lines.')

def plot_data(repo_dir):
    #Eventually should save plots as files 
    # TODO Change plots to scatter and plot 3D against distance to runway(x)

    #X_Error vs X
    plt.figure(1)
    plt.rcParams["figure.autolayout"] = True
    columns = ["error_x", "error_y","phi", "theta", "psi", "x", "y", "h"]
    df = pd.read_csv(f"{repo_dir}/plots/2023-11-27/combined_states.csv", usecols=columns)
    plt.plot(df.error_x, df.x)
    plt.xlabel("NN Error X")
    plt.ylabel("X")

    #X_Error vs Y
    plt.figure(2)
    plt.rcParams["figure.autolayout"] = True
    columns = ["error_x", "error_y","phi", "theta", "psi", "x", "y", "h"]
    df = pd.read_csv(f"{repo_dir}/plots/2023-11-27/combined_states.csv", usecols=columns)
    plt.plot(df.error_x, df.y)
    plt.xlabel("NN Error X")
    plt.ylabel("Y")
 

    #X_Error vs H
    plt.figure("3")
    plt.rcParams["figure.autolayout"] = True
    columns = ["error_x", "error_y","phi", "theta", "psi", "x", "y", "h"]
    df = pd.read_csv(f"{repo_dir}/plots/2023-11-27/combined_states.csv", usecols=columns)
    plt.plot(df.error_x, df.h)
    plt.xlabel("NN Error X")
    plt.ylabel("H")


    #X_Error vs Phi
    plt.figure("4")
    plt.rcParams["figure.autolayout"] = True
    columns = ["error_x", "error_y","phi", "theta", "psi", "x", "y", "h"]
    df = pd.read_csv(f"{repo_dir}/plots/2023-11-27/combined_states.csv", usecols=columns)
    plt.plot(df.error_x, df.phi)
    plt.xlabel("NN Error X")
    plt.ylabel("Phi")


    #X_Error vs Theta
    plt.figure("5")
    plt.rcParams["figure.autolayout"] = True
    columns = ["error_x", "error_y","phi", "theta", "psi", "x", "y", "h"]
    df = pd.read_csv(f"{repo_dir}/plots/2023-11-27/combined_states.csv", usecols=columns)
    plt.plot(df.error_x, df.theta)
    plt.xlabel("NN Error X")
    plt.ylabel("Theta")


    #X_Error vs Psi
    plt.figure("6")
    plt.rcParams["figure.autolayout"] = True
    columns = ["error_x", "error_y","phi", "theta", "psi", "x", "y", "h"]
    df = pd.read_csv(f"{repo_dir}/plots/2023-11-27/combined_states.csv", usecols=columns)
    plt.plot(df.error_x, df.psi)
    plt.xlabel("NN Error X")
    plt.ylabel("Psi")
    

    #Y_Error vs X
    plt.figure("7")
    plt.rcParams["figure.autolayout"] = True
    columns = ["error_x", "error_y","phi", "theta", "psi", "x", "y", "h"]
    df = pd.read_csv(f"{repo_dir}/plots/2023-11-27/combined_states.csv", usecols=columns)
    plt.plot(df.error_y, df.x)
    plt.xlabel("NN Error Y")
    plt.ylabel("X")
  

    #Y_Error vs Y
    plt.figure("8")
    plt.rcParams["figure.autolayout"] = True
    columns = ["error_x", "error_y","phi", "theta", "psi", "x", "y", "h"]
    df = pd.read_csv(f"{repo_dir}/plots/2023-11-27/combined_states.csv", usecols=columns)
    plt.plot(df.error_y, df.y)
    plt.xlabel("NN Error Y")
    plt.ylabel("Y")


    #Y_Error vs H
    plt.figure("9")
    plt.rcParams["figure.autolayout"] = True
    columns = ["error_x", "error_y","phi", "theta", "psi", "x", "y", "h"]
    df = pd.read_csv(f"{repo_dir}/plots/2023-11-27/combined_states.csv", usecols=columns)
    plt.plot(df.error_y, df.h)
    plt.xlabel("NN Error Y")
    plt.ylabel("H")
  

    #Y_Error vs Phi
    plt.figure("10")
    plt.rcParams["figure.autolayout"] = True
    columns = ["error_x", "error_y","phi", "theta", "psi", "x", "y", "h"]
    df = pd.read_csv(f"{repo_dir}/plots/2023-11-27/combined_states.csv", usecols=columns)
    plt.plot(df.error_y, df.phi)
    plt.xlabel("NN Error Y")
    plt.ylabel("Phi")
    plt.xlabel("NN Error Y")
    plt.ylabel("Theta")
   


    #Y_Error vs Psi
    plt.figure("12")
    plt.rcParams["figure.autolayout"] = True
    columns = ["error_x", "error_y","phi", "theta", "psi", "x", "y", "h"]
    df = pd.read_csv(f"{repo_dir}/plots/2023-11-27/combined_states.csv", usecols=columns)
    plt.plot(df.error_y, df.psi)
    plt.xlabel("NN Error Y")
    plt.ylabel("Psi")


    plt.show()



if __name__ == '__main__':
    this_dir = Path(__file__).parent
    repo_dir = this_dir.resolve()

    today = date.today()
    save_dir = Path(f"{repo_dir}/plots/{today.year}-{today.month}-{today.day}")
    if not save_dir.exists():
        save_dir.mkdir()
    
    combine_data(save_dir, repo_dir)
    plot_data(repo_dir)