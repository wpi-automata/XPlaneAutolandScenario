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
            writer.writerow(['error_h', 'error_y','phi', 'theta', 'psi', 'x', 'y', 'h'])

    f = open(str(statepath), 'a')
    writer = csv.writer(f)

    with open(f"{repo_dir}/data/processed-states.csv") as states_file: #This needs to change 
        with open(f"{repo_dir}/errors/2024-3-3/generated_states.csv") as gen_states_file:
            csv_reader1 = csv.reader(states_file, delimiter=',')
            csv_reader2 = csv.reader(gen_states_file, delimiter=',')
            line_count = 0
            for row1, row2 in zip(csv_reader1, csv_reader2):
                if line_count == 0:
                    line_count += 1
                else:
                    error_h = float(row2[1]) - float(row2[2])
                    error_y = float(row1[4]) - float(row2[0])
                    phi = row1[0]
                    theta = row1[1]
                    psi = row1[2]
                    x = row1[3]
                    y = row1[4]
                    h = row1[5]
                    writer.writerow([error_h, error_y, phi, theta, psi, x, y, h])
                    line_count += 1
            print(f'Processed {line_count} lines.')

def plot_data(repo_dir): #This has not been updated to new model since I switched over to MATLAB for plotting 
    #Eventually should save plots as files 
    # TODO Change plots to scatter and plot 3D against distance to runway(x)

    #X is horizontal distance from plane to runway
    #Y is cross-track error 
    #All 3D Plots are plotted with NN Error on the Z Axis and Horizontal Distance (X) on the X-axis

    #X_Error (Horizontal Distance Error) vs X (Horizontal Distance), 2D plot
    plt.figure("Figure 1")
    plt.rcParams["figure.autolayout"] = True
    # ax = fig1.add_subplot(projection = "3d")

    columns = ["error_h", "error_y","phi", "theta", "psi", "x", "y", "h"]
    df = pd.read_csv(f"{repo_dir}/plots/2024-1-15/combined_states.csv", skiprows=lambda x: x%10, usecols=columns)

    plt.scatter(df.error_h, df.x)
    plt.title("Height Error (NN) vs Height (True h)")
    plt.ylabel("Height(True h)")
    plt.xlabel("Height Error (Predicted H)")

    #3D PLOTS =========================================================================================
    #X_Error (Horizontal Distance Error) vs Y (Crosstrack Error)
    fig2 = plt.figure("Figure 2")
    plt.rcParams["figure.autolayout"] = True
    ax = fig2.add_subplot(projection = "3d")
    ax.scatter( df.x, df.y, df.error_x)
    plt.title("Horizontal Distance Error (NN) vs Crosstrack vs Horizontal Distance")
    ax.set_xlabel("Horizontal Distance (True X)")
    ax.set_ylabel("Crosstrack Error (Y)")
    ax.set_zlabel("Horizontal Distance Error (Predicted X)")

    # #X_Error (Horizontal Distance Error) vs Height 
    fig3 = plt.figure("Figure 3")
    plt.rcParams["figure.autolayout"] = True
    ax = fig3.add_subplot(projection = "3d")
    ax.scatter( df.x, df.h, df.error_x)
    plt.title("Horizontal Distance Error (NN) vs Height vs Horizontal Distance")
    ax.set_xlabel("Horizontal Distance (True X)")
    ax.set_ylabel("Height (y)")
    ax.set_zlabel("Horizontal Distance Error (Predicted X)")


    # #X_Error (Horizontal Distance Error) vs Phi (Roll)
    fig4 = plt.figure("Figure 4")
    plt.rcParams["figure.autolayout"] = True
    ax = fig4.add_subplot(projection = "3d")
    ax.scatter( df.x, df.phi, df.error_x)
    plt.title("Horizontal Distance Error (NN) vs Roll (Phi) vs Horizontal Distance")
    ax.set_xlabel("Horizontal Distance (True X)")
    ax.set_ylabel("Roll (Phi)")
    ax.set_zlabel("Horizontal Distance Error (Predicted X)")



    # X_Error (Horizontal Distance Error) vs Theta (Pitch)
    fig5 = plt.figure("Figure 5")
    plt.rcParams["figure.autolayout"] = True
    ax = fig5.add_subplot(projection = "3d")
    ax.scatter( df.x, df.theta, df.error_x)
    plt.title("Horizontal Distance Error (NN) vs Pitch (Theta) vs Horizontal Distance")
    ax.set_xlabel("Horizontal Distance (True X)")
    ax.set_ylabel("Theta")
    ax.set_zlabel("Horizontal Distance Error (Predicted X)")


    # #X_Error (Horizontal Distance Error) vs Psi (yaw)
    fig6 = plt.figure("Figure 6")
    plt.rcParams["figure.autolayout"] = True
    ax = fig6.add_subplot(projection = "3d")
    ax.scatter( df.x, df.psi, df.error_x)
    plt.title("Horizontal Distance Error (NN) vs Yaw (psi) vs Horizontal Distance")
    ax.set_xlabel("Horizontal Distance (True X)")
    ax.set_ylabel("Psi (yaw)")
    ax.set_zlabel("Horizontal Distance Error (Predicted X)")
    

    #Y_Error (Crosstrack error) vs X (Horizontal Distance), 2D plot
    plt.figure("Figure 7")
    plt.rcParams["figure.autolayout"] = True

    plt.scatter(df.error_y, df.x)
    plt.title("Crosstrack Error (NN) vs Horizontal Distance (true x)")
    plt.ylabel("Horizontal Distance (True x)")
    plt.xlabel("Crosstrack Error(Predicted y)")

    #3D PLOTS =========================================================================================
    #Y_Error (Crosstrack error) vs Y (Crosstrack Error)
    fig8 = plt.figure("Figure 8")
    plt.rcParams["figure.autolayout"] = True
    ax = fig8.add_subplot(projection = "3d")
    ax.scatter( df.x, df.y, df.error_y)
    plt.title("Crosstrack Error (NN) vs Crosstrack vs Horizontal Distance")
    ax.set_xlabel("Horizontal Distance (True X)")
    ax.set_ylabel("Crosstrack Error (Y)")
    ax.set_zlabel("Crosstrack Error (NN, Pred Y)")

    #Y_Error (Crosstrack error) vs Height
    fig9 = plt.figure("Figure 9")
    plt.rcParams["figure.autolayout"] = True
    ax = fig9.add_subplot(projection = "3d")
    ax.scatter( df.x, df.h, df.error_y)
    plt.title("Crosstrack Error (NN) vs Height vs Horizontal Distance")
    ax.set_xlabel("Horizontal Distance (True X)")
    ax.set_ylabel("Height")
    ax.set_zlabel("Crosstrack Error (NN, Pred Y)")


    # Y_Error (Crosstrack error) vs Phi (Roll)
    fig10 = plt.figure("Figure 10")
    plt.rcParams["figure.autolayout"] = True
    ax = fig10.add_subplot(projection = "3d")
    ax.scatter( df.x, df.phi, df.error_y)
    plt.title("Crosstrack Error (NN) vs Roll (Phi) vs Horizontal Distance")
    ax.set_xlabel("Horizontal Distance (True X)")
    ax.set_ylabel("Roll (Phi)")
    ax.set_zlabel("Crosstrack Error (NN, Pred Y)")



    # Y_Error (Crosstrack error) vs Theta (Pitch)
    fig11 = plt.figure("Figure 11")
    plt.rcParams["figure.autolayout"] = True
    ax = fig11.add_subplot(projection = "3d")
    ax.scatter( df.x, df.theta, df.error_y)
    plt.title("Crosstrack Error (NN) vs Roll (Phi) vs Horizontal Distance")
    ax.set_xlabel("Horizontal Distance (True X)")
    ax.set_ylabel("Pitch (theta)")
    ax.set_zlabel("Crosstrack Error (NN, Pred Y)")


    # Y_Error (Crosstrack error) vs Psi (yaw)
    fig12 = plt.figure("Figure 12")
    ax.set_ylabel("Yaw (psi)")
    ax.set_zlabel("Crosstrack Error (NN, Pred Y)")

    plt.show()



if __name__ == '__main__':
    this_dir = Path(__file__).parent
    repo_dir = this_dir.resolve()

    today = date.today()
    save_dir = Path(f"{repo_dir}/plots/{today.year}-{today.month}-{today.day}")
    if not save_dir.exists():
        save_dir.mkdir()
    
    combine_data(save_dir, repo_dir)
    #plot_data(repo_dir) #Switched over to MATLAB for plotting 