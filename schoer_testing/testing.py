import os
import matplotlib.pyplot as plt
import numpy as np
from run_autoland_ads import autoland_tester

def load_file(filename):
    return np.loadtxt(filename, dtype=float)

def config_plots():
    fig,axs = plt.subplots(3,1,sharex=True) # psi, y, phi, (u, theta)
    axs[0].set_title('$\psi$: Yaw (heading)')
    axs[1].set_title('Y: Lateral Deviation wrt Runway')
    axs[2].set_title('$\phi$: Roll')
    # axs[3].set_title('u')
    # axs[4].set_title('$\\theta$')
    axs[len(axs)-1].set_xlabel('Time')
    return fig, axs

def plot_traj(axs, states):
    dt = .1
    n = states.shape[0]
    t = np.arange(0,n*dt,dt)
    axs[0].plot(t, states[:,8])
    axs[1].plot(t, states[:,10], label='$y$')
    axs[2].plot(t, states[:,6])
    # axs[3].plot(t, states[:,0])
    # axs[4].plot(t, states[:,7])

    # plot ydot calculated from y
    axs[1].plot(t[1:], np.diff(states[:,10])/dt, label='Diff $\dot{y}$')

    # Calculate ydot from velocity and orientation
    '''
            u      - longitudinal velocity (m/s)
            v      - lateral velocity (m/s)
            w      - vertical velocity (m/s)
            p      - roll velocity (deg/s)
            q      - pitch velocity (deg/s)
            r      - yaw velocity (deg/s)
            phi    - roll angle (deg)
            theta  - pitch angle (deg)
            psi    - yaw angle (deg)
            x      - horizontal distance (m)
            y      - lateral deviation (m)
            h      - aircraft altitude (m)
    '''
    u = states[:,0]
    v = states[:,1]
    psi = states[:,8]
    axs[1].plot(t, u*np.sin(np.radians(psi)) + v*np.cos(np.radians(psi)), label='$\dot{y}$')
    axs[1].legend()

    # Plot goals
    tt = [t[0], t[-1]]
    axs[0].plot(tt, [0,0], 'k:')
    axs[1].plot(tt, [0,0], 'k:')
    axs[2].plot(tt, [0,0], 'k:')

def data_plotter(dir):
    files = [os.path.join(dir,f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir,f))]
    fig,axs = config_plots()
    for f in files:
        states = load_file(f)
        plot_traj(axs, states)
    plt.legend()
    plt.show()

def data_driver(dir, save_data=False):
    NUM_TRAJS = 1
    TRAJ_LEN = 500
    DIFF_IN_INIT = 0 #.001
    FILE_BASE_NAME = 'zero_-10'

    dx,dy=(-35205.421875, 40957.0234375)

    for i in range(NUM_TRAJS):
        ddx = dx - DIFF_IN_INIT*i

        testname = FILE_BASE_NAME + f'_{i}.txt'
        filename = os.path.join(dir,testname)
        autoland_tester(traj_len=TRAJ_LEN,
                         local_start=(ddx,dy),
                         filename=filename,
                         print_rate=TRAJ_LEN/10,
                         save_data=save_data)
        print(f'{i} is done')

if __name__ == '__main__':
    run_sim = False
    save_data = True

    dir = '/home/an25749/nasa_uli/XPlaneAutolandScenario/schoer_testing/data/ydot_tuning'
    if run_sim:
        data_driver(dir, save_data)
    else:
        data_plotter(dir)