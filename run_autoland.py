#!/usr/bin/env python3
import math
import sys
import time
import matplotlib.pyplot as plt

from xplane_autoland.controllers.glideslope_controller import GlideSlopeController
from xplane_autoland.xplane_connect.driver import XPlaneDriver

def plot_live(param_data, dt, fig_id):
    '''
    plot given param over time
    '''
    plt.figure(fig_id)
    plt.plot(dt, param_data, 'b-')
    plt.pause(1e-4)

if __name__ == '__main__':
    plane = XPlaneDriver()
    plane.reset()

    dt = 0.1
    max_time = 300

    gsc = GlideSlopeController(gamma=3, dt=dt)

    plotting = True
    # plotting
    if plotting:
        y_data = []
        ydot = []
        t_data = []
        fig1=plt.figure("y plot")
        ax = plt.gca()
        ax.set_xlim([0,max_time+10])
        # fig2 = plt.figure("ydot plot")
        # ax = plt.gca()
        # ax.set_xlim([0,max_time+10])

    try:
        plane.pause(False)
        last_time = time.time()
        t0 = last_time
        last_y = plane.get_statevec()[-2]
        for step in range(math.ceil(max_time/dt)):
            state = plane.get_statevec()
            h = state[-1]
            elevator, aileron, rudder, throttle = gsc.control(state)
            # the runway slopes down so this works fine
            if h <= gsc.runway_elevation:
                # disable throttle once you've landed
                plane.send_ctrl(elevator, aileron, rudder, -1)
                break
            plane.send_ctrl(elevator, aileron, rudder, throttle)
            time_diff = dt - (time.time() - last_time)
            if time_diff > 0:
                time.sleep(time_diff)
            last_time = time.time()

            if plotting:
                y_data.append(state[-2])
                # ydot.append((state[-2]-last_y)/dt)
                t_data.append(last_time - t0)
                plot_live(y_data, t_data, 1)
                # plot_live(ydot, t_data, 2)
                time.sleep(1e-3)
            last_y = state[-2]

        # run the simulation for 10 more seconds to complete landing
        for step in range(math.ceil(10/dt)):
            state = plane.get_statevec()
            # use the controller to keep it straight
            elevator, aileron, rudder, _ = gsc.control(state)
            throttle = -1
            plane.send_brake(1)
            plane.send_ctrl(elevator, aileron, rudder, throttle)
            time.sleep(dt)

        print('Done')
        plane.pause(False)
    except KeyboardInterrupt:
        print('Interrupted -- Pausing sim and exiting')
        plane.pause(True)
        sys.exit(130)