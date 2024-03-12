#!/usr/bin/env python3
import argparse
import math
import sys
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from xplane_autoland.controllers.safe_gs_controller import ydot_controller
from xplane_autoland.controllers.glideslope_controller import GlideSlopeController
from xplane_autoland.xplane_connect.vision_driver import XPlaneVisionDriver
from xplane_autoland.xplane_connect.driver import XPlaneDriver
from xplane_autoland.vision.perception import AutolandPerceptionModel

def autoland_tester(local_start=(-35205.421875, 40957.0234375), traj_len=100, filename='testing.txt', print_rate=100, save_data=False):
    plane = XPlaneDriver(local_start=local_start)
    plane.pause(True)

    dt = 0.1
    max_time = 600

    # gsc = GlideSlopeController(gamma=3, dt=dt)
    gsc = ydot_controller(gamma=3, dt=dt)
    # the altitude of the runway threshold
    h_thresh = gsc.runway_threshold_height
    start_elev = plane._start_elev
    slope = float(start_elev - h_thresh) / plane._start_ground_range

    # distance from the runway crossing (decrease to start closer)
    # vision mode works best at 9000m and less (up until right before landing)
    # need to train more at higher distances and close up
    x_val = 9000
    init_h = slope * x_val + h_thresh
    # can set the state arbitrarily (see reset documentation for semantics)
    plane.reset(init_x=x_val, init_h=init_h)
    plane.pause(False)

    states = list()

    verbose = False

    try:
        for step in range(math.ceil(max_time/dt)):
            state = plane.get_statevec()
            phi, theta, psi, x, y, h = state[-6:]
            err_h = None

            states.append(state)
            
            steps = len(states)
            if verbose:
                if steps % print_rate == 0:
                    print(f'Num:{steps}')
            if len(states) == traj_len:
                if save_data:
                    save_trajectory(states, filename)
                plane.pause(True)
                return

            elevator, aileron, rudder, throttle = gsc.control(state, err_h=err_h)
            
            if h <= gsc.runway_elevation and x <= 0:
                # disable throttle once you've landed
                plane.send_ctrl(0, 0, 0, -1)
                print("Successfully landed")
                plane.pause(False)
                # run the simulation for 10 more seconds to complete landing
                for step in range(math.ceil(10/dt)):
                    state = plane.get_statevec()
                    # use the controller to keep it straight
                    elevator, aileron, rudder, _ = gsc.control(state)
                    throttle = -1
                    plane.send_brake(1)
                    plane.send_ctrl(0, 0, rudder, 0)
                    time.sleep(dt)
                if save_data:
                    save_trajectory(states,filename)
                return
            
            plane.send_ctrl(elevator, aileron, rudder, throttle)
            plane.pause(False)
            time.sleep(dt)

        print('Done')
        plane.pause(True)
        h = input("Press any key to end.")
    except KeyboardInterrupt:
        print('Interrupted -- Pausing sim and exiting')
        plane.pause(True)
        sys.exit(130)

def save_trajectory(states, filename):
    rel_dir, name = os.path.split(filename)
    full_dir = os.path.join(os.getcwd(), rel_dir)
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    
    filename = os.path.join(full_dir, name)
    states_hist = np.vstack(states)
    np.savetxt(filename, states_hist)
    return
