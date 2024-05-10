#!/usr/bin/env python3
import argparse
import math
import sys
import time
from datetime import date
from pathlib import Path
import csv

from src.xplane_autoland.controllers.glideslope_controller import GlideSlopeController
from src.xplane_autoland.xplane_connect.vision_driver import XPlaneVisionDriver
from src.xplane_autoland.xplane_connect.driver import XPlaneDriver
from src.xplane_autoland.vision.perception import AutolandPerceptionModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the autoland scenario at KMWH Grant County International Airport Runway 04. You must start XPlane and choose the airport + a Cessna separately.")
    parser.add_argument("--model", help="The path to model parameters (*.pt) for a vision network. Note must have XPlane fullscreen for screenshots", default=None)
    args = parser.parse_args()

    ##Save state information to a file##
    this_dir = Path(__file__).parent
    repo_dir = this_dir.resolve()

    today = date.today()
    save_dir = Path(f"{repo_dir}/autoland_errors/{today.year}-{today.month}-{today.day}")
    if not save_dir.exists():
        save_dir.mkdir()

    statepath = Path(f"{save_dir}/Trial15-12464_states.csv") #12464 corresponds to start distance from the runway
    if not statepath.is_file():
        with open(str(statepath), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['phi, theta, psi, x, y, y_pred, h, h_err, h_err_pred, y_err_NN, h_err_NN'])

    f = open(str(statepath), 'a')
    writer = csv.writer(f)
    ####################################

    WITH_VISION = False
    if args.model:
        WITH_VISION=True
        model = AutolandPerceptionModel(resnet_version="50")
        model.load(args.model)
        model.eval()
        plane = XPlaneVisionDriver(model)
    else:
        plane = XPlaneDriver()

    plane.pause(True)

    dt = 0.1
    max_time = 600

    gsc = GlideSlopeController(gamma=3, dt=dt)
    # the altitude of the runway threshold
    h_thresh = gsc.runway_threshold_height
    start_elev = plane._start_elev
    slope = float(start_elev - h_thresh) / plane._start_ground_range

    # distance from the runway crossing (decrease to start closer)
    # vision mode works best at 9000m and less (up until right before landing)
    # need to train more at higher distances and close up
    x_val = 12464
    init_h = slope * x_val + h_thresh
    # can set the state arbitrarily (see reset documentation for semantics)
    plane.reset(init_x=x_val, init_h=init_h)
    plane.pause(False)

    try:
        last_time = time.time()
        for step in range(math.ceil(max_time/dt)):
            state = plane.get_statevec()
            phi, theta, psi, x, y, h = state[-6:]
            print(x)
            h_err_pred = None

            if WITH_VISION:
                plane.pause(True)
                y_pred, h_err_pred = plane.est_pos_state()
                # update state vector with y estimate
                # err_h will be used directly
                state[-2] = y_pred

                #Calculations to get NN Errors and true height error
                h_err = gsc.get_glideslope_height_at(x) - h
                y_err_NN = y - y_pred
                h_err_NN = h_err - h_err_pred

                writer.writerow([phi, theta, psi, x, y, y_pred, h, h_err, h_err_pred, y_err_NN, h_err_NN])
                # print("Y: %f", y_1)

            elevator, aileron, rudder, throttle = gsc.control(state, err_h=h_err_pred)
            # the runway slopes down so this works fine
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
                break
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
