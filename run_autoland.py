#!/usr/bin/env python3
import math
import sys
import time

from xplane_autoland.controllers.glideslope_controller import GlideSlopeController
from xplane_autoland.xplane_connect.vision_driver import XPlaneVisionDriver
from xplane_autoland.vision.perception import AutolandPerceptionModel

if __name__ == '__main__':

    model = AutolandPerceptionModel(resnet_version="50")
    model.load("/home/ma25944/github_repos/XPlaneAutolandScenario/models/vision/2023-8-10/best_model_params.pt")
    model.eval()

    plane = XPlaneVisionDriver(model)
    plane.pause(True)

    dt = 0.1
    max_time = 300

    gsc = GlideSlopeController(gamma=3, dt=dt)
    h_thresh = gsc._h_thresh
    start_elev = plane._start_elev
    slope = float(start_elev - h_thresh) / plane._start_ground_range

    x_val = 5000
    init_h = slope * x_val + h_thresh
    plane.reset(init_downtrack=5000, init_elev=init_h)

    try:
        last_time = time.time()
        for step in range(math.ceil(max_time/dt)):
            plane.pause(True)
            state = plane.get_statevec()
            phi, theta, psi, x, y, h = state[-6:]

            est_state = plane.est_statevec()
            print("x diff", x - est_state[-3])
            print("y diff", y - est_state[-2])
            # use estimates
            state[-3] = est_state[-3]
            state[-2] = est_state[-2]

            elevator, aileron, rudder, throttle = gsc.control(state)
            # the runway slopes down so this works fine
            if h <= gsc.runway_elevation:
                # disable throttle once you've landed
                plane.send_ctrl(elevator, aileron, rudder, -1)
                break
            plane.send_ctrl(elevator, aileron, rudder, throttle)
            plane.pause(False)
            time.sleep(dt)

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
