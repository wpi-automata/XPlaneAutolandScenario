#!/usr/bin/env python3
import math
import sys
import time

from xplane_autoland.controllers.glideslope_controller import GlideSlopeController
from xplane_autoland.xplane_connect.driver import XPlaneDriver

if __name__ == '__main__':
    plane = XPlaneDriver()
    plane.pause(True)
    plane.reset()

    dt = 0.1
    max_time = 300

    gsc = GlideSlopeController(gamma=3, dt=dt)

    try:
        plane.pause(False)
        last_time = time.time()
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