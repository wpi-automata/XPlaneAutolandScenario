#!/usr/bin/env python3
import math
import sys
import time

from xplane_autoland.controllers.glideslope_controller import GlideSlopeController
from xplane_autoland.xplane_connect.xpc3 import XPlaneConnect
from xplane_autoland.xplane_connect.xpc3_helper import get_autoland_statevec, reset, sendCTRL, sendBrake

if __name__ == '__main__':
    client = XPlaneConnect()
    reset(client)

    dt = 0.1
    max_time = 300

    gsc = GlideSlopeController(client, gamma=3, dt=dt)

    try:
        client.pauseSim(False)
        last_time = time.time()
        for step in range(math.ceil(max_time/dt)):
            state = get_autoland_statevec(client)
            h = state[-1]
            elevator, aileron, rudder, throttle = gsc.control(state)
            # the runway slopes down so this works fine
            if h <= gsc.runway_elevation:
                # disable throttle once you've landed
                sendCTRL(client, elevator, aileron, rudder, -1)
                break
            sendCTRL(client, elevator, aileron, rudder, throttle)
            time_diff = dt - (time.time() - last_time)
            if time_diff > 0:
                time.sleep(time_diff)
            last_time = time.time()

        # run the simulation for 10 more seconds to complete landing
        for step in range(math.ceil(10/dt)):
            state = get_autoland_statevec(client)
            # use the controller to keep it straight
            elevator, aileron, rudder, _ = gsc.control(state)
            throttle = -1
            sendBrake(client, 1)
            sendCTRL(client, elevator, aileron, rudder, throttle)
            time.sleep(dt)

        print('Done')
        client.pauseSim(True)
    except KeyboardInterrupt:
        print('Interrupted -- Pausing sim and exiting')
        client.pauseSim(True)
        sys.exit(130)