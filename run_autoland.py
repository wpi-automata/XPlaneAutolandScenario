#!/usr/bin/env python3
import math
import sys
import time

from xplane_autoland.controllers.glideslope_controller import GlideSlopeController
from xplane_autoland.xplane_connect.xpc3 import XPlaneConnect
from xplane_autoland.xplane_connect.xpc3_helper import get_autoland_statevec, reset, sendCTRL

## Potential info
# dist = dist_from_glideslope(client)
# linv = body_frame_velocity(client)
# HUSMI, UBGUY = get_glideslope_points()
# # 50 ft TCH -> m -> + the agl at that point
# tch = 50 * 0.3048 + 223

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
            # TODO: once it hits the ground, stop generating autoland controls and taxi (or just stop)
            elevator, aileron, rudder, throttle = gsc.control(state)
            sendCTRL(client, elevator, aileron, rudder, throttle)
            time_diff = dt - (time.time() - last_time)
            if time_diff > 0:
                time.sleep(time_diff)
            last_time = time.time()
        print('Done')
        client.pauseSim(True)
    except KeyboardInterrupt:
        print('Interrupted -- Pausing sim and exiting')
        client.pauseSim(True)
        sys.exit(130)