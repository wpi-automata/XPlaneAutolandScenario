#!/usr/bin/env python3
import math
import sys
import time
import numpy as np
import pandas as pd

sys.path.append('./src')

from src.xplane_autoland.controllers.cbf_controller import CBFController, Airplane, Airplane_upd
from src.xplane_autoland.controllers.glideslope_controller import GlideSlopeController
from src.xplane_autoland.xplane_connect.driver import XPlaneDriver

def saveData(folder, filename, state, controls, initial=False):
    labels = ["u", "v", "w", "p", "q", "r", "phi", "theta", "psi", "x", "y", "h", "elev", "throttle", "aileron", "rudder"]
    values = np.concatenate([state,controls])
    values =  np.array(values).reshape((1,len(values)))

    if initial:
        outData = pd.DataFrame(values,index=[0],columns=labels)
        outData.to_csv(folder + "/"+ filename,index=False,index_label=False)
    else:
        outData = pd.DataFrame(values,index=[0])
        outData.to_csv(folder + "/"+ filename, index=False, index_label=False, header=None, mode='a')

if __name__ == '__main__':
    plane = XPlaneDriver()
    plane.reset()

    save_data = False

    dt = 1e-2
    
    max_time = 400
    u_deg = 65.

    target_h=361

    init_state = plane.get_statevec()
    cbf_c = CBFController(initial_state = init_state, goal_h=target_h, dt= dt, dynamics=Airplane_upd(dt=dt), u_bounds=[[-1,1],[0,1],[-1,1],[-1,1]])
    cbf_c.setup(gamma=1., scaling=u_deg, u_ref = True)

    # for the landing portion
    gsc = GlideSlopeController(gamma=3, dt=dt)

    # to save data
    if save_data:
        filename = 'aircraft_data2.csv'
        folder = r'C:\Users\AM28303\OneDrive - MIT Lincoln Laboratory\Desktop\code\XPlaneAutolandScenario'
        saveData(folder, filename, init_state, np.zeros(4), initial=True)

    try:
        plane.pause(False)
        last_time = time.time()
        for step in range(math.ceil(max_time/dt)):
            plane.pause(True)

            state = plane.get_statevec()
            h = state[-1]
            print('state', state)
            # the runway slopes down so this works fine
            if h <= target_h:
                # disable throttle once you've landed
                print("landed", plane.get_statevec())
                plane.send_ctrl(elevator, aileron, rudder, -1)
                plane.pause(False)
                break

            # reference controls
            e_ref, a_ref, r_ref, t_ref = gsc.control(state)
            print('ref controls: ', e_ref, a_ref, r_ref, t_ref)

            # update specs to account for state-dependent cbfs
            cbf_c.s.ready_to_sim = False
            cbf_c.add_specs(state)
            # update sim variables
            cbf_c.update_sim([e_ref, t_ref, a_ref, r_ref], state)

            elevator, throttle, aileron, rudder = cbf_c.s.simulate(num_steps=1, dt = dt, plotting=False)[0]

            print('upd controls: ', elevator, aileron, rudder, throttle)
            
            if save_data:
                saveData(folder, filename, state, [elevator, aileron, rudder, throttle], initial=False)

            plane.pause(False)
            plane.send_ctrl(elevator, aileron, rudder, throttle)
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
        plane.pause(True)
    except KeyboardInterrupt:
        print('Interrupted -- Pausing sim and exiting')
        plane.pause(True)
        sys.exit(130)