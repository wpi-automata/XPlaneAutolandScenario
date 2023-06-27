import csv
import itertools
import mss
import numpy as np
import time

from xplane_autoland.xplane_connect.driver import XPlaneDriver
from xplane_autoland.controllers.glideslope_controller import GlideSlopeController


if __name__ == '__main__':
    driver = XPlaneDriver()
    driver.pause(True)
    gsc    = GlideSlopeController(gamma=3)

    h_thresh = gsc._h_thresh
    start_elev = driver._start_elev
    slope = float(start_elev - h_thresh) / driver._start_ground_range

    save_dir = 'data'
    f = open(f'./{save_dir}/states.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['phi', 'theta', 'psi', 'x', 'y', 'h', 'imagename'])
    imgid = 0
    sct = mss.mss()

    # parameter sweeps
    x_sweep      = np.arange(0., driver._start_ground_range, 1.)

    dphi_sweep   = np.arange(-10, 10, 1)
    dtheta_sweep = np.arange(-10, 10, 1)
    dpsi_sweep   = np.arange(-10, 10, 1)
    dx_sweep     = np.arange(-100, 100, 10)
    dy_sweep     = np.arange(-100, 100, 10)
    dh_sweep     = np.arange(-100, 100, 10)


    try:
        # sample points around the glideslope
        for x in np.arange(0., 12464., 100.):
            h = slope * x + h_thresh
            for dphi, dtheta, dpsi, dx, dy, dh in itertools.product(dphi_sweep,
                                                                    dtheta_sweep,
                                                                    dpsi_sweep,
                                                                    dx_sweep,
                                                                    dy_sweep,
                                                                    dh_sweep):
                if h+dh < h_thresh:
                    continue
                elif x+dx < 0:
                    continue
                driver.set_orient_pos(dphi, dtheta, dpsi, x+dx, dy, h+dh)
                state = driver.get_statevec()
                fname = f'./{save_dir}/images/image{imgid}.png'
                sct.shot(mon=1, output=fname)
                phi, theta, psi, x, y, h = state[-6:]
                writer.writerow([phi, theta, psi, x, y, h, fname])
                time.sleep(0.001)
                imgid += 1
    except KeyboardInterrupt:
        print('Interrupted.', flush=True)