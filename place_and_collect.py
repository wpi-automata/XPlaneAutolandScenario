import argparse
import csv
import itertools
import mss
import numpy as np
from pathlib import Path
import sys
import time
from tqdm import tqdm

from xplane_autoland.xplane_connect.driver import XPlaneDriver
from xplane_autoland.controllers.glideslope_controller import GlideSlopeController


dphi_sweep   = np.arange(-5, 6, 1)
dtheta_sweep = np.arange(-5, 6, 1)
dpsi_sweep   = np.arange(-5, 6, 1)
dx_sweep     = np.arange(-100, 101, 50)
dy_sweep     = np.arange(-100, 101, 50)
dh_sweep     = np.arange(-100, 101, 50)


def data_for_x(driver, x_center):
    print(f"Saving data for x={x_center}")
    gsc    = GlideSlopeController(gamma=3)

    h_thresh = gsc._h_thresh
    start_elev = driver._start_elev
    slope = float(start_elev - h_thresh) / driver._start_ground_range

    save_dir = 'data'
    statepath = Path(f"./{save_dir}/states.csv")
    if not statepath.is_file():
        with open(str(statepath), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['phi', 'theta', 'psi', 'x', 'y', 'h', 'imagename'])

    f = open(str(statepath), 'a')
    writer = csv.writer(f)
    sct = mss.mss()

    try:
        # sample points around the glideslope
        h = slope * x_center + h_thresh
        for dphi, dtheta, dpsi, dx, dy, dh in itertools.product(dphi_sweep,
                                                                dtheta_sweep,
                                                                dpsi_sweep,
                                                                dx_sweep,
                                                                dy_sweep,
                                                                dh_sweep):
            if h+dh < h_thresh:
                continue
            elif x_center+dx < 0:
                continue

            # set time to 8am
            driver._client.sendDREF("sim/time/zulu_time_sec", 8 * 3600 + 8 * 3600)

            orient_pos = [dphi, dtheta, dpsi, x_center+dx, dy, h+dh]
            driver.set_orient_pos(*orient_pos)
            state = driver.get_statevec()
            actual_orient_pos = state[-6:]
            abs_diff = np.abs(orient_pos - actual_orient_pos)
            rel = np.abs(orient_pos)
            rel[rel < 1] = 1.
            per_diff = abs_diff / rel
            if not np.all(per_diff < 0.05):
                print(f'Warning: more than 5% difference between requested and actual state')
                print(f'% Diff: {per_diff}')
                print(f'Requested: {orient_pos}')
                print(f'Actual: {actual_orient_pos}')
            nv_pairs = zip(['phi', 'theta', 'psi', 'x', 'y', 'h'], actual_orient_pos)
            statestr = '_'.join([p0 + str(int(p1)) for p0, p1 in nv_pairs])
            fname = f'./{save_dir}/images/image_{statestr}.png'
            sct.shot(mon=1, output=fname)
            phi, theta, psi, x, y, h = actual_orient_pos
            writer.writerow([phi, theta, psi, x, y, h, fname])
            time.sleep(0.001)
    except KeyboardInterrupt:
        print('Interrupted.', flush=True)

    f.close()


def sweep_x(driver):
    # parameter sweeps
    x_sweep      = np.arange(0., driver._start_ground_range, 100.)
    for x_center in range(x_sweep):
        data_for_x(x_center)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Save data for training a vision-based state estimator")
    parser.add_argument("x_center", type=float, help="Which x value to collect data around")
    args = parser.parse_args()

    driver = XPlaneDriver()
    driver.pause(True)
    driver.reset()
    time.sleep(4)
    print('Starting...')

    data_for_x(driver, args.x_center)