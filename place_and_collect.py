import argparse
import csv
import itertools
import mss
import numpy as np
from pathlib import Path
import random
import time
from tqdm import tqdm

from xplane_autoland.xplane_connect.driver import XPlaneDriver
from xplane_autoland.controllers.glideslope_controller import GlideSlopeController


dphi_sweep   = [-5, -2, 0, 2, 5]
dtheta_sweep = [-5, -2, 0, 2, 5]
dpsi_sweep   = [-5, -2, 0, 2, 5]
dx_sweep     = np.arange(-100, 101, 20)
dy_sweep     = np.arange(-100, 101, 20)
dh_sweep     = np.arange(-100, 101, 20)


def data_for_x(driver, x_center, prob, save_dir="data"):
    print(f"Saving data for x={x_center}")
    gsc    = GlideSlopeController(gamma=3)

    h_thresh = gsc._h_thresh
    start_elev = driver._start_elev
    slope = float(start_elev - h_thresh) / driver._start_ground_range

    statepath = Path(f"./{save_dir}/states.csv")
    if not statepath.is_file():
        with open(str(statepath), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['phi', 'theta', 'psi', 'x', 'y', 'h', 'imagename'])

    f = open(str(statepath), 'a')
    writer = csv.writer(f)
    writer.writerow([-1, -1, -1, -1, -1, -1, "divider"])
    sct = mss.mss()

    try:
        # sample points around the glideslope
        h_center = slope * x_center + h_thresh
        print(f"Base h = {h_center}")
        for dphi, dtheta, dpsi, dx, dy, dh in itertools.product(dphi_sweep,
                                                                dtheta_sweep,
                                                                dpsi_sweep,
                                                                dx_sweep,
                                                                dy_sweep,
                                                                dh_sweep):
            if h_center+dh < h_thresh:
                continue
            elif x_center+dx < 0:
                continue
            elif random.random() > prob:
                continue

            driver.reset()
            # set time to 8am
            driver._client.sendDREF("sim/time/zulu_time_sec", 8 * 3600 + 8 * 3600)

            orient_pos = [dphi, dtheta, dpsi, x_center+dx, dy, h_center+dh]
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
    print(f'Last Image Name: {fname}')


def sweep_x(driver, prob):
    # parameter sweeps
    x_sweep      = np.arange(0., driver._start_ground_range, 100.)
    for x_center in range(x_sweep):
        data_for_x(driver, x_center, prob=prob)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sample training data for a vision-based state estimator")
    parser.add_argument("x_center", type=float, help="Which x value to collect data around")
    parser.add_argument("--seed", type=int, help="Set the random seed", default=1)
    parser.add_argument("--prob", type=float, help="Probability of saving a particular configuration", default=0.1)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    driver = XPlaneDriver()
    driver.pause(True)
    driver.reset()
    time.sleep(4)
    print('Starting...')

    save_dir = "data"
    with open(f"./{save_dir}/config.txt", "w") as f:
        f.write(f"Save Directory: {save_dir}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Prob: {args.prob}\n")
        f.write("\n")
        f.write("Sweep Values:\n")
        f.write(f"dphi_sweep: {dphi_sweep}\n")
        f.write(f"dtheta_sweep: {dtheta_sweep}\n")
        f.write(f"dpsi_sweep: {dpsi_sweep}\n")
        f.write(f"dx_sweep: {dx_sweep}\n")
        f.write(f"dy_sweep: {dy_sweep}\n")
        f.write(f"dh_sweep: {dh_sweep}\n")

    data_for_x(driver, args.x_center, prob=args.prob, save_dir=save_dir)