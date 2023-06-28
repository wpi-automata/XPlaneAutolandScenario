import csv
import itertools
import mss
import numpy as np
import time
from tqdm import tqdm

from xplane_autoland.xplane_connect.driver import XPlaneDriver
from xplane_autoland.controllers.glideslope_controller import GlideSlopeController


if __name__ == '__main__':
    driver = XPlaneDriver()
    driver.pause(True)
    time.sleep(4)
    print('Starting...')
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

    # dphi_sweep   = np.arange(-5, 6, 5)
    # dtheta_sweep = np.arange(-5, 6, 5)
    # dpsi_sweep   = np.arange(-5, 6, 5)
    dphi_sweep   = [0]
    dtheta_sweep = [0]
    dpsi_sweep   = [0]
    dx_sweep     = np.arange(-100, 101, 100)
    dy_sweep     = np.arange(-100, 101, 100)
    dh_sweep     = np.arange(-100, 101, 100)


    try:
        # sample points around the glideslope
        for x in tqdm(np.arange(0., 12464., 100.)):
            driver.reset()
            # set time to 8am
            driver._client.sendDREF("sim/time/zulu_time_sec", 8 * 3600 + 8 * 3600)
            h = slope * x + h_thresh
            print(f'Base: x={x}, h={h}')
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
                orient_pos = [dphi, dtheta, dpsi, x+dx, dy, h+dh]
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
                imgid += 1
            print(f"Last image: {imgid-1}")
    except KeyboardInterrupt:
        print('Interrupted.', flush=True)