import argparse
import csv
import itertools
import mss
import numpy as np
from pathlib import Path
import random
import time
#from tqdm import tqdm

from PIL import Image
import torch
from torchvision import transforms

from src.xplane_autoland.xplane_connect.driver import XPlaneDriver
from src.xplane_autoland.controllers.glideslope_controller import GlideSlopeController
from src.xplane_autoland.vision.perception import AutolandPerceptionModel


# for sigmas, chosen so that rarely ever goes beyond given value
# dividing by 3 so that 3sigma is a bit of a bound
max_degrees   = 15.
dphi_sigma    = max_degrees/3
dtheta_sigma  = max_degrees/3
dpsi_sigma    = max_degrees/3
dx_bounds     = [-150, 150]
dy_bounds     = [-150, 150]
dh_bounds     = [-150, 150]


model = AutolandPerceptionModel()
transform = model.preprocess
to_tensor = transforms.PILToTensor()


def data_for_x(driver, x_center, num_samples, save_dir):
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
    entries = set()
    sct = mss.mss()

    try:
        # sample points around the glideslope
        h_center = slope * x_center + h_thresh
        print(f"Base h = {h_center}")
        n = 0
        while n < num_samples:
            dphi = random.normalvariate(0., dphi_sigma)
            dtheta = random.normalvariate(0., dtheta_sigma)
            dpsi = random.normalvariate(0., dpsi_sigma)
            dx = float(random.randint(*dx_bounds))
            dy = float(random.randint(*dy_bounds))
            dh = float(random.randint(*dh_bounds))
            if h_center+dh < gsc._runway_elev:
                continue

            if n % 100 == 0:
                driver.reset()
                time.sleep(10)
            # set time to 8am
            driver._client.sendDREF("sim/time/zulu_time_sec", 8 * 3600 + 8 * 3600)

            orient_pos = (dphi, dtheta, dpsi, x_center+dx, dy, h_center+dh)
            if orient_pos in entries:
                continue
            entries.add(orient_pos)
            driver.set_orient_pos(*orient_pos)
            time.sleep(0.25)
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
            fname = f'./{save_dir}/images/image_{statestr}.pt'
            sct_img = sct.grab(sct.monitors[1])
            pil_img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            #pil_img.show()
            img = to_tensor(pil_img)
            img = transform(img)
            torch.save(img, fname)
            phi, theta, psi, x, y, h = actual_orient_pos
            writer.writerow([phi, theta, psi, x, y, h, fname])
            n += 1
            time.sleep(0.15)
    except KeyboardInterrupt:
        print('Interrupted.', flush=True)

    f.close()
    print(f'Last Image Name: {fname}')


def sweep_x(driver, num_samples, save_dir):
    # parameter sweeps
    x_sweep      = np.arange(0., driver._start_ground_range, 100.)
    for x_center in x_sweep:
        data_for_x(driver, x_center, num_samples, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sample training data for a vision-based state estimator")
    parser.add_argument("x_center", type=float, help="Which x value to collect data around")
    parser.add_argument("--seed", type=int, help="Set the random seed", default=1)
    parser.add_argument("--num-samples", type=float, help="How many samples to collect for this value of x", default=400)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    driver = XPlaneDriver()
    driver.pause(True)
    driver.reset()
    time.sleep(4)
    print('Starting...')

    save_dir = Path("./dataWPI_17000") #Need to make sure we change this 
    if not save_dir.exists():
        save_dir.mkdir()
    images_dir = save_dir / "images"
    if not images_dir.exists():
        images_dir.mkdir()
    with open(f"./{save_dir}/config.txt", "w") as f:
        f.write(f"Save Directory: {save_dir}\n")
        f.write(f"Seed: {args.seed}\n")
    sweep_x(driver, args.num_samples, save_dir=save_dir)
    #data_for_x(driver, args.x_center, args.num_samples, save_dir=save_dir)