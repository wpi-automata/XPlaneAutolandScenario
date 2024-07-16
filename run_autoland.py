#!/usr/bin/env python3
import argparse
import math
import sys
import time
from datetime import date
from pathlib import Path
import csv
import mss
from PIL import Image
import torch
from torchvision import transforms

from src.xplane_autoland.controllers.glideslope_controller import GlideSlopeController
from src.xplane_autoland.xplane_connect.vision_driver import XPlaneVisionDriver
from src.xplane_autoland.xplane_connect.driver import XPlaneDriver
from src.xplane_autoland.vision.perception import AutolandPerceptionModel

basic_model = AutolandPerceptionModel()
transform = basic_model.preprocess
to_tensor = transforms.PILToTensor()

def collect_image(sct, pos, save_dir, writer): #Code from place_and_collect.py
    nv_pairs = zip(['phi', 'theta', 'psi', 'x', 'y', 'h'], pos)
    statestr = '_'.join([p0 + str(int(p1)) for p0, p1 in nv_pairs])
    fname = f'{save_dir}/images/image_{statestr}.pt'
    sct_img = sct.grab(sct.monitors[1])
    pil_img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
    img = to_tensor(pil_img)
    img = transform(img)
    torch.save(img, fname)
    phi, theta, psi, x, y, h = pos
    # print(f"x: {x}. y: {y}. h: {h}")
    writer.writerow([phi, theta, psi, x, y, h, fname])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the autoland scenario at KMWH Grant County International Airport Runway 04. You must start XPlane and choose the airport + a Cessna separately.")
    parser.add_argument("--model", help="The path to model parameters (*.pt) for a vision network. Note must have XPlane fullscreen for screenshots", default=None)
    parser.add_argument("--collect" , help="Boolean varible indicating if state information and images of the autolanding will be collected", default=False)
    parser.add_argument("--start-dist", help="The starting distance of the plane from the runway.", default=3000)
    parser.add_argument("--gamma", help="The desired angle for the plane to follow along the glidescope", default=3.5)
    parser.add_argument("--lateral-dev", help="How far the plane should laterally deviate from the glidescope", default=0)
    parser.add_argument("--keep_h", help="Was the model trained with or without h_err calculations", default=True)
    args = parser.parse_args()

    ##Save state information to a file##
    this_dir = Path(__file__).parent
    repo_dir = this_dir.resolve()

    if args.collect:
        print("Collecting images and states")
        img_dir = Path(f"/home/achadbo/Desktop/Autoland/7-11-2024/{args.start_dist}")
        if not img_dir.exists():
            img_dir.mkdir()
        img_dir = Path(f"{img_dir}/Gamma{args.gamma}_LatDev{args.lateral_dev}")
        if not img_dir.exists():
            img_dir.mkdir()
        
        img_path = Path(f"{img_dir}/states.csv")
        if not img_path.is_file():
            with open(str(img_path), 'w') as f:
                img_writer = csv.writer(f)
                img_writer.writerow(['phi', 'theta', 'psi', 'x', 'y', 'h', 'imagename'])

        f2 = open(str(img_path), 'a')
        img_writer = csv.writer(f2)
        
        images_dir = img_dir / "images"
        if not images_dir.exists():
            images_dir.mkdir()
    ####################################

    WITH_VISION = False
    if args.model:
        WITH_VISION=True
        print(args.model)
        model = AutolandPerceptionModel(resnet_version="50", keep_h=args.keep_h)
        model.load(args.model)
        model.eval()
        plane = XPlaneVisionDriver(model, start_ground_range=float(args.start_dist), keep_h=args.keep_h)

        today = date.today()
        save_dir = Path(f"{repo_dir}/autoland_errors/{today.year}-{today.month}-{today.day}")
        if not save_dir.exists():
            save_dir.mkdir()

        statepath = Path(f"{save_dir}/Trial15-12464_states.csv") #12464 corresponds to start distance from the runway
        if not statepath.is_file():
            with open(str(statepath), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['phi, theta, psi, x, y, y_pred, h, h_err, h_err_pred, y_err_NN, h_err_NN'])

        f = open(str(statepath), 'a')
        writer = csv.writer(f)        
    else:
        plane = XPlaneDriver()

    plane.pause(True)

    dt = 0.1
    max_time = 600

    target_deg = float(args.gamma)
    gsc = GlideSlopeController(gamma=target_deg, dt=dt)
    # the altitude of the runway threshold
    h_thresh = gsc.runway_threshold_height
    start_elev = plane._start_elev
    #slope = float(start_elev - h_thresh) / plane._start_ground_range
    slope = math.tan(math.radians(target_deg))

    # distance from the runway crossing (decrease to start closer)
    # vision mode works best at 9000m and less (up until right before landing)
    # need to train more at higher distances and close up
    x_val = float(args.start_dist)
    init_h = slope * x_val + h_thresh
    init_y = int(args.lateral_dev)
    # can set the state arbitrarily (see reset documentation for semantics)
    plane.reset(init_x=x_val, init_h=init_h, init_y=init_y)
    plane.pause(False)

    sct = mss.mss()
    try:
        last_time = time.time()
        for step in range(math.ceil(max_time/dt)):
            state = plane.get_statevec()
            phi, theta, psi, x, y, h = state[-6:]
            #Debug 
            #print(x)
            h_err_pred = None

            if WITH_VISION:
                plane.pause(True)
                if args.keep_h:
                    y_pred, h_err_pred = plane.est_pos_state()
                    #Calculations to get NN Errors and true height error
                    h_err = gsc.get_glideslope_height_at(x) - h
                    h_err_NN = h_err - h_err_pred
                else:
                    y_pred = plane.est_pos_state()
                    h_err = None
                    h_err_NN = None

                # update state vector with y estimate
                state[-2] = y_pred
                y_err_NN = y - y_pred
                
                writer.writerow([phi, theta, psi, x, y, y_pred, h, h_err, h_err_pred, y_err_NN, h_err_NN]) #Write to the file at each iteration 
                # print("Y: %f", y_1)

            if args.collect:
                collect_image(sct, [phi, theta, psi, x, y, h], img_dir, img_writer)
            elevator, aileron, rudder, throttle = gsc.control(state, err_h=h_err_pred)
            # the runway slopes down so this works fine
            if h <= gsc.runway_elevation and x <= 0:
                # disable throttle once you've landed
                plane.send_ctrl(0, 0, 0, -1)
                print("Successfully landed")
                plane.pause(False)
                # run the simulation for 10 more seconds to complete landing
                for step in range(math.ceil(10/dt)):
                    state = plane.get_statevec()
                    # use the controller to keep it straight
                    elevator, aileron, rudder, _ = gsc.control(state)
                    throttle = -1
                    plane.send_brake(1)
                    plane.send_ctrl(0, 0, rudder, 0)
                    time.sleep(dt)
                break
            plane.send_ctrl(elevator, aileron, rudder, throttle)
            plane.pause(False)
            time.sleep(dt)

        print('Done')
        plane.pause(True)
        h = input("Press any key to end.")
    except KeyboardInterrupt:
        print('Interrupted -- Pausing sim and exiting')
        plane.pause(True)
        sys.exit(130)
