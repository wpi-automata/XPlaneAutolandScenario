#!/usr/bin/env python3
import math
import sys
import time

from xplane_autoland.controllers.glideslope_controller import GlideSlopeController
from xplane_autoland.xplane_connect.driver import XPlaneDriver

# Evaluating network as sidecar
from xplane_autoland.vision.perception import AutolandPerceptionModel
import mss
from PIL import Image
import torch
from torchvision import transforms

if __name__ == '__main__':

    # Need to stop and take image
    model = AutolandPerceptionModel(resnet_version="50")
    model.load("/home/ma25944/github_repos/XPlaneAutolandScenario/src/xplane_autoland/vision/models/2023-7-24/best_model_params.pt")
    model.eval()
    sct = mss.mss()


    plane = XPlaneDriver()
    plane.pause(True)
    plane.reset()

    dt = 0.1
    max_time = 300
    to_tensor = transforms.PILToTensor()

    gsc = GlideSlopeController(gamma=3, dt=dt)

    try:
        plane.pause(False)
        last_time = time.time()
        for step in range(math.ceil(max_time/dt)):
            state = plane.get_statevec()
            phi, theta, psi, x, y, h = state[-6:]

            plane.pause(True)
            sct_img = sct.grab(sct.monitors[1])
            pil_img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            img = to_tensor(pil_img)
            img = model.preprocess(img)
            orient_alt = torch.FloatTensor([phi, theta, psi, h])
            img, orient_alt = img[None, :, :, :], orient_alt[None, :]
            with torch.no_grad():
                pred_x, pred_y = model(img, orient_alt).flatten()
                pred_x *= 12464
                pred_y *= 500
            print(f"Error: x={x-pred_x}, y={y-pred_y}")
            plane.pause(False)

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