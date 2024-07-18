import sys
sys.path.append('/home/agchadbo/XPlaneAutolandScenario/src/xplane_autoland')
from xplane_autoland.xplane_connect.driver import XPlaneDriver

from xplane_autoland.vision.perception import AutolandPerceptionModel
import mss
from PIL import Image
import torch
from torchvision import transforms

to_tensor = transforms.PILToTensor()


class XPlaneVisionDriver(XPlaneDriver):
    """
    This class estimates x and y state variables from an image
    """
    def __init__(self, state_estimator, home_heading=53.7,
                 local_start=(-35285.421875, 40957.0234375),
                 start_ground_range=12464, start_elev = 1029.45,
                 t=(-25159.26953, 33689.8125), keep_h=True):
        super().__init__(home_heading, local_start, start_ground_range,
                         start_elev, t, keep_h)
        self._state_estimator = state_estimator
        self._orient_norm_divisor = torch.FloatTensor([180., 180., 180., float(start_ground_range)])
        self._keep_h = keep_h

        # Need to stop and take image
        self._sct = mss.mss()

    def est_pos_state(self):
        """
        Estimates the position state using a vision network

        Returns:
            y -- lateral deviation
            err_h -- error in height relative to a glideslope
        """
        sct_img = self._sct.grab(self._sct.monitors[1])
        pil_img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        img = to_tensor(pil_img)
        img = self._state_estimator.preprocess(img)
        phi, theta, psi = self.est_orient_state()
        _, _, h  = self.get_pos_state()
        orient_alt = torch.FloatTensor([phi, theta, psi, h])
        orient_alt /= self._orient_norm_divisor
        img, orient_alt = img[None, :, :, :], orient_alt[None, :]

        with torch.no_grad():
            label_mult = 50. # must match the normalization used in AutolandImageDataset when training network
            if self._keep_h:
                y_err, h_err = self._state_estimator(img, orient_alt).flatten()
                y_err *= label_mult
                h_err *= label_mult
                return y_err.item(), h_err.item()
            else:
                y_err = self._state_estimator(img, orient_alt).flatten()
                y_err *= label_mult
                return y_err.item()
            # print("Got here")
            # print("Y: %f", y_err)
            # print("H: %f", h_err)

    def get_no_h_state(self, img, orient_alt): #TODO: (For Ava) Delete this function. It's used for 1 irrelvant thing
        with torch.no_grad():
            label_mult = 50. 
            y_err = self._state_estimator(img, orient_alt).flatten()
            y_err *= label_mult

        return y_err.item()


if __name__ == "__main__":
    model = AutolandPerceptionModel(resnet_version="50")
    model.load("/home/agchadbo/XPlaneAutolandScenario/models/2024-3-3/best_model_params.pt")
    model.eval()
    vision_driver = XPlaneVisionDriver(model)