import sys
sys.path.append('/home/colette/XPlaneAutolandScenario/src/xplane_autoland')
from xplane_connect.driver import XPlaneDriver

from vision.perception import AutolandPerceptionModel
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
                 t=(-25159.26953, 33689.8125)):
        super().__init__(home_heading, local_start, start_ground_range,
                         start_elev, t)
        self._state_estimator = state_estimator
        self._orient_norm_divisor = torch.FloatTensor([180., 180., 180., 2000.])

        # Need to stop and take image
        self._sct = mss.mss()

    def est_pos_state(self, monitor):
        """
        Estimates the position state using a vision network

        Returns:
            y -- lateral deviation
            err_h -- error in height relative to a glideslope
        """
        sct_img = self._sct.grab(self._sct.monitors[monitor]) 
        pil_img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        img = to_tensor(pil_img)
        img = self._state_estimator.preprocess(img)
        phi, theta, psi = self.est_orient_state()
        _, _, h  = self.get_pos_state()
        orient_alt = torch.FloatTensor([phi, theta, psi, h])
        orient_alt /= self._orient_norm_divisor
        img, orient_alt = img[None, :, :, :], orient_alt[None, :]

        with torch.no_grad():
            label_mult = 150. # must match the normalization used in AutolandImageDataset when training network
            y_err, h_err = self._state_estimator(img, orient_alt).flatten()
            y_err *= label_mult
            h_err *= label_mult
            print("Got here")
            print("Y: %f", y_err)
            print("H: %f", h_err)

        return y_err.item(), h_err.item()


if __name__ == "__main__":
    model = AutolandPerceptionModel(resnet_version="50")
    model.load("/home/colette/XPlaneAutolandScenario/models/2024-3-3/best_model_params.pt")
    model.eval()
    vision_driver = XPlaneVisionDriver(model)