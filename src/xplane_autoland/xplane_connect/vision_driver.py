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
                 t=(-25159.26953, 33689.8125)):
        super().__init__(home_heading, local_start, start_ground_range,
                         start_elev, t)
        self._state_estimator = state_estimator
        self._orient_norm_divisor = torch.FloatTensor([180., 180., 180., 2000.])

        #(No) need to stop and take image
        #self._sct = mss.mss()

    def est_pos_state(self, im, orient):
        #Commented out unnecessary lines
        # sct_img = self._sct.grab(self._sct.monitors[1])
        # pil_img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        # img = to_tensor(img)
        #Debugger
        #print(f"Got here")

        img = self._state_estimator.preprocess(im)

        # phi, theta, psi = self.est_orient_state()
        # _, _, h  = self.get_pos_state()
        #orient_alt = torch.FloatTensor([phi, theta, psi, h])

        orient_alt = orient

        #orient_alt /= self._orient_norm_divisor
        
        img, orient_alt = img[None, :, :, :], orient_alt[None, :]

        with torch.no_grad():
            pred_x, pred_y = self._state_estimator(img, orient_alt).flatten()
            pred_x *= 1000
            pred_y *= 1000
            pred_x = pred_x.item()
            pred_y = pred_y.item()
            print(pred_x)
            print(pred_y)
        return pred_x, pred_y


if __name__ == "__main__":
    model = AutolandPerceptionModel(resnet_version="50")
    model.load("/home/XPlaneAutolandScenario_ColetteCopy/models/vision/2023-8-10/best_model_params.pt")
    model.eval()
    vision_driver = XPlaneVisionDriver(model)