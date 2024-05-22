from src.xplane_autoland.xplane_connect.driver import XPlaneDriver

from src.xplane_autoland.vision.perception import AutolandPerceptionModel
import mss
from PIL import Image
import torch
from torchvision import transforms

to_tensor = transforms.PILToTensor()
to_img = transforms.ToPILImage(mode="RGB")


class XPlaneVisionDriver(XPlaneDriver):
    """
    This class estimates x and y state variables from an image
    """

    def __init__(
        self,
        state_estimator,
        home_heading=53.7,
        local_start=(-35285.421875, 40957.0234375),
        start_ground_range=12464,
        start_elev=1029.45,
        t=(-25159.26953, 33689.8125),
    ):
        super().__init__(home_heading, local_start, start_ground_range, start_elev, t)
        self._state_estimator = state_estimator
        self._orient_norm_divisor = torch.FloatTensor([180.0, 180.0, 180.0, 2000.0])

        # Need to stop and take image
        self._sct = mss.mss()

    def est_pos_state(
        self, img, orient_alt
    ):  # Refactor to take in image and orientation data for error data generation
        """
        Estimates the position state using a vision network

        Returns:
            y -- lateral deviation
            err_h -- error in height relative to a glideslope
        """
        # Take out pre-processing and image generation (already done)

        # sct_img = self._sct.grab(self._sct.monitors[1])
        # pil_img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        # img = to_tensor(pil_img)
        # img = self._state_estimator.preprocess(img)
        # phi, theta, psi = self.est_orient_state()
        # _, _, h  = self.get_pos_state()
        # orient_alt = torch.FloatTensor([phi, theta, psi, h])
        # orient_alt /= self._orient_norm_divisor
        # img, orient_alt = img[0, :, :, :], orient_alt[0, :]
        # print(img.size())
        # img1 = to_img(img)
        # img1.show()

        with torch.no_grad():
            label_mult = 150.0  # must match the normalization used in AutolandImageDataset when training network
            y_err, h_err = self._state_estimator(img, orient_alt).flatten()
            y_err *= label_mult
            h_err *= label_mult
            # Debug statements
            # print("Got here")
            # print("Y: %f", y_err)

        return y_err.item(), h_err.item()


if __name__ == "__main__":
    model = AutolandPerceptionModel(resnet_version="50")
    model.load(
        "/home/agchadbo/XPlaneAutolandScenario/models/2023-12-6/best_model_params.pt"
    )
    model.eval()
    vision_driver = XPlaneVisionDriver(model)
