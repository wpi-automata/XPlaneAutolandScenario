import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

tensor1 = torch.load('image_phi0_theta0_psi0_x1_y-8_h411.pt')
tensor2 = torch.load('image_phi0_theta0_psi0_x1_y-8_h411WPI.pt')
transform = T.ToPILImage()

img1 = transform(tensor1)
img2 = transform(tensor2)

img1.show()
img2.show()