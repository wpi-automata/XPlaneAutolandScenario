import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

tensor1 = torch.load('dataWPI_12464/images/image_phi0_theta4_psi4_x142_y-111_h523.pt')
tensor2 = torch.load('image_phi-1_theta-1_psi-1_x109_y-31_h445.pt')
tensor3 = torch.load('image_phi-1_theta-1_psi5_x10491_y-117_h960.pt')
tensor4 = torch.load('image_phi5_theta-5_psi1_x11564_y-7_h1008.pt')

transform = T.ToPILImage()

img1 = transform(tensor1)
img2 = transform(tensor2)
img3 = transform(tensor3)
img4 = transform(tensor4)

img1.show()
img2.show()
img3.show()
img4.show()
