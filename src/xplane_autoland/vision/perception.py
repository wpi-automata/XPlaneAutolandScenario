import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

class AutolandPerceptionModel(nn.Module):
    def __init__(self, resnet_version="50", freeze=True):
        super(AutolandPerceptionModel, self).__init__()
        if str(resnet_version) == "50":
            resnet = resnet50
            weights = ResNet50_Weights.DEFAULT
        elif str(resnet_version) == "18":
            resnet = resnet18
            weights = ResNet18_Weights.DEFAULT
        else:
            raise ValueError(f"Unrecognized resnet version: {resnet_version}")
        self.transform = weights.transforms(antialias=True)
        self.resnet = resnet(weights=weights)
        if freeze:
            # freeze all the layers
            for param in self.resnet.parameters():
                param.requires_grad = False

        # replace the last layer (same number of features but this resets the weights
        # and resets requires_grad to True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.resnet.fc.out_features)
        # going to concatenate the 4d orient_alt vector
        self.fc1 = nn.Linear(self.resnet.fc.out_features+4, 512)
        self.fc_final = nn.Linear(self.fc1.out_features, 2)

    def forward(self, img, orient_alt):
        #Had to remove a tensor dimension as img and orient_alt were each a dimension too large 
        #Not sure if this is a potential source of error- not necessary when running run_autoland.py
        img = torch.squeeze(img,1)
        orient_alt = torch.squeeze(orient_alt,1)

        #Debugger
        #print(img.size())
        conv_out = self.resnet(img)
        #More Debuggers
        #print(conv_out.size())
        #print(orient_alt.size())
        x = torch.cat((conv_out, orient_alt), dim=1)
        x = self.fc1(x)
        return self.fc_final(x)

    def preprocess(self, img):
        return self.transform(img)

    def load(self, params_file):
        self.load_state_dict(torch.load(params_file))