import torch
from torch import nn
import torchvision
from torchvision import transforms as tr

class ResNet50Energy(nn.Module):
    def __init__(self):
        super(ResNet50Energy, self).__init__()
        self.model = torchvision.models.resnet50()
        self.model.load_state_dict(torch.load("./rn50.pt", map_location='cpu'))
        self.linear = self.model.fc
        self.avg_pool2d = self.model.avgpool
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.normalizer = tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.last_dim = 2048

    def forward(self, x, vx=None):
        x = self.normalizer(x)
        out = self.model(x)
        out = self.avg_pool2d(out)
        return out.view(out.size(0), -1)


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50()
        self.model.load_state_dict(torch.load("./rn50.pt", map_location='cpu'))
        self.avg_pool2d = self.model.avgpool
        self.normalizer = tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.dummy = torch.nn.Parameter(torch.tensor([0.]), requires_grad=False)
        print('loaded resnet50')

    def device(self):
        return self.dummy.device

    def forward(self, x, vx=None):
        x = x.to(device=self.dummy.device, dtype=self.dummy.dtype)
        x = self.normalizer(x)
        out = self.model(x)

        return out

class WideResNet502(nn.Module):
    def __init__(self):
        super(WideResNet502, self).__init__()
        import torchvision.models as models
        self.model = models.wide_resnet50_2(weights=torchvision.models.Wide_ResNet50_2_Weights.DEFAULT)
        self.normalizer = tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.dummy = torch.nn.Parameter(torch.tensor([0.]), requires_grad=False)
        print('loaded wideresnet')

    def device(self):
        return self.dummy.device

    def forward(self, x, vx=None):
        x = x.to(device=self.dummy.device, dtype=self.dummy.dtype)
        x = self.normalizer(x)
        out = self.model(x)

        return out

class WideResNet1012(nn.Module):
    def __init__(self):
        super(WideResNet1012, self).__init__()
        import torchvision.models as models
        self.model = models.wide_resnet101_2(weights=torchvision.models.Wide_ResNet101_2_Weights.DEFAULT)
        self.normalizer = tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.dummy = torch.nn.Parameter(torch.tensor([0.]), requires_grad=False)
        print('loaded wideresnet')

    def device(self):
        return self.dummy.device

    def forward(self, x, vx=None):
        x = x.to(device=self.dummy.device, dtype=self.dummy.dtype)
        x = self.normalizer(x)
        out = self.model(x)

        return out

