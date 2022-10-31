import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_norm(x: torch.Tensor, module: nn.modules.batchnorm) -> torch.Tensor:
    return F.batch_norm(input=x,
                        running_mean=module.running_mean,
                        running_var=module.running_var,
                        weight=module.weight,
                        bias=module.bias,
                        training=module.training,
                        momentum=module.momentum,
                        eps=module.eps)


class BottleNeck2d(nn.Module):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 spatial_stride: int = 1) -> None:
        super(BottleNeck2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               padding=1,
                               stride=spatial_stride,
                               bias=False)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        if inplanes != planes * 4 or spatial_stride != 1:
            downsample = [
                nn.Conv2d(inplanes,
                          planes * 4,
                          kernel_size=1,
                          stride=spatial_stride,
                          bias=False),
                nn.BatchNorm2d(planes * 4)
            ]
            self.downsample = nn.Sequential(*downsample)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out).relu_()

        out = self.conv2(out)
        out = self.bn2(out).relu_()

        out = self.conv3(out)
        out = self.bn3(out)

        if hasattr(self, 'downsample'):
            x = self.downsample(x)

        return out.add_(x).relu_()


class BottleNeck3d(nn.Module):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 spatial_stride: int = 1) -> None:
        super(BottleNeck3d, self).__init__()

        self.conv1 = nn.Conv3d(inplanes,
                               planes,
                               kernel_size=(3, 1, 1),
                               padding=(1, 0, 0),
                               bias=False)
        self.conv2 = nn.Conv3d(planes,
                               planes,
                               stride=(1, spatial_stride, spatial_stride),
                               kernel_size=(1, 3, 3),
                               padding=(0, 1, 1),
                               bias=False)

        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm3d(planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.bn3 = nn.BatchNorm3d(planes * 4)

        if inplanes != planes * 4 or spatial_stride != 1:
            downsample = [
                nn.Conv3d(inplanes,
                          planes * 4,
                          kernel_size=1,
                          stride=(1, spatial_stride, spatial_stride),
                          bias=False),
                nn.BatchNorm3d(planes * 4)
            ]
            self.downsample = nn.Sequential(*downsample)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 4:
            return self.forward_2d(x)

        out = self.conv1(x)
        out = self.bn1(out).relu_()

        out = self.conv2(out)
        out = self.bn2(out).relu_()

        out = self.conv3(out)
        out = self.bn3(out)

        if hasattr(self, 'downsample'):
            x = self.downsample(x)

        return out.add_(x).relu_()

    def forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        out = F.conv2d(x, self.conv1.weight.sum(2))
        out = batch_norm(out, self.bn1).relu_()

        out = F.conv2d(out,
                       self.conv2.weight.squeeze(2),
                       stride=self.conv2.stride[-1],
                       padding=1)
        out = batch_norm(out, self.bn2).relu_()

        out = F.conv2d(out, self.conv3.weight.squeeze(2))
        out = batch_norm(out, self.bn3)

        if hasattr(self, 'downsample'):
            x = F.conv2d(x,
                         self.downsample[0].weight.squeeze(2),
                         stride=self.downsample[0].stride[-1])
            x = batch_norm(x, self.downsample[1])

        return out.add_(x).relu_()


class I3DResNet(nn.Module):
    def __init__(self,
                 layers: list = [3, 4, 6, 3],
                 image_classes: int = 1000,
                 video_classes: int = 400,
                 pretrain: str = 'resnet50-11ad3fa6.pth') -> None:
        super(I3DResNet, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3,
                               self.inplanes,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.pool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(BottleNeck2d, 64, layers[0])
        self.layer2 = self._make_layer(BottleNeck2d, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BottleNeck3d, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BottleNeck3d, 512, layers[3], stride=2)

        self.fc2d = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.BatchNorm1d(2048), nn.Dropout(0.2),
                                  nn.Linear(2048, image_classes))

        self.fc3d = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                                  nn.Linear(2048, video_classes * 2),
                                  nn.BatchNorm1d(video_classes * 2),
                                  nn.ReLU(inplace=True), nn.Dropout(0.2),
                                  nn.Linear(video_classes * 2, video_classes))

        if pretrain is not None:
            self.init_from_2d(pretrain)

    def _make_layer(self,
                    block: nn.Module,
                    planes: int,
                    num_blocks: int,
                    stride: int = 1) -> nn.Module:
        layers = [block(self.inplanes, planes, spatial_stride=stride)]
        self.inplanes = planes * 4
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def init_from_2d(self, pretrain: str) -> None:
        param2d = torch.load(pretrain, 'cpu')
        param3d = self.state_dict()
        for key in param3d:
            if key in param2d:
                weight = param2d[key]
                if len(weight.shape) == 4 and len(param3d[key].shape) == 5:
                    t = param3d[key].shape[2]
                    weight = weight.unsqueeze(2)
                    weight = weight.expand(-1, -1, t, -1, -1)
                    weight = weight / t
                param3d[key] = weight
            else:
                print('Missing key: %s' % key)
        self.load_state_dict(param3d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flag_3d = False
        if len(x.shape) == 5:
            flag_3d = True
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4)
            x = x.reshape(-1, C, H, W)
            x = x.contiguous()

        x = self.conv1(x)
        x = self.bn1(x).relu_()
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        if flag_3d is True:
            x = x.reshape(B, T, -1, H // 8, W // 8)
            x = x.permute(0, 2, 1, 3, 4)
            x = x.contiguous()

        x = self.layer3(x)
        x = self.layer4(x)

        x = self.fc3d(x) if flag_3d else self.fc2d(x)
        return x


if __name__ == '__main__':
    model = I3DResNet(pretrain=None)
    image = torch.randn(2, 3, 224, 224)
    video = torch.randn(2, 3, 8, 224, 224)
    print('Image output shape', model(image).shape)
    print('Video output shape', model(video).shape)
