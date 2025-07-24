import torch
from torch import nn
import torch.nn.functional as F
import math
import random

from util.unet import UNet


class WNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=1, depth=5, wf=6, padding=True, batch_norm=True, up_mode='upconv'):
        super(WNet, self).__init__()

        self.unet1 = UNet(in_channels=in_channels, n_classes=n_classes, depth=depth, wf=wf, padding=padding,
                          batch_norm=batch_norm, up_mode=up_mode)

        self.unet2 = UNet(in_channels=in_channels + n_classes, n_classes=n_classes, depth=depth, wf=wf, padding=padding,
                          batch_norm=batch_norm, up_mode=up_mode)

        self.final_activation = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        init_set = {
            nn.Conv2d,
            nn.ConvTranspose2d,
            nn.Linear,
        }
        for m in self.modules():
            if type(m) in init_set:
                nn.init.kaiming_normal_(
                    m.weight.data, mode='fan_out', nonlinearity='relu', a=0
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        output1_logits = self.unet1(input_batch)
        output1_prob = self.final_activation(output1_logits)

        input2 = torch.cat([input_batch, output1_prob], dim=1)

        output2_logits = self.unet2(input2)
        output2_prob = self.final_activation(output2_logits)

        return output1_prob, output2_prob


class SegmentationAugmentation(nn.Module):
    def __init__(
            self, flip=None, offset=None, scale=None, rotate=None, noise=None
    ):
        super(SegmentationAugmentation, self).__init__()

        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise

    def forward(self, input_g, label_g):
        transform_t = self._build2dTransformMatrix()
        transform_t = transform_t.expand(input_g.shape[0], -1, -1)
        transform_t = transform_t.to(input_g.device, torch.float32)
        affine_t = F.affine_grid(transform_t[:, :2],
                                 input_g.size(), align_corners=False)

        augmented_input_g = F.grid_sample(input_g,
                                          affine_t, padding_mode='border',
                                          align_corners=False)
        augmented_label_g = F.grid_sample(label_g.to(torch.float32),
                                          affine_t, padding_mode='border',
                                          align_corners=False)

        if self.noise:
            noise_t = torch.randn_like(augmented_input_g)
            noise_t *= self.noise

            augmented_input_g += noise_t

        return augmented_input_g, augmented_label_g > 0.5

    def _build2dTransformMatrix(self):
        transform_t = torch.eye(3)

        for i in range(2):
            if self.flip:
                if random.random() > 0.5:
                    transform_t[i, i] *= -1

            if self.offset:
                offset_float = self.offset
                random_float = (random.random() * 2 - 1)
                transform_t[2, i] = offset_float * random_float

            if self.scale:
                scale_float = self.scale
                random_float = (random.random() * 2 - 1)
                transform_t[i, i] *= 1.0 + scale_float * random_float

        if self.rotate:
            angle_rad = random.random() * math.pi * 2
            s = math.sin(angle_rad)
            c = math.cos(angle_rad)

            rotation_t = torch.tensor([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]])

            transform_t @= rotation_t

        return transform_t
