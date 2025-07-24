import math
from torch import nn as nn
from util.logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.WARN)
log.setLevel(logging.INFO)


class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batchnorm = nn.BatchNorm3d(1)

        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8)

        # 输出头优化
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))  # 替代 view
        self.dropout = nn.Dropout(p=0.3)
        self.head_linear = nn.Linear(conv_channels * 8, 2)
        self.head_bn = nn.BatchNorm1d(2)
        self.relu = nn.ReLU(inplace=True)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if  m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batchnorm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)

        x = self.global_pool(block_out).view(block_out.size(0), -1)
        x = self.dropout(x)
        x = self.head_linear(x)
        x = self.head_bn(x)
        #x = self.relu(x)

        return x, self.head_softmax(x)

class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels, conv_channels, kernel_size=3, padding=1, bias=True,
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            conv_channels, conv_channels, kernel_size=3, padding=1, bias=True,
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(2, 2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.maxpool(block_out)

# 模块中追加一个测试类，方便模块功能测试
class modelCheck:
    def __init__(self,arg):
        self.arg = arg
        log.info("init {}".format(type(self).__name__))
    def main(self):
        model = LunaModel()  # 实例化模型
        # 统计参数总数
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型总参数: {total_params:,}")
        # 查看各层参数形状
        for name, param in model.named_parameters():
            print(f"{name}: 形状={param.shape}, 可训练={param.requires_grad}")

if __name__ == "__main__":
    checkmodel = modelCheck('参数检查').main()
