import torch
from torch import nn

# Squeeze-and-Excitation (SE) 模块
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        se = x.mean((2, 3))  # 全局平均池化
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se).view(batch_size, channels, 1, 1)
        return x * se  # 将权重应用于输入

# 修改后的 Bottleneck 模块
class Bottleneck(nn.Module):
    extention = 4

    def __init__(self, inplanes, planes, stride, downsample=None, use_se=True, activation='relu', norm_type='bn'):
        '''
        :param inplanes: 输入block的通道数
        :param planes: 在block中间处理的通道数
        :param stride: 步长
        :param downsample: 下采样操作
        :param use_se: 是否使用SE模块
        :param activation: 激活函数类型
        :param norm_type: 归一化类型：'bn'为批量归一化，'gn'为组归一化
        '''
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = self._get_norm_layer(planes, norm_type)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = self._get_norm_layer(planes, norm_type)

        self.conv3 = nn.Conv2d(planes, planes * self.extention, kernel_size=1, stride=1, bias=False)
        self.bn3 = self._get_norm_layer(planes * self.extention, norm_type)

        # 激活函数选择
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'swish':
            self.activation = nn.SiLU()  # Swish 激活函数

        # 判断是否使用 SE 模块
        self.se = SEBlock(planes * self.extention) if use_se else None

        self.downsample = downsample
        self.stride = stride

    def _get_norm_layer(self, planes, norm_type):
        if norm_type == 'bn':
            return nn.BatchNorm2d(planes)
        elif norm_type == 'gn':
            return nn.GroupNorm(32, planes)  # 使用32个组
        else:
            raise ValueError('Unsupported normalization type')

    def forward(self, x):
        residual = x

        # 卷积操作
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 是否使用 SE 模块
        if self.se is not None:
            out = self.se(out)

        # 判断是否需要下采样
        if self.downsample is not None:
            residual = self.downsample(x)

        out = torch.add(out, residual)
        out = self.activation(out)

        return out

# 修改后的 ResNet 模块
class ResNet(nn.Module):
    def __init__(self, block, layers, num_class, use_se=True, activation='relu', norm_type='bn'):
        self.inplane = 64
        super(ResNet, self).__init__()

        # Stem部分
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._get_norm_layer(self.inplane, norm_type)
        self.activation = self._get_activation_layer(activation)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 构建网络层
        self.stage1 = self.make_layer(block, 64, layers[0], stride=1, use_se=use_se, activation=activation, norm_type=norm_type)
        self.stage2 = self.make_layer(block, 128, layers[1], stride=2, use_se=use_se, activation=activation, norm_type=norm_type)
        self.stage3 = self.make_layer(block, 256, layers[2], stride=2, use_se=use_se, activation=activation, norm_type=norm_type)
        self.stage4 = self.make_layer(block, 512, layers[3], stride=2, use_se=use_se, activation=activation, norm_type=norm_type)

        # 后续部分
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.extention, num_class)

    def _get_norm_layer(self, planes, norm_type):
        if norm_type == 'bn':
            return nn.BatchNorm2d(planes)
        elif norm_type == 'gn':
            return nn.GroupNorm(32, planes)  # 使用32个组
        else:
            raise ValueError('Unsupported normalization type')

    def _get_activation_layer(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'swish':
            return nn.SiLU()  # Swish 激活函数

    def forward(self, x):
        # stem部分
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.maxpool(out)

        # block部分
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        # 分类
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def make_layer(self, block, plane, block_num, stride=1, use_se=True, activation='relu', norm_type='bn'):
        block_list = []
        downsample = None
        if stride != 1 or self.inplane != plane * block.extention:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplane, plane * block.extention, stride=stride, kernel_size=1, bias=False),
                self._get_norm_layer(plane * block.extention, norm_type)
            )

        conv_block = block(self.inplane, plane, stride=stride, downsample=downsample, use_se=use_se, activation=activation, norm_type=norm_type)
        block_list.append(conv_block)
        self.inplane = plane * block.extention

        for i in range(1, block_num):
            block_list.append(block(self.inplane, plane, stride=1, use_se=use_se, activation=activation, norm_type=norm_type))
        return nn.Sequential(*block_list)
