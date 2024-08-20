import mlx.nn as nn
import mlx.core as mx
import math


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm(in_planes)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            padding=1,
            bias=False,
        )

    def __call__(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        return mx.concatenate([out, x], -1)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm(in_planes)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_planes,
            inter_planes,
            kernel_size=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm(inter_planes)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            inter_planes,
            out_planes,
            kernel_size=3,
            padding=1,
            bias=False,
        )

    def __call__(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        return mx.concatenate([out, x], -1)


class TransitionLayer(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(TransitionLayer, self).__init__()
        self.bn1 = nn.BatchNorm(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            stride=1,
            kernel_size=1,
            bias=False,
        )
        self.relu1 = nn.ReLU()
        self.avgpool = nn.AvgPool2d(2)

    def __call__(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        return self.avgpool(out)


class DenseNet(nn.Module):
    def __init__(
        self,
        block,
        nblocks,
        num_classes,
        growth_rate,
        reduction,
    ):
        super(DenseNet, self).__init__()
        in_planes = 2 * growth_rate
        self.growth_rate = growth_rate

        self.conv1 = nn.Conv2d(
            3,
            in_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.dense1 = self._make_dense_layer(
            block,
            in_planes,
            nblocks[0],
        )
        in_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(in_planes * reduction))
        self.trans1 = TransitionLayer(in_planes, out_planes)  # downsampling

        in_planes = out_planes
        self.dense2 = self._make_dense_layer(
            block,
            in_planes,
            nblocks[1],
        )
        in_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(in_planes * reduction))
        self.trans2 = TransitionLayer(in_planes, out_planes)  # downsampling

        in_planes = out_planes
        self.dense3 = self._make_dense_layer(
            block,
            in_planes,
            nblocks[2],
        )
        in_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(in_planes * reduction))
        self.trans3 = TransitionLayer(in_planes, out_planes)  # downsampling

        in_planes = out_planes
        self.dense4 = self._make_dense_layer(
            block,
            in_planes,
            nblocks[3],
        )
        in_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm(in_planes)
        self.avg = nn.AvgPool2d(4)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_planes, num_classes)

    def _make_dense_layer(self, block_class, in_planes, nblock):
        layers = []
        for _ in range(nblock):
            layers.append(block_class(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def __call__(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = self.relu(self.bn(out))
        out = self.avg(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)
        return out
