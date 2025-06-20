import torch
import torch.nn as nn
from spikingjelly.activation_based.neuron import ParametricLIFNode
from spikingjelly.activation_based import layer


def conv1d(in_channels, out_channels, kernel_size=3):
    return nn.Sequential(
        layer.SeqToANNContainer(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, stride=1, bias=False),
            nn.BatchNorm1d(out_channels),
        ),
        ParametricLIFNode(init_tau=2.0, detach_reset=True, step_mode='m')
    )


class SEWBlock1D(nn.Module):
    def __init__(self, in_channels, mid_channels, connect_f=None):
        super(SEWBlock1D, self).__init__()
        self.connect_f = connect_f
        self.conv = nn.Sequential(
            conv1d(in_channels, mid_channels),
            conv1d(mid_channels, in_channels),
        )

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        if self.connect_f == 'ADD':
            out = out + x
        elif self.connect_f == 'AND':
            out = out * x
        elif self.connect_f == 'IAND':
            out = x * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)
        return out


class PlainBlock1D(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(PlainBlock1D, self).__init__()
        self.conv = nn.Sequential(
            conv1d(in_channels, mid_channels),
            conv1d(mid_channels, in_channels),
        )

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(BasicBlock1D, self).__init__()
        self.conv = nn.Sequential(
            conv1d(in_channels, mid_channels),
            layer.SeqToANNContainer(
                nn.Conv1d(mid_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm1d(in_channels, track_running_stats=False),
            ),
        )
        self.sn = ParametricLIFNode(init_tau=2.0, detach_reset=True, step_mode='m')
        
    def forward(self, x: torch.Tensor):
        return self.sn(x + self.conv(x))


class ResNet1D(nn.Module):
    def __init__(self, layer_list, num_classes, connect_f=None):
        super(ResNet1D, self).__init__()
        in_channels = 1  # Single channel for audio
        conv = []

        for cfg_dict in layer_list:
            channels = cfg_dict['channels']
            mid_channels = cfg_dict.get('mid_channels', channels)

            if in_channels != channels:
                conv.append(conv1d(in_channels, channels))

            in_channels = channels

            if 'num_blocks' in cfg_dict:
                num_blocks = cfg_dict['num_blocks']
                block_type = cfg_dict.get('block_type', 'sew')
                
                for _ in range(num_blocks):
                    if block_type == 'sew':
                        conv.append(SEWBlock1D(in_channels, mid_channels, connect_f))
                    elif block_type == 'plain':
                        conv.append(PlainBlock1D(in_channels, mid_channels))
                    elif block_type == 'basic':
                        conv.append(BasicBlock1D(in_channels, mid_channels))
                    else:
                        raise NotImplementedError(f"Block type {block_type} not implemented")

            if 'k_pool' in cfg_dict:
                k_pool = cfg_dict['k_pool']
                conv.append(layer.SeqToANNContainer(nn.MaxPool1d(k_pool, k_pool)))

        conv.append(nn.Flatten(2))

        self.conv = nn.Sequential(*conv)

        with torch.no_grad():
            x = torch.zeros([1, 1, 16000])  # Standard audio length
            for m in self.conv.modules():
                if isinstance(m, nn.MaxPool1d):
                    x = m(x)
            out_features = x.numel() * in_channels

        self.out = nn.Linear(out_features, num_classes, bias=True)

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2, 3)  # [T, N, 1, L]
        x = self.conv(x)
        return self.out(x.mean(0))


def SEWResNet1D(connect_f):
    layer_list = [
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
    ]
    num_classes = 35  # Number of speech commands in the dataset
    return ResNet1D(layer_list, num_classes, connect_f)


def PlainNet1D():
    layer_list = [
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'plain', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'plain', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'plain', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'plain', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'plain', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'plain', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'plain', 'k_pool': 2},
    ]
    num_classes = 35
    return ResNet1D(layer_list, num_classes)


def SpikingResNet1D():
    layer_list = [
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 32, 'up_kernel_size': 1, 'mid_channels': 32, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
    ]
    num_classes = 35
    return ResNet1D(layer_list, num_classes) 