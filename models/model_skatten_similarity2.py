from collections import OrderedDict
from functools import reduce
from torch.nn import functional as F
import torch
from torch import nn
from thop import profile
from torchstat import stat
import sys
sys.path.append("/root/autodl-tmp/msfe/")


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate, growth_rate, bn_size)
            self.add_module('denselayer%d' % (i + 1), layer)


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module(
            'conv1',
            nn.Conv2d(
                num_input_features,
                bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                bias=False
            )
        ),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module(
            'conv2',
            nn.Conv2d(
                bn_size * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )
        )

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False
            )
        )
        # self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # grouping, 通道分组
    # b, num_channels, h, w =======>  b, groups, channels_per_group, h, w
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # channel shuffle, 通道洗牌
    x = torch.transpose(x, 1, 2).contiguous()
    # x.shape=(batchsize, channels_per_group, groups, height, width)
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class atten(nn.Module):
    def __init__(self, in_channels, out_channels,M=3,r=16,L=32):
        super(atten, self).__init__()
        d = max(in_channels // r, L)  # 计算向量Z 的长度d
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList()  # 根据分支数量 添加 不同核的卷积操作
        for i in range(M):
            # 为提高效率，原论文中 扩张卷积5x5为 （3X3，dilation=2）来代替。 且论文中建议组卷积G=32
            self.conv.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1 + i, dilation=1 + i,
                          bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 自适应pool到指定维度    这里指定为1，实现 GAP
        self.fc1 = nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False),
                                 nn.BatchNorm2d(d),
                                 nn.ReLU(inplace=True))  # 降维
        self.fc2 = nn.Conv2d(d, out_channels * (M+1), 1, 1, bias=False)  # 升维
        self.softmax = nn.Softmax(dim=1)  # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1

    def forward(self, input):
        batch_size=input.size(0)
        output=[]
        #the part of split
        for i,conv in enumerate(self.conv):
            output.append(conv(input))
        output.append(input)
        #the part of fusion
        U=reduce(lambda x,y:x+y,output) # 逐元素相加生成 混合特征U
        s=self.global_pool(U)
        z=self.fc1(s)  # S->Z降维
        a_b=self.fc2(z) # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b
        a_b=a_b.reshape(batch_size,self.M+1,self.out_channels,-1) #调整形状，变为 两个全连接层的值
        a_b=self.softmax(a_b) # 使得两个全连接层对应位置进行softmax
        #the part of selection
        a_b=list(a_b.chunk(self.M+1,dim=1))#split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) # 将所有分块  调整形状，即扩展两维
        V=list(map(lambda x,y:x*y,output,a_b)) # 权重与对应  不同卷积核输出的U 逐元素相乘
        V = reduce(lambda x, y: x + y, V)  # 两个加权后的特征 逐元素相加

        return V


class group_atten(nn.Module):
    def __init__(self, in_channels, groups=8):
        super(group_atten, self).__init__()
        self.group_channel = in_channels // groups
        self.groups = groups
        self.casm_list = nn.ModuleList([atten(self.group_channel, self.group_channel) for i in range(groups)])

    def forward(self, x):
        # channel split
        output = torch.split(x, self.group_channel, dim=1)
        out = []
        for i in range(self.groups):
            output_i = output[i]
            output_i = self.casm_list[i](output_i)
            avg_out = torch.mean(output_i, dim=1, keepdim=True)
            max_out, _ = torch.max(output_i, dim=1, keepdim=True)
            out.append(avg_out)
            out.append(max_out)
        out = torch.cat(out, dim=1)
        out = channel_shuffle(out, self.groups)

        return out


class CLA(nn.Module):
    def __init__(self, low_channels, high_channels, groups=8, K=0.25):
        super(CLA, self).__init__()

        self.group_atten1 = group_atten(low_channels, groups)
        self.group_atten2 = group_atten(high_channels, groups)

        self.K = K

        self.conv = nn.Sequential(
            nn.Conv2d(4 * groups, high_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(high_channels)
        )

    def forward(self, x_low, x_high):
        low1 = self.group_atten1(x_low)
        high1 = self.group_atten2(x_high)

        N, C, H1, W1 = low1.shape
        _, _, H2, W2 = high1.shape

        select_number = int(self.K * H1 * W1)

        low1_flatten = low1.view(N, C, -1)
        high1_transpose = torch.transpose(high1.view(N, C, -1), 1, 2).contiguous()

        similarity = torch.matmul(high1_transpose, low1_flatten)
        similarity = torch.softmax(similarity, dim=2)

        topk_index = torch.topk(similarity, k=select_number, dim=2)[1]
        topk_similarity = torch.topk(similarity, k=select_number, dim=2)[0]

        result = []
        for batch in range(N):
            batch_index = topk_index[batch, :, :]
            batch_similarity = topk_similarity[batch, :, :]
            batch_value = low1_flatten[batch, :, :][:, batch_index]
            out = torch.sum(batch_value * batch_similarity, dim=2)
            out = out.reshape(C, H2, W2)

            out = out.unsqueeze(dim=0)
            result.append(out)
        result = torch.cat(result, dim=0)
        result = torch.cat([result, high1], dim=1)

        x_resize = self.conv(result)
        x_resize = x_resize + x_high
        x_resize = F.relu(x_resize, inplace=True)

        return x_resize


class multi_aggre(nn.Module):
    def __init__(self):
        super(multi_aggre, self).__init__()

        self.cla_1 = CLA(256, 512, groups=32, K=0.5)
        self.cla_2 = CLA(512, 1024, groups=32, K=0.5)
        self.cla_3 = CLA(512, 1024, groups=32, K=0.5)


    def forward(self, three_stage, four_stage, five_stage):
        x2_1 = self.cla_1(three_stage, four_stage)
        x2_2 = self.cla_2(four_stage, five_stage)

        x_out = self.cla_3(x2_1, x2_2)
        return x_out


class DenseNet_skatten_similarity2(nn.Module):
    def __init__(
            self,
            bn_size=4,
            growth_rate=32,
            block_config=(6, 12, 24, 16),
            num_init_features=64):

        super(DenseNet_skatten_similarity2, self).__init__()
        self.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        'conv0',
                        nn.Conv2d(
                            3,
                            num_init_features,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False
                        )
                    ),
                    ('norm0', nn.BatchNorm2d(num_init_features)),
                    ('relu0', nn.ReLU(inplace=True)),
                    (
                        'pool0',
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                    ),
                ]
            )
        )

        num_features = num_init_features  # 64
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2
                )
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.bn1_3 = nn.BatchNorm2d(256)
        self.bn1_4 = nn.BatchNorm2d(512)
        self.bn1_5 = nn.BatchNorm2d(1024)

        self.multi_aggre = multi_aggre()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.35),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x1_1 = self.features.conv0(x)
        x1_1 = self.features.norm0(x1_1)
        x1_1 = self.features.relu0(x1_1)
        x1_1 = self.features.pool0(x1_1)
        # print(x1_1.shape)

        x1_2 = self.features.denseblock1(x1_1)
        x1_2 = self.features.transition1(x1_2)
        x12 = F.avg_pool2d(x1_2, kernel_size=2, stride=2)
        # print(x1_2.shape)

        # 1024
        x1_3 = self.features.denseblock2(x12)
        x1_3 = self.features.transition2(x1_3)
        x13 = F.avg_pool2d(x1_3, kernel_size=2, stride=2)
        # print(x1_3.shape)

        # 1024
        x1_4 = self.features.denseblock3(x13)
        x1_4 = self.features.transition3(x1_4)
        x14 = F.avg_pool2d(x1_4, kernel_size=2, stride=2)
        # print(x1_4.shape)

        # 1024
        x1_5 = self.features.denseblock4(x14)

        x2_3 = self.bn1_3(x1_3)
        x2_3 = F.relu(x2_3, inplace=True)

        x2_4 = self.bn1_4(x1_4)
        x2_4 = F.relu(x2_4, inplace=True)

        x2_5 = self.bn1_5(x1_5)
        x2_5 = F.relu(x2_5, inplace=True)


        x_out = self.multi_aggre(x2_3, x2_4, x2_5)
        # print(x_out.shape)
        f = F.adaptive_avg_pool2d(x_out, (1, 1))
        v = f.view(f.size(0), -1)
        out = self.classifier(v)

        return out


if __name__ == '__main__':
    model = DenseNet_skatten_similarity2()
    # model_weight_path = '/root/autodl-tmp/models_dense1024_nocla/dense121_1024_nocla_model29_auc_0.8988.pth'
    # model_state = torch.load(model_weight_path)
    # model.load_state_dict(model_state['model'])
    # x = torch.randn(1, 3, 1024, 768)
    # flops, params = profile(model, (x,))
    # print('flops: ', flops, 'params: ', params)

    stat(model, (3, 1024, 768))









