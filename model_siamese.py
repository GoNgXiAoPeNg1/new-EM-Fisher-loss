import pdb
import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class MaxPoolingWithArgmax2D(nn.Module):
    def __init__(self, kernel_size=3, stride=2, padding=1):
        super(MaxPoolingWithArgmax2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.maxpool_with_argmax = nn.MaxPool2d(self.kernel_size, self.stride, self.padding, return_indices=True)

    def forward(self, inputs):
        outputs, indices = self.maxpool_with_argmax(inputs)
        return outputs, indices


class MaxUnpooling2D(nn.Module):
    def __init__(self, kernel_size=3, stride=2, padding=0):
        super(MaxUnpooling2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.max_unpool = nn.MaxUnpool2d(self.kernel_size, self.stride, self.padding)

    def forward(self, inputs, indices):
        outputs = self.max_unpool(inputs, indices)
        return outputs


class Den_Resnet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(Den_Resnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool_with_argmax = MaxPoolingWithArgmax2D()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=1)

        self.space_inf = ConvRelu(2048, 1, activate=True)
        # self.roi = nn.AdaptiveMaxPool2d((5, 5))
        self.cor_bn = nn.BatchNorm2d(32 * 32)
        self.cor_space_inf = ConvRelu(1024, 1, activate=True)
        self.cor_space_bn = nn.BatchNorm2d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp1 = nn.Linear(4096, 4096)
        self.mlp2 = nn.Linear(4096, 2048)

        self.deconv1 = nn.ConvTranspose2d(2048, 1024, padding=1, kernel_size=3, stride=1, bias=False)
        self.bn_deconv1 = nn.BatchNorm2d(1024)
        self.relu_deconv1 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(1024, 512, padding=1,output_padding=1, kernel_size=3, stride=2, bias=False)
        self.bn_deconv2 = nn.BatchNorm2d(512)
        self.relu_deconv2 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(512, 256, padding=1,output_padding=1, kernel_size=3, stride=2, bias=False)
        self.bn_deconv3 = nn.BatchNorm2d(256)
        self.relu_deconv3 = nn.ReLU()

        self.deconv4 = nn.ConvTranspose2d(256, 64, padding=1, kernel_size=3, stride=1, bias=False)
        self.bn_deconv4 = nn.BatchNorm2d(64)
        self.relu_deconv4 = nn.ReLU()
        self.max_unpool = MaxUnpooling2D()

        self.deconv5 = nn.ConvTranspose2d(64, 64, padding=1,output_padding=1, kernel_size=3, stride=2, bias=False)
        self.bn_deconv5 = nn.BatchNorm2d(64)
        self.relu_deconv5 = nn.ReLU()

        self.deconv6 = nn.ConvTranspose2d(64, num_classes, padding=2,output_padding=0,kernel_size=7, stride=2, bias=False)
        self.bn_deconv6 = nn.BatchNorm2d(2)
        self.relu_deconv6 = nn.ReLU()

        # initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, input1,input2):

        x1 = self.conv1(input1)
        x2 = self.conv1(input2)
        x1 = self.bn1(x1)
        x2 = self.bn1(x2)
        x1 = self.relu(x1)
        x2 = self.relu(x2)
        # block_1_1, mask1 = self.maxpool_with_argmax(x1)
        # block_1_2, mask2 = self.maxpool_with_argmax(x2)
        # pdb.set_trace()
        block_1_1 = self.maxpool(x1)
        block_1_2 = self.maxpool(x2)

        block_2_1 = self.layer1(block_1_1)
        block_2_2 = self.layer1(block_1_2)

        block_3_1 = self.layer2(block_2_1)
        block_3_2 = self.layer2(block_2_2)

        block_4_1 = self.layer3(block_3_1)
        block_4_2 = self.layer3(block_3_2)

        block_5_1 = self.layer4(block_4_1)
        block_5_2 = self.layer4(block_4_2)
        #------------------------------------------------------------------------

        space_inf_1 = self.space_inf(block_5_1)
        space_inf_2 = self.space_inf(block_5_2)

        # 空间协同 空间上的attentino  空间上加权  sigmod
        cor_1 = []
        cor_2 = []
        for i in range(space_inf_1.shape[0]):
            filter2 = torch.reshape(space_inf_2[i], (space_inf_1.shape[2] * space_inf_1.shape[3], 1, 1, 1))
            cor_1.append(F.conv2d(torch.unsqueeze(space_inf_1[i], dim=0), filter2))

            filter1 = torch.reshape(space_inf_1[i], (space_inf_2.shape[2] * space_inf_2.shape[3], 1, 1, 1))
            cor_2.append(F.conv2d(torch.unsqueeze(space_inf_2[i], dim=0), filter1))

        cor_1 = torch.cat(cor_1, dim=0)
        cor_2 = torch.cat(cor_2, dim=0)

        cor_1 = self.cor_bn(cor_1)
        cor_2 = self.cor_bn(cor_2)
        cor_1 = F.relu(cor_1)
        cor_2 = F.relu(cor_2)

        cor_space_1 = self.cor_space_inf(cor_1)
        cor_space_2 = self.cor_space_inf(cor_2)
        cor_space_1 = self.cor_space_bn(cor_space_1)
        cor_space_2 = self.cor_space_bn(cor_space_2)
        cor_space_1 = F.sigmoid(cor_space_1)
        cor_space_2 = F.sigmoid(cor_space_2)

        # 全局协同+channel加权  即channel上的attention   ponling用自己的方法

        center_1_view = block_5_1.view((-1, 2048, block_5_1.shape[2] * block_5_1.shape[3]))
        center_1_topk, index = torch.topk(center_1_view, int(block_5_1.shape[2] * block_5_1.shape[3] / 1), dim=2)
        feature_1 = torch.mean(center_1_topk, dim=2)

        center_2_view = block_5_2.view((-1, 2048, block_5_2.shape[2] * block_5_2.shape[3]))
        center_2_topk, index = torch.topk(center_2_view, int(block_5_2.shape[2] * block_5_2.shape[3] / 2), dim=2)
        feature_2 = torch.mean(center_2_topk, dim=2)
         
        feature_12 = torch.cat((feature_1, feature_2), dim=1)
        attention_12 = F.sigmoid(self.mlp2(
            F.tanh(self.mlp1(feature_12.view(-1, 4096))))).view(-1, 2048, 1, 1)

        block_5_1 = block_5_1 * attention_12
        block_5_2 = block_5_2 * attention_12
        block_5_1 = block_5_1 * cor_space_1
        block_5_2 = block_5_2 * cor_space_2
        #---------------------------------------------------------------------------
        x_1 = self.deconv1(block_5_1)
        x_2 = self.deconv1(block_5_2)
        x_1 = self.bn_deconv1(x_1)
        x_2 = self.bn_deconv1(x_2)
        x_1 = self.relu_deconv1(x_1)
        x_2 = self.relu_deconv1(x_2)
        x_1 += block_4_1
        x_2 += block_4_2

        x_1 = self.deconv2(x_1)
        x_2 = self.deconv2(x_2)
        x_1 = self.bn_deconv2(x_1)
        x_2 = self.bn_deconv2(x_2)
        x_1 = self.relu_deconv2(x_1)
        x_2 = self.relu_deconv2(x_2)
        x_1 += block_3_1
        x_2 += block_3_2


        x_1 = self.deconv3(x_1)
        x_2 = self.deconv3(x_2)
        x_1 = self.bn_deconv3(x_1)
        x_2 = self.bn_deconv3(x_2)
        x_1 = self.relu_deconv3(x_1)
        x_2 = self.relu_deconv3(x_2)
        x_1 += block_2_1
        x_2 += block_2_2

        x_1 = self.deconv4(x_1)
        x_2 = self.deconv4(x_2)
        x_1 = self.bn_deconv4(x_1)
        x_2 = self.bn_deconv4(x_2)
        x_1 = self.relu_deconv4(x_1)
        x_2 = self.relu_deconv4(x_2)
        x_1 += block_1_1
        x_2 += block_1_2


        x_1 = self.deconv5(x_1)
        x_2 = self.deconv5(x_2)
        x_1 = self.bn_deconv5(x_1)
        x_2 = self.bn_deconv5(x_2)
        x_1 = self.relu_deconv5(x_1)
        x_2 = self.relu_deconv5(x_2)

        x_1 = self.deconv6(x_1)
        x_2 = self.deconv6(x_2)
        x_1 = self.bn_deconv6(x_1)
        x_2 = self.bn_deconv6(x_2)
        x_1 = self.relu_deconv6(x_1)
        x_2 = self.relu_deconv6(x_2)
        x_1 = x_1[:, :, :-1, :-1]
        x_2 = x_2[:, :, :-1, :-1]
       
        return x_1,x_2


def crate_Den_Resnet_model(**kwargs):
    model = Den_Resnet(Bottleneck, [3, 4, 6, 3], **kwargs)
    checkpoint = torch.load('/home/gongxp/mlmr/githubcode/siamase_pytorch/resnet50_origin.pth')
    model.load_state_dict(checkpoint,strict=False)
    return model


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int, activate=True):
        super(ConvRelu, self).__init__()
        self.activate = activate
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        # self.bn = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.activate:
            # x = self.bn(x)
            x = self.activation(x)
        return x



class DecoderBlockResnet(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlockResnet, self).__init__()
        self.in_channels = in_channels

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels, activate=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


