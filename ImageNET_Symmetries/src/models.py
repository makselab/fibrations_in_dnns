from torch.nn import Module, ModuleList, Sequential
from torch.nn import Conv2d, BatchNorm2d, AdaptiveAvgPool2d, MaxPool2d
from torch.nn import Linear, ReLU, Identity
from torch.nn.init import kaiming_normal_, constant_, kaiming_uniform_
from torch import flatten, stack

from coloring.coloring import fibration_conv2d, fibration_linear, fibration_pooling_ch, opfibration_conv2d, opfibration_linear, opfibration_pooling_ch

# ===============================================================
# ===============================================================

import torch.nn as nn


class DenseLayer(Module):
    def __init__(self, in_features, out_features, linear=False):
        super(DenseLayer, self).__init__()

        self.fc = nn.Linear(in_features, out_features, bias=True)
        self.act_layer = Identity() if linear else ReLU()
        name_act = 'linear' if linear else 'relu' 
        self.fc.bias.data.fill_(0.0)
        kaiming_uniform_(self.fc.weight, nonlinearity=name_act)

    def forward(self, x):
        return self.act_layer(self.fc(x))

class MLP(Module):
    def __init__(self, in_dim, num_features=[2000,2000,2000], num_outputs=10):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.num_hidden_layers = len(num_features)

        #self.layers_to_log = [-(i * 2 + 1) for i in range(num_hidden_layers + 1)]

        self.layers = nn.ModuleList()
        self.layers.append(DenseLayer(in_features=in_dim, out_features=num_features[0], linear=False))
        for i in range(self.num_hidden_layers - 1):
            self.layers.append(DenseLayer(in_features=num_features[i], out_features=num_features[i+1], linear=False))
        self.layers.append(DenseLayer(in_features=num_features[-1], out_features=num_outputs, linear=True))

        self.act_type = 'relu'

        self.fibration_colors = None
        self.opfibration_colors = None
        self.covering_colors = None

    def forward(self, x):
        feature_list = []
        out = self.layers[0](x)
        feature_list.append(out)

        for idx_l in range(1,self.num_hidden_layers):
            out = self.layers[idx_l](out)
            feature_list.append(out)

        out = self.layers[self.num_hidden_layers](out)
        return out, feature_list

    def fibration_coloring(self, threshold=0.8, bias=False):

        coloring = []

        # First Layer ---------------------------------------
        term_bias = self.layers[0].fc.bias.data if bias else None
        
        clusters = fibration_linear(weights=self.layers[0].fc.weight.data, 
                                    in_clusters=None, 
                                    threshold=threshold, 
                                    first_layer = True,
                                    bias = term_bias)

        coloring.append(clusters)

        # Second Layer ---------------------------------------
        term_bias = self.layers[1].fc.bias.data if bias else None

        clusters = fibration_linear(weights=self.layers[1].fc.weight.data, 
                                    in_clusters=clusters, 
                                    threshold=threshold, 
                                    first_layer = False,
                                    bias = term_bias)

        coloring.append(clusters)

        # Third Layer ---------------------------------------
        term_bias = self.layers[2].fc.bias.data if bias else None
        clusters = fibration_linear(weights=self.layers[2].fc.weight.data, 
                                    in_clusters=clusters, 
                                    threshold=threshold, 
                                    first_layer = False,
                                    bias = term_bias)

        coloring.append(clusters)

        self.fibration_colors = coloring


    def opfibration_coloring(self, threshold=0.5):

        coloring = [None, None, None]

        # 3th Layer ---------------------------------------
        clusters = opfibration_linear(weights=self.layers[3].fc.weight.data, 
                            out_clusters=None, 
                            threshold=threshold, 
                            last_layer = True)

        fibers_vs_time[2].append(clusters)

        # 2th Layer ---------------------------------------
        clusters = opfibration_linear(weights=self.layers[2].fc.weight.data, 
                            out_clusters=clusters, 
                            threshold=threshold, 
                            last_layer = False)

        fibers_vs_time[1].append(clusters)

        # 1st Layer ---------------------------------------
        clusters = opfibration_linear(weights=self.layers[1].fc.weight.data, 
                            out_clusters=clusters, 
                            threshold=threshold, 
                            last_layer = False)

        fibers_vs_time[0].append(clusters)

        self.opfibration_colors = coloring

    def covering_coloring(self, fib_thr=0.8, op_thr = 0.5, bias=False):
        self.fibration_coloring(fib_thr, bias)
        self.opfibration_coloring(op_thr)

        self.covering_colors = []

        for ii in range(3):
            self.covering_colors.append(stack((self.fibration_colors[ii], self.opfibration_colors[ii]), dim=1))

# ===============================================================
# ===============================================================

# 0, 2, 4: Conv2d (stride=1 , padd=0)
# 6, 8, 10: Linear
# 1, 3, 5, 7, 9: ReLU 

class ConvNet(Module):
    def __init__(self, num_classes=2):
        """
        Convolutional Neural Network with 3 convolutional layers followed by 3 fully connected layers
        """
        super().__init__()
        self.last_filter_output = 2 * 2
        self.num_conv_outputs = 128 * self.last_filter_output
        self.pool = MaxPool2d(2, 2)

        self.layers = ModuleList()
        self.layers.append(Conv2d(3, 32, 5))
        self.layers.append(ReLU())
        self.layers.append(Conv2d(32, 64, 3))
        self.layers.append(ReLU())
        self.layers.append(Conv2d(64, 128, 3))
        self.layers.append(ReLU())
        self.layers.append(Linear(self.num_conv_outputs, 128))
        self.layers.append(ReLU())
        self.layers.append(Linear(128, 128))
        self.layers.append(ReLU())
        self.layers.append(Linear(128, num_classes))

        self.act_type = 'relu'

        self.deltas = {}
        self.fibration_colors = None
        self.opfibration_colors = None
        self.covering_colors = None

    def forward(self, x):
        x1 = self.pool(self.layers[1](self.layers[0](x)))
        x1.register_hook(lambda grad: self.store_delta(0, grad))
        
        x2 = self.pool(self.layers[3](self.layers[2](x1)))
        x2.register_hook(lambda grad: self.store_delta(1, grad))
        
        x3 = self.pool(self.layers[5](self.layers[4](x2)))
        x3 = x3.view(-1, self.num_conv_outputs)
        x3.register_hook(lambda grad: self.store_delta(2, grad))
        
        x4 = self.layers[7](self.layers[6](x3))
        x4.register_hook(lambda grad: self.store_delta(3, grad))
        
        x5 = self.layers[9](self.layers[8](x4))
        x5.register_hook(lambda grad: self.store_delta(4, grad))

        x6 = self.layers[10](x5)

        return x6, [x1, x2, x3, x4, x5]

    def store_delta(self, layer_idx, grad):
        """
        Store the gradient of the loss with respect to the output of a layer.

        Args:
            layer_idx (int): Index of the layer.
            grad (Tensor): Gradient tensor.
        """
        self.deltas[layer_idx] = grad

    def get_deltas(self):
        """
        Returns:
            dict: Dictionary where keys are layer indices and values are gradients.
        """
        return self.deltas

    def fibration_coloring(self, threshold=0.8):

        coloring = []
        clusters = fibration_conv2d(weights=self.layers[0].weight.data, 
                        in_clusters=None, 
                        threshold=threshold, 
                        first_layer = True)

        coloring.append(clusters)

        clusters = fibration_conv2d(weights=self.layers[2].weight.data, 
                                    in_clusters=clusters, 
                                    threshold=threshold, 
                                    first_layer = False)

        coloring.append(clusters)

        clusters = fibration_conv2d(weights=self.layers[4].weight.data, 
                                    in_clusters=clusters, 
                                    threshold=threshold, 
                                    first_layer = False)

        coloring.append(clusters)

        clusters = fibration_pooling_ch(clusters,self.last_filter_output)

        clusters = fibration_linear(weights=self.layers[6].weight.data, 
                                    in_clusters=clusters, 
                                    threshold=threshold, 
                                    first_layer = False)

        coloring.append(clusters)

        clusters = fibration_linear(weights=self.layers[8].weight.data, 
                                    in_clusters=clusters, 
                                    threshold=threshold, 
                                    first_layer = False)

        coloring.append(clusters)

        self.fibration_colors = coloring


    def opfibration_coloring(self, threshold=0.5):

        coloring = [None, None, None, None, None]
        clusters = opfibration_linear(weights=self.layers[10].weight.data, 
                            out_clusters=None, 
                            threshold=threshold, 
                            last_layer = True)

        coloring[4] = clusters

        clusters = opfibration_linear(weights = self.layers[8].weight.data, 
                            out_clusters=clusters, 
                            threshold=threshold, 
                            last_layer = False)

        coloring[3] = clusters

        clusters = opfibration_linear(weights= self.layers[6].weight.data, 
                            out_clusters=clusters, 
                            threshold=threshold, 
                            last_layer = False)

        clusters = opfibration_pooling_ch(clusters, self.last_filter_output)

        coloring[2] = clusters

        clusters = opfibration_conv2d(weights = self.layers[4].weight.data, 
                            out_clusters = clusters, 
                            threshold = threshold, 
                            last_layer = False)

        coloring[1] = clusters

        clusters = opfibration_conv2d(weights = self.layers[2].weight.data, 
                            out_clusters = clusters, 
                            threshold = threshold, 
                            last_layer = False)

        coloring[0] = clusters

        self.opfibration_colors = coloring

    def covering_coloring(self, fib_thr=0.8, op_thr = 0.5):
        self.fibration_coloring(fib_thr)
        self.opfibration_coloring(op_thr)

        self.covering_colors = []

        for ii in range(5):
            self.covering_colors.append(stack((self.fibration_colors[ii], self.opfibration_colors[ii]), dim=1))

# ===============================================================
# ===============================================================

class SequentialWithKeywordArguments(Sequential):
    def forward(self, input, **kwargs):
        for module in self:
            input = module(input, **kwargs)
        return input


class BasicBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super().__init__()

        self.conv1 = Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = BatchNorm2d(planes)
        self.relu = ReLU(inplace=True)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, feature_list):
        identity = x

        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        feature_list.append(out1)

        out2 = self.conv2(out1)
        out2 = self.bn2(out2)

        if self.downsample is not None:
            identity = self.downsample(x)

        out2 += identity
        out2 = self.relu(out2)

        feature_list.append(out2)

        return out2

class ResNet18(Module):

    def __init__(self, num_classes):
        super().__init__()

        self.inplanes = 64

        self.conv1 = Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = BatchNorm2d(self.inplanes)
        self.relu = ReLU(inplace=True)

        self.act_type = 'relu'

        self.layer1 = self._make_layer(64 , 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.output_pool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512 * BasicBlock.expansion, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                kaiming_normal_(m.weight, nonlinearity="relu") # mode="fan_out"
                constant_(m.bias, 0.0)
            if isinstance(m, Linear):
                kaiming_normal_(m.weight, nonlinearity="linear") # mode="fan_out"
                constant_(m.bias, 0.0)        

            if isinstance(m, BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride):
        downsample = None

        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = SequentialWithKeywordArguments(
                Conv2d(self.inplanes, planes * BasicBlock.expansion, kernel_size=1, stride=stride),
                BatchNorm2d(planes * BasicBlock.expansion))

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * BasicBlock.expansion

        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes,planes, stride=1))

        return SequentialWithKeywordArguments(*layers)

    def forward(self, x):
        feature_list = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        feature_list.append(x)

        x = self.layer1(x, feature_list=feature_list)
        x = self.layer2(x, feature_list=feature_list)
        x = self.layer3(x, feature_list=feature_list)
        x = self.layer4(x, feature_list=feature_list)

        feature_list.pop(-1)

        x = self.output_pool(x)
        x = flatten(x, 1)

        feature_list.append(x)

        x = self.fc(x)

        return x, feature_list
