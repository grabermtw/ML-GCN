import torchvision.models as models
from torch.nn import Parameter
from corrected_reflectance_util import *
import torch
import torch.nn as nn

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNResnet(nn.Module):
    def __init__(self, model, num_weather_classes, num_terrain_classes, in_channel=300, t=0, adj_file=None):
        super(GCNResnet, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        
        self.num_weather_classes = num_weather_classes
        self.num_terrain_classes = num_terrain_classes
        self.pooling = nn.MaxPool2d(4, 4)

        self.weather_gc1 = GraphConvolution(in_channel, 1024)
        self.weather_gc2 = GraphConvolution(1024, 2048)
        self.weather_relu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_weather_classes + num_terrain_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())

        self.terrain_gc1 = GraphConvolution(in_channel, 1024)
        self.terrain_gc2 = GraphConvolution(1024, 2048)
        self.terrain_relu = nn.LeakyReLU(0.2)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp):
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)

        inp = inp[0]
        adj = gen_adj(self.A).detach()
        weather_x = self.weather_gc1(inp, adj)
        weather_x = self.weather_relu(weather_x)
        weather_x = self.weather_gc2(weather_x, adj)

        #weather_x = weather_x.transpose(0, 1)
        weather_x = weather_x.view(2048,-1)
        weather_x = torch.matmul(feature, weather_x)
        weather_x = weather_x.view(-1,16)


        terrain_x = self.terrain_gc1(inp, adj)
        terrain_x = self.terrain_relu(terrain_x)
        terrain_x = self.terrain_gc2(terrain_x, adj)

        #terrain_x = terrain_x.transpose(0, 1)
        terrain_x = terrain_x.view(2048,-1)
        terrain_x = torch.matmul(feature, terrain_x)
        terrain_x = terrain_x.view(-1,16)

        return weather_x, terrain_x

    def get_config_optim(self, lr, lrp):
        params = [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.weather_gc1.parameters(), 'lr': lr},
                {'params': self.weather_gc2.parameters(), 'lr': lr},
                {'params': self.terrain_gc1.parameters(), 'lr': lr},
                {'params': self.terrain_gc2.parameters(), 'lr': lr},
                ]
        return params



def gcn_resnet101(num_weather_classes, num_terrain_classes, t, pretrained=False, adj_file=None, in_channel=300):
    model = models.resnet101(pretrained=pretrained)
    return GCNResnet(model, num_weather_classes, num_terrain_classes, t=t, adj_file=adj_file, in_channel=in_channel)
