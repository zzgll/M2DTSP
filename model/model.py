import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.nn.modules.batchnorm import _BatchNorm
import torch_geometric.nn as gnn
from torch import Tensor
from collections import OrderedDict
from torch_geometric.data import Batch
from einops.layers.torch import Rearrange, Reduce
import pickle
import math

class NodeLevelBatchNorm(_BatchNorm):


    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(NodeLevelBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        return torch.functional.F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={eps}, ' \
               'affine={affine}'.format(**self.__dict__)


class GraphConvBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = gnn.GraphConv(in_channels, out_channels)
        self.norm = NodeLevelBatchNorm(out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        data.x = F.relu(self.norm(self.conv(x, edge_index)))

        return data

class ConvLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        self.conv1 = GraphConvBn(num_input_features, int(growth_rate * bn_size))
        self.conv2 = GraphConvBn(int(growth_rate * bn_size), growth_rate)

    def bn_function(self, data):
        concated_features = torch.cat(data.x, 1)
        data.x = concated_features

        data = self.conv1(data)

        return data

    def forward(self, data):
        if isinstance(data.x, Tensor):
            data.x = [data.x]

        data = self.bn_function(data)
        # data = self.conv1(data)
        data = self.conv2(data)

        return data



class MultiScaleBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        for i in range(num_layers):
            # 每一层除了第一层，输入输出都是growth_rate
            in_channels = num_input_features if i == 0 else growth_rate
            layer = ConvLayer(in_channels, growth_rate, bn_size)
            self.add_module('layer%d' % (i + 1), layer)

    def forward(self, data):
        layer_outputs = [data.x]  # 保存所有层的输出

        # 遍历每一层
        for name, layer in self.items():
            data = layer(data)
            layer_outputs.append(data.x)

        # 在所有层处理完毕后，只拼接所有层的输出与最后一层的输出
        data.x = torch.cat(layer_outputs, dim=1)

        return data


class GraphMSNet(nn.Module):
    def __init__(self, num_input_features, out_dim, growth_rate=32, block_config=(3, 3, 3, 3), bn_sizes=[2, 3, 4, 4]):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', GraphConvBn(num_input_features, 32))]))
        num_input_features = 32

        for i, num_layers in enumerate(block_config):
            block = MultiScaleBlock(
                num_layers, num_input_features, growth_rate=growth_rate, bn_size=bn_sizes[i]
            )
            self.features.add_module('block%d' % (i + 1), block)
            num_input_features += int(num_layers * growth_rate)

            trans = GraphConvBn(num_input_features, num_input_features // 2)
            self.features.add_module("transition%d" % (i + 1), trans)
            num_input_features = num_input_features // 2

        self.classifer = nn.Linear(num_input_features, out_dim)

    def forward(self, data):
        data = self.features(data)
        x = gnn.global_mean_pool(data.x, data.batch)
        x = self.classifer(x)

        return x

class DilaCNN(nn.Module):
    def __init__(self, dilaSize, filterSize=256, dropout=0.15):
        super(DilaCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(filterSize, filterSize, kernel_size=3, padding=dilaSize, dilation=dilaSize),
            nn.ReLU(),
            nn.Conv1d(filterSize, filterSize, kernel_size=3, padding=dilaSize, dilation=dilaSize),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layers(x)


class ProteinSeqModule(nn.Module):
    def __init__(self, feaSize, filterSize=256, outputSize=128, dilaSizeList=[1, 2, 4, 8, 16], dropout=0.5):
        super(ProteinSeqModule, self).__init__()
        self.branches = nn.ModuleList([DilaCNN(dilaSize, filterSize, dropout) for dilaSize in dilaSizeList])
        self.linear_in = nn.Linear(feaSize, filterSize)
        self.linear_out = nn.Linear(filterSize * len(dilaSizeList), outputSize)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = self.linear_in(x)  # => batchSize × seqLen × filterSize
        x = x.transpose(1, 2)  # => batchSize × filterSize × seqLen

        # 每块空洞卷积输出
        branch_outputs = [branch(x) for branch in self.branches]

        # 输出拼接
        concatenated = torch.cat(branch_outputs, dim=1)

        # 最大池化
        pooled = torch.max(concatenated, dim=2)[0]
        # 全连接层修改维度
        output = self.linear_out(pooled)
        output = self.act(output)

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        assert output_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        self.fc_query = nn.Linear(input_dim, output_dim)
        self.fc_key = nn.Linear(input_dim, output_dim)
        self.fc_value = nn.Linear(input_dim, output_dim)
        self.fc_out = nn.Linear(output_dim, output_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        batch_size = x1.size(0)

        query = self.fc_query(x1).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.fc_key(x2).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.fc_value(x2).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        attention_weights = self.softmax(attention_scores)

        context = torch.matmul(attention_weights, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        output = self.fc_out(context)

        return output

class M2DTSP(nn.Module):
    def __init__(self, block_num, vocab_protein_size, embedding_size=128, filter_num=32, out_dim=1):
        super().__init__()
        # 我们主要看图神经网络的输入参数
        self.protein_encoder = GraphMSNet(num_input_features=41, out_dim=filter_num*3, block_config=[3, 3, 3], bn_sizes=[2, 2, 2])

        self.ligand_encoder = GraphMSNet(num_input_features=18, out_dim=filter_num*3, block_config=[8, 8, 8], bn_sizes=[2, 2, 2])

        embed_dim = 96
        self.embed_smile = nn.Embedding(65, embed_dim)
        self.embed_prot = nn.Embedding(26, embed_dim)
        self.onehot_smi_net = ProteinSeqModule(embed_dim, 256, embed_dim)
        self.onehot_prot_net = ProteinSeqModule(embed_dim, 256, embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(192, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim)
        )

    def forward(self, data):
        protein = data.protein
        # protein为蛋白质序列结构信息


        smi = data.molecule_target
        # smi为药物序列信息

        # 加载 PKL 文件
        with open('compound_features_davis_68.pkl', 'rb') as file:
            features_dict = pickle.load(file)

        result_vectors = []

        # 遍历每个表达式，提取特征向量
        for expression in smi:

            if expression in features_dict:
                result_vectors.append(features_dict[expression])
            else:
                print(f"Warning: Expression '{expression}' not found in features_dict.")


        # 将列表转换为 PyTorch tensor
        drugFeature = torch.tensor(result_vectors)

        protein = Batch.from_data_list(protein)

        # seq为蛋白质序列信息
        seq = protein.target

        proteinFeature_onehot = self.embed_prot(seq)
        # 提取蛋白质序列特征向量
        proteinFeature_onehot = self.onehot_prot_net(proteinFeature_onehot)

        # 提取蛋白质结构特征向量
        protein_x = self.protein_encoder(protein)

        # 提取药物结构特征向量
        ligand_x = self.ligand_encoder(data)

        drugFeature = drugFeature.to('cuda:0')


        # 使用多头注意力机制
        protein_attention = MultiHeadAttention(input_dim=96, output_dim=96, num_heads=16).to('cuda:0')
        ligand_attention = MultiHeadAttention(input_dim=96, output_dim=96, num_heads=16).to('cuda:0')

        # 进行注意力融合
        protein_fusion = protein_attention(protein_x, proteinFeature_onehot)
        ligand_fusion = ligand_attention(ligand_x, drugFeature)

        # 融合蛋白质和药物特征
        x = torch.cat([protein_fusion, ligand_fusion], dim=-1)

        x = self.classifier(x)

        return x
