# coding=utf-8
import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from torch_geometric.nn.conv.gcn_conv import GCNConv
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
from transformers import RobertaTokenizer, RobertaForMaskedLM
import math
import numpy as np
import torch.backends.cudnn as cudnn


class ListModule(nn.Module):
    """
    Abstract list layer class.
    """

    def __init__(self, *args):
        """
        Model initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)


#对每一个节点序列进行池化   获得节点的表示
class MyPooler(nn.Module):
    def __init__(self, hidden_size):
        hidden_size
        super().__init__()
        self.dense = nn.Linear(768, 768)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):

        first_token_tensor = hidden_states[0, :]  #取第一个字的表示进行池化
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class GraphEncoder(nn.Module):#开始初始化的时候是词表中的信息
    def __init__(self, num_nodes, num_relations, gnn_layers, embedding_size, initilized_embedding, dropout_ratio=0.3):
        super(GraphEncoder, self).__init__()
        self.num_nodes = num_nodes #1734
        self.num_relations = num_relations #8
        self.gnn_layers = gnn_layers #2
        self.embedding_size = embedding_size #768
        self.dropout_ratio = dropout_ratio #0.3

        self.node_embedding = nn.Embedding(num_nodes, embedding_size) #1734*768
        self.node_embedding.from_pretrained(torch.from_numpy(np.load(initilized_embedding)), freeze=False) #进行初始化

        self.dropout = nn.Dropout(dropout_ratio)

        self.gnn = [] #两层 RGCNConv(768, 768, num_relations=8)
        for layer in range(gnn_layers):
            self.gnn.append(RGCNConv(embedding_size, embedding_size,num_relations))  # if rgcn is too slow, you can use gcn
        self.gnn = ListModule(*self.gnn)

    def forward(self, nodes, edges, types):
      
        batch_size = nodes.size(0)
        device = nodes.device

        # (batch_size, num_nodes, output_size)
        node_embeddings = []
        for bid in range(batch_size):
            embed = self.node_embedding(nodes[bid, :])
            edge_index = torch.as_tensor(edges[bid], dtype=torch.long, device=device)
            edge_type = torch.as_tensor(types[bid], dtype=torch.long, device=device)
            for lidx, rgcn in enumerate(self.gnn):
                if lidx == len(self.gnn) - 1:
                    embed = rgcn(embed, edge_index=edge_index,edge_type=edge_type)
                else:
                    embed = self.dropout(F.relu(rgcn(embed, edge_index=edge_index,edge_type=edge_type)))
            node_embeddings.append(embed)
        node_embeddings = torch.stack(node_embeddings, 0)  # [batch_size, num_node, embedding_size]

        return node_embeddings


