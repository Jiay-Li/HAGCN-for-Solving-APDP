import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import sys
sys.path.append("..")

from nets.gcn_layers import ResidualGatedGCNLayer



class ResidualGatedGCNModel(nn.Module):

    def __init__(self, embedding_dim):
        super(ResidualGatedGCNModel, self).__init__()

        self.node_dim = 2
        self.embedding_dim = embedding_dim
        self.num_layers = 6
        self.aggregation = 'mean'

        # Node and edge embedding layers/lookups
        self.edges_values_embedding = nn.Linear(1, self.embedding_dim, bias=False)

        # Define GCN Layers
        gcn_layers = []
        for layer in range(self.num_layers):
            gcn_layers.append(ResidualGatedGCNLayer(self.embedding_dim, self.aggregation))
        self.gcn_layers = nn.ModuleList(gcn_layers)




    def forward(self, node_embeddings, edge_value):

        # Node and edge embedding
        edges_embeder = self.edges_values_embedding(edge_value.unsqueeze(3).to(torch.float32))
        node_embeder = node_embeddings

        # GCN layers
        for layer in range(self.num_layers):
            node_embeder, edges_embeder = self.gcn_layers[layer](node_embeder, edges_embeder)

        return node_embeder

