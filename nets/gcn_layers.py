import torch
import torch.nn.functional as F
import torch.nn as nn

class BatchNormNode(nn.Module):
    """
    Batch normalization for node features.
    """
    def __init__(self, hidden_dim):
        super(BatchNormNode, self).__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)

    def forward(self, x):
        x_trans = x.transpose(1, 2).contiguous()  # Reshape input: (batch_size, hidden_dim, num_nodes)
        x_trans_bn = self.batch_norm(x_trans)
        x_bn = x_trans_bn.transpose(1, 2).contiguous()  # Reshape to original shape
        return x_bn


class BatchNormEdge(nn.Module):
    """
        Batch normalization for edge features.
    """

    def __init__(self, hidden_dim):
        super(BatchNormEdge, self).__init__()
        self.batch_norm = nn.BatchNorm2d(hidden_dim, track_running_stats=False)

    def forward(self, e):
        e_trans = e.transpose(1, 3).contiguous()  # Reshape input: (batch_size, num_nodes, num_nodes, hidden_dim)
        e_trans_bn = self.batch_norm(e_trans)
        e_bn = e_trans_bn.transpose(1, 3).contiguous()  # Reshape to original
        return e_bn


class NodeFeatures(nn.Module):

    def __init__(self, hidden_dim, aggregation="mean"):
        super(NodeFeatures, self).__init__()
        self.aggregation = aggregation
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)


    def forward(self,node_embeder_in, edge_gate):

        W1_node_i = self.W1(node_embeder_in)
        W2_node_j = self.W2(node_embeder_in)
        W2_node_j = W2_node_j.unsqueeze(1)
        gate_W2_node_j = edge_gate * W2_node_j

        if self.aggregation == "mean":
            node_new = W1_node_i + torch.sum(gate_W2_node_j, dim=2) / (1e-20 + torch.sum(edge_gate, dim=2))
        elif self.aggregation == "sum":
            node_new = W1_node_i + torch.sum(gate_W2_node_j, dim=2)

        return node_new


class EdgeFeatures(nn.Module):
    def __init__(self, hidden_dim):
        super(EdgeFeatures, self).__init__()
        self.W3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
    def forward(self, node_embeder_in, edges_embeder_in):
        W3_edge = self.W3(edges_embeder_in)
        W_node = self.W(node_embeder_in)
        W5_node = W_node.unsqueeze(2)
        W4_node = W_node.unsqueeze(1)
        edge_new = W3_edge + W4_node + W5_node
        return edge_new


class ResidualGatedGCNLayer(nn.Module):
    """
    Convnet layer with gating and residual connection.
    """
    def __init__(self, hidden_dim, aggregation="sum"):
        super(ResidualGatedGCNLayer, self).__init__()

        self.node_feat = NodeFeatures(hidden_dim, aggregation)
        self.edge_feat = EdgeFeatures(hidden_dim)
        self.bn_node = BatchNormNode(hidden_dim)
        self.bn_edge = BatchNormEdge(hidden_dim)

    def forward(self, node_embeder, edges_embeder):

        edges_embeder_in = edges_embeder
        node_embeder_in = node_embeder

        # Edge convolution
        edges_embeder_tmp = self.edge_feat(node_embeder_in, edges_embeder_in)

        # Compute edge gates
        edge_gate = torch.sigmoid(edges_embeder_tmp)

        # Node convolution
        node_embeder_tmp = self.node_feat(node_embeder_in, edge_gate)

        # Batch normalization
        edges_embeder_tmp = self.bn_edge(edges_embeder_tmp)
        node_embeder_tmp = self.bn_node(node_embeder_tmp)
        # ReLU Activation
        edges_embeder = F.relu(edges_embeder_tmp)
        node_embeder = F.relu(node_embeder_tmp)
        # Residual connection
        node_embeder_new = node_embeder_in + node_embeder
        edges_embeder_new = edges_embeder_in + edges_embeder
        return node_embeder_new, edges_embeder_new


