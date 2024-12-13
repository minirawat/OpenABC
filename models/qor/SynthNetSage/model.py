import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import SAGEConv  # Import GraphSAGE layer from PyTorch Geometric

allowable_synthesis_features = {
    'synth_type' : [0,1,2,3,4,5,6]
}

def get_synth_feature_dims():
    return list(map(len, [
        allowable_synthesis_features['synth_type']
]))

full_synthesis_feature_dims = get_synth_feature_dims()


allowable_features = {
    'node_type' : [0,1,2],
    'num_inverted_predecessors' : [0,1,2]
}

def get_node_feature_dims():
    return list(map(len, [
        allowable_features['node_type']
    ]))


full_node_feature_dims = get_node_feature_dims()


class NodeEncoder(torch.nn.Module):

    def __init__(self, emb_dim, use_relu=True):
        super(NodeEncoder, self).__init__()

        self.node_type_embedding = torch.nn.Embedding(full_node_feature_dims[0], emb_dim)
        self.use_relu = use_relu
        torch.nn.init.xavier_uniform_(self.node_type_embedding.weight.data)

    def forward(self, x):
        # First feature is node type, second feature is inverted predecessor
        x_embedding = self.node_type_embedding(x[:, 0])
        x_embedding = torch.cat((x_embedding, x[:,1].reshape(-1,1)), dim=1)
        if self.use_relu:
          x_embedding = F.relu(x_embedding)  
        return x_embedding


class GCNConv(MessagePassing):
    def __init__(self, in_emb_dim, out_emb_dim):
        super(GCNConv, self).__init__(aggr='add')
        self.linear = torch.nn.Linear(in_emb_dim, out_emb_dim)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = self.linear(x)
        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, node_encoder, num_layer, input_dim, emb_dim, gnn_type='sage'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''
        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.node_emb_size = input_dim
        self.node_encoder = node_encoder

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.convs.append(SAGEConv(input_dim, emb_dim))  # First GraphSAGE layer
        self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(1, num_layer):
            self.convs.append(SAGEConv(emb_dim, emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):

        #gate_type, node_type, edge_index = batched_data.gate_type, batched_data.node_type, batched_data.edge_index
        edge_index = batched_data.edge_index

        x = torch.cat([batched_data.node_type.reshape(-1, 1),batched_data.num_inverted_predecessors.reshape(-1, 1)], dim=1)
        h = self.node_encoder(x)

        for layer in range(self.num_layer):

            h = self.convs[layer](h, edge_index)
            h = self.batch_norms[layer](h)

            if layer != self.num_layer - 1:
                h = F.relu(h)

        return h

class GNN(torch.nn.Module):

    def __init__(self, node_encoder, input_dim, num_layer = 2, emb_dim = 128,gnn_type = 'gcn',graph_pooling = "mean"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.graph_pooling = graph_pooling

        self.gnn_node = GNN_node(node_encoder,num_layer,input_dim,emb_dim,gnn_type = gnn_type)
        self.pool1 = global_mean_pool
        self.pool2 = global_max_pool

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph1 = self.pool1(h_node, batched_data.batch)
        h_graph2 = self.pool2(h_node,batched_data.batch)
        return torch.cat([h_graph1,h_graph2],dim=1)


class SynthFlowEncoder(torch.nn.Module):
    def __init__(self, emb_dim, activation_fn=None, dropout_prob=0.0, padding_idx=None):
        """
        Args:
            num_features (int): Number of distinct feature types (e.g., vocabulary size).
            emb_dim (int): Embedding dimension for each feature.
            activation_fn (callable, optional): Activation function to apply after concatenating embeddings.
            dropout_prob (float): Dropout probability to apply after embedding lookup.
            padding_idx (int, optional): Index to treat as padding (ignored in embedding lookup).
        """
        super(SynthFlowEncoder, self).__init__()

        # Embedding layer with optional padding_idx for handling padded values
        self.synth_emb = torch.nn.Embedding(full_synthesis_feature_dims[0], emb_dim, padding_idx=padding_idx)
        
        # Xavier initialization
        torch.nn.init.xavier_uniform_(self.synth_emb.weight.data)

        # Dropout layer for regularization
        self.dropout = torch.nn.Dropout(p=dropout_prob)
        
        # Optional activation function
        self.activation_fn = activation_fn

    def forward(self, x):
        # Embed the features (batch_size, full_synthesis_feature_dims[0])
        x_embedding = self.synth_emb(x[:, 0])  # First feature (embedded)
        
        # Stack embeddings of remaining features (concatenate embeddings for all features)
        embeddings = [x_embedding]
        for i in range(1, x.shape[1]):
            embeddings.append(self.synth_emb(x[:, i]))  # Embed each feature
        
        # Concatenate all embeddings along the feature dimension
        x_embedding = torch.cat(embeddings, dim=1)
        
        # Apply dropout for regularization
        x_embedding = self.dropout(x_embedding)

        # Optionally apply activation function
        if self.activation_fn:
            x_embedding = self.activation_fn(x_embedding)
        
        return x_embedding


class SynthConv(torch.nn.Module):
    def __init__(self, inp_channel=1, out_channel=3, ksize=6, stride_len=1, use_bn=False, activation_fn=None):
        """
        Args:
            inp_channel (int): Number of input channels.
            out_channel (int): Number of output channels.
            ksize (int): Kernel size for the convolution.
            stride_len (int): Stride length for the convolution.
            use_bn (bool): Whether to apply batch normalization after the convolution.
            activation_fn (callable, optional): Activation function to apply after the convolution.
        """
        super(SynthConv, self).__init__()
        
        # 1D Convolution
        self.conv1d = torch.nn.Conv1d(inp_channel, out_channel, kernel_size=ksize, stride=stride_len)
        
        # Optional batch normalization
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = torch.nn.BatchNorm1d(out_channel)
        
        # Optional activation function (e.g., ReLU, LeakyReLU)
        self.activation_fn = activation_fn

    def forward(self, x):
        # Reshape the input from [batch_size, seq_length] to [batch_size, 1, seq_length]
        x = x.unsqueeze(1)  # Adds a channel dimension

        # Apply convolution
        x = self.conv1d(x)

        # Apply batch normalization (if needed)
        if self.use_bn:
            x = self.bn(x)
        
        # Apply activation function (if provided)
        if self.activation_fn:
            x = self.activation_fn(x)
        
        # Flatten the output back to [batch_size, -1]
        return x.view(x.size(0), -1)

import torch
import torch.nn.functional as F

class SynthNet(torch.nn.Module):

    def __init__(self, node_encoder, synth_encoder, n_classes, synth_input_dim, node_input_dim, gnn_embed_dim=256, num_fc_layer=3, hidden_dim=128):
        super(SynthNet, self).__init__()
        self.num_layers = num_fc_layer
        self.hidden_dim = hidden_dim
        self.node_encoder = node_encoder
        self.synth_encoder = synth_encoder
        self.node_enc_outdim = node_input_dim
        self.synth_enc_outdim = synth_input_dim
        self.gnn_emb_dim = gnn_embed_dim
        self.n_classes = n_classes

        # Synthesis Convolution parameters
        self.synconv_in_channel = 1
        self.synconv_out_channel = 1
        self.synconv_stride_len = 3

        # Synth Conv1 output
        self.synconv1_ks = 6
        self.synconv1_out_dim_flatten = 1 + (self.synth_enc_outdim - self.synconv1_ks) / self.synconv_stride_len

        # Synth Conv2 output
        self.synconv2_ks = 9
        self.synconv2_out_dim_flatten = 1 + (self.synth_enc_outdim - self.synconv2_ks) / self.synconv_stride_len

        # Synth Conv3 output
        self.synconv3_ks = 12
        self.synconv3_out_dim_flatten = 1 + (self.synth_enc_outdim - self.synconv3_ks) / self.synconv_stride_len

        # Convolutional layers for synthesis flow
        self.gnn = GNN(self.node_encoder, self.node_enc_outdim+1, gnn_type='sage')
        self.synth_conv1 = SynthConv(self.synconv_in_channel, self.synconv_out_channel, ksize=self.synconv1_ks, stride_len=self.synconv_stride_len)
        self.synth_conv2 = SynthConv(self.synconv_in_channel, self.synconv_out_channel, ksize=self.synconv2_ks, stride_len=self.synconv_stride_len)
        self.synth_conv3 = SynthConv(self.synconv_in_channel, self.synconv_out_channel, ksize=self.synconv3_ks, stride_len=self.synconv_stride_len)

        # Fully connected layers
        self.fcs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # Input to FCs: GNN embedding + synthesis conv outputs
        self.in_dim_to_fcs = int(self.gnn_emb_dim + self.synconv1_out_dim_flatten + self.synconv3_out_dim_flatten + self.synconv2_out_dim_flatten)
        self.fcs.append(torch.nn.Linear(self.in_dim_to_fcs, self.hidden_dim))

        for layer in range(1, self.num_layers - 1):
            self.fcs.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))

        self.fcs.append(torch.nn.Linear(self.hidden_dim, self.n_classes))

    def forward(self, batch_data):
        graphEmbed = self.gnn(batch_data)
        # Synthesis flow processing
        synthFlow = batch_data.synVec
        h_syn = self.synth_encoder(synthFlow.reshape(-1, 20))
        synconv1_out = self.synth_conv1(h_syn)
        synconv2_out = self.synth_conv2(h_syn)
        synconv3_out = self.synth_conv3(h_syn)

        # Concatenate embeddings
        concatenatedInput = torch.cat([graphEmbed, synconv1_out, synconv2_out, synconv3_out], dim=1)

        # Fully connected layers
        x = F.relu(self.fcs[0](concatenatedInput))
        for layer in range(1, self.num_layers - 1):
            x = F.relu(self.fcs[layer](x))

        x = self.fcs[-1](x)
        return x
