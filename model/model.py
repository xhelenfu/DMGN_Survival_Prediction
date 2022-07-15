import numpy as np 
import torch
import torch.nn as nn
from torch_geometric.data import Data

from .components import ResNet, ConvLayer2D, OutputBlock, PVEM, RMRM

class Network(nn.Module):

    def __init__(self, opts, n_out_features, n_markers, batch_size, device, 
                 n_cont_cols, n_classes_cat):
        super(Network, self).__init__()
        channels = [opts.initial_depth, opts.initial_depth*2, opts.initial_depth*4, opts.initial_depth*8]
    
        # CNN backbone
        resnet_type = 'resnet50'

        self.fv_dim = 128
        self.n_clinical = 9
        self.n_pixel = 6*6
        self.n_nodes = self.n_clinical + self.n_pixel

        n_aux_classes = [1]*n_cont_cols + n_classes_cat

        self.edge_index = (self.get_edges(self.n_clinical, self.n_nodes)).to(device)
        
        self.batch_size = batch_size

        # Model architecture
        self.resnet_stage_0 = ResNet(0, resnet_type, n_markers, activation='leaky_relu')
        self.resnet_stage_1 = ResNet(1, resnet_type, channels[0], activation='leaky_relu')
        self.resnet_stage_2 = ResNet(2, resnet_type, channels[1], activation='leaky_relu')
        self.resnet_stage_3 = ResNet(3, resnet_type, channels[2], activation='leaky_relu')

        self.conv_1x1 = ConvLayer2D(channels[3], self.fv_dim, 1, 1, 0)

        self.clinical_mlps = PVEM(self.fv_dim, self.n_clinical, n_aux_classes)

        self.graph_net = RMRM(self.fv_dim)

        self.gap = nn.AvgPool2d(kernel_size=(6,6))

        self.output_mlp = OutputBlock(self.fv_dim*(self.n_clinical+1), n_out_features)


    def get_edges(self, n_clinical, n_nodes):
        node_ids = np.expand_dims(np.arange(n_nodes, dtype=int), 0)

        self_edges = np.concatenate((node_ids, node_ids), 0)

        c_array_asc = np.expand_dims(np.arange(n_clinical), 0)

        all_edges = self_edges[:]

        # Edges for each pixel 
        for i in range(n_clinical, n_nodes):
            i_array = np.expand_dims(np.array([i]*n_clinical), 0)

            # Image to clinical
            inter_edges_ic = np.concatenate((i_array, c_array_asc), 0)
            # Clinical to image
            inter_edges_ci = np.concatenate((c_array_asc, i_array), 0)

            inter_edges_i = np.concatenate((inter_edges_ic, inter_edges_ci), 1)

            all_edges = np.concatenate((all_edges, inter_edges_i), 1)

        all_edges = torch.tensor(all_edges, dtype=torch.long)
        return all_edges


    def forward(self, in_img, in_clinical):
        
        # Extract image features
        rstage_0 = self.resnet_stage_0(in_img)
        rstage_1 = self.resnet_stage_1(rstage_0)
        rstage_2 = self.resnet_stage_2(rstage_1)
        rstage_3 = self.resnet_stage_3(rstage_2)

        # Reduce dimensionality
        conv_1x1 = self.conv_1x1(rstage_3)
        conv_1x1_reshape = torch.reshape(conv_1x1, (self.batch_size, self.fv_dim, -1))
        
        # [batch_size, n_pixel, fv_dim]
        conv_1x1_reshape = torch.transpose(conv_1x1_reshape, 1, 2)

        # Extract clinical features [batch_size, n_clinical, fv_dim]
        clinical_fvs, clinical_preds = self.clinical_mlps(in_clinical)

        # Turn [batch_size, n_nodes, fv_dim] to [batch_size*n_nodes, fv_dim]
        for ind in range(self.batch_size):
            if ind == 0:
                batch_semantic_fvs = clinical_fvs[0,:,:]
                batch_semantic_fvs = torch.cat((batch_semantic_fvs, conv_1x1_reshape[0,:,:]),0)
            else:
                batch_semantic_fvs = torch.cat((batch_semantic_fvs, clinical_fvs[ind,:,:]),0)
                batch_semantic_fvs = torch.cat((batch_semantic_fvs, conv_1x1_reshape[ind,:,:]),0)

        batch_edge_index = self.edge_index.clone()
        for ind in range(1, self.batch_size):
            next_edge_index = self.edge_index + self.n_nodes*ind
            batch_edge_index = torch.cat((batch_edge_index, next_edge_index), 1)

        data = Data(x=batch_semantic_fvs, edge_index=batch_edge_index)

        # RMRM
        batch_graph_fvs = self.graph_net(data)

        # Reshape RMRM outputs
        for ind in range(self.batch_size):
            if ind == 0:
                graph_fvs_c = batch_graph_fvs[:self.n_clinical,:].unsqueeze(0)
                graph_fvs_i = batch_graph_fvs[self.n_clinical:self.n_clinical+self.n_pixel,:].unsqueeze(0)
            else:
                graph_fvs_c = torch.cat((graph_fvs_c,
                                         batch_graph_fvs[ind*self.n_nodes:ind*self.n_nodes+self.n_clinical,:].unsqueeze(0)),
                                         0)
                graph_fvs_i = torch.cat((graph_fvs_i,
                                         batch_graph_fvs[ind*self.n_nodes+self.n_clinical:ind*self.n_nodes+self.n_clinical+self.n_pixel,:].unsqueeze(0)),
                                         0)                

        # [batch_size, fv_dim, n_pixel]
        graph_fvs_i = torch.transpose(graph_fvs_i, 1, 2)
        # Vectorise
        gap = self.gap(torch.reshape(graph_fvs_i, (self.batch_size, self.fv_dim, 6, 6))).squeeze(-1).squeeze(-1)

        combined = torch.cat((graph_fvs_c, gap.unsqueeze(1)), 1)
        combined = torch.reshape(combined, (self.batch_size, -1))

        # Get predictions
        feature_preds = self.output_mlp(combined)
        
        return feature_preds, clinical_preds