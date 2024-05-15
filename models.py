#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 10:43:04 2022

@author: aldomoscatelli
"""
import torch
from torch_geometric.nn import GINConv,GINEConv
from utils import linear_assignment

class GNOME(torch.nn.Module):
    def __init__(self, args, node_dim,edge_dim):
        super().__init__()
        self.args = args
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.setup_layers()
    
    def setup_layers(self):
        self.edges_features_layer = torch.nn.Linear(self.edge_dim,self.args.gnn_size[0])
        self.gnn_layers = torch.nn.ModuleList([])
        num_ftrs = self.node_dim + self.args.rw_k
        self.pre_emb = torch.nn.Linear(num_ftrs+self.args.rw_k,self.args.gnn_size[0])
        self.num_gnn_layers = len(self.args.gnn_size)
        hidden_size = self.args.gnn_size[0]
        if self.args.edge_features:
            for i in range(self.num_gnn_layers):
                self.gnn_layers.append(GINEConv(torch.nn.Sequential(
                        torch.nn.Linear(hidden_size, self.args.gnn_size[i]),
                        torch.nn.ReLU(),
                        torch.nn.Linear(self.args.gnn_size[i], self.args.gnn_size[i])
                    )))    
                hidden_size = self.args.gnn_size[i]
        else : 
            for i in range(self.num_gnn_layers):
                self.gnn_layers.append(GINConv(torch.nn.Sequential(
                        torch.nn.Linear(hidden_size, self.args.gnn_size[i]),
                        torch.nn.ReLU(),
                        torch.nn.Linear(self.args.gnn_size[i], self.args.gnn_size[i])
                    )))    
                hidden_size = self.args.gnn_size[i]

        self.out_emb = torch.nn.Sequential(
            torch.nn.Linear(self.args.gnn_size[0]*(self.num_gnn_layers), 2*self.args.gnn_size[0]*(self.num_gnn_layers)),
            torch.nn.ReLU(),
            torch.nn.Linear(2*self.args.gnn_size[0]*(self.num_gnn_layers), self.args.gnn_size[0]*(self.num_gnn_layers))
        )

    def node_embedding_block(self, edge_index, features,edges_features):
        if self.args.edge_features :
            edges_features = self.edges_features_layer(edges_features)

        node_feature_matrices = []
        feat = features
        batch = torch.zeros(len(features),dtype = torch.int64).cuda()
        for i in range(self.num_gnn_layers-1):
            if self.args.edge_features :
                features = self.gnn_layers[i](features, edge_index,edges_features)
            else:
                features = self.gnn_layers[i](features, edge_index)
            if i&1:
                features += feat
                feat = features
            features = torch.nn.functional.relu(features)
            node_feature_matrices.append(features)

        if self.args.edge_features :
            features = self.gnn_layers[-1](features, edge_index,edges_features)
        else : 
            features = self.gnn_layers[-1](features, edge_index)
        features = torch.nn.functional.relu(features)
        node_feature_matrices.append(features)
        node_feature_matrices = torch.cat(node_feature_matrices,1)

        return node_feature_matrices

    
    def forward(self, data, return_matching=True):

        edge_index_1 = data["edge_index_1"].to(device='cuda:0', dtype=torch.long)
        edge_index_2 = data["edge_index_2"].to(device='cuda:0', dtype=torch.long)
        
        G_1, G_2 = data['G_1'], data['G_2']
       
        n1, n2 = len(data["labels_1"]), len(data["labels_2"])
        features_1 = G_1.ndata['features'].cuda()
        features_2 = G_2.ndata['features'].cuda()
        e_features1 = data['e_features1'].cuda()
        e_features2 = data['e_features2'].cuda()
        RW_1 = data['PE_1'].cuda()
        RW_2 = data['PE_2'].cuda()
        features_1 = torch.cat((features_1,RW_1),1)
        features_2 = torch.cat((features_2,RW_2),1)
        
        features_1 = self.pre_emb(features_1) 
        features_2 = self.pre_emb(features_2) 

        node_embedding_1 = self.node_embedding_block(edge_index_1, features_1,e_features1)
        node_embedding_2 = self.node_embedding_block(edge_index_2, features_2,e_features2)

        metric_embedding_1 = self.out_emb(node_embedding_1)
        metric_embedding_2 = self.out_emb(node_embedding_2)

        substitution_distance_matrix = [torch.cdist(metric_embedding_1, metric_embedding_2)]

        if n1 > n2:
            ones = torch.ones((n1-n2,1)).cuda()
            deletion_distance_matrix =torch.mul(torch.norm(metric_embedding_1,dim=1),ones)
            full_distance_matrix = [
            torch.cat((substitution_distance_matrix, deletion_distance_matrix.T), dim=1)]
        if n2 > n1:
            ones = torch.ones((n2-n1,1)).cuda()
            deletion_distance_matrix =torch.mul(torch.norm(metric_embedding_2,dim=1),ones)
            full_distance_matrix = [
            torch.cat((substitution_distance_matrix, deletion_distance_matrix), dim=0)]

        if n1 == n2:
            full_distance_matrix = substitution_distance_matrix   
    
        loss,matching = linear_assignment(full_distance_matrix)

        if return_matching:
            return loss, matching   
        else :
            return loss
