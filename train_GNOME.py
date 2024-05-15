#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 10:43:04 2022

@author: aldomoscatelli
"""
from os import path

import random
from datetime import datetime, timedelta, time

import numpy as np
import torch
from torch import nn
import dgl

from tqdm.auto import tqdm

from models import GNOME
from utils import squared_error
from utils import Metric
from scipy import sparse as sp


def process_pair(data, global_labels,global_edges_labels,args):
   
    edges_1 = data["graph_1"]
    edges_2 = data["graph_2"]

    edges_1 = np.array(edges_1, dtype=np.int64)
    edges_2 = np.array(edges_2, dtype=np.int64)
    
    G_1 = dgl.DGLGraph((edges_1[:,0], edges_1[:,1]))
    G_2 = dgl.DGLGraph((edges_2[:,0], edges_2[:,1]))
        
    data["e_lab1"] = data["e_lab1"]
    data["e_lab2"] = data["e_lab2"]
   
    edges_1 = torch.from_numpy(edges_1.T).type(torch.long)
    edges_2 = torch.from_numpy(edges_2.T).type(torch.long)
  
    data["edge_index_1"] = edges_1
    data["edge_index_2"] = edges_2

    features_1, features_2 = [], []

    for n in data["labels_1"]:
        features_1.append([1.0 if global_labels[n] == i else 0.0 for i in global_labels.values()])

    for n in data["labels_2"]:
        features_2.append([1.0 if global_labels[n] == i else 0.0 for i in global_labels.values()])
    
    G_1.ndata['features'] = torch.FloatTensor(np.array(features_1))
    G_2.ndata['features'] = torch.FloatTensor(np.array(features_2))
    
    e_features1, e_features2 = [], []
    
    for e in data["e_lab1"]:
        e_features1.append([1.0 if global_edges_labels[e] == i else 0.0 for i in global_edges_labels.values()])
        
    for e in data["e_lab2"]:
        e_features2.append([1.0 if global_edges_labels[e] == i else 0.0 for i in global_edges_labels.values()])

    G_1.edata['features'] = torch.FloatTensor(np.array(e_features1))
    G_2.edata['features'] = torch.FloatTensor(np.array(e_features2))

    data['e_features1'] = G_1.edata['features']
    data['e_features2'] = G_2.edata['features']
    data['G_1'] = G_1
    data['G_2'] = G_2
    
    if np.linalg.norm(G_1.adj(scipy_fmt="csr").todense()-G_1.adj(scipy_fmt="csr").todense().T) > 0 :
        G_1 = dgl.to_bidirected(G_1,copy_ndata = True)
    if np.linalg.norm(G_2.adj(scipy_fmt="csr").todense()-G_2.adj(scipy_fmt="csr").todense().T) > 0 :
        G_2 = dgl.to_bidirected(G_2,copy_ndata = True)
    
    PE_1 = RW(G_1,args.rw_k)
    PE_2 = RW(G_2,args.rw_k)
    data['PE_1'] = PE_1
    data['PE_2'] = PE_2
    data["target"] = torch.from_numpy(np.array(data["ged"])).float()
    return data
    
def RW(G,k):
    A = G.adj(scipy_fmt="csr")
    Dinv = sp.diags(dgl.backend.asnumpy(G.in_degrees()).clip(1) ** -1.0, dtype=float)
    RW = Dinv*A 
    M = RW
    # Iterate
    nb_pos_enc = k
    PE = [torch.from_numpy(M.diagonal()).float()]
    M_power = M
    for _ in range(nb_pos_enc-1):
        M_power = M_power * M
        PE.append(torch.from_numpy(M_power.diagonal()).float())
    PE = torch.stack(PE,dim=-1)
    return PE
    

class Trainer(object):
    def __init__(self, args, training_pairs, validation_pairs, testing_pairs, device):
        self.args = args
    
        self.training_pairs = training_pairs
        self.validation_pairs = validation_pairs
        self.testing_pairs = testing_pairs

        self.initial_label_enumeration()
        
        self.epoch = 0
        self.best_val_mse = None
        self.val_mses = []
        self.best_val_metric = None
        self.val_metrics = []
        self.losses = []
        self.early_stop = False
        self.counter = 0
        self.epoch_times = []

        self.device = device
        self.setup_model()
        self.initialize_model()
        self.mse_val = []
        
    def setup_model(self):
        self.model = GNOME(self.args, self.node_dim,self.edge_dim)
        print(self.model)
        
        self.model = self.model.to(self.device)

        self.best_model = GNOME(self.args, self.node_dim,self.edge_dim)
        
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                          lr=self.args.learning_rate,
                          weight_decay=self.args.weight_decay)
        self.lrs = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.0000001, max_lr=self.args.learning_rate, step_size_up=2000, step_size_down=2000, cycle_momentum=False)
        
    def initialize_model(self):
        if path.exists(self.args.exp_dir):
            checkpoint_files = []
            if len(checkpoint_files) > 0:
                checkpoint_path = path.join(self.args.exp_dir, checkpoint_files[-1])
                print('Loading existing checkpoint: {}'.format(checkpoint_path))
                checkpoint = torch.load(checkpoint_path)
                
                self.model.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                
                self.best_val_mse = checkpoint['best_val_mse']
                self.val_mses = checkpoint['val_mses']
                if 'best_val_metric' in checkpoint:
                    self.best_val_metric = checkpoint['best_val_metric']
                    self.val_metrics = checkpoint['val_metrics']
                self.losses = checkpoint['losses']
                self.early_stop = checkpoint['early_stop']
                self.epoch = checkpoint['epoch'] + 1
                self.counter = checkpoint['counter']
                self.epoch_times = checkpoint['epoch_times']
                print('Starting from epoch {}'.format(self.epoch))
                    
    def save_checkpoint(self, is_best_model=False):
        if not is_best_model:
            save_path = path.join(self.args.exp_dir, 'checkpoint_{:04d}.pt'.format(self.epoch))
            torch.save({
                'epoch': self.epoch,
                'counter': self.counter,
                'best_val_mse': self.best_val_mse,
                'val_mses': self.val_mses, 
                'best_val_metric': self.best_val_metric,
                'val_metrics': self.val_metrics, 
                'losses': self.losses,
                'early_stop': self.early_stop,
                'model': self.model.state_dict(),
                'epoch_times': self.epoch_times,
                'optimizer': self.optimizer.state_dict()
            }, save_path)
            
        else: 
            save_path = path.join(self.args.exp_dir, 'best_checkpoint.pt')
            torch.save({
                'epoch': self.epoch,
                'best_val_mse': self.best_val_mse,
                'best_val_metric': self.best_val_metric,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, save_path)
        


    def create_batches(self):
        random.shuffle(self.training_pairs_dgl)
        batches = []
        for graph in range(0, len(self.training_pairs_dgl), self.args.batch_size):
            batches.append(self.training_pairs_dgl[graph:graph+self.args.batch_size])
        return batches

    def process_batch(self, batch):
        self.optimizer.zero_grad()
        losses = torch.tensor([0.0]).to(self.device)
        for data in batch:
            target = data["target"].to(self.device)

            prediction, matches = self.model(data)

            losses = losses + torch.nn.functional.mse_loss(target, prediction)
        losses = losses/len(batch)
        losses.backward()
        self.optimizer.step()
        self.lrs.step()
        
        loss = losses.item()
        return loss
    
    def track_state(self, val_mse, is_final_epoch=False):
        if self.best_val_mse is None:
            self.best_val_mse = val_mse
            self.val_mses.append(val_mse)
            self.save_checkpoint(is_best_model=True)
        else:
            min_mse = min(self.val_mses[-self.args.patience:])
            if val_mse > min_mse + self.args.delta:
                self.val_mses.append(val_mse)

                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.args.patience}')
                if self.counter >= self.args.patience:
                    self.early_stop = True
            else: 
                print(f'Validation MSE decreased ({min_mse:.6f} --> {val_mse:.6f}).')
                self.val_mses.append(val_mse)
                self.counter = 0

            if val_mse < self.best_val_mse:
                print(f'Best MSE ({self.best_val_mse:.6f} --> {val_mse:.6f}).  Will save best model.')
                self.best_val_mse = val_mse
                self.save_checkpoint(is_best_model=True)
            
        if self.early_stop or is_final_epoch or self.epoch % self.args.save_frequency == 0:
            self.save_checkpoint()

    def track_metric_state(self, val_metric, is_final_epoch=False):
        if self.best_val_metric is None: 
            self.best_val_metric = val_metric
            self.val_metrics.append(val_metric)
            self.save_checkpoint(is_best_model=True)
        else:
            min_metric = min(self.val_metrics[-self.args.patience:])
            if val_metric > min_metric + self.args.delta:
                print("val and min metric",val_metric,min_metric)
                self.val_metrics.append(val_metric)

                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.args.patience}')
                if self.counter >= self.args.patience:
                    self.early_stop = True
            else: 
                print(f'Validation Metric decreased ({min_metric:.6f} --> {val_metric:.6f}).')
                self.val_metrics.append(val_metric)
                self.counter = 0

            if val_metric < self.best_val_metric:
                print(f'Best Metric ({self.best_val_metric:.6f} --> {val_metric:.6f}).  Will save best model.')
                self.best_val_metric = val_metric
                self.save_checkpoint(is_best_model=True)
            
        if self.early_stop or is_final_epoch or self.epoch % self.args.save_frequency == 0:
            self.save_checkpoint()

    def fit(self):
        print("\nModel training.\n")
        
        self.model.train()
        epochs = tqdm(range(self.epoch, self.args.epochs), leave=True, desc="Epoch")
        
        iters_per_stat = self.args.iters_per_stat if 'iters_per_stat' in self.args else 10
        metric = Metric(self.validation_pairs)
        evaluation_frequency = 1
        for self.epoch in epochs:
            if self.early_stop:
                print('Early stopping!')
                break

            epoch_start_time = datetime.now()
            
            self.model.train()
            batches = self.create_batches()
            self.loss_sum = 0.0
            main_index = 0
            for index, batch in tqdm(enumerate(batches), total=len(batches), mininterval=1):
                loss_score = self.process_batch(batch)
                main_index = main_index + len(batch)
                self.loss_sum = self.loss_sum + loss_score * len(batch)
                loss = self.loss_sum/main_index

                if index % iters_per_stat == 0:
                    print("[Epoch %04d][Iter %d/%d] (Loss=%.05f)" % (self.epoch, index, len(batches), round(loss, 5)))


            epoch_duration = (datetime.now() - epoch_start_time)
            self.epoch_times.append(epoch_duration.total_seconds())
            print('[Epoch {}]: Finish in {:.2f} sec ({:.2f} min).'.format(
                    self.epoch, epoch_duration.total_seconds(), epoch_duration.total_seconds() / 60))
            self.losses.append(loss)
            validation_mse = self.score(test_pairs=self.validation_pairs_dgl)
            baseline_var = self.baseline_variance(test_pairs=self.validation_pairs_dgl)
            print("Validation MSE: {:.05f}, Baseline: {:.05f}".format(validation_mse, baseline_var))
            self.mse_val.append(validation_mse)
            is_final_epoch=(self.epoch+1 == self.args.epochs)

            if self.epoch % evaluation_frequency == 0 or is_final_epoch:
                eval_start_time = datetime.now()
                
                validation_predictions = self.predict(self.validation_pairs)
                #valid_mae = metric.mae(validation_predictions)
                valid_mse = metric.mse(validation_predictions)

                validation_metric = valid_mse
                
                self.track_metric_state(validation_metric, is_final_epoch=is_final_epoch)
                print(f'Evaluation: {validation_metric:.05f} (finishes in {(datetime.now() - eval_start_time).total_seconds()})')
            
            print("learning rate : ",self.lrs.get_last_lr())
        validation_mse = self.score(test_pairs=self.validation_pairs_dgl)
        print("Final Validation MSE: ", validation_mse)

        
    def score(self, test_pairs):
        self.model.eval()
        scores = []
        for data in tqdm(test_pairs, mininterval=2):
            prediction, matches = self.model(data)
            prediction = prediction.detach().cpu().numpy()
            scores.append(squared_error(prediction, data))
        return np.mean(scores)
    
    def load_best_model(self, load_path=None):
        if self.best_model is None:
            self.best_model = self.model.clone()
            
        if load_path is None:
            load_path = path.join(self.args.exp_dir, 'best_checkpoint.pt' )
        print('Load best model from {}'.format(load_path))
        checkpoint = torch.load(load_path)

        self.best_model.load_state_dict(checkpoint['model'])
        self.best_model = self.best_model.to(self.device)
        self.best_model.eval()
        
    def load_model(self, epoch=None, load_path=None):
        if load_path is None:
            load_path = path.join(self.args.exp_dir, 'checkpoint_{:04d}.pt'.format(epoch))
        print('Load model from {}'.format(load_path))
        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint['model'])
        self.model = self.best_model.to(self.device)
        self.model.eval()
            
    def score_best_model(self, test_pairs, load_path=None):
        self.load_best_model(load_path=load_path)
        self.best_model.eval()
        scores = []
        for data in tqdm(test_pairs, mininterval=2):
            prediction, matches = self.best_model(data)
            prediction = prediction.detach().cpu().numpy()
            scores.append(squared_error(prediction, data))
        return np.mean(scores)
    
    def predict_best_model(self, test_pairs, load_path=None):
        self.load_best_model(load_path=load_path)
        self.best_model.eval()
        prediction = []
        matches = []
        for data in tqdm(test_pairs, mininterval=2):
            one_prediction, match = self.best_model(data)
            prediction.append(one_prediction.detach().cpu().numpy())
            matches.append(match)
        prediction = np.concatenate(prediction)
        #prediction = np.array(prediction)
        return prediction,matches
        
    def predict(self, test_pairs):
        self.model.eval()
        prediction = []
        for data in tqdm(test_pairs, mininterval=2):
            one_prediction, match = self.model(data)
            prediction.append(one_prediction.detach().cpu().numpy())#attention 0
        #prediction = np.array(prediction)
        prediction = np.concatenate(prediction)
        
        return prediction
    
    def baseline_variance(self, test_pairs):
        self.model.eval()
        average_ged = np.mean([data["target"].detach().numpy()for data in test_pairs])
        base_error = np.mean([(data["target"].detach().numpy()-average_ged)**2 for data in test_pairs])
        return base_error
    
    def initial_label_enumeration(self):
        print("\nEnumerating unique labels.\n")
        graph_pairs = self.training_pairs + self.testing_pairs + self.validation_pairs
        self.global_labels = set()
        self.global_edges_labels = set()
        for data in graph_pairs:
            self.global_labels = self.global_labels.union(set(data["labels_1"]))
            self.global_labels = self.global_labels.union(set(data["labels_2"]))
            self.global_edges_labels = self.global_edges_labels.union(set(data["e_lab1"]))
            self.global_edges_labels = self.global_edges_labels.union(set(data["e_lab2"]))
        self.global_labels = list(self.global_labels)
        self.global_labels = {val:index  for index, val in enumerate(self.global_labels)}
        self.global_edges_labels = list(self.global_edges_labels)
        self.global_edges_labels = {val:index  for index, val in enumerate(self.global_edges_labels)}
        self.node_dim = len(self.global_labels)
        self.edge_dim = len(self.global_edges_labels)
        
        
        print('Graph Pairs Preprocessing... \n')
        self.training_pairs_dgl = [process_pair(graph_pair, self.global_labels,self.global_edges_labels,self.args) for graph_pair in tqdm(self.training_pairs, mininterval=2)]
        self.testing_pairs_dgl = [process_pair(graph_pair, self.global_labels,self.global_edges_labels,self.args) for graph_pair in tqdm(self.testing_pairs, mininterval=2)]
        self.validation_pairs_dgl = [process_pair(graph_pair, self.global_labels,self.global_edges_labels,self.args) for graph_pair in tqdm(self.validation_pairs, mininterval=2)] 