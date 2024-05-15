#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 10:43:04 2022

@author: aldomoscatelli
"""
import numpy as np
import pandas as pd
import json
import random
import time
import os
from copy import deepcopy

class data_loader():
    def __init__(self,path,split_ratio):
        self.path = path
        self.dataframe = pd.read_csv(path, delimiter = ";",header = None,
                                     names = ["Graph1","Graph2","exactGT","time","GED"])
        if "mao" in self.path:
            self.rad = 'MAO/'
        elif "alkane" in self.path:
            self.rad = 'Alkane/'
            self.dataframe.drop(self.dataframe.index[22500:],0,inplace=True)
        else :
            print("data set traitment not implemented yet")
        self.tvt_dict = self.shuffle_and_split(split_ratio)
        self.final_folder = ""
        
    def shuffle_and_split(self,split_ratio):
        data = self.dataframe
        graph_set = set(data['Graph1'])
        graph_set = list(graph_set)
        random.shuffle(graph_set)
        size = len(graph_set)
        train_set = graph_set[:int(split_ratio[0]*size)]
        val_set = graph_set[int(split_ratio[0]*size):int(split_ratio[1]*size)]
        test_set = graph_set[int(split_ratio[1]*size):]
        is_train = [data['Graph1'].iloc[i] in train_set and data['Graph2'].iloc[i] in train_set for i in range(len(data))] 
        is_val = [data['Graph1'].iloc[i] in val_set and data['Graph2'].iloc[i] in val_set for i in range(len(data))]
        is_test = [data['Graph1'].iloc[i] in test_set and data['Graph2'].iloc[i] in test_set for i in range(len(data))]
        train_data = data.iloc[np.where(is_train)]
        val_data = data.iloc[np.where(is_val)]
        test_data = data.iloc[np.where(is_test)]
        return {'train' : train_data,'val' : val_data,'test' : test_data}
        
    def pairs(self):
        tvt = self.tvt_dict
        rad = self.rad
        res = {'train' : [], 'val' : [], 'test' : []}
        ext = ".ct"
        if rad == 'MAO/' :
            for k,data in tvt.items():
                for i in range(len(data)):
                    g1 = data.iloc[i]['Graph1'].split('.')[0]
                    id1 = g1[-2:]
                    if g1[8] == '0' :
                        g1 = g1.replace('0','',1)
                    g1 = rad+g1+ext
                    g2 = data.iloc[i]['Graph2'].split('.')[0]
                    id2 = g2[-2:]
                    if g2[8] == '0' :
                        g2 = g2.replace('0','',1)
                    g2 = rad+g2+ext
                    ged = float(data.iloc[i]['GED'])
                    res[k].append(list([g1,g2,id1,id2,ged]))
        elif rad == 'Alkane/' :
            for k,data in tvt.items():
                for i in range(len(data)):
                    if data.iloc[i]['Graph1'].split('.')[0][-3:] != '001' and data.iloc[i]['Graph2'].split('.')[0][-3:] != '001':
                        g1 = data.iloc[i]['Graph1'].split('.')[0]
                        id1 = g1[-3:]
        
                        g1 = rad+g1+ext
                        g2 = data.iloc[i]['Graph2'].split('.')[0]
                        id2 = g2[-3:]
                        g2 = rad+g2+ext
                        ged = float(data.iloc[i]['GED'])
                        res[k].append(list([g1,g2,id1,id2,ged]))
        else :
            print("error data set not implemented")
        self.tvt_dict = res
        
    def extract_from_ct(self,file):
        with open(file) as f:
            lines = f.readlines()
            i = 0
            n_label = []
            list_arc = []
            for line in lines:
                if i == 1:
                    nb_node = int(line.split(' ')[0])
                elif i>1 and i<=nb_node+1 :
                    n_label.append(line.split(' ')[-1][0])
                else :
                    if i!=0:
                        spl = line.split(' ')
                        list_arc.append(list([int(spl[0])-1,int(spl[1])-1]))
                i+=1
            return list([n_label,list_arc])
        
    def pairs_to_json(self):
        rad = "pair_"
        ext = ".json"
        tvt = self.tvt_dict
        seconds = time.time()
        lt = time.localtime(seconds)
        marker = str(lt.tm_mday)+'_'+str(lt.tm_mon)+'_'+str(lt.tm_hour)+'_'+str(lt.tm_min)+'_'+str(lt.tm_sec)
        rep = self.rad[:-1] + '_graphs_pairs'+marker +'/'
        os.makedirs(rep)
        self.final_folder = rep
        for k,v in tvt.items():
            list_file = []
            for pair in v:
                pair_dict = {}
                g1 = self.extract_from_ct(pair[0])
                pair_dict["labels_1"] = g1[0]
                pair_dict["graph_1"] = g1[1]
                g2 = self.extract_from_ct(pair[1])
                pair_dict["labels_2"] = g2[0]
                pair_dict["graph_2"] = g2[1]
                pair_dict["ged"] = pair[4]
                pair_dict["id_1"] = pair[2]
                pair_dict["id_2"] = pair[3]
                fileName = rad+pair_dict["id_1"]+'_'+pair_dict["id_2"]+ext
                list_file.append(list([fileName,pair_dict]))
            os.makedirs(rep+k+'/')
            for i in range(len(list_file)):
                pair_json = list_file[i][1]
                fileName = rep+k+'/'+list_file[i][0]
                file = open(fileName, "w")
                json.dump(pair_json, file)
                file.close()

def merge_dict(dflist):
    res = deepcopy(dflist[0])
    for i in range(len(dflist)) :
        for k,v in res.items():
            v.update(dflist[i][k])
    return res

def set_to_dict(pair,fold,dataset):
    list_file =[]
    for i in range(len(pair)):
        pair_dict = {}
        p = pair.iloc[i]
        if "MAO" in dataset:
            g1 = p['Graph1'].split('.')[0]
            key = int(g1[-2:])
            for k,v in fold['nom'].items():
                if v.split('.')[0] == key :
                    key = k
            pair_dict["labels_1"] = fold['nodes'][str(key)]
            pair_dict["graph_1"] = fold['edges'][str(key)]
            pair_dict["e_lab1"] = fold['e_label'][str(key)]
            g2 = p['Graph2'].split('.')[0]
            key = int(g2[-2:])
            for k,v in fold['nom'].items():
                if v.split('.')[0] == key :
                    key = k
        else:
            g1 = p['Graph1']
            key = int(g1)
            pair_dict["labels_1"] = fold['nodes'][str(key)]
            pair_dict["graph_1"] = fold['edges'][str(key)]
            pair_dict["e_lab1"] = fold['e_label'][str(key)]
            g2 = p['Graph2']
            key = int(g2)
        
        pair_dict["labels_2"] = fold['nodes'][str(key)]
        pair_dict["graph_2"] = fold['edges'][str(key)]
        pair_dict["e_lab2"] = fold['e_label'][str(key)]
        pair_dict["ged"] = float(p['GED'])
        pair_dict["id_1"] = g1
        pair_dict["id_2"] = g2
        list_file.append(pair_dict)
    return list_file
    
def split_dic(data,key_list):
    out_dic = {}
    nom = {key: data['nom'][key] for key in data['nom'].keys() if int(key) in key_list}
    nodes = {key: data['nodes'][key] for key in data['nodes'].keys() if int(key) in key_list}
    edges = {key: data['edges'][key] for key in data['edges'].keys() if int(key) in key_list}
    e_label = {key: data['e_label'][key] for key in data['e_label'].keys() if int(key) in key_list}
    out_dic['nom']=nom
    out_dic['nodes']=nodes
    out_dic['edges']=edges
    out_dic['e_label']=e_label
    
    return out_dic
