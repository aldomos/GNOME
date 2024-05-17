#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 10:43:04 2022

@author: aldomoscatelli
"""
import os
from os import path
from argparse import Namespace
from utils import Metric,tab_printer,save_csv
import json
import glob
import torch
import random
from train_GNOME import Trainer
import argparse
import extract_pairs
import pandas as pd


def parameter_parser():
	parser = argparse.ArgumentParser(description="Run GNOME.")

	parser.add_argument("--dataset",
						nargs="?",
						default="Linux_csv/Linux.csv",
						help="csv with graphs pairs and GED values.")

	parser.add_argument("--epochs",
						type=int,
						default=200,
						help="Number of training epochs. Default is 200.")

	parser.add_argument("--batch_size",
						type=int,
						default=32,
						help="Number of graph pairs in a mini-batch. Default is 32.")

	parser.add_argument("--gnn_size",
						type=int,
						default=[64,64,64,64,64,64,64,64],
						nargs='+',
						help="List of output shape of each GIN layers. Default is 8*64.")

	parser.add_argument("--learning_rate",
						type=float,
						default=0.0001,
						help="Learning rate. Default is 0.0001.")

	parser.add_argument("--weight_decay",
						type=float,
						default=5*10**-4,
						help="Optimizer weight decay. Default is 5*10^-4 or 0.5 for IMDB.")

	parser.add_argument("--nb_folds",
						type=int,
						default=5,
						help="Number of folds. Default is 5.")

	parser.add_argument("--patience",
						type=int,
						default=10,
						help="Patience for early stopping. Default is 10.")
	
	parser.add_argument("--size_valid", 
		     			type = int,
						default = 200,
						help = "Number of Graphs in validation set Linux = 200 , AIDS = 140 , IMDB = 300 ")
	
	
	parser.add_argument("--basedir", type=str)

	parser.add_argument("--exp_label", type=str)

	parser.add_argument("--save_frequency", type=int, default=1)
	
	parser.add_argument("--rw_k",type=int, default = 11)

	parser.add_argument("--cross_val", default = False)

	parser.add_argument("--edge_features", default = False)

	return parser.parse_args()

def main():

	args = parameter_parser()

	args.exp_label = f"{args.batch_size}_{args.learning_rate}_{args.weight_decay}_{args.rw_k}"
	
	tab_printer(args)
	
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	args.exp_dir = path.join(args.basedir, args.exp_label)
	if path.exists(args.exp_dir):
		print('Experiment directory already exists...')
	else:
		print('Creating experiment directory: {}'.format(args.exp_dir))
		os.makedirs(args.exp_dir)
	


	if args.cross_val == True :
		dataset_name = args.dataset.split('.')[0].split('/')[-1]
		print('Cross Validation mode')
		dflist = []
		for fold in range(args.nb_folds):
			fold_location = glob.glob(os.path.join("CVfolder/", str(fold), "*.json"))
			dflist.append(json.load(open(fold_location[0])))
			df = pd.read_csv(args.dataset, delimiter = ",")
		
		for fold in range(args.nb_folds):
			test_fold = fold
			valid_fold = (fold+1)%args.nb_folds
			train_folds = [j for j in range(0, args.nb_folds) if j!=test_fold and j!=valid_fold]
		
			training_fold = extract_pairs.merge_dict([dflist[j] for j in train_folds])
			validation_fold = dflist[valid_fold]
			testing_fold = dflist[test_fold]
		
			training_set =  df[df.Graph1.isin(list(training_fold['nom'].values()))&df.Graph2.isin(list(training_fold['nom'].values()))]
			validation_set = df[df.Graph1.isin(list(validation_fold['nom'].values()))&df.Graph2.isin(list(training_fold['nom'].values()))]
			testing_set = df[df.Graph1.isin(list(testing_fold['nom'].values()))&df.Graph2.isin(list(training_fold['nom'].values()))]
			l = [dflist[j] for j in train_folds]
			l.append(dflist[valid_fold])
			validation_fold = extract_pairs.merge_dict(l)
			l = [dflist[j] for j in train_folds]
			l.append(dflist[test_fold])
			testing_fold = extract_pairs.merge_dict(l)
			
			training_pairs = extract_pairs.set_to_dict(training_set, training_fold,args.dataset)
			random.shuffle(training_pairs)
			validation_pairs = extract_pairs.set_to_dict(validation_set, validation_fold,args.dataset)
			random.shuffle(validation_pairs)
			testing_pairs = extract_pairs.set_to_dict(testing_set, testing_fold,args.dataset)
			random.shuffle(testing_pairs)
		
			
			trainer = Trainer(args, training_pairs, validation_pairs, testing_pairs, device)
			trainer.fit()


			test_results = []
			train_loss = []
			val_mse = []
			metric = Metric(trainer.testing_pairs)
			test_predictions, test_matches,test_emb = trainer.predict_best_model(trainer.testing_pairs)

			print(test_predictions[0:10])
			print(test_matches[0])

			train_loss.append(trainer.losses)

			val_mse.append(trainer.mse_val)
			

			test_mse = metric.mse(test_predictions)

			test_mae = metric.mae(test_predictions)

			test_results.append(Namespace(mse=test_mse, mae=test_mae)) 

			print(f'[Fold Test]: mse {test_mse:.05f}, mae {test_mae:.05f}')

			save_csv(trainer,test_predictions,test_matches,dataset_name,args,test_mse,test_mae,test_fold)
			
			print("val_mse" ,val_mse)



	else :
		dataset_name = args.dataset.split('_')[0]
		train_folder = dataset_name+'_train/'
		test_folder = dataset_name+'_test/'
		train_location = glob.glob(os.path.join("CVfolder/", train_folder, "*.json"))
		test_location = glob.glob(os.path.join("CVfolder/", test_folder, "*.json"))

		data = json.load(open(train_location[0]))
		test_data = json.load(open(test_location[0]))
		df = pd.read_csv(args.dataset, delimiter = ",")
		list_key = list(data['nom'].values())
		random.shuffle(list_key)

		val_size = args.size_valid
		val_key = list_key[:val_size]
		train_key = list_key[val_size:]
		training_fold = extract_pairs.split_dic(data,train_key)
		validation_fold = extract_pairs.split_dic(data,val_key)
		testing_fold = test_data

		training_set =  df[df.Graph1.isin(train_key)&df.Graph2.isin(train_key)]
		validation_set = df[df.Graph1.isin(train_key)&df.Graph2.isin(val_key)]
		testing_set = df[df.Graph1.isin(train_key)&df.Graph2.isin(list(testing_fold['nom'].values()))]
		l = [training_fold,validation_fold]
		validation_pair_set = extract_pairs.merge_dict(l)
		l = [training_fold,testing_fold]
		testing_pair_set = extract_pairs.merge_dict(l)

		training_pairs = extract_pairs.set_to_dict(training_set, training_fold,args.dataset)
		random.shuffle(training_pairs)
		validation_pairs = extract_pairs.set_to_dict(validation_set, validation_pair_set,args.dataset)
		random.shuffle(validation_pairs)
		testing_pairs = extract_pairs.set_to_dict(testing_set, testing_pair_set,args.dataset)
		random.shuffle(testing_pairs)

		trainer = Trainer(args, training_pairs, validation_pairs, testing_pairs, device)
		trainer.fit()

		test_results = []
		train_loss = []
		val_mse = []
		metric = Metric(trainer.testing_pairs)
		test_predictions, test_matches = trainer.predict_best_model(trainer.testing_pairs)

		print(test_predictions[0:10])
		print(test_matches[0])

		train_loss.append(trainer.losses)

		val_mse.append(trainer.mse_val)

		test_mse = metric.mse(test_predictions)

		test_mae = metric.mae(test_predictions)

		test_results.append(Namespace(mse=test_mse, mae=test_mae)) 

		print(f'[Fold Test]: mse {test_mse:.05f}, mae {test_mae:.05f}')

		save_csv(trainer,test_predictions,test_matches,dataset_name,args,test_mse,test_mae)

		print("val_mse" ,val_mse)

	return test_results
	
if __name__ == "__main__":
	results = main()  
