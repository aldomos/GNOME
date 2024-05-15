from texttable import Texttable
import torch
import numpy as np
import ot
from pandas import DataFrame

def linear_assignment(cost_matrix):
    num_pts_1 = cost_matrix.shape[0]
    a = (torch.ones(num_pts_1)).cuda()
    loss = torch.tensor([0.0]).cuda()
    mat_match = ot.emd(a, a, cost_matrix)
    matching = torch.where(mat_match != 0)
    loss += torch.sum(mat_match*cost_matrix)
    return loss,matching

def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def squared_error(prediction, data):
    if type(data) == list:
        target = np.array([d_instance['target'].detach().numpy() for d_instance in data])
    else:
        target = data["target"].detach().numpy()
    
    score = (prediction-target)**2
    return score

def save_csv(trainer,test_predictions,test_matches,dataset_name,args,test_mse,test_mae,test_fold=None):
    csvdict = {}
    csvdict['Graph1'] = []
    csvdict['Graph2'] = []
    csvdict['GED'] = []
    csvdict['Pred'] = []
    csvdict['match'] = []
    for i in range(len(trainer.testing_pairs)):
        csvdict['Graph1'].append(trainer.testing_pairs[i]["id_1"])
        csvdict['Graph2'].append(trainer.testing_pairs[i]["id_2"])
        csvdict['GED'].append(trainer.testing_pairs[i]["ged"])
        csvdict['Pred'].append(test_predictions[i])
        csvdict['match'].append(test_matches[i])
    df_res = DataFrame(csvdict, columns= ['Graph1', 'Graph2', 'GED', 'Pred','match'])

    if args.cross_val == True :
        name = '/'+dataset_name+'_fold'+str(test_fold)+'_kval_'+str(args.rw_k)+'_nbcouches_'+str(args.gnn_size)+'_lr_'+str(args.learning_rate)+ '_mse_'+str(round(test_mse,5))+'mae_'+str(round(test_mae,5))+'.csv'
        df_res.to_csv (name, index = None, header=True, encoding='utf-8', sep=';')
    else :
        nom = '/'+dataset_name+'_kval_'+str(args.rw_k)+'_nbcouches_'+str(args.gnn_size)+'_'+'lr_'+str(args.learning_rate)+ '_'+ '_' +'mse_'+str(round(test_mse,5))+'mae_'+str(round(test_mae,5))+'.csv'
        df_res.to_csv (nom, index = None, header=True, encoding='utf-8', sep=';')
        

class Metric():
    def __init__(self, instances):
        self.instances = instances
        self.ged = []
        for i,entry in enumerate(instances):
            self.ged.append(entry['ged'])
        self.ged = np.array(self.ged)
        
    def mse(self, predictions):
        score_list = (predictions - self.ged)**2
        print('GED : ',self.ged[0:10])
        print('output : ',predictions[0:10])
        mse = np.mean(score_list)
        print("validation MSE : {:.05f}".format(mse))
        return mse
        
    def mae(self, predictions):
        score_list = np.absolute(predictions - self.ged)
        print('GED : ',self.ged[0:10])
        print('output : ',predictions[0:10])
        mae = np.mean(score_list)
        print("validation MAE : {:.05f}".format(mae))
        return mae
        