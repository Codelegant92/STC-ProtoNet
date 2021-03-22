import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import torch.utils.data.sampler
import os
import glob
import random
import time

import configs
import backbone
import data.feature_loader as feat_loader
from data.datamgr import SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file, get_best_file , get_assigned_file

def feature_evaluation(cl_data_file, model, n_way = 5, n_support = 5, n_query = 15, max_n_way=12, min_n_way=3, fixed_way=2, max_shot=10, min_shot=1, adaptation = False):
    class_list = cl_data_file.keys()
    if n_way != -1:
        selected_n_way = n_way
    else:
        selected_n_way = np.random.randint(min_n_way, max_n_way+1)
    model.n_way = selected_n_way
    select_class = random.sample(class_list, selected_n_way)
    z_all  = []
    if n_support != -1:
        sampled_n_support = n_support
    else:
        sampled_n_support = max_shot
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append( [ np.squeeze( img_feat[perm_ids[i]]) for i in range(sampled_n_support+n_query) ] )     # stack each batch

    z_all = torch.from_numpy(np.array(z_all) )
    model.n_query = n_query
    if adaptation:
        scores  = model.set_forward_adaptation(z_all, is_feature = True)
    else:
        scores, regularizer_loss  = model.set_forward(z_all, n_support, is_feature = True)
    pred = scores.data.cpu().numpy().argmax(axis = 1)
    y = np.repeat(range( selected_n_way ), n_query )
    acc = np.mean(pred == y)*100 
    return acc

if __name__ == '__main__':
    params = parse_args('test')
    acc_all = []
    iter_num = 1000
    n_query = 100
    few_shot_params = dict(n_way=params.test_n_way, n_query = n_query, max_n_way=params.test_max_way, min_n_way=params.test_min_way, \
            max_shot=params.max_shot, min_shot=params.min_shot, fixed_way=params.fixed_way) 

    
    if params.method == 'protonet':
        model           = ProtoNet( model_dict[params.model], **few_shot_params )
    else:
       raise ValueError('Unknown method')

    model = model.cuda()

    #checkpoint_dir = '%s/checkpoints/%s/%s_%s_regularizer' %(configs.save_dir, params.dataset, params.model, params.method)
    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    
    if params.train_aug:
        checkpoint_dir += '_aug'
    
    if params.train_n_way != -1:
        checkpoint_dir += '_%d-way_' %( params.train_n_way )
    else:
        checkpoint_dir += '_random-way_'
    if params.train_n_shot != -1:
        checkpoint_dir += '%d-shot' % ( params.train_n_shot )
    else:
        checkpoint_dir += 'random-shot'

    #modelfile   = get_resume_file(checkpoint_dir)
    print(checkpoint_dir)
    
    if params.save_iter != -1:
        modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
    else:
         modelfile   = get_best_file(checkpoint_dir, params.test_n_way)
    if modelfile is not None:
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])

    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" +str(params.save_iter)
    else:
        split_str = split
    
    novel_file = os.path.join( checkpoint_dir.replace("checkpoints","features"), split_str + ".hdf5")
        
    print('feature file: '+ novel_file)
    cl_data_file = feat_loader.init_loader(novel_file)

    for i in range(iter_num):
        acc = feature_evaluation(cl_data_file, model, n_support=params.test_n_shot, adaptation = params.adaptation, **few_shot_params)
        acc_all.append(acc)

    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
    with open('./record/results.txt' , 'a') as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime()) 
        aug_str = '-aug' if params.train_aug else ''
        aug_str += '-adapted' if params.adaptation else ''
        if params.method in ['baseline', 'baseline++'] :
            exp_setting = '%s-%s-%s-%s%s %sshot %sway_test' %(params.dataset, split_str, params.model, params.method, aug_str, params.test_n_shot, params.test_n_way )
        else:
            exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test %squery' %(params.dataset, split_str, params.model, params.method, aug_str , params.train_n_shot , params.train_n_way, params.test_n_way, params.test_n_shot )
        acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num))
        f.write( 'Time: %s, Setting: %s, Acc: %s \n' %(timestamp,exp_setting,acc_str)  )
