import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.protonet import ProtoNet
from io_utils import model_dict, parse_args, get_resume_file
from scipy.io import loadmat

def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):    
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
       raise ValueError('Unknown optimization, please define by yourself')
    
    max_acc = 0

    feature_dim = 576
    num_class = 22
    mean_var = 10
    kernel_dict = loadmat('meanvar1_featuredim'+str(feature_dim)+'_class'+str(num_class)+'.mat')
    mean_logits_np = kernel_dict['mean_logits'] #num_class X num_dense
    mean_logits = mean_var * torch.FloatTensor(mean_logits_np)
    mean_logits = mean_logits.cuda()
    for epoch in range(start_epoch,stop_epoch):
        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        model.train()
        model.train_loop(epoch+1, base_loader, mean_logits, optimizer, n_support=params.train_n_shot ) #model are called by reference, no need to return 
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
     
        acc = model.test_loop(val_loader, n_support=-1)
        if acc > max_acc:
            print("best model for random-way, random-shot! save ...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch+1, 'state':model.state_dict()}, outfile)
    return model

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    base_file = configs.data_dir[params.dataset] + 'base.json' 
    base_file_unk = configs.data_dir[params.dataset] + 'base_unk.json'
    base_file_sil = configs.data_dir[params.dataset] + 'base_sil.json'
    val_file   = configs.data_dir[params.dataset] + 'val.json' 
    val_file_unk = configs.data_dir[params.dataset] + 'val_unk.json'
    val_file_sil = configs.data_dir[params.dataset] + 'val_sil.json'
         
    image_size = 40
    optimization = 'Adam'

    if params.stop_epoch == -1: 
        if params.train_n_shot < 5:
            params.stop_epoch = 600
        else:
            params.stop_epoch = 400
     
    

    
    #n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
    n_query = 16

    train_few_shot_params    = dict(n_way=params.train_n_way, n_query = n_query, max_n_way=params.train_max_way, \
            min_n_way=params.train_min_way, max_shot=params.max_shot, min_shot=params.min_shot, fixed_way=params.fixed_way) 

      
    base_datamgr            = SetDataManager(image_size, n_support=params.train_n_shot, n_eposide=100, **train_few_shot_params)
    base_loader             = base_datamgr.get_data_loader( base_file , [base_file_unk, base_file_sil], aug = params.train_aug )
         
    val_few_shot_params     = dict(n_way=-1, n_query = n_query, max_n_way=params.test_max_way, min_n_way=params.test_min_way, \
            max_shot=params.max_shot, min_shot=params.min_shot, fixed_way=params.fixed_way, n_eposide=1000) 
    val_datamgr             = SetDataManager(image_size, n_support=-1, **val_few_shot_params)
    val_loader              = val_datamgr.get_data_loader( val_file, [val_file_unk, val_file_sil], aug = False) 
        
    if params.method == 'protonet':  
        model           = ProtoNet( model_dict[params.model], **train_few_shot_params )
    else:
       raise ValueError('Unknown method')

    model = model.cuda()

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s_regularizer' %(configs.save_dir, params.dataset, params.model, params.method)
    
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    
    if params.train_n_way != -1:
        params.checkpoint_dir += '_%d-way_' %( params.train_n_way )
    else:
        params.checkpoint_dir += '_random-way_'
    if params.train_n_shot != -1:
        params.checkpoint_dir += '%d-shot' % ( params.train_n_shot )
    else:
        params.checkpoint_dir += 'random-shot'
        
 
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.load_state_dict(tmp['state'])
    elif params.warmup: #We also support warmup from pretrained baseline feature, but we never used in our paper
        baseline_checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, 'baseline')
        if params.train_aug:
            baseline_checkpoint_dir += '_aug'
        warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
        tmp = torch.load(warmup_resume_file)
        if tmp is not None: 
            state = tmp['state']
            state_keys = list(state.keys())
            for i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.feature.load_state_dict(state)
        else:
            raise ValueError('No warm_up file')
    model = train(base_loader, val_loader,  model, optimization, start_epoch, stop_epoch, params)
