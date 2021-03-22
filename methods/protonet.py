# This code is modified from https://github.com/jakesnell/prototypical-networks 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

class ProtoNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_query, max_n_way, min_n_way, max_shot, min_shot, fixed_way):
        super(ProtoNet, self).__init__( model_func,  n_way, n_query, max_n_way, min_n_way, max_shot, min_shot, fixed_way)
        self.loss_fn = nn.CrossEntropyLoss()


    def set_forward(self,x,n_support,selected_mean_logits = None,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)
        z_support   = z_support.contiguous()
        if selected_mean_logits == None:
            selected_mean_logits = torch.randn(self.n_way, 64).cuda()
        if n_support == -1:
            z_support = z_support.view(self.n_way, self.max_shot, -1)
            z_proto = []
            class_num = []
            intra_class_dist = []
            for i in range(self.n_way):
                selected_n_support = np.random.randint(self.min_shot, self.max_shot+1)
                selected_z_support = z_support[i, :selected_n_support+1, :] # shape: [selected_n_support, feat_dim]
                selected_z_proto = selected_z_support.mean(0).view(1, -1)
                intra_class_dist.append(torch.square(selected_mean_logits[i] - selected_z_proto).sum())
                z_proto.append(selected_z_proto)
                class_num.append(selected_n_support)
                
                #selected_n_support = np.random.randint(self.min_shot, self.max_shot+1)
                #z_proto.append(z_support[i, :selected_n_support+1, :].mean(0).view(1, -1))               
                
            z_proto = torch.cat(z_proto, axis=0)
            #z_proto_mean = z_proto.mean(0).view(1, -1)
            #inter_class_dist = torch.square(z_proto - z_proto_mean).sum(axis=1)
        else:

            selected_n_support = n_support
            class_num = [selected_n_support] * self.n_way
            z_proto     = z_support.view(self.n_way, selected_n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
            intra_class_dist = [0]*self.n_way
        #print("%d-way, %d-shot, %d-query" % (self.n_way, n_support, self.n_query))
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )
        '''
        intra_class_dist_query = []
        z_proto_query = []
        for i in range(self.n_way):
            z_query_i = z_query[i*self.n_query:(i+1)*self.n_query, :]
            z_proto_query_i = z_query_i.mean(0).view(1, -1)
            intra_class_dist_query.append(torch.square(z_query_i - z_proto_query_i).sum())
            z_proto_query.append(z_proto_query_i)
        z_proto_query = torch.cat(z_proto_query, axis=0)
        z_proto_query_mean = z_proto_query.mean(0).view(1, -1)
        inter_class_dist_query = torch.square(z_proto_query - z_proto_query_mean).sum(axis=1)
        '''
        class_num = torch.FloatTensor(class_num).cuda()
        regularizer_loss = torch.matmul(class_num, torch.cuda.FloatTensor(intra_class_dist)) / class_num.sum()
        
        #regularizer_loss_query = (self.n_way / self.n_query) * (torch.FloatTensor(intra_class_dist_query).sum() / inter_class_dist_query.sum())

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        # return scores
        return scores, regularizer_loss

    def set_forward_loss(self, x, n_support, selected_mean_logits): # selected_mean_logits: [self.n_way, feat_dim]
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        #scores = self.set_forward(x, n_support)
        scores, regularizer_loss = self.set_forward(x, n_support, selected_mean_logits)
        classification_loss = self.loss_fn(scores, y_query)
        total_loss = classification_loss + 0.5*torch.log(regularizer_loss)
        #print(classification_loss, regularizer_loss)
        #print(classification_loss, regularizer_loss, regularizer_loss_query)
        #return self.loss_fn(scores, y_query )
        return total_loss


def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
