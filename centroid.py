from scipy.io import loadmat
import torch

feature_dim = 512
num_class = 22
mean_var = 10
kernel_dict = loadmat('meanvar1_featuredim'+str(feature_dim)+'_class'+str(num_class)+'.mat')
mean_logits_np = kernel_dict['mean_logits'] #num_class X num_dense
mean_logits = mean_var * torch.FloatTensor(mean_logits_np)
