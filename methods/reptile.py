import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

class Reptile(MetaTemplate):
	def __init__(self, model_func, n_way, n_support):
		super(Reptile, self).__init__(model_func, n_way, n_support, change_way = False)

		self.loss_fn = nn.CrossEntropyLoss()
		self.classifier = backbone.Linear_fw(self.feat_dim, n_way)
		self.classifier.bias.data.fill_(0)

		self.n_task = 4
		self.task_update_num = 5
		self.train_inner_lr = 0.02
		self.train_outer_lr = 0.1

	def forward(self, x):
		out = self.feature.forward(x)
		scores = self.classifier.forward(out)
		return scores

	def set_forward_train(self, x, is_feature=False):
		pass

	def set_forward(self, x, is_feature=False):
		assert is_feature == False, 'Reptile do not support fixed feature'
		x = x.cuda()
		x_var = Variable(x)
		#print(x_var.shape)
		x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view(self.n_way*self.n_support, *x.size()[2:])
		#print(x_a_i.shape)
		x_b_i = x_var[:,self.n_support:,:,:,:].contiguous().view(self.n_way*self.n_query,	*x.size()[2:])
		
		#print(x_b_i.shape)
		y_a_i = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_support))).cuda()

		fast_parameters = list(self.parameters())
		for weight in self.parameters():
			weight.fast = None
		self.zero_grad()

		delta_parameters = []
		for task_step in range(self.task_update_num):
			scores = self.forward(x_a_i)
			set_loss = self.loss_fn(scores, y_a_i)
			grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True)
			fast_parameters = []
			for k, weight in enumerate(self.parameters()):
				if weight.fast is None:
					weight.fast = weight - self.train_inner_lr * grad[k]
				else:
					weight.fast = weight.fast - self.train_inner_lr * grad[k]
				fast_parameters.append(weight.fast)
		#print(fast_parameters)
		#print(list(self.parameters))
		for weight_after, weight_before in zip(fast_parameters, list(self.parameters())):
			delta_parameters.append(weight_after-weight_before)
		scores = self.forward(x_b_i)
		return delta_parameters, set_loss, scores

	def set_forward_adaptation(self, x, is_feature=False):
		raise ValueError('Reptile performs further adaptation simply by increasing task_update_num')

	def set_forward_loss(self, x):
		pass

	def train_loop(self, epoch, train_loader, optimizer):
		print_freq = 10
		avg_loss = 0
		task_count = 0
		delta_all = []

		for i, (x,_) in enumerate(train_loader):
			self.n_query = x.size(1) - self.n_support
			assert self.n_way == x.size(0), "Reptile do not support way change"

			delta_parameters, loss, scores = self.set_forward(x)
			avg_loss = avg_loss + loss.item()
			if len(delta_all) == 0:
				delta_all = delta_parameters
			else:
				for i, weight in enumerate(delta_parameters):
					delta_all[i] += weight

			task_count += 1

			if task_count == self.n_task:
				outer_lr = self.train_outer_lr * ( 1 - epoch*i/120000.)
				for k, weight in enumerate(self.parameters()):
					weight = weight + outer_lr / self.n_task * delta_all[k]
				task_count = 0
				delta_all = []
			if i % print_freq == 0:
				print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))


	def test_loop(self, test_loader, return_std=False):
		correct = 0
		count = 0
		acc_all = []

		iter_num = len(test_loader)
		for i, (x, _) in enumerate(test_loader):
			self.n_query = x.size(1) - self.n_support
			assert self.n_way == x.size(0), "Reptile do not support way change"
			correct_this, count_this = self.correct(x)
			acc_all.append(correct_this/ count_this * 100)

		acc_all = np.asarray(acc_all)
		acc_mean = np.mean(acc_all)
		acc_std = np.std(acc_all)
		print('%d Test Acc = %4.2f%% += %4.2f%%' %(iter_num, acc_mean, 1.96*acc_std/np.sqrt(iter_num)))
		if return_std:
			return acc_mean, acc_std
		else:
			return acc_mean