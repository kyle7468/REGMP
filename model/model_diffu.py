
import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from model.models_layer import GATLayer
from model.base_class import BaseGNN
from helper import *
from model.function_diffu import EdgeODEFuncAtt
from model.block_diffu import EdgeAttODEblock
from model.early_stop_solver import EarlyStopInt


# class BaseGNN(MessagePassing):
# 	def __init__(self, edge_index, edge_type, num_rel, params=None):
# 		super(BaseGNN, self).__init__()
# 		self.p		= params
# 		self.act	= torch.tanh
# 		self.bceloss = torch.nn.BCELoss()
# 		self.edge_index = edge_index
# 		self.edge_type = edge_type
# 		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
#
# 	def loss(self, pred, true_label):
# 		return self.bceloss(pred, true_label)
#
# 	def getNFE(self):
# 		return self.odeblock.odefunc.nfe + self.odeblock.reg_odefunc.odefunc.nfe


class EdgeAttentionDiffusion_ConvE(BaseGNN):
	def __init__(self, edge_index, edge_type, params=None):
		new_x, edge_index, edge_type = self.construct_new_graph(edge_index, edge_type, params)
		super(EdgeAttentionDiffusion_ConvE, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.x = get_param(new_x.shape)
		print(self.x.shape)
		self.x.data = new_x
		self.f = EdgeODEFuncAtt
		block = EdgeAttODEblock
		self.device = params.device
		time_tensor = torch.tensor([0, self.T]).to(self.device)
		self.odeblock = block(self.f, self.regularization_fns, params, edge_index, edge_type, self.device, t=time_tensor).to(self.device)
		# self.relation = get_param()
		self.bn_out = torch.nn.BatchNorm1d(params.embed_dim)
		self.dropout_output = nn.Dropout(0.3)
		self.W_Z = get_param((params.init_dim, params.embed_dim))

	def construct_new_graph(self, edge_index, edge_type, params):
		# print('params', params)
		const_alpha = 0.2
		edge_num = edge_index.size(1)
		new_node_ids = torch.arange(edge_num, device=params.device) + params.num_ent
		x = get_param((params.num_ent, params.init_dim))
		edge_feature = get_param((params.num_rel * 2, params.init_dim))
		new_node_x = x[edge_index[0]]
		new_node_x.mul_(const_alpha).add_(edge_feature[edge_type], alpha=1. - const_alpha)
		new_x = torch.cat([x, new_node_x], dim=0).to('cuda')
		edge_index_slow = [
			torch.stack([edge_index[0], new_node_ids]),
			torch.stack([new_node_ids, edge_index[1]]),
		]
		new_edge_index = torch.cat([*edge_index_slow], dim=1)

		return new_x, new_edge_index, edge_type


	def set_solver_m2(self):
		self.odeblock.test_integrator.m2_weight = self.m2.weight.data.detach().clone().to(self.device)
		self.odeblock.test_integrator.m2_bias = self.m2.bias.detach().clone().to(self.device)

	def set_solver_data(self, data):
		self.odeblock.test_integrator.data = data

	def cleanup(self):
		del self.odeblock.test_integrator.m2
		torch.cuda.empty_cache()

	def forward(self, sub, rel):
		self.odeblock.set_x0(self.x)
		# x = self.x
		# with torch.no_grad():
		# 	self.set_solver_m2()

		z = self.odeblock(self.x)

		# print(z.shape)

		z = F.elu(z)

		z_self = torch.matmul(z, self.W_Z)
		z = z + z_self
		z = self.bn_out(z)


		# z = self.bn_out(z)
		z = self.dropout_output(z)
		# z = self.m2(z)

		# relation_node_emb = z[self.p.num_ent:,]



		edge_feature = self.odeblock.odefunc.edge_feature
		edge_feature = torch.matmul(edge_feature, self.odeblock.odefunc.W)
		# if not self.training:
		# 	print(edge_feature)

		sub_emb = torch.index_select(z, 0, sub)
		rel_emb = torch.index_select(edge_feature, 0, rel)


		stk_inp = self.concat(sub_emb, rel_emb)
		x1 = self.odeblock.odefunc.bn0(stk_inp)
		x1 = self.odeblock.odefunc.m_conv1(x1)
		x1 = self.odeblock.odefunc.bn1(x1)
		x1 = F.relu(x1)
		x1 = self.odeblock.odefunc.feature_drop(x1)
		x1 = x1.view(-1, self.odeblock.odefunc.flat_sz)
		x1 = self.odeblock.odefunc.fc(x1)
		x1 = self.odeblock.odefunc.hidden_drop2(x1)
		x1 = self.odeblock.odefunc.bn2(x1)
		x1 = F.relu(x1)

		x1 = torch.mm(x1, z[:self.p.num_ent,].detach().transpose(1, 0))
		x1 += self.odeblock.odefunc.bias1.expand_as(x1)

		score = torch.sigmoid(x1)
		# print(score.shape)
		return score

	def concat(self, e1_embed, rel_embed):
		e1_embed	= e1_embed. view(-1, 1, self.odeblock.odefunc.embed_dim)
		rel_embed	= rel_embed.view(-1, 1, self.odeblock.odefunc.embed_dim)
		stack_inp	= torch.cat([e1_embed, rel_embed], 1)
		stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.odeblock.odefunc.k_w, self.odeblock.odefunc.k_h))
		return stack_inp