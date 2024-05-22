import torch

from helper import *
from model.compgcn_conv import CompGCNConv
from model.compgcn_conv_basis import CompGCNConvBasis
from torch import nn
# from model.models_layer import SpGraphAttentionLayer, GATLayer
from model.models_layer import GATLayer

class BaseModel(torch.nn.Module):
	def __init__(self, params):
		super(BaseModel, self).__init__()

		self.p		= params
		self.act	= torch.tanh
		self.bceloss	= torch.nn.BCELoss()

	def loss(self, pred, true_label):
		return self.bceloss(pred, true_label)
		
class CompGCNBase(BaseModel):
	def __init__(self, edge_index, edge_type, num_rel, params=None):
		super(CompGCNBase, self).__init__(params)

		self.edge_index		= edge_index
		self.edge_type		= edge_type
		self.p.gcn_dim		= self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
		self.init_embed		= get_param((self.p.num_ent,   self.p.init_dim))
		self.device		= self.edge_index.device

		if self.p.num_bases > 0:
			self.init_rel  = get_param((self.p.num_bases,   self.p.init_dim))
		else:
			if self.p.score_func == 'transe': 	self.init_rel = get_param((num_rel,   self.p.init_dim))
			else: 					self.init_rel = get_param((num_rel*2, self.p.init_dim))

		if self.p.num_bases > 0:
			self.conv1 = CompGCNConvBasis(self.p.init_dim, self.p.gcn_dim, num_rel, self.p.num_bases, act=self.act, params=self.p)
			self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None
		else:
			self.conv1 = CompGCNConv(self.p.init_dim, self.p.gcn_dim,      num_rel, act=self.act, params=self.p)
			self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

	def forward_base(self, sub, rel, drop1, drop2):

		r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
		x, r	= self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)
		x	= drop1(x)
		x, r	= self.conv2(x, self.edge_index, self.edge_type, rel_embed=r) 	if self.p.gcn_layer == 2 else (x, r)
		x	= drop2(x) 							if self.p.gcn_layer == 2 else x

		sub_emb	= torch.index_select(x, 0, sub)
		rel_emb	= torch.index_select(r, 0, rel)

		return sub_emb, rel_emb, x


class CompGCN_TransE(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.drop, self.drop)
		obj_emb				= sub_emb + rel_emb

		x	= self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)		
		score	= torch.sigmoid(x)

		return score

class CompGCN_DistMult(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.drop, self.drop)
		obj_emb				= sub_emb * rel_emb

		x = torch.mm(obj_emb, all_ent.transpose(1, 0))
		x += self.bias.expand_as(x)

		score = torch.sigmoid(x)
		return score

class CompGCN_ConvE(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)

		self.bn0		= torch.nn.BatchNorm2d(1)
		self.bn1		= torch.nn.BatchNorm2d(self.p.num_filt)
		self.bn2		= torch.nn.BatchNorm1d(self.p.embed_dim)
		
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		self.hidden_drop2	= torch.nn.Dropout(self.p.hid_drop2)
		self.feature_drop	= torch.nn.Dropout(self.p.feat_drop)
		self.m_conv1		= torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)

		flat_sz_h		= int(2*self.p.k_w) - self.p.ker_sz + 1
		flat_sz_w		= self.p.k_h 	    - self.p.ker_sz + 1
		self.flat_sz		= flat_sz_h*flat_sz_w*self.p.num_filt
		self.fc			= torch.nn.Linear(self.flat_sz, self.p.embed_dim)

	def concat(self, e1_embed, rel_embed):
		e1_embed	= e1_embed. view(-1, 1, self.p.embed_dim)
		rel_embed	= rel_embed.view(-1, 1, self.p.embed_dim)
		stack_inp	= torch.cat([e1_embed, rel_embed], 1)
		stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))
		return stack_inp

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)
		stk_inp				= self.concat(sub_emb, rel_emb)
		x				= self.bn0(stk_inp)
		x				= self.m_conv1(x)
		x				= self.bn1(x)
		x				= F.relu(x)
		x				= self.feature_drop(x)
		x				= x.view(-1, self.flat_sz)
		x				= self.fc(x)
		x				= self.hidden_drop2(x)
		x				= self.bn2(x)
		x				= F.relu(x)

		x = torch.mm(x, all_ent.transpose(1,0))
		x += self.bias.expand_as(x)

		score = torch.sigmoid(x)
		return score




class EdgeAttention_ConvE(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		new_x, edge_index, edge_type = self.construct_new_graph(edge_index, edge_type, params)
		super(EdgeAttention_ConvE, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.x = get_param(new_x.shape)
		self.x.data = new_x
		heads = params.heads
		self.p = params
		# new_x, edge_index, edge_type = self.construct_new_graph(edge_index, edge_type, params)
		# self.x = new_x
		self.init_dim = params.init_dim
		self.init_rel_dim = params.init_dim
		self.embed_dim = params.embed_dim
		self.hidden_dim = self.embed_dim // heads
		self.do = 0.3
		self.alpha = 0.2
		self.edge_index = edge_index
		self.edge_type = edge_type

		self.W_entities = get_param((self.init_dim, self.embed_dim))

		self.attentions = [
			GATLayer(self.p.num_ent, self.init_dim, self.hidden_dim, self.init_rel_dim, self.do, self.alpha,
					 concat=True) for _ in range(heads)]
		for i, attention in enumerate(self.attentions):
			self.add_module('attention_{}'.format(i), attention)

		self.attentions2 = [
			GATLayer(self.p.num_ent, self.embed_dim, self.hidden_dim, self.init_rel_dim, self.do, self.alpha,
					 concat=True) for _ in range(heads)]
		for i, attention in enumerate(self.attentions2):
			self.add_module('attention2_{}'.format(i), attention)



		self.out_att = GATLayer(self.p.num_ent, self.embed_dim, self.embed_dim, self.init_rel_dim, self.do, self.alpha,
								concat=False)

		self.bn = torch.nn.BatchNorm1d(self.embed_dim)
		self.bn_x = torch.nn.BatchNorm1d(self.embed_dim)
		# self.bn = torch.nn.BatchNorm2d(1)
		self.gamma = params.gamma

		num_filter = params.num_filt
		ker_sz = params.ker_sz
		self.k_w = params.k_w
		self.k_h = params.k_h
		self.m_conv1 = torch.nn.Conv2d(1, out_channels=num_filter, kernel_size=(ker_sz, ker_sz), stride=1, padding=0,
									   bias=False)
		flat_sz_h = int(2 * self.k_w) - ker_sz + 1
		flat_sz_w = self.k_h - ker_sz + 1
		self.flat_sz = flat_sz_h * flat_sz_w * num_filter
		self.fc = torch.nn.Linear(self.flat_sz, self.embed_dim)
		self.W = get_param((self.embed_dim, self.embed_dim))
		self.edge_feature = get_param((self.p.num_rel * 2, self.init_dim))

		self.bn0 = torch.nn.BatchNorm2d(1)
		self.bn1 = torch.nn.BatchNorm2d(num_filter)
		self.bn2 = torch.nn.BatchNorm1d(self.embed_dim)

		self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
		self.hidden_drop2 = torch.nn.Dropout(self.p.hid_drop2)
		self.feature_drop = torch.nn.Dropout(self.p.feat_drop)

		self.register_parameter('bias1', Parameter(torch.zeros(self.p.num_ent)))

		self.dropout_layer = nn.Dropout(self.do)
		self.dropout_layer2 = nn.Dropout(self.do)
		self.dropout_layer3 = nn.Dropout(self.do)

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
		print(new_node_ids.shape, edge_num)
		edge_index_con = [
			torch.stack([new_node_ids[:edge_num//2], new_node_ids[edge_num//2:]])
		]
		new_edge_index = torch.cat([*edge_index_slow, *edge_index_con], dim=1)

		return new_x, new_edge_index, edge_type


	def forward(self, sub, rel):

			# x_0 = F.normalize(
			# 	self.x, p=2, dim=1)
			# x_0 = self.x

			# x = self.gat1(x, edge_index)
			# x = F.relu(x)
			# x = F.dropout(x, training=self.training)
			# x = self.gat2(x, edge_index)
			# print(data.edge_feature.shape)
			edge_feature = self.edge_feature
			# edge_feature = F.normalize(
			#     edge_feature, p=2, dim=1)
			# edge_feature = F.dropout(edge_feature, 0.3, training=self.training)
			# edge_feature = self.Wliner(edge_feature)
			# edge_type = torch.cat((data.edge_type, data.r2r_edge_type, data.n2e_edge_type))

			edge_type = self.edge_type
			# print('edge_feature', edge_feature.shape)
			# x = self.bn_in(x)
			# edge_feature = self.bn_in2(edge_feature)
			# print(x_0.shape, edge_index.shape, data.edge_feature.shape, edge_type.shape)
			# x = self.edge_attention(x, edge_index, edge_feature[edge_type])
			# x_0 = F.dropout(x_0, 0.3, training=self.training)
			# print('edge_index_shape:', self.edge_index.shape)
			# print('edge_type_shape:', self.edge_type.shape)
			self.edge_index = self.edge_index[[1, 0]]
			edge_index = self.edge_index

			x = torch.cat([att(edge_index, self.x, edge_feature[edge_type])
						   for att in self.attentions], dim=1)
			x = self.dropout_layer(x)
			# print(x.shape)

			x = torch.cat([att(edge_index, x, edge_feature[edge_type])
						   for att in self.attentions2], dim=1)
			x = self.dropout_layer2(x)
			# print(x.shape)

			# r_1 = F.elu(r_1)




			x = self.out_att(edge_index, x, edge_feature[edge_type])
			x = F.elu(x)


			edge_features = torch.zeros((self.p.num_rel * 2, self.p.embed_dim)).to('cuda')
			for i in range(self.p.num_rel * 2):
				edge_features[i] = torch.mean(
					torch.index_select(x[self.p.num_ent:, ], dim=0, index=torch.nonzero(edge_type == i).squeeze()),
					dim=0)

			edge_features = torch.matmul(edge_features, self.W)

			x_self = torch.matmul(self.x, self.W_entities)
			x = x + x_self
			x = self.bn(x)
			x = self.dropout_layer3(x)


			# x = self.bn2(x)
			# relation_emb = self.bn3(relation_emb)
			# x = F.elu(x)
			# x = F.normalize(x, p=2, dim=1)
			# x = F.tanh(x)
			# x = F.relu(x)
			sub_emb = torch.index_select(x, 0, sub)
			rel_emb = torch.index_select(edge_features, 0, rel)


			stk_inp = self.concat(sub_emb, rel_emb)
			x1 = self.bn0(stk_inp)
			x1 = self.m_conv1(x1)
			x1 = self.bn1(x1)
			x1 = F.relu(x1)
			x1 = self.feature_drop(x1)
			x1 = x1.view(-1, self.flat_sz)
			x1 = self.fc(x1)
			x1 = self.hidden_drop2(x1)
			x1 = self.bn2(x1)
			x1 = F.relu(x1)

			x1 = torch.mm(x1, x[:self.p.num_ent,].transpose(1, 0))
			x1 += self.bias1.expand_as(x1)

			score = torch.sigmoid(x1)
			return score

	def concat(self, e1_embed, rel_embed):
		e1_embed	= e1_embed. view(-1, 1, self.embed_dim)
		rel_embed	= rel_embed.view(-1, 1, self.embed_dim)
		stack_inp	= torch.cat([e1_embed, rel_embed], 1)
		stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.k_w, self.k_h))
		return stack_inp


