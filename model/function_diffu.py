import torch
from torch import nn
from torch_geometric.utils import softmax
import torch_sparse
from torch_geometric.utils.loop import add_remaining_self_loops
from model.utils import MaxNFEException, squareplus
from model.base_class import ODEFunc
import torch.nn.functional as F
from model.models_layer import GATLayer
from helper import *

class EdgeODEFuncAtt(ODEFunc):
  def __init__(self, edge_index, edge_type, params=None):
    super(EdgeODEFuncAtt, self).__init__(params, None, device='cuda')
    heads = params.heads
    self.p = params
    # new_x, edge_index, edge_type = self.construct_new_graph(edge_index, edge_type, params)
    # self.x = new_x
    self.init_dim = params.init_dim
    self.init_rel = params.init_dim
    self.embed_dim = params.embed_dim
    self.hidden_dim = self.embed_dim // heads
    self.do = 0.3
    self.alpha = 0.2
    self.edge_index = edge_index
    self.edge_type = edge_type

    self.W_entities = get_param((self.init_dim, self.embed_dim))

    self.attentions = [
      GATLayer(self.p.num_ent, self.init_dim, self.init_dim, self.init_rel, self.do, self.alpha,
               concat=True) for _ in range(heads)]
    for i, attention in enumerate(self.attentions):
      self.add_module('attention_{}'.format(i), attention)
    self.out_att = GATLayer(self.p.num_ent, self.embed_dim, self.embed_dim, self.embed_dim, self.do, self.alpha,
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
    self.W = get_param((self.init_dim, self.embed_dim))
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
    # self.edge_feature_final = None

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


  def forward(self, t, x):  # t is needed when called by the integrator

    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException

    self.nfe += 1
    # if self.nfe % 1000 == 0:
    #   print('nfe', self.nfe)

    edge_feature = self.edge_feature
    edge_type = self.edge_type
    self.edge_index = self.edge_index[[1, 0]]
    edge_index = self.edge_index

    # ax = torch.cat([att(edge_index, x, edge_feature[edge_type])
    #                for att in self.attentions], dim=1)
    # ax = self.dropout_layer(ax)
    # ax = torch.mean(torch.stack([att(edge_index, x, edge_feature[edge_type])
    #                 for att in self.attentions], dim=0), dim=0)
    attention = torch.stack([att(edge_index, x, edge_feature[edge_type])
                             for att in self.attentions], dim=1)
    # print(attention.shape)
    ax = torch.mean(torch.stack(
      [torch_sparse.spmm(self.edge_index, attention[:, idx], x.shape[0], x.shape[0], x) for idx in
       range(self.p.heads)], dim=0), dim=0)


    # edge_feature = torch.matmul(edge_feature, self.W)

    # ax = self.out_att(edge_index, ax, edge_feature[edge_type])
    # ax = F.elu(ax)

    # x_self = torch.matmul(x, self.W_entities)
    # ax = ax + x_self
    # ax = self.bn(ax)
    # ax = self.dropout_layer2(ax)
    # x = self.bn_x(x)

    # todo would be nice if this was more efficient

    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train
    # print('alpha', alpha)
    f = alpha * (ax - x)
    # f = (1 - alpha) * ax - alpha * x
    # f = alpha * ax
    # f = self.bn(f)
    # print("f shape", f.shape, x.shape)
    # f = ax
    # self.edge_feature_final = edge_feature
    # self.edge_feature = edge_feature
    # x_self = torch.matmul(x, self.W_entities)
    # f = self.alpha_train * (ax - x_self)
    # if self.opt['add_source']:
    #   f = f + self.beta_train * self.x0
    f = f + self.beta_train * self.x0
    # f = F.elu(f)
    return f
