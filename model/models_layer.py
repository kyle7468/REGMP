from torch import nn
import torch
import torch.nn.functional as F
# from os import PRIO_PGRP
from torch import nn
import torch
import torch.nn.functional as F
from .helper import *
from .message_passing import MessagePassing
# from torch_geometric.utils import softmax
from torch_scatter import scatter_max, scatter_add
from .message_passing import scatter_

CUDA = torch.cuda.is_available()

def softmax(src, index, num_nodes=None):
  r"""Computes a sparsely evaluated softmax.
  Given a value tensor :attr:`src`, this function first groups the values
  along the first dimension based on the indices specified in :attr:`index`,
  and then proceeds to compute the softmax individually for each group.

  Args:
      src (Tensor): The source tensor.
      index (LongTensor): The indices of elements for applying the softmax.
      num_nodes (int, optional): The number of nodes, *i.e.*
          :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

  :rtype: :class:`Tensor`
  """
  # num_nodes = maybe_num_nodes(index, num_nodes)

  out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
  out = out.exp()
  out = out / (
          scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

  return out


class GATLayer(torch.nn.Module):

  def __init__(self, num_nodes, in_features, out_features, nrela_dim, dropout, alpha, concat=True):
    super(GATLayer, self).__init__()

    self.in_features = in_features
    self.out_features = out_features
    self.num_nodes = num_nodes
    self.alpha = alpha
    self.concat = concat
    self.nrela_dim = nrela_dim

    # self.W = get_param((2 * in_features + nrela_dim, out_features))
    self.W = get_param((2 * in_features, out_features))
    self.a = get_param((out_features, 1))

    self.dropout = nn.Dropout(dropout)
    self.leakyrelu = nn.LeakyReLU(self.alpha)

  def forward(self, edge_index, x, edge_type_embed):

    # print('edge: ', x.size(), edge_type_embed.size())
    # edge_h = torch.cat(
    #   (x[edge_index[0, :], :], x[edge_index[1, :], :], edge_type_embed[:, :]), dim=1)
    edge_h = torch.cat(
      (x[edge_index[0, :], :], x[edge_index[1, :], :]), dim=1)

    edge_h = torch.matmul(edge_h, self.W)
    # print('edge_h: ', edge_h.size())

    alpha = self.leakyrelu(torch.matmul(edge_h, self.a).squeeze())

    # print('alpha: ', alpha.size())
    alpha = softmax(alpha, edge_index[0], x.size(0))
    # return alpha
    alpha = self.dropout(alpha)

    # print('alpha after: ', alpha.size())
    out = self.path_message(edge_h, edge_index, size=x.size(0), edge_norm=alpha)

    if self.concat:
      # if this layer is not last layer,
      return F.elu(out)
    else:
      # if this layer is last layer,
      return out
    # return out

  def path_message(self, x, edge_index, size, edge_norm):
    # print('x bef: ', edge_norm)
    # print('x bef: ', x)

    x = edge_norm.unsqueeze(1) * x

    # print('x aff: ', x)
    out = scatter_('add', x, edge_index[0], dim_size=size)
    # print('out', out.size())
    return out





















