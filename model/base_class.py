import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from model.utils import Meter
from model.regularized_ODE_function import RegularizedODEfunc
import model.regularized_ODE_function as reg_lib
import six

REGULARIZATION_FNS = {
    "kinetic_energy": reg_lib.quadratic_cost,
    "jacobian_norm2": reg_lib.jacobian_frobenius_regularization_fn,
    "total_deriv": reg_lib.total_derivative,
    "directional_penalty": reg_lib.directional_derivative
}


def create_regularization_fns(args):
    regularization_fns = []
    regularization_coeffs = []

    for arg_key, reg_fn in six.iteritems(REGULARIZATION_FNS):
        if args[arg_key] is not None:
            regularization_fns.append(reg_fn)
            regularization_coeffs.append(args[arg_key])

    regularization_fns = regularization_fns
    regularization_coeffs = regularization_coeffs
    return regularization_fns, regularization_coeffs


class ODEblock(nn.Module):
    def __init__(self, odefunc, regularization_fns, params, edge_index, edge_type, device, t):
        super(ODEblock, self).__init__()
        self.opt = vars(params)
        self.t = t

        self.odefunc = odefunc(edge_index, edge_type, params)

        self.nreg = len(regularization_fns)
        # self.reg_odefunc = RegularizedODEfunc(self.odefunc, regularization_fns)

        if self.opt['adjoint']:
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint

        self.train_integrator = odeint
        # self.test_integrator = None
        self.set_tol()

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()
        # self.reg_odefunc.odefunc.x0 = x0.clone().detach()

    # def set_rel_emb(self, rel_emb):
    #     self.odefunc.rel_emb = rel_emb
    #
    # def set_rel_emb_hidden(self, rel_emb_hidden):
    #     self.odefunc.rel_emb_hidden = rel_emb_hidden

    def set_tol(self):
        self.atol = self.opt['tol_scale'] * 1e-7
        self.rtol = self.opt['tol_scale'] * 1e-9
        if self.opt['adjoint']:
            self.atol_adjoint = self.opt['tol_scale_adjoint'] * 1e-7
            self.rtol_adjoint = self.opt['tol_scale_adjoint'] * 1e-9

    def reset_tol(self):
        self.atol = 1e-7
        self.rtol = 1e-9
        self.atol_adjoint = 1e-7
        self.rtol_adjoint = 1e-9

    def set_time(self, time):
        self.t = torch.tensor([0, time]).to(self.device)

    def __repr__(self):
        return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
               + ")"


class ODEFunc(MessagePassing):

    # currently requires in_features = out_features
    def __init__(self, opt, data, device):
        super(ODEFunc, self).__init__()
        self.opt = vars(opt)
        self.device = device
        self.edge_index = None
        self.edge_weight = None
        self.attention_weights = None
        self.alpha_train = nn.Parameter(torch.tensor(0.0))
        self.beta_train = nn.Parameter(torch.tensor(0.0))
        self.x0 = None
        self.nfe = 0
        self.alpha_sc = nn.Parameter(torch.ones(1))
        self.beta_sc = nn.Parameter(torch.ones(1))

    def __repr__(self):
        return self.__class__.__name__


class BaseGNN(MessagePassing):
    def __init__(self, edge_index, edge_type, num_rel, params=None):
        super(BaseGNN, self).__init__()
        # self.opt = opt
        self.T = params.time
        # self.num_classes = dataset.num_classes
        # self.num_features = dataset.data.num_features
        # self.num_features = 64
        # self.num_nodes = dataset.data.num_nodes
        # self.device = device
        self.fm = Meter()
        self.bm = Meter()

        self.p = params
        self.act = torch.tanh
        self.bceloss = torch.nn.BCELoss()
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.register_parameter('bias', nn.Parameter(torch.zeros(self.p.num_ent)))
        # self.regularization_fns, self.regularization_coeffs = create_regularization_fns(params)
        self.regularization_fns, self.regularization_coeffs = [], []
        # self.m2 = nn.Linear(params.embed_dim, dataset.num_classes)


    def getNFE(self):
        # return self.odeblock.odefunc.nfe + self.odeblock.reg_odefunc.odefunc.nfe
        return self.odeblock.odefunc.nfe

    def resetNFE(self):
        self.odeblock.odefunc.nfe = 0
        # self.odeblock.reg_odefunc.odefunc.nfe = 0

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)

    def __repr__(self):
        return self.__class__.__name__

