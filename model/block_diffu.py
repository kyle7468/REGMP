import torch
from model.models_layer import  GATLayer
from model.base_class import ODEblock
from model.utils import get_rw_adj


class EdgeAttODEblock(ODEblock):
  def __init__(self, odefunc, regularization_fns, params, edge_index, edge_type, device, t=torch.tensor([0, 1]), gamma=0.5):
    super(EdgeAttODEblock, self).__init__(odefunc, regularization_fns, params, edge_index, edge_type, device, t)
    self.odefunc = odefunc(edge_index, edge_type, params)

    # self.reg_odefunc.odefunc.edge_index = self.odefunc.edge_index

    if self.opt['adjoint']:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint
    self.train_integrator = odeint
    # self.test_integrator = odeint
    self.set_tol()

  def forward(self, x):
    # z = self.odefunc(None, x)
    # print(z.shape)
    # return z
    t = self.t.type_as(x)

    integrator = self.train_integrator

    func = self.odefunc
    state = x

    if self.opt["adjoint"] and self.training:
      state_dt = integrator(
        func, state, t,
        method=self.opt['method'],
        options={'step_size': self.opt['step_size'], 'max_iters': self.opt['max_iters']},
        adjoint_method=self.opt['adjoint_method'],
        adjoint_options={'step_size': self.opt['adjoint_step_size']},
        atol=self.atol,
        rtol=self.rtol,
        adjoint_atol=self.atol_adjoint,
        adjoint_rtol=self.rtol_adjoint)
    else:
      state_dt = integrator(
        func, state, t,
        method=self.opt['method'],
        options={'step_size': self.opt['step_size'], 'max_iters': self.opt['max_iters']},
        atol=self.atol,
        rtol=self.rtol)

    z = state_dt[1]
    return z
