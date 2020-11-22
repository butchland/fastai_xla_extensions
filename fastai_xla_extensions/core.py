# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/00_core.ipynb (unless otherwise specified).

__all__ = ['XLA_AVAILABLE', 'PickableOpt', 'XLAOptimProxy', 'XLAOptCallback']

# Cell
#colab
#hide_output
XLA_AVAILABLE = True
try:
    import torch_xla.core.xla_model as xm
except ImportError as e:
    XLA_AVAILABLE = False

# Internal Cell
if not globals().get('XLA_AVAILABLE'):
    from types import SimpleNamespace
    import torch.cuda
    def fake_opt_step(opt,barrier=False):
        opt.step()
    def fake_device(n=None, devkind=None):
        gpu_available = torch.cuda.is_available()
        return torch.device(torch.cuda.current_device()) if gpu_available else torch.device('cpu')
    #xm = SimpleNamespace(
    #    optimizer_step = fake_opt_step,
    #    xla_device = fake_device
    #)


# Cell
from fastcore.foundation import GetAttr
from fastai.optimizer import Optimizer
from copy import deepcopy

class PickableOpt(Optimizer):
  def __init__(self, opt):
    # copy the passed optimizer dictionary
    self.__dict__ = deepcopy(opt.__dict__)
    self.opt = opt

  def __getstate__(self):
    v = self.state_dict()
    v['param_groups'] = self.param_groups
    return v

# Cell
class XLAOptimProxy(GetAttr):
    _default='opt'
    "Proxy optimizer to override `opt.step` with Pytorch XLA sync method `xm.optimizer_step` "
    def __init__(self,opt, barrier=True):
        self.opt = PickableOpt(opt)
        self._barrier = barrier

    def xla_step(self):
        xm.optimizer_step(self.opt,barrier=self._barrier) # sync on gradient update

    @property
    def barrier(self): return self._barrier
    @barrier.setter
    def barrier(self,v): self._barrier = v

# Cell
from fastai.callback.core import Callback
from fastai.data.core import DataLoaders
from fastai.vision.all import to_device


class XLAOptCallback(Callback):
    'Callback to replace `opt.step` with `xm.optimizer_step(opt)` as required to run on TPU'
    def __init__(self, barrier=True):
        self._barrier = barrier

    def before_fit(self):
        'replace opt with proxy which calls `xm.optimizer_step` instead of `opt.step` and set `dls.device` and model to `xla_device`'
        to_device(self.dls, device=xm.xla_device())
        self.model.to(self.dls.device)
        if self.learn.opt is not None:
            if not isinstance(self.learn.opt,XLAOptimProxy):
                opt = self.learn.opt
                self.learn.opt = XLAOptimProxy(opt, barrier=self._barrier)

    def after_fit(self):
        'restore original opt '
        if isinstance(self.learn.opt, XLAOptimProxy):
            opt = self.learn.opt.opt
            self.learn.opt = opt
    @property
    def barrier(self): return self._barrier
    @barrier.setter
    def barrier(self,v): self._barrier = v

# Cell
if globals().get('XLA_AVAILABLE'):
    from fastcore.foundation import defaults
    if hasattr(defaults,'callbacks'):
        if XLAOptCallback not in defaults.callbacks:
            defaults.callbacks.append(XLAOptCallback)
    else:
        defaults.callbacks = [XLAOptCallback]

# Cell
if globals().get('XLA_AVAILABLE'):
    from fastcore.foundation import patch
    from fastai.learner import Learner
    from fastai.callback.hook import summary as orig_summary
    @patch
    def xlasummary(self:Learner):
        to_device(self.dls, device=xm.xla_device())
        return orig_summary(self)
