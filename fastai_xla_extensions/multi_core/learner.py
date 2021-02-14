# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03b_multi_core.learner.ipynb (unless otherwise specified).

__all__ = ['make_xla_child_learner', 'xla_run_method']

# Internal Cell
from ..utils import xla_imported

# Internal Cell
from .base import *
from ..misc_utils import *

# Internal Cell
# import sys
# def xla_imported():
#     return 'torch_xla' in sys.modules

# Internal Cell
try:
    import torch_xla
except ImportError:
    pass

# Internal Cell
if xla_imported():
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp


# Cell

from fastai.callback.progress import ProgressCallback

from fastai.learner import Learner
def make_xla_child_learner(rank, sync_valid,learner_args, add_args, ctrl_args):
    "create a learner using passed parameters"
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    dls = build_distributed_dataloaders(learner_args.pop('base_dls'),
                                       rank, world_size, sync_valid=sync_valid)

    model = learner_args.pop('wrapped_model').to(device)

    learner = Learner(dls, model,**learner_args)
    learner.__stored_args__ = {**learner.__stored_args__, **add_args}
    learner.to_multi_xla(device, rank, sync_valid=sync_valid)
    if not ctrl_args['use_progress'] and 'progress' in L(learner.cbs).attrgot('name'):
        learner.remove_cbs(ProgressCallback)

    return learner

# Cell
def xla_run_method(rank, fit_method, learner_args, add_args, fit_args, ctrl_args):
    "run fit method on spawned process"
    sync_valid = True
    learner = make_xla_child_learner(rank, sync_valid, learner_args, add_args, ctrl_args)
    fit_method(learner, **fit_args)
    learner.save('_xla_tmp_model')
    xm.mark_step()

# Cell
from fastcore.basics import defaults, patch_to, patch

_extra_args = ['concat_pool', 'arch', 'n_out', 'pretrained','normalize']

@patch
def pack_learner_args(self:Learner):
    "pack learner args into dict to pass to spawned process"
    learner_args = {**self.__stored_args__}
    learner_args['wrapped_model'] =  xmp.MpModelWrapper(self.model)
    learner_args['base_dls'] = self.dls
   # fetch only cbs not in defaults
    if ProgressCallback not in defaults.callbacks:
        defaults.callbacks.append(ProgressCallback)
    learner_args['cbs'] = [cb for cb in self.cbs
                      if cb.name not in L(defaults.callbacks).attrgot('name')]

    add_args = {}
    for arg in _extra_args:
        if arg in learner_args:
            add_args[arg] = learner_args.pop(arg)
    return learner_args, add_args

# Cell
@patch
def reload_child_model(self:Learner):
    "reload model built by spawned processes"
    # blatantly stolen from fastai LRFinder after_fit :)
    tmp_f = self.path/self.model_dir/'_xla_tmp_model.pth'
    if tmp_f.exists():
        self.opt.zero_grad()
        self.load('_xla_tmp_model', with_opt=False)
        os.remove(tmp_f)
        self.create_opt()

# Cell

from fastcore.foundation import L

@patch
def pre_xla_fit(self:Learner, ctrl_args={}):
    "prepare learner for running spawned processes"
    progress_removed = False
    if 'progress' in L(self.cbs).attrgot('name'):
        self.remove_cbs(ProgressCallback)
        progress_removed = True
    ctrl_args['use_progress'] = progress_removed
    return ctrl_args

@patch
def post_xla_fit(self:Learner, ctrl_args):
    "clean up learner after running spawned processes"
    if ctrl_args['use_progress']:
        self.add_cbs(ProgressCallback)

# Cell

from fastcore.meta import delegates

@patch
@delegates(Learner.fit, but='num_cores,start_method')
def xla_fit(self:Learner, n_epoch, num_cores=8, start_method='fork', **kwargs):
    """call fit in multicore tpu environment"""
    ctrl_args = self.pre_xla_fit()
    learner_args, add_args = self.pack_learner_args()
    fit_args={**kwargs}

    fit_args['n_epoch'] = n_epoch
    xmp.spawn(xla_run_method,
              args=(Learner.fit, learner_args, add_args, fit_args, ctrl_args),
              nprocs=num_cores,
              start_method=start_method)

    self.reload_child_model()
    self.post_xla_fit(ctrl_args)

# Cell
from fastai.learner import Learner
from fastai.callback.schedule import *
@patch
@delegates(Learner.fit_one_cycle, but='num_cores,start_method')
def xla_fit_one_cycle(self:Learner, n_epoch, num_cores=8, start_method='fork', **kwargs):
    """call fit_one_cycle in multicore tpu environment"""
    ctrl_args = self.pre_xla_fit()
    learner_args, add_args = self.pack_learner_args()
    fit_args={**kwargs}
    fit_args['n_epoch'] = n_epoch
    xmp.spawn(xla_run_method,
              args=(Learner.fit_one_cycle, learner_args, add_args, fit_args, ctrl_args),
              nprocs=num_cores,
              start_method=start_method)

    self.reload_child_model()
    self.post_xla_fit(ctrl_args)