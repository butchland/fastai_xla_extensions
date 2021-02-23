# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03b_multi_core.learner.ipynb (unless otherwise specified).

__all__ = ['master_cbs', 'make_xla_child_learner', 'setup_fit_cbs', 'xla_run_method', 'tmp_files', 'prep_fit_args']

# Internal Cell
from ..utils import xla_imported

# Internal Cell
from .base import *
from ..misc_utils import *
from .callback import *

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

# Internal Cell
from fastcore.basics import patch
from fastai.learner import Learner
from fastcore.meta import delegates
from fastcore.foundation import L


# Cell
from fastai.learner import Learner
from fastcore.basics import patch
@patch(as_prop=True)
def master_cbs(self:Learner):
    "list all cbs to be run on the master ordinal thread"
    if not hasattr(self,'_master_cbs'):
        self._master_cbs = L()
    return self._master_cbs


# Cell
@patch
def add_master_cb(self:Learner, cb):
    "add a master callback"
    if not hasattr(self,'_master_cbs'):
        self._master_cbs = L()
    if isinstance(cb, type): cb = cb()
#     cb.learn = self
#     setattr(self, cb.name, cb)
    self._master_cbs.append(cb)

@patch
def add_master_cbs(self:Learner, cbs):
    "add master callbacks"
    L(cbs).map(self.add_master_cb)



# Cell

@patch
def grab_master_cbs(self:Learner, cb_cls):
    "find instance of `cb_cls` in master_cbs"
    return L(cb for cb in self._master_cbs if isinstance(cb, cb_cls))

@patch
def remove_master_cb(self:Learner, cb):
    "remove a cb from master callbacks"
    if isinstance(cb, type): self.remove_master_cbs(self.grab_master_cbs(cb))
    else:
#         cb.learn = None
#         if hasattr(self, cb.name): delattr(self, cb.name)
        if cb in self._master_cbs: self._master_cbs.remove(cb)
    return self

@patch
def remove_master_cbs(self:Learner, cbs):
    "remove callbacks from master callbacks"
    L(cbs).map(self.remove_master_cb)
    return self

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
    master_cbs = learner_args.pop('master_cbs')
    if master_cbs is None:
        master_cbs = L()
    learner = Learner(dls, model,**learner_args)
    learner.__stored_args__ = {**learner.__stored_args__, **add_args}

    learner.to_multi_xla(device, rank, sync_valid=sync_valid)

    if not ctrl_args['use_progress'] and 'progress' in L(learner.cbs).attrgot('name'):
        learner.remove_cbs(ProgressCallback)

    if rank == 0:
        learner.add_cbs(master_cbs)

    return learner

# Cell
def setup_fit_cbs(rank, fit_args):
    "add master cbs to cbs fit args if rank 0"
    master_cbs = L(fit_args.pop('master_cbs'))
    if rank != 0:
        master_cbs = L()
    if 'cbs' in fit_args:
        cbs = L(fit_args.pop('cbs'))
    else:
        cbs = L()
    if len(master_cbs) > 0 or len(cbs) > 0:
        fit_args['cbs'] = [*cbs, *master_cbs]
    return fit_args

# Cell
def xla_run_method(rank, fit_method, learner_args, add_args, fit_args, ctrl_args):
    "run fit method on spawned process"
    sync_valid = True
    learner = make_xla_child_learner(rank, sync_valid, learner_args, add_args, ctrl_args)
    fit_args = setup_fit_cbs(rank, fit_args)
    fit_method(learner, **fit_args)
    xm.rendezvous('xla_run_method')
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
    default_cbs = [cls() for cls in defaults.callbacks]
    learner_args['cbs'] = [cb for cb in self.cbs
                      if cb.name not in L(default_cbs).attrgot('name')]

    learner_args['master_cbs'] = self.master_cbs

    # remove extra args from learner args (in __stored_args__ but not in init args)
    add_args = {}
    for arg in _extra_args:
        if arg in learner_args:
            add_args[arg] = learner_args.pop(arg)
    return learner_args, add_args

# Cell
import os

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
from pathlib import Path

tmp_files = ['_paramsched_hps.pkl', '_rec_attr.pkl']
@patch
def delete_tmp_files(self:Learner):
    '''remove files created by spawned process prior to
    potentially recreating them'''
    for fn in tmp_files:
        fn = Path(fn)
        if fn.is_file():
            fn.unlink()


@patch
def pre_xla_fit(self:Learner, ctrl_args={}):
    "prepare learner for running spawned processes"
    progress_removed = False
    if 'progress' in L(self.cbs).attrgot('name'):
        self.remove_cbs(ProgressCallback)
        progress_removed = True
    ctrl_args['use_progress'] = progress_removed
    self.delete_tmp_files()
    return ctrl_args

@patch
def post_xla_fit(self:Learner, ctrl_args):
    "clean up learner after running spawned processes"
    self.recorder.reload_attrs()
    self.recorder.reload_hps()
    if ctrl_args['use_progress']:
        self.add_cbs(ProgressCallback)

# Cell
def prep_fit_args(n_epoch, master_cbs, **kwargs):
    "prepare fit method args for running spawned processes"
    fit_args={**kwargs}
    fit_args['master_cbs'] = master_cbs
    fit_args['n_epoch'] = n_epoch
    return fit_args

# Cell

from fastcore.meta import delegates

@patch
@delegates(Learner.fit, but='num_cores,start_method,master_cbs')
def xla_fit(self:Learner, n_epoch, num_cores=8,
            start_method='fork', master_cbs=None, **kwargs):
    """call fit in multicore tpu environment"""
    ctrl_args = self.pre_xla_fit()
    learner_args, add_args = self.pack_learner_args()

    fit_args = prep_fit_args(n_epoch, master_cbs, **kwargs)

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
@delegates(Learner.fit_one_cycle, but='num_cores,start_method,master_cbs')
def xla_fit_one_cycle(self:Learner, n_epoch, num_cores=8,
                      start_method='fork', master_cbs=None, **kwargs):
    """call fit_one_cycle in multicore tpu environment"""
    ctrl_args = self.pre_xla_fit()
    learner_args, add_args = self.pack_learner_args()

    fit_args = prep_fit_args(n_epoch, master_cbs, **kwargs)

    xmp.spawn(xla_run_method,
              args=(Learner.fit_one_cycle, learner_args, add_args, fit_args, ctrl_args),
              nprocs=num_cores,
              start_method=start_method)

    self.reload_child_model()
    self.post_xla_fit(ctrl_args)