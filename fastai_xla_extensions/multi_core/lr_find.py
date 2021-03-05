# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03d_multi_core.lr_find.ipynb (unless otherwise specified).

__all__ = ['SkipValidationCallback', 'XLALRFinder', 'xla_run_lr_find']

# Internal Cell
from ..utils import xla_imported
from .base import *
from .callback import *
from .learner import *
from ..misc_utils import *
from ..core import *


# Internal Cell
try:
    import torch_xla
except:
    pass


# Internal Cell
if xla_imported():
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp

# Cell

from fastai.callback.core import Callback
from fastai.learner import CancelValidException
class SkipValidationCallback(Callback):
    order,run_valid = -9, False
    # raise CancelValidException before XLATrainingCallback.before_validate
    # to prevent call to wrap_parallel_loader on before_validate
    def before_validate(self):
        raise CancelValidException()

    def after_cancel_validate(self):
        xm.mark_step()


# Cell

from fastai.callback.schedule import ParamScheduler, SchedExp
from fastcore.xtras import is_listy
from fastcore.imports import noop
class XLALRFinder(ParamScheduler):
    "Training with exponentially growing learning rate"
    def __init__(self, start_lr=1e-7, end_lr=10, num_it=100, stop_div=True):
        if is_listy(start_lr):
            self.scheds = {'lr': [SchedExp(s, e) for (s,e) in zip(start_lr,end_lr)]}
        else: self.scheds = {'lr': SchedExp(start_lr, end_lr)}
        self.num_it,self.stop_div = num_it,stop_div
        self.skip_batch = False



    def before_fit(self):
        super().before_fit()
        # no need to save orig weights
        # since learner instances are transient on spawned procs
        # self.learn.save('_tmp')
        self.best_loss = float('inf')
        self.skip_batch = False

    def before_epoch(self):
        # dont report losses while running lrfind (override sync_recorder)
        if not xm.is_master_ordinal():
            return
        if hasattr(self.learn, 'sync_recorder'):
            self.learn.logger = noop
            self.learn.sync_recorder._sync_stats_log = noop

    def before_batch(self):
        if self.skip_batch:
            return
        self._update_val(self.train_iter/self.num_it)

    def after_batch(self):
        if self.skip_batch:
            return
        super().after_batch()
        smooth_loss = self.smooth_loss.item() # move xla tensor to cpu
        if smooth_loss < self.best_loss:
            self.best_loss = smooth_loss

        # handle continuation of batch iteration until all batches exhausted
        if smooth_loss > 4*self.best_loss and self.stop_div:
            # print(f'xla {xm.get_ordinal()}: stop stats collection due to loss')
            self.skip_batch = True
            self.copy_losses_and_lrs()
            self.synced_cancel.trigger_cancel_fit()
            return


        if self.train_iter >= self.num_it:
            # print(f'xla {xm.get_ordinal()}: stop stats collection due to num_iter')
            # return and stop updating losses
            self.skip_batch = True
            self.copy_losses_and_lrs()
            self.synced_cancel.trigger_cancel_fit()
            return

    def copy_losses_and_lrs(self):
        if xm.is_master_ordinal():
            losses = [loss.item() for loss in self.recorder.losses]
            iters = self.recorder.iters[:]
            values = self.recorder.values[:]

            self.plot_data = {'lrs': self.recorder.lrs[:],
                              'losses': losses,
                              'iters': iters,
                              'values': values}
            if hasattr(self,'hps'):
                self.plot_data['hps']  = {**self.hps}

    def after_fit(self):
        super().after_fit()
        # no need to load old weights since these will be transient
        # self.learn.opt.zero_grad() #Need to zero the gradients of the model before detaching the optimizer for future fits
        # tmp_f = self.path/self.model_dir/'_tmp.pth'
        # if tmp_f.exists():
        #     self.learn.load('_tmp', with_opt=True)
        #     os.remove(tmp_f)
        if not self.skip_batch:
            self.copy_losses_and_lrs()
        if xm.is_master_ordinal():
            with open('_plt_loss.pkl','wb') as f:
                pickle.dump(self.plot_data,f)


# Cell
from fastai.learner import Learner
from fastai.callback.schedule import SuggestedLRs
from fastcore.basics import patch
from fastai.torch_core import tensor
@patch
def get_suggested_lrs(self:Learner, num_it):
    'compute Suggested LRs'
    lrs,losses = tensor(self.recorder.lrs[num_it//10:-5]),tensor(self.recorder.losses[num_it//10:-5])
    if len(losses) == 0: return
    lr_min = lrs[losses.argmin()].item()
    grads = (losses[1:]-losses[:-1]) / (lrs[1:].log()-lrs[:-1].log())
    lr_steep = lrs[grads.argmin()].item()
    return SuggestedLRs(lr_min/10.,lr_steep)


# Cell
import pickle
from fastai.learner import Recorder
from fastcore.basics import patch
@patch
def reload_lr_find_attrs(self:Recorder, fn='_plt_loss.pkl'):
    if isinstance(fn,str):
        fn = Path(fn)

    if not fn.is_file():
        return

    with open(fn,'rb') as f:
        d = pickle.load(f)
        self.lrs,self.losses = d['lrs'],d['losses']
        self.values, self.iters = d['values'], d['iters']
        if 'hps' in d:
            self.hps = d['hps']
    # delete file after
    if fn.is_file():
        fn.unlink()


# Cell

def xla_run_lr_find(rank, learner_args, add_args, lr_find_args, ctrl_args):
    xm.rendezvous('start_xla_run_lr_find')
    # print(f'xla {rank} : start run lrfind')
    sync_valid = True
    learner = make_xla_child_learner(rank, sync_valid, learner_args, add_args, ctrl_args)

    num_it = lr_find_args['num_it']
    n_epoch = num_it//len(learner.dls.train) + 1
    learner.opt = None
    learner.create_opt()
    cb = XLALRFinder(**lr_find_args)

    skip_valid_cb = SkipValidationCallback()

    with learner.no_logging():
        learner.fit(n_epoch, cbs=[cb, skip_valid_cb])



# Cell

from pathlib import Path
from fastai.learner import Learner
from fastcore.basics import patch
from fastcore.meta import delegates

@patch
@delegates(Learner.lr_find)
def xla_lr_find(self:Learner, num_cores=8, start_method='fork', **kwargs):
    lr_find_args = {
        'start_lr': 1e-7,
        'end_lr': 10.,
        'num_it': 100,
        'stop_div': True
    }
    fn = Path('_plt_loss.pkl')
    if fn.is_file():
        fn.unlink()
    # remove show_plot and suggestions param
    show_plot = kwargs.pop('show_plot', True)
    suggestions = kwargs.pop('suggestions',True)
    # override default with kwargs
    lr_find_args = {**lr_find_args, **kwargs}

    ctrl_args = self.pre_xla_fit()
    learner_args, add_args = self.pack_learner_args()
    xmp.spawn(xla_run_lr_find,
              args=(learner_args, add_args, lr_find_args, ctrl_args),
              nprocs=num_cores,
              start_method=start_method)
    self.post_xla_fit(ctrl_args)
    self.recorder.reload_lr_find_attrs()
    if show_plot:
        # show_loss()
        self.recorder.plot_lr_find()
    if suggestions:
        return self.get_suggested_lrs(lr_find_args['num_it'])