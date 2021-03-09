# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03e_multi_core.inference.ipynb (unless otherwise specified).

__all__ = ['setup_inference_args', 'save_pred_results', 'xla_run_inference', 'reload_pred_results',
           'prep_inference_args']

# Cell
try:
    import torch_xla
except ImportError:
    pass

# Cell

from fastai.vision.all import *
from ..utils import xla_imported
from ..misc_utils import *
from ..core import XLAOptCallback
from .base import *
from .learner import *
from .callback import *

# Cell

if xla_imported():
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp

# Cell
from fastai.vision.all import *


# Cell
from fastai.learner import _ConstantFunc
# from fastcore.basics import patch
# from fastai.learner import Learner

@patch
def inner_get_preds(self:Learner, ds_idx=1, dl=None, with_input=False, with_decoded=False, with_loss=False, act=None,
                inner=False, reorder=True, cbs=None, **kwargs):

    xla_rank = getattr(self,'xla_rank',None)
    if xla_rank is None:
        return

    if dl is None:
        dl = self.dls[ds_idx].new(shuffled=False, drop_last=False)
    else:
        try: len(dl)
        except TypeError as e:
            raise TypeError("`dl` is something other than a single `DataLoader` object")
        if not isinstance(dl, TPUDistributedDL):
            world_size = kwargs.pop('world_size', xm.xrt_world_size())
            seed = kwargs.pop('dl_seed',42)
            dl = TPUDistributedDL(dl, xla_rank, world_size=world_size, seed=seed)

    if reorder and hasattr(dl, 'get_idxs'):
        idxs = dl.dl.get_idxs()
        dl = dl.new(get_idxs = _ConstantFunc(idxs))
        rank_idxs = dl.get_idxs()
        rank_idxs_len = len(rank_idxs)

    #handle save_preds and save_targs across ranks
    save_preds = kwargs.pop('save_preds',None)
    if save_preds is not None:
        kwargs['save_preds'] = save_preds + str(xla_rank) # add rank to filename
    save_targs = kwargs.pop('save_targs',None)
    if save_targs is not None:
        kwargs['save_targs'] = save_targs + str(xla_rank) # add rank to filename

    cb = GatherPredsCallback(with_input=with_input, with_loss=with_loss, **kwargs)
    ctx_mgrs = self.validation_context(cbs=L(cbs)+[cb], inner=inner)
    if with_loss:
        ctx_mgrs.append(self.loss_not_reduced())

    with ContextManagers(ctx_mgrs):
        self._do_epoch_validate(dl=dl)

        if act is None:
            act = getattr(self.loss_func, 'activation', noop)

        res = cb.all_tensors()

        pred_i = 1 if with_input else 0
        if res[pred_i] is not None:
            if act != noop:
                # compute activation on tpu device and detach after
                tmp_pred = res[pred_i].to(xm.xla_device())
                tmp_res = act(tmp_pred)
                res[pred_i] = self.to_detach(tmp_res)

            if with_decoded:
                res.insert(pred_i+2, getattr(self.loss_func, 'decodes', noop)(res[pred_i]))

        if reorder and hasattr(dl, 'get_idxs'):
            t_idxs = tensor(rank_idxs)
            start_idx = xla_rank * rank_idxs_len
            t_idxs = t_idxs - tensor(start_idx) # broadcast
            sorted_idxs = t_idxs.argsort()
            res = nested_reorder(res, sorted_idxs )

        return tuple(res)
    self._end_cleanup()


# Cell

def setup_inference_args(rank, inference_args):
    master_cbs = ifnone(inference_args.pop('master_cbs', None),[])
    return inference_args, master_cbs


# Cell

import pickle
def save_pred_results(rank, results):
    fn = f'preds{rank}.pkl'
    fn = Path(fn)
    with open(fn,'wb') as f:
        pickle.dump(results, f)

# Cell

def xla_run_inference(rank, learner_args, add_args, inference_args, ctrl_args):
    sync_valid = True
    learner = make_xla_child_learner(rank, sync_valid, learner_args, add_args, ctrl_args)
    pred_args, master_cbs = setup_inference_args(rank, inference_args)

    if rank == 0 and len(master_cbs) > 0:
        learner.add_cbs(master_cbs)

    learner.synced_cancel.before_fit()

    if rank == 0:
        learner.sync_recorder.orig_logger = learner.logger

    results = learner.inner_get_preds(**pred_args)
    xm.rendezvous('xla_run_inference')

    save_pred_results(rank, results)
    xm.mark_step()


# Cell
from fastcore.foundation import L

def reload_pred_results(num_files, n_samples):
    all_preds = L()
    for rank in range(num_files):
        fn = f'preds{rank}.pkl'

        fn = Path(fn)
        if fn.is_file():
            with open(fn,'rb') as f:
                rank_preds = pickle.load(f)
                all_preds.append(rank_preds)
        else:
            raise RuntimeException(f'Missing preds file for rank {rank}')

    for rank in range(num_files):
        fn = f'preds{rank}.pkl'
        fn = Path(fn)
        fn.unlink()

    n_items = len(all_preds[0]) # num items per preds

    all_res = []
    for i in range(n_items):
        items = all_preds.itemgot(i)

        if isinstance(items[0], torch.Tensor):
            all_items = torch.cat(tuple(items))
        elif is_listy(items[0]):
            all_items = [*items]
        else:
            all_items = items
        all_res.append(all_items)
    res = []
    for i, pred in enumerate(all_res):
        pred = pred[:n_samples] # take only first
        res.append(pred)
    return res



# Cell

@patch
def pre_xla_inference(self:Learner):
    ctrl_args = {}
    progress_removed = False
    if 'progress' in L(self.cbs).attrgot('name'):
        self.remove_cbs(ProgressCallback)
        progress_removed = True
    ctrl_args['use_progress'] = progress_removed
    return ctrl_args

# Cell

@patch
def post_xla_inference(self:Learner, ctrl_args):
    if ctrl_args['use_progress']:
        self.add_cbs(ProgressCallback)
    self.recorder.reload_attrs()

# Cell

def prep_inference_args(**kwargs):
    return kwargs

# Cell

#export

@patch
@delegates(Learner.get_preds, but='num_cores,start_method,master_cbs')
def xla_get_preds(self:Learner, ds_idx=1, dl=None,
                  with_input=False, with_decoded=False,
                  with_loss=False, act=None, inner=False,
                  reorder=True, cbs=None, num_cores=8,
                  start_method='fork', master_cbs=None,**kwargs):
    ctrl_args = self.pre_xla_inference()
    learner_args, add_args = self.pack_learner_args()

    inference_args = prep_inference_args(ds_idx=ds_idx, dl=dl,
                                         with_input=with_input, with_decoded=with_decoded,
                                         with_loss=with_loss,
                                         act=act, inner=inner,
                                         reorder=reorder,
                                         cbs=cbs, master_cbs=master_cbs, **kwargs)
    if dl:
        n_results = len(dl.dataset)
    else:
        n_results = len(self.dls.loaders[ds_idx].dataset)

    xmp.spawn(xla_run_inference,
              args=(learner_args, add_args, inference_args, ctrl_args),
              nprocs=num_cores,
              start_method=start_method)

    all_results = reload_pred_results(num_cores, n_results)
    self.post_xla_inference(ctrl_args)
    return all_results
