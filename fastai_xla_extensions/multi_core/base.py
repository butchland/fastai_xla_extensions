# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03_multi_core.base.ipynb (unless otherwise specified).

__all__ = ['revert_tensor', 'recast2tensor', 'round_to_multiple', 'TPUDistributedDL', 'after_batch', 'bs', 'device',
           'build_distributed_dataloaders', 'make_fastai_dataloaders', 'wrap_parallel_loader', 'XLATrainingCallback',
           'pack_metric', 'make_tensor', 'pack_metrics', 'restore_metrics', 'SyncedAvgSmoothLoss',
           'SyncRecorderCallback', 'xm_save', 'do_one_loop']

# Cell

from fastai.vision.all import *
from ..utils import xla_imported
from ..misc_utils import *
from ..core import XLAOptCallback

# Internal Cell
try:
    import torch_xla
except ImportError:
    pass

# Cell

if xla_imported():
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

# Internal Cell

import time
import torch
from fastcore.foundation import L
from fastai.data.core import DataLoaders
import math
from fastcore.basics import store_attr
from operator import attrgetter
from fastai.data.load import _FakeLoader

from fastai.torch_core import TensorBase
import random
from fastcore.basics import patch

# Cell

def revert_tensor(o):
    "Remove tensor subclass and revert to `torch.Tensor`"
    try:
        o.__class__ = torch.Tensor
    except:
        raise RuntimeError(f'could not convert {o} to torch.Tensor')
    return o

def recast2tensor(o):
    "Recast `fastai.torch_core.TensorBase` subclassed tensors to torch.Tensors"
    if isinstance(o,TensorBase):
        # return plain tensor since pl.parallelloader doesn't
        # seem to work with tensor subclasses
        # return torch.as_tensor(o.numpy())
        # TODO: recreate bug in notebook gist to file bug to torch_xla team
        return revert_tensor(o)
    return o

def round_to_multiple(number,multiple):
    "round up batch samples to fill number of cores"
    return int(math.ceil(number/multiple)*multiple)

# Cell

from fastai.data.core import TfmdDL

class TPUDistributedDL(TfmdDL):
    """A `TfmdDL` which splits a batch into equal size pieces for each TPU core
       It also recasts the output of a batch from a TensorBase subclass to
       a regular tensor since the XLA Parallel loader doesn't seem to be compatible
       to it.
       Code implementation was based on @tmabraham's `TPUDistributedDL` implementation
       here: https://github.com/tmabraham/fastai_tpu/blob/master/fastai_v2/tpu_distributed_dl.py
    """
    _default = 'dl'
    def __init__(self,dl,rank,world_size, seed=42):
        store_attr()
        self.bs,self.device,self.num_workers, \
        self.drop_last,self.dataset,self.offs,fake, self.shuffle = \
            attrgetter('bs','device','num_workers',
                       'drop_last','dataset','offs','fake_l', 'shuffle')(dl)
        self.fake_l = _FakeLoader(self, fake.pin_memory, fake.num_workers, fake.timeout,
                                  persistent_workers=fake.persistent_workers)
        self.epoch = 0
        random.seed(self.seed)
        # setting inner dl rng
        self.dl.rng = random.Random(random.randint(0,2**32-1))
        self.reset_rng()

    def reset_rng(self):
        random.seed(self.seed + self.epoch)
        # setting outer dl rng
        self.rng = random.Random(random.randint(0,2**32-1))

    def __len__(self):
        return round_to_multiple(len(self.dl),self.world_size)//self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_idxs(self):
        idxs = self.dl.get_idxs()
        # do your own shuffling which factors in self.epoch + self.seed in
        # generating a random sequence (underlying self.dl does not)
        if self.shuffle:
            idxs = self.shuffle_fn(idxs)
        self.n = len(idxs)
        # we assumed n was dl.n but we really care about number of idxs
        # add extra samples to make it evenly divisible
        self.n_padded = round_to_multiple(self.n,self.world_size)
        idxs += (idxs * (self.n_padded//self.n))[:self.n_padded-self.n]
        # idx needs to be repeated when n_padded>>n
        # slice padded idxs so that each rank gets self.n_padded//self.world_size tensors
        start_pos = self.rank*self.n_padded//self.world_size
        end_pos = (self.rank+1)*self.n_padded//self.world_size
        return idxs[start_pos:end_pos]

    def before_iter(self):
        self.dl.before_iter()

    def randomize(self):
        self.reset_rng()
        self.dl.randomize()

    def after_batch(self,b):
        b = self.dl.after_batch(b)
        # recast tensor subclasses to plain tensors
        # undoing work of self.retain()
        tb = [recast2tensor(o) for o in b]
        b = tuple(tb)
        return b

    def after_iter(self):
        self.dl.after_iter()

    def create_batches(self,samps):
        return self.dl.create_batches(samps)

    def to(self, device):
        self.dl.device = device
        self.device = device
        return self

    def one_batch(self):
        return self.dl.one_batch()

    def new(self, dataset=None, cls=None, **kwargs):
        new_dl = self.dl.new(dataset=dataset, cls=cls, **kwargs)
        use_rank = self.rank
        use_size = self.world_size
        seed = self.seed

        new_dl = TPUDistributedDL(new_dl,
                            rank=use_rank,
                            world_size=use_size,
                            seed=seed)

        return new_dl

# Cell

import torch
import torch.utils.data as th_data
import torch.utils.data.distributed as th_distrib

# Cell

from fastcore.basics import patch

# import torch.utils.data as th_data
from fastcore.transform import Pipeline
# import torch.utils.data.distributed as th_distrib

# from torch.data.utils.DataLoader
# def __setattr__(self, attr, val):
#     if self.__initialized and attr in (
#             'batch_size', 'batch_sampler', 'sampler', 'drop_last', 'dataset', 'persistent_workers'):
#         raise ValueError('{} attribute should not be set after {} is '
#                             'initialized'.format(attr, self.__class__.__name__))
@patch
def __setattr__(self:th_data.DataLoader, attr, val):
    'remove sampler,batch_sampler from list of attrs which should not be set after init'
    initialized = getattr(self,'__initialized', False)
    if initialized and attr in (
            'batch_size', 'drop_last', 'dataset', 'persistent_workers'):
        raise ValueError('{} attribute should not be set after {} is '
                            'initialized'.format(attr, self.__class__.__name__))
    super(th_data.DataLoader, self).__setattr__(attr, val)

@patch(as_prop=True)
def after_batch(self:th_data.DataLoader):
    'return empty pipeline when fastai learner looks for after_batch'
    return Pipeline()

@patch(as_prop=True)
def bs(self:th_data.DataLoader):
    'return fastai synonym for torch batch size'
    return self.batch_size

@patch(as_prop=True)
def device(self:th_data.DataLoader):
    'return null device'
    device = getattr(self,'_device', torch.device('cpu'))
    return device

@patch
def to(self:th_data.DataLoader, device):
    'add impl for to(device)'
    # self._device = device
    setattr(self,'_device',device) # fix 'can't set attrib error'
    return self

@patch
def set_distributed_sampler(self:th_data.DataLoader, rank, world_size):
    'replace sampler with torch distributed sampler'
    distrib_sampler = th_distrib.DistributedSampler(self.dataset,
                                                    num_replicas=world_size,
                                                    rank=rank, shuffle=True)
    self.sampler = distrib_sampler
    batch_sampler_klass = self.batch_sampler.__class__
    self.batch_sampler = batch_sampler_klass(self.sampler,
                                             self.batch_size,
                                             self.drop_last)

# Cell
import torch.utils.data as th_data

def build_distributed_dataloaders(dls, rank, world_size, sync_valid=False):
    """Wrap dataloaders with distributed TPU aware dataloader """
    new_loaders = []
    for i,dl in enumerate(dls.loaders):
        if i == 0 or sync_valid:
            use_rank = rank
            use_size = world_size
        else:
            use_rank = 0
            use_size = 1
        if isinstance(dl, th_data.DataLoader):
            if i == 0: # set train dl to use distrib sampler
                dl.set_distributed_sampler(use_rank, use_size)
        else: # fastai dataloader
            dl = TPUDistributedDL(dl,
                                rank=use_rank,
                                world_size=use_size)
        new_loaders += [dl]
    return DataLoaders(*new_loaders, path=dls.path, device=dls.device)

# Cell
from fastcore.meta import delegates
from fastai.data.block import DataBlock

@delegates(DataBlock.dataloaders,but='datablock,rank,world_size,sync_valid,device')
def make_fastai_dataloaders(datablock, source, rank, world_size, device=None, path='.', sync_valid=False, verbose=False,**kwargs):
    "create fastai-based dataloaders from a datablock and wrap a tpu distributed dataloader around them"
    dls = datablock.dataloaders(source=source, path=path, device=device, **kwargs)
    distrib_dls = build_distributed_dataloaders(dls, rank, world_size, sync_valid=sync_valid)
    return distrib_dls

# Cell

def wrap_parallel_loader(loader, device):
    'wraps a tpu distributed loader or a torch dataloader (with distributed sampler) with xla parallel loader'
    para_loader = pl.ParallelLoader(loader, [device])
    loop_loader = para_loader.per_device_loader(device)
    return loop_loader

# Internal Cell

from fastai.learner import Recorder
from fastai.callback.core import Callback
from fastai.learner import CancelValidException

# Cell

class XLATrainingCallback(Callback):
    "A callback for training as a spawned process on multi-core TPUs"
    run_before = Recorder
    run_valid = False
    order = -5 # after TrainEvalCallback
    def __init__(self, device, rank=0, sync_valid=False):
        self.pdevice = device
        self.rank = rank
        self.sync_valid = sync_valid

    def before_fit(self):
        if not getattr(self.learn,'inner_xla', False):
            return # skip if not spawned
        xm.master_print('start fit')

    def before_epoch(self):
        # set the epoch on train only to make sure shuffle produces same seq
        # across all ranks
        if not getattr(self.learn,'inner_xla',False):
            return # skip if not spawned

        if hasattr(self.learn.dls.train,'sampler'):
            if hasattr(self.learn.dls.train.sampler,'set_epoch'):
                self.learn.dls.train.sampler.set_epoch(self.learn.epoch)
        elif hasattr(self.learn.dls.train,'set_epoch'):
            self.learn.dls.train.set_epoch(self.learn.epoch)

        if self.sync_valid: # update epoch on valid if sync_valid
            if hasattr(self.learn.dls.valid,'sampler'):
                if hasattr(self.learn.dls.valid.sampler,'set_epoch'):
                    self.learn.dls.valid.sampler.set_epoch(self.learn.epoch)
            elif hasattr(self.learn.dls.valid,'set_epoch'):
                self.learn.dls.valid.set_epoch(self.learn.epoch)

    def before_train(self):
        if not getattr(self.learn,'inner_xla',False):
            return # skip if not spawned

        self.learn.dl = wrap_parallel_loader(self.dls.train, self.pdevice)

    def before_validate(self):
        "Set the model in validation mode"
        if not getattr(self.learn,'inner_xla',False):
            return # skip if not spawned

        if self.rank != 0 and not self.sync_valid:
        # no need to compute valid loss/ metric if not master if not sync valid
            raise CancelValidException()

        if not isinstance(self.learn.dl, pl.PerDeviceLoader):
            self.learn.dl = wrap_parallel_loader(self.learn.dl, self.pdevice)


# Internal Cell

import copy
from fastai.learner import _maybe_item
from fastprogress.fastprogress import format_time

# Cell
def pack_metric(metrics):
    "extract counts and totals from avg metrics and avg losses into a list"
    counts = metrics.attrgot('count',0)
    totals = metrics.attrgot('total',0)
    metrics_list = counts + totals
    return metrics_list

def make_tensor(o, device):
    "convert a scalar or tensor into a float tensor and move them to `device`"
    if not isinstance(o, torch.Tensor):
        o = torch.tensor(o)
    return o.float().to(device)

def pack_metrics(all_metrics, device):
    "pack train and valid metrics into a list of float tensors and move them to `device`"
    metrics_list = pack_metric(all_metrics['train_mets']) + pack_metric(all_metrics['valid_mets'])
    return [make_tensor(item,device) for item in metrics_list ]

def restore_metrics(reduced_metrics, all_metrics):
    "restore list of float tensors (count and values) back into train and valid metrics"
    n_train = len(all_metrics['train_mets'])
    n_valid = len(all_metrics['valid_mets'])
    train_counts = reduced_metrics[:n_train]
    train_totals = reduced_metrics[n_train: n_train*2]
    valid_counts = reduced_metrics[n_train*2: n_train*2 + n_valid]
    valid_totals = reduced_metrics[n_train*2 + n_valid:]
    for i,metric in enumerate(all_metrics['train_mets']):
        if hasattr(metric,'count'):
            metric.count = train_counts[i].clone().detach().long()
        if hasattr(metric,'total'):
            metric.total = train_totals[i].clone().detach()
    for i,metric in enumerate(all_metrics['valid_mets']):
        if hasattr(metric,'count'):
            metric.count = valid_counts[i].clone().detach().long()
        if hasattr(metric,'total'):
            metric.total = valid_totals[i].clone().detach()
    return all_metrics

# Cell

from fastai.learner import AvgSmoothLoss

class SyncedAvgSmoothLoss(AvgSmoothLoss):
    "Smooth average of the losses (exponentially weighted with `beta`) synced across all ranks"
    def __init__(self, beta=0.98):
        super(SyncedAvgSmoothLoss, self).__init__(beta=beta)

    def accumulate(self, learn):
        self.count += 1
        # get loss across all ranks
        synced_loss = xm.all_reduce(xm.REDUCE_SUM, learn.loss.mean())
        avg_synced_loss = synced_loss/xm.xrt_world_size()
        self.val = torch.lerp(avg_synced_loss, self.val, self.beta)


# Cell
class SyncRecorderCallback(Callback):
    """A `Callback` to sync the metrics from each rank and update statistics
       accordingly so it will display correctly in the progress callback
    """
    order  = 55 # after Recorder, before ProgressCallback

    def before_fit(self):
        if not getattr(self.learn,'inner_xla',False):
            return # skip if not spawned

        # replace AvgSmoothLoss  with SyncedAvgSmoothLoss which
        # uses mean loss across all ranks per batch to compute smooth loss
        # instead of just using one rank's mean loss
        if not isinstance(self.recorder.smooth_loss, SyncedAvgSmoothLoss):
            orig_beta = self.recorder.smooth_loss.beta
            self.recorder.smooth_loss = SyncedAvgSmoothLoss(beta=orig_beta)
            self.recorder.smooth_loss.reset()

        if not xm.is_master_ordinal():
            return

        if 'progress' in self.learn.cbs.attrgot('name',None):
            self._sync_stats_log = self.progress._write_stats
        else:
            self._sync_stats_log = self.learn.logger

    def before_epoch(self):
        if not getattr(self.learn,'inner_xla',False):
            return # skip if not spawned

        self.sync_log = copy.copy(self.recorder.log)

    def after_epoch(self):
        if not getattr(self.learn,'inner_xla',False):
            return # skip if not spawned

        if 'recorder' not in self.learn.cbs.attrgot('name'):
            all_metrics = {
                'train_mets': L([]),
                'valid_mets': L([]),
            }
        else:
            all_metrics = {
                'train_mets': self.recorder._train_mets,
                'valid_mets': self.recorder._valid_mets,
            }
        # send metrics data to sync ranks across spawned processes
        device = self.learn.xla_training.pdevice
        packed_metrics = pack_metrics(all_metrics, device) # convert metrics to tensor list on TPU
        reduced_metrics = xm.all_reduce(xm.REDUCE_SUM, packed_metrics)
        xm.mark_step()
        if xm.is_master_ordinal():
            all_metrics = restore_metrics(reduced_metrics, all_metrics) # convert list to metric objects
            for m in self.recorder._train_mets:
                self.sync_log += _maybe_item(m)

            for m in self.recorder._valid_mets:
                self.sync_log += _maybe_item(m)

            self.learn.final_record = self.sync_log[1:].copy()
            del self.recorder.values[-1] # remove last entry added by recorder
            self.recorder.values.append(self.learn.final_record) # add updated metrics
            if self.recorder.add_time:
                updated_time = (time.time() - self.recorder.start_epoch)
                self.sync_log.append(format_time(updated_time))
            self.recorder.log = self.sync_log
            self._sync_stats_log(self.sync_log) # write_stats to output
            self.learn.logger = self.orig_logger # restore orig logger after skipping recorder.logger(log)

    def after_validate(self):
        if not getattr(self.learn,'inner_xla',False):
            return # skip if not spawned

        if xm.is_master_ordinal():
            self.orig_logger = self.learn.logger
            self.learn.logger = noop # write to logger disabled so calling recorder.logger(log) wont print

# Cell
from fastcore.imports import noop
#from fastcore.basics import patch
from fastai.learner import Learner
from fastai.callback.progress import ProgressCallback
from fastcore.xtras import join_path_file
#from fastai.torch_core import get_model

# Cell

#copied from `torch_xla.core.xla_model.save` with the addition of rendezvous as a param

def xm_save(data, file_or_path, master_only=True, global_master=False, rendezvous=True):
    """Saves the input data into a file.

    The saved data is transferred to PyTorch CPU device before being saved, so a
    following `torch.load()` will load CPU data.
    Care must be taken when working with views. Instead of saving views it's
    recommended that you recreate them after the tensors have been loaded and
    moved to their destination device(s).

    Args:
    data: The input data to be saved. Any nested combination of Python objects
        (list, tuples, sets, dicts, ...).
    file_or_path: The destination for the data saving operation. Either a file
        path or a Python file object. If `master_only` is ``False`` the path or
        file objects must point to different destinations as otherwise all the
        writes from the same host will override each other.
    master_only (bool, optional): Whether only the master device should save the
        data. If False, the `file_or_path` argument should be a different file or
        path for each of the ordinals taking part to the replication, otherwise
        all the replicas on the same host will be writing to the same location.
        Default: True
    global_master (bool, optional): When ``master_only`` is ``True`` this flag
        controls whether every host's master (if ``global_master`` is ``False``)
        saves the content, or only the global master (ordinal 0).
        Default: False
    """
    should_write_data = not master_only or xm.is_master_ordinal(
        local=not global_master)

    cpu_data = xm._maybe_convert_to_cpu(data, convert=should_write_data)
    if should_write_data:
        torch.save(cpu_data, file_or_path)
    if rendezvous:
        xm.rendezvous('torch_xla.core.xla_model.save')


# Cell
from fastai.callback.tracker import SaveModelCallback
from fastcore.basics import patch
@patch
def _save(self:SaveModelCallback, name):
    'save best model using `rendezvous=False`'
    if getattr(self.learn,'inner_xla', False):
        self.last_saved_path = self.learn.save(name, with_opt=self.with_opt,
                                           rendezvous=False)
    else:
        self.last_saved_path = self.learn.save(name, with_opt=self.with_opt)

# Internal Cell
@patch
@delegates(Learner.save, but='rendezvous')
def save(self:Learner, file, **kwargs):
    file = join_path_file(file, self.path/self.model_dir, ext='.pth')
    with_opt = kwargs.pop('with_opt', self.opt is not None)
    pickle_protocol = kwargs.pop('pickle_protocol', 2)

    state = self.model.state_dict()
    if with_opt:
        # add opt state to state to be saved
        opt_state = self.opt.state_dict()
        state = {'model': state, 'opt':opt_state}
    if getattr(self,'inner_xla',False):
        xm_save(state, file, **kwargs) # use xm_save instead of torch.save
    else:
        # use default if not spawned
        torch.save(state,file,pickle_protocol=pickle_protocol)
    return file

# Internal Cell
@patch
def to_multi_xla(self:Learner,device, rank, sync_valid=False):
    "Sets up the learner on the spawned process for multi core TPU training"
    # add xla info on learner
    self.inner_xla = True
    self.xla_rank = rank
    if 'xla_training' not in self.cbs.attrgot('name'):
        self.dls.device = None
        self.add_cbs([XLATrainingCallback(device, rank, sync_valid=sync_valid),
                      XLAOptCallback()])
        self.opt = None # clear opt to ensure

    else:
        self.xla_training.pdevice = device
        self.xla_training.rank = rank
        self.xla_training.sync_valid = sync_valid

    if sync_valid and 'sync_recorder' not in self.cbs.attrgot('name'):
        self.add_cbs(SyncRecorderCallback)
    elif not sync_valid:
        self.remove_cbs(SyncRecorderCallback)

    if rank != 0: # progress bar only for rank 0
        self.remove_cbs(ProgressCallback)
    self.logger = xm.master_print

# Cell
# for testing
def do_one_loop(dl, rank, world_size, device, wrap_parallel=True):
    "test one loop for a tpu distributed dataloader"
    n_batches = len(dl)
    print(f'xla: {rank} world_size: {world_size} n_batches:{n_batches}')

    if wrap_parallel:
        print(f'xla: {rank} wrapping ploader')
        pdl = wrap_parallel_loader(dl, device=device)
    else:
        pdl = dl
    for i,b in enumerate(pdl):
        if i > 1:
            break
        xb, yb = b
        print(f'xla: {rank} iter:{i} xb type {type(xb)} yb type: {type(yb)}')
        print(f'xla: {rank} iter:{i} xb.shape {xb.shape} yb.shape: {yb.shape}')
        print(f'xla: {rank} iter:{i} xb.device {xb.device} yb.device: {yb.device}')
        print(f'xla: {rank} iter:{i} xb.dtype {xb.dtype} yb.device: {yb.dtype}')