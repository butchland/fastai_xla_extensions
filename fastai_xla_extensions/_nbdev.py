# AUTOGENERATED BY NBDEV! DO NOT EDIT!

__all__ = ["index", "modules", "custom_doc_links", "git_url"]

index = {"XLAOptimProxy": "00_core.ipynb",
         "DeviceMoverTransform": "00_core.ipynb",
         "isAffineCoordTfm": "00_core.ipynb",
         "isDeviceMoverTransform": "00_core.ipynb",
         "has_affinecoord_tfm": "00_core.ipynb",
         "has_devicemover_tfm": "00_core.ipynb",
         "get_last_affinecoord_tfm_idx": "00_core.ipynb",
         "insert_batch_tfm": "00_core.ipynb",
         "Learner.setup_input_device_mover": "00_core.ipynb",
         "XLAOptCallback": "00_core.ipynb",
         "Learner.to_xla": "00_core.ipynb",
         "Learner.detach_xla": "00_core.ipynb",
         "xla_imported": "01_utils.ipynb",
         "print_aten_ops": "01_utils.ipynb",
         "download_torch_dsets": "02_cifar_loader.ipynb",
         "load_torch_items": "02_cifar_loader.ipynb",
         "load_classes": "02_cifar_loader.ipynb",
         "CifarNP2ImageTransform": "02_cifar_loader.ipynb",
         "Int2TensorTransform": "02_cifar_loader.ipynb",
         "CifarImageTransform": "02_cifar_loader.ipynb",
         "CifarImage2FloatTransform": "02_cifar_loader.ipynb",
         "make_torch_tfms": "02_cifar_loader.ipynb",
         "CifarTupleTransform": "02_cifar_loader.ipynb",
         "TupleTorchDS": "02_cifar_loader.ipynb",
         "make_cifar_item_tfm": "02_cifar_loader.ipynb",
         "i2t_tfm": "02_cifar_loader.ipynb",
         "cfnp2img_tfm": "02_cifar_loader.ipynb",
         "cfimg_tfm": "02_cifar_loader.ipynb",
         "cfimg2float_tfm": "02_cifar_loader.ipynb",
         "make_cifar_tls": "02_cifar_loader.ipynb",
         "make_cifar_dl": "02_cifar_loader.ipynb",
         "make_fastai_cifar_dls": "02_cifar_loader.ipynb",
         "revert_tensor": "03_multi_core.base.ipynb",
         "recast2tensor": "03_multi_core.base.ipynb",
         "round_to_multiple": "03_multi_core.base.ipynb",
         "TPUDistributedDL": "03_multi_core.base.ipynb",
         "build_distributed_dataloaders": "03_multi_core.base.ipynb",
         "make_fastai_dataloaders": "03_multi_core.base.ipynb",
         "wrap_parallel_loader": "03_multi_core.base.ipynb",
         "XLATrainingCallback": "03_multi_core.base.ipynb",
         "pack_metric": "03_multi_core.base.ipynb",
         "make_tensor": "03_multi_core.base.ipynb",
         "pack_metrics": "03_multi_core.base.ipynb",
         "restore_metrics": "03_multi_core.base.ipynb",
         "SyncRecorderCallback": "03_multi_core.base.ipynb",
         "xm_save": "03_multi_core.base.ipynb",
         "Learner.save": "03_multi_core.base.ipynb",
         "Learner.to_multi_xla": "03_multi_core.base.ipynb",
         "do_one_loop": "03_multi_core.base.ipynb",
         "TfmdTorchDS": "03a_multi_core.torch_compat.ipynb",
         "to_list": "03a_multi_core.torch_compat.ipynb",
         "has_setup": "03a_multi_core.torch_compat.ipynb",
         "run_setups": "03a_multi_core.torch_compat.ipynb",
         "TorchDatasetBuilder": "03a_multi_core.torch_compat.ipynb",
         "VocabularyMapper": "03a_multi_core.torch_compat.ipynb",
         "to": "03a_multi_core.torch_compat.ipynb",
         "make_torch_dataloaders": "03a_multi_core.torch_compat.ipynb",
         "FileNamePatternLabeller": "03a_multi_core.torch_compat.ipynb",
         "make_xla_child_learner": "03b_multi_core.learner.ipynb",
         "xla_run_method": "03b_multi_core.learner.ipynb",
         "Learner.pack_learner_args": "03b_multi_core.learner.ipynb",
         "Learner.reload_child_model": "03b_multi_core.learner.ipynb",
         "Learner.pre_xla_fit": "03b_multi_core.learner.ipynb",
         "Learner.post_xla_fit": "03b_multi_core.learner.ipynb",
         "Learner.xla_fit": "03b_multi_core.learner.ipynb",
         "Learner.xla_fit_one_cycle": "03b_multi_core.learner.ipynb"}

modules = ["core.py",
           "utils.py",
           "cifar_loader.py",
           "misc_utils.py",
           "multi_core/base.py",
           "multi_core/torch_compat.py",
           "multi_core/learner.py"]

doc_url = "https://butchland.github.io/fastai_xla_extensions/"

git_url = "https://github.com/butchland/fastai_xla_extensions/tree/master/"

def custom_doc_links(name): return None
