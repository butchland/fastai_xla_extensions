# FastAI XLA Extensions Library
> The FastAI XLA Extensions library package allows your fastai/Pytorch models to run on TPUs using the Pytorch-XLA library.


## Install

`pip install git+https://github.com/butchland/fastai_xla_extensions`

## How to use

### Configure the Pytorch XLA package 

The Pytorch xla package requires an environment supporting TPUs (Kaggle kernels, GCP or Colab environments required)

If running on Colab, make sure the Runtime Type is set to TPU.


```
#colab
import os
assert os.environ['COLAB_TPU_ADDR'], 'Make sure to select TPU from Edit > Notebook settings > Hardware accelerator'
```

```
#colab
VERSION = "20200325"  #@param ["1.5" , "20200325", "nightly"]
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version $VERSION
```

Install fastai2 and the fastai_xla_extensions packages

```
!pip install fastai2
```

```
#colab
!pip install git+https://github.com/butchland/fastai_xla_extensions
```

### Import the libraries
Import the pytorch xla, fastai2 and fastai_xla_extensions libraries

```
#colab
import torch_xla.core.xla_model as xm
```

```
from fastai2.vision.all import *
```

```
#colab
from fastai_xla_extensions.core import *
```

### Example
Build a Pets classifier -- adapted from fastai course [Lesson 5 notebook](https://github.com/fastai/course-v4/blob/master/nbs/05_pet_breeds.ipynb)

```
path = untar_data(URLs.MNIST_SAMPLE)
```

```
Path.BASE_PATH = path
```

```
pat = r'(.+)_\d+.jpg$'
```

```
datablock = DataBlock(
    blocks=(ImageBlock,CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(seed=42),
    get_y=using_attr(RegexLabeller(pat),'name'),
    item_tfms=Resize(460),
    batch_tfms=aug_transforms(size=224,min_scale=0.75)
)
```

Get a TPU device

```
#colab
tpu = xm.xla_device()
```

Set the dataloaders to use the TPU instead of a CPU

```
dls = datablock.dataloaders(path, device=tpu)
```

Wrap the optimizer function with the XLA Optimizer

```
#colab
opt_func = XLAOptFuncWrapper(Adam)
```

```
learner = cnn_learner(dls, resnet34, metrics=accuracy, opt_func=opt_func)
                      
```

```
#colab
learner.fine_tune(5)
```

## Status
The fastai XLA extensions library is still in very early development phase (not even alpha) which means there's still a lot of things not working. 

Use it at your own risk.

If you wish to contribute to the project, fork it and make pull request. 

This project uses [nbdev](https://nbdev.fast.ai/) -- a jupyter notebook first development environment and is being developed on [Colab](https://colab.research.google.com).

