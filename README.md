# FastAI XLA Extensions Library
> The FastAI XLA Extensions library package allows your fastai/Pytorch models to run on TPUs using the Pytorch-XLA library.


<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Install" data-toc-modified-id="Install-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Install</a></span></li><li><span><a href="#How-to-use" data-toc-modified-id="How-to-use-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>How to use</a></span></li></ul></div>

## Install

`pip install git+https://github.com/butchland/fastai_xla_extensions`

## How to use

Configure the pytorch xla library

```
VERSION = "20200325"  #@param ["1.5" , "20200325", "nightly"]
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version $VERSION
```




    2



```
import torch_xla.core.xla_model as xm
```

```
from fastai2.vision.all import *
```

```
from fastai_xla_extensions.core import *
```

```
path = untar_data(URLs.PETS)
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

```
tpu = xm.xla_device()
```

```
dls = datablock.dataloaders(path, device=tpu)
```

```
opt_func = XLAOptFuncWrapper(Adam)
```

```
learner = cnn_learner(dls, resnet34, metrics=accuracy, opt_func=opt_func)
                      
```

```
learner.fine_tune(5)
```
