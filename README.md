# FastAI XLA Extensions Library
> The FastAI XLA Extensions library package allows your fastai/Pytorch models to run on TPUs using the Pytorch-XLA library.


## Install

`pip install git+https://github.com/butchland/fastai_xla_extensions`

## How to use

### Configure the Pytorch XLA package 

The Pytorch xla package requires an environment supporting TPUs (Kaggle kernels, GCP or Colab environments required)

If running on Colab, make sure the Runtime Type is set to TPU.


Install fastai2 and the fastai_xla_extensions packages

```
#hide_output
#colab
!pip install fastai2  > /dev/null
```

```
!pip install git+https://github.com/butchland/fastai_xla_extensions
```

Install Pytorch-XLA

```
#hide_output
#colab
VERSION = "20200707"  #@param ["1.5" , "20200325","20200707", "nightly"]
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version $VERSION
```

### Import the libraries
Import the fastai2 and fastai_xla_extensions libraries

```
#colab
import fastai_xla_extensions.core
```

**IMPORTANT: Make sure to import fastai_xla_extensions.core BEFORE importing fastai2 libraries** 

If you don't, fastai2 will not default to use tpu device but will instead use the cpu.

```
from fastai2.vision.all import *
```

### Example
Build a MNIST classifier -- adapted from fastai course [Lesson 4 notebook](https://github.com/fastai/course-v4/blob/master/nbs/04_mnist_basics.ipynb)

Load MNIST dataset 

```
path = untar_data(URLs.MNIST_TINY)
```





Create Fastai DataBlock


_Note that batch transforms are currently
set to none as they seem to slow the training
on the TPU (for investigation)._

```
datablock = DataBlock(
    blocks=(ImageBlock,CategoryBlock),
    get_items=get_image_files,
    splitter=GrandparentSplitter(),
    get_y=parent_label,
    item_tfms=Resize(28),
    batch_tfms=[]
)
```

```
#colab
datablock.summary(path)
```

    Setting-up type transforms pipelines
    Collecting items from /root/.fastai/data/mnist_tiny
    Found 1428 items
    2 datasets of sizes 709,699
    Setting up Pipeline: PILBase.create
    Setting up Pipeline: parent_label -> Categorize -- {'vocab': None, 'add_na': False}
    
    Building one sample
      Pipeline: PILBase.create
        starting from
          /root/.fastai/data/mnist_tiny/train/7/703.png
        applying PILBase.create gives
          PILImage mode=RGB size=28x28
      Pipeline: parent_label -> Categorize -- {'vocab': (#2) ['3','7'], 'add_na': False}
        starting from
          /root/.fastai/data/mnist_tiny/train/7/703.png
        applying parent_label gives
          7
        applying Categorize -- {'vocab': (#2) ['3','7'], 'add_na': False} gives
          TensorCategory(1)
    
    Final sample: (PILImage mode=RGB size=28x28, TensorCategory(1))
    
    
    Setting up after_item: Pipeline: Resize -- {'size': (28, 28), 'method': 'crop', 'pad_mode': 'reflection'} -> ToTensor
    Setting up before_batch: Pipeline: 
    Setting up after_batch: Pipeline: IntToFloatTensor -- {'div': 255.0, 'div_mask': 1}
    
    Building one batch
    Applying item_tfms to the first sample:
      Pipeline: Resize -- {'size': (28, 28), 'method': 'crop', 'pad_mode': 'reflection'} -> ToTensor
        starting from
          (PILImage mode=RGB size=28x28, TensorCategory(1))
        applying Resize -- {'size': (28, 28), 'method': 'crop', 'pad_mode': 'reflection'} gives
          (PILImage mode=RGB size=28x28, TensorCategory(1))
        applying ToTensor gives
          (TensorImage of size 3x28x28, TensorCategory(1))
    
    Adding the next 3 samples
    
    No before_batch transform to apply
    
    Collating items in a batch
    
    Applying batch_tfms to the batch built
      Pipeline: IntToFloatTensor -- {'div': 255.0, 'div_mask': 1}
        starting from
          (TensorImage of size 4x3x28x28, TensorCategory([1, 1, 1, 1], device='xla:1'))
        applying IntToFloatTensor -- {'div': 255.0, 'div_mask': 1} gives
          (TensorImage of size 4x3x28x28, TensorCategory([1, 1, 1, 1], device='xla:1'))


Create the dataloader

```
dls = datablock.dataloaders(path)
```

```
#colab
dls.device
```




    device(type='xla', index=1)



```
#colab
dls.show_batch()
```


![png](docs/images/output_23_0.png)


Create a Fastai CNN Learner

```
#colab
learner = cnn_learner(dls, resnet18, metrics=accuracy)
                      
```

    Downloading: "https://download.pytorch.org/models/resnet18-5c106cde.pth" to /root/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth


    


```
#colab
learner.summary()
```




    Sequential (Input shape: ['64 x 3 x 28 x 28'])
    ================================================================
    Layer (type)         Output Shape         Param #    Trainable 
    ================================================================
    Conv2d               64 x 64 x 14 x 14    9,408      False     
    ________________________________________________________________
    BatchNorm2d          64 x 64 x 14 x 14    128        True      
    ________________________________________________________________
    ReLU                 64 x 64 x 14 x 14    0          False     
    ________________________________________________________________
    MaxPool2d            64 x 64 x 7 x 7      0          False     
    ________________________________________________________________
    Conv2d               64 x 64 x 7 x 7      36,864     False     
    ________________________________________________________________
    BatchNorm2d          64 x 64 x 7 x 7      128        True      
    ________________________________________________________________
    ReLU                 64 x 64 x 7 x 7      0          False     
    ________________________________________________________________
    Conv2d               64 x 64 x 7 x 7      36,864     False     
    ________________________________________________________________
    BatchNorm2d          64 x 64 x 7 x 7      128        True      
    ________________________________________________________________
    Conv2d               64 x 64 x 7 x 7      36,864     False     
    ________________________________________________________________
    BatchNorm2d          64 x 64 x 7 x 7      128        True      
    ________________________________________________________________
    ReLU                 64 x 64 x 7 x 7      0          False     
    ________________________________________________________________
    Conv2d               64 x 64 x 7 x 7      36,864     False     
    ________________________________________________________________
    BatchNorm2d          64 x 64 x 7 x 7      128        True      
    ________________________________________________________________
    Conv2d               64 x 128 x 4 x 4     73,728     False     
    ________________________________________________________________
    BatchNorm2d          64 x 128 x 4 x 4     256        True      
    ________________________________________________________________
    ReLU                 64 x 128 x 4 x 4     0          False     
    ________________________________________________________________
    Conv2d               64 x 128 x 4 x 4     147,456    False     
    ________________________________________________________________
    BatchNorm2d          64 x 128 x 4 x 4     256        True      
    ________________________________________________________________
    Conv2d               64 x 128 x 4 x 4     8,192      False     
    ________________________________________________________________
    BatchNorm2d          64 x 128 x 4 x 4     256        True      
    ________________________________________________________________
    Conv2d               64 x 128 x 4 x 4     147,456    False     
    ________________________________________________________________
    BatchNorm2d          64 x 128 x 4 x 4     256        True      
    ________________________________________________________________
    ReLU                 64 x 128 x 4 x 4     0          False     
    ________________________________________________________________
    Conv2d               64 x 128 x 4 x 4     147,456    False     
    ________________________________________________________________
    BatchNorm2d          64 x 128 x 4 x 4     256        True      
    ________________________________________________________________
    Conv2d               64 x 256 x 2 x 2     294,912    False     
    ________________________________________________________________
    BatchNorm2d          64 x 256 x 2 x 2     512        True      
    ________________________________________________________________
    ReLU                 64 x 256 x 2 x 2     0          False     
    ________________________________________________________________
    Conv2d               64 x 256 x 2 x 2     589,824    False     
    ________________________________________________________________
    BatchNorm2d          64 x 256 x 2 x 2     512        True      
    ________________________________________________________________
    Conv2d               64 x 256 x 2 x 2     32,768     False     
    ________________________________________________________________
    BatchNorm2d          64 x 256 x 2 x 2     512        True      
    ________________________________________________________________
    Conv2d               64 x 256 x 2 x 2     589,824    False     
    ________________________________________________________________
    BatchNorm2d          64 x 256 x 2 x 2     512        True      
    ________________________________________________________________
    ReLU                 64 x 256 x 2 x 2     0          False     
    ________________________________________________________________
    Conv2d               64 x 256 x 2 x 2     589,824    False     
    ________________________________________________________________
    BatchNorm2d          64 x 256 x 2 x 2     512        True      
    ________________________________________________________________
    Conv2d               64 x 512 x 1 x 1     1,179,648  False     
    ________________________________________________________________
    BatchNorm2d          64 x 512 x 1 x 1     1,024      True      
    ________________________________________________________________
    ReLU                 64 x 512 x 1 x 1     0          False     
    ________________________________________________________________
    Conv2d               64 x 512 x 1 x 1     2,359,296  False     
    ________________________________________________________________
    BatchNorm2d          64 x 512 x 1 x 1     1,024      True      
    ________________________________________________________________
    Conv2d               64 x 512 x 1 x 1     131,072    False     
    ________________________________________________________________
    BatchNorm2d          64 x 512 x 1 x 1     1,024      True      
    ________________________________________________________________
    Conv2d               64 x 512 x 1 x 1     2,359,296  False     
    ________________________________________________________________
    BatchNorm2d          64 x 512 x 1 x 1     1,024      True      
    ________________________________________________________________
    ReLU                 64 x 512 x 1 x 1     0          False     
    ________________________________________________________________
    Conv2d               64 x 512 x 1 x 1     2,359,296  False     
    ________________________________________________________________
    BatchNorm2d          64 x 512 x 1 x 1     1,024      True      
    ________________________________________________________________
    AdaptiveAvgPool2d    64 x 512 x 1 x 1     0          False     
    ________________________________________________________________
    AdaptiveMaxPool2d    64 x 512 x 1 x 1     0          False     
    ________________________________________________________________
    Flatten              64 x 1024            0          False     
    ________________________________________________________________
    BatchNorm1d          64 x 1024            2,048      True      
    ________________________________________________________________
    Dropout              64 x 1024            0          False     
    ________________________________________________________________
    Linear               64 x 512             524,288    True      
    ________________________________________________________________
    ReLU                 64 x 512             0          False     
    ________________________________________________________________
    BatchNorm1d          64 x 512             1,024      True      
    ________________________________________________________________
    Dropout              64 x 512             0          False     
    ________________________________________________________________
    Linear               64 x 2               1,024      True      
    ________________________________________________________________
    
    Total params: 11,704,896
    Total trainable params: 537,984
    Total non-trainable params: 11,166,912
    
    Optimizer used: <function Adam at 0x7ff1f355eae8>
    Loss function: FlattenedLoss of CrossEntropyLoss()
    
    Model frozen up to parameter group number 2
    
    Callbacks:
      - TrainEvalCallback
      - XLAOptCallback
      - Recorder
      - ProgressCallback



Using the `lr_find` works 

```
#colab
learner.lr_find()
```








    SuggestedLRs(lr_min=0.03630780577659607, lr_steep=1.9054607491852948e-06)




![png](docs/images/output_28_2.png)


Fine tune model


```
#colab
learner.fine_tune(1, base_lr=1e-2)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.745400</td>
      <td>0.949072</td>
      <td>0.706724</td>
      <td>00:15</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.330027</td>
      <td>0.192076</td>
      <td>0.932761</td>
      <td>00:09</td>
    </tr>
  </tbody>
</table>


Unfreeze the model

```
#colab
learner.unfreeze()
```

Run the LR Finder again. 

```
#colab
learner.lr_find()
```








    SuggestedLRs(lr_min=0.00020892962347716094, lr_steep=9.12010818865383e-07)




![png](docs/images/output_34_2.png)


Further fine-tuning

```
#colab
learner.fit_one_cycle(1,slice(7e-4),pct_start=0.99)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.155878</td>
      <td>0.152143</td>
      <td>0.959943</td>
      <td>00:11</td>
    </tr>
  </tbody>
</table>


Model params are using TPU

```
#colab
list(learner.model.parameters())[0].device
```




    device(type='xla', index=1)



```
#colab
learner.unfreeze()
```

```
#colab
learner.fit_one_cycle(4,lr_max=slice(1e-6,1e-4))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.097287</td>
      <td>0.154531</td>
      <td>0.955651</td>
      <td>00:06</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.074006</td>
      <td>0.140767</td>
      <td>0.965665</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.064490</td>
      <td>0.134700</td>
      <td>0.971388</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.055005</td>
      <td>0.129603</td>
      <td>0.971388</td>
      <td>00:02</td>
    </tr>
  </tbody>
</table>


```
#colab
learner.fit_one_cycle(4,lr_max=slice(4e-6,5e-4))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.031957</td>
      <td>0.118141</td>
      <td>0.974249</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.022216</td>
      <td>0.110391</td>
      <td>0.972818</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.023388</td>
      <td>0.116325</td>
      <td>0.971388</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.019499</td>
      <td>0.114521</td>
      <td>0.975680</td>
      <td>00:02</td>
    </tr>
  </tbody>
</table>


Plot loss seems to be working fine.

```
#colab
learner.recorder.plot_loss()
```


![png](docs/images/output_43_0.png)


## Status
The fastai XLA extensions library is still in very early development phase (not even alpha) which means there's still a lot of things not working. 

Use it at your own risk.

If you wish to contribute to the project, fork it and make pull request. 

This project uses [nbdev](https://nbdev.fast.ai/) -- a jupyter notebook first development environment and is being developed on [Colab](https://colab.research.google.com).

