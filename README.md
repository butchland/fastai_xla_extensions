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
VERSION = "20200325"  #@param ["1.5" , "20200325", "nightly"]
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version $VERSION
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  4139  100  4139    0     0  21557      0 --:--:-- --:--:-- --:--:-- 21557
    Updating TPU and VM. This may take around 2 minutes.
    Updating TPU runtime to pytorch-dev20200325 ...
    Collecting cloud-tpu-client
      Downloading https://files.pythonhosted.org/packages/56/9f/7b1958c2886db06feb5de5b2c191096f9e619914b6c31fdf93999fdbbd8b/cloud_tpu_client-0.10-py3-none-any.whl
    Requirement already satisfied: oauth2client in /usr/local/lib/python3.6/dist-packages (from cloud-tpu-client) (4.1.3)
    Collecting google-api-python-client==1.8.0
    [?25l  Downloading https://files.pythonhosted.org/packages/9a/b4/a955f393b838bc47cbb6ae4643b9d0f90333d3b4db4dc1e819f36aad18cc/google_api_python_client-1.8.0-py3-none-any.whl (57kB)
    [K     |████████████████████████████████| 61kB 2.6MB/s 
    [?25hRequirement already satisfied: rsa>=3.1.4 in /usr/local/lib/python3.6/dist-packages (from oauth2client->cloud-tpu-client) (4.6)
    Requirement already satisfied: pyasn1-modules>=0.0.5 in /usr/local/lib/python3.6/dist-packages (from oauth2client->cloud-tpu-client) (0.2.8)
    Requirement already satisfied: pyasn1>=0.1.7 in /usr/local/lib/python3.6/dist-packages (from oauth2client->cloud-tpu-client) (0.4.8)
    Requirement already satisfied: six>=1.6.1 in /usr/local/lib/python3.6/dist-packages (from oauth2client->cloud-tpu-client) (1.12.0)
    Requirement already satisfied: httplib2>=0.9.1 in /usr/local/lib/python3.6/dist-packages (from oauth2client->cloud-tpu-client) (0.17.4)
    Requirement already satisfied: google-auth-httplib2>=0.0.3 in /usr/local/lib/python3.6/dist-packages (from google-api-python-client==1.8.0->cloud-tpu-client) (0.0.3)
    Requirement already satisfied: google-auth>=1.4.1 in /usr/local/lib/python3.6/dist-packages (from google-api-python-client==1.8.0->cloud-tpu-client) (1.17.2)
    Requirement already satisfied: google-api-core<2dev,>=1.13.0 in /usr/local/lib/python3.6/dist-packages (from google-api-python-client==1.8.0->cloud-tpu-client) (1.16.0)
    Requirement already satisfied: uritemplate<4dev,>=3.0.0 in /usr/local/lib/python3.6/dist-packages (from google-api-python-client==1.8.0->cloud-tpu-client) (3.0.1)
    Requirement already satisfied: setuptools>=40.3.0 in /usr/local/lib/python3.6/dist-packages (from google-auth>=1.4.1->google-api-python-client==1.8.0->cloud-tpu-client) (47.3.1)
    Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.6/dist-packages (from google-auth>=1.4.1->google-api-python-client==1.8.0->cloud-tpu-client) (4.1.0)
    Requirement already satisfied: googleapis-common-protos<2.0dev,>=1.6.0 in /usr/local/lib/python3.6/dist-packages (from google-api-core<2dev,>=1.13.0->google-api-python-client==1.8.0->cloud-tpu-client) (1.52.0)
    Requirement already satisfied: protobuf>=3.4.0 in /usr/local/lib/python3.6/dist-packages (from google-api-core<2dev,>=1.13.0->google-api-python-client==1.8.0->cloud-tpu-client) (3.10.0)
    Requirement already satisfied: requests<3.0.0dev,>=2.18.0 in /usr/local/lib/python3.6/dist-packages (from google-api-core<2dev,>=1.13.0->google-api-python-client==1.8.0->cloud-tpu-client) (2.23.0)
    Requirement already satisfied: pytz in /usr/local/lib/python3.6/dist-packages (from google-api-core<2dev,>=1.13.0->google-api-python-client==1.8.0->cloud-tpu-client) (2018.9)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0dev,>=2.18.0->google-api-core<2dev,>=1.13.0->google-api-python-client==1.8.0->cloud-tpu-client) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0dev,>=2.18.0->google-api-core<2dev,>=1.13.0->google-api-python-client==1.8.0->cloud-tpu-client) (1.24.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0dev,>=2.18.0->google-api-core<2dev,>=1.13.0->google-api-python-client==1.8.0->cloud-tpu-client) (2020.6.20)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0dev,>=2.18.0->google-api-core<2dev,>=1.13.0->google-api-python-client==1.8.0->cloud-tpu-client) (2.9)
    Uninstalling torch-1.5.1+cu101:
    Installing collected packages: google-api-python-client, cloud-tpu-client
      Found existing installation: google-api-python-client 1.7.12
        Uninstalling google-api-python-client-1.7.12:
          Successfully uninstalled google-api-python-client-1.7.12
    Successfully installed cloud-tpu-client-0.10 google-api-python-client-1.8.0
    Done updating TPU runtime
      Successfully uninstalled torch-1.5.1+cu101
    Uninstalling torchvision-0.6.1+cu101:
      Successfully uninstalled torchvision-0.6.1+cu101
    Copying gs://tpu-pytorch/wheels/torch-nightly+20200325-cp36-cp36m-linux_x86_64.whl...
    - [1 files][ 83.4 MiB/ 83.4 MiB]                                                
    Operation completed over 1 objects/83.4 MiB.                                     
    Copying gs://tpu-pytorch/wheels/torch_xla-nightly+20200325-cp36-cp36m-linux_x86_64.whl...
    \ [1 files][114.5 MiB/114.5 MiB]                                                
    Operation completed over 1 objects/114.5 MiB.                                    
    Copying gs://tpu-pytorch/wheels/torchvision-nightly+20200325-cp36-cp36m-linux_x86_64.whl...
    / [1 files][  2.5 MiB/  2.5 MiB]                                                
    Operation completed over 1 objects/2.5 MiB.                                      
    Processing ./torch-nightly+20200325-cp36-cp36m-linux_x86_64.whl
    Requirement already satisfied: future in /usr/local/lib/python3.6/dist-packages (from torch==nightly+20200325) (0.16.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from torch==nightly+20200325) (1.18.5)
    [31mERROR: fastai 1.0.61 requires torchvision, which is not installed.[0m
    Installing collected packages: torch
    Successfully installed torch-1.5.0a0+d6149a7
    Processing ./torch_xla-nightly+20200325-cp36-cp36m-linux_x86_64.whl
    Installing collected packages: torch-xla
    Successfully installed torch-xla-1.6+e788e5b
    Processing ./torchvision-nightly+20200325-cp36-cp36m-linux_x86_64.whl
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from torchvision==nightly+20200325) (1.12.0)
    Requirement already satisfied: torch in /usr/local/lib/python3.6/dist-packages (from torchvision==nightly+20200325) (1.5.0a0+d6149a7)
    Requirement already satisfied: pillow>=4.1.1 in /usr/local/lib/python3.6/dist-packages (from torchvision==nightly+20200325) (7.0.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from torchvision==nightly+20200325) (1.18.5)
    Requirement already satisfied: future in /usr/local/lib/python3.6/dist-packages (from torch->torchvision==nightly+20200325) (0.16.0)
    Installing collected packages: torchvision
    Successfully installed torchvision-0.6.0a0+3c254fb
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    The following package was automatically installed and is no longer required:
      libnvidia-common-440
    Use 'apt autoremove' to remove it.
    The following NEW packages will be installed:
      libomp5
    0 upgraded, 1 newly installed, 0 to remove and 33 not upgraded.
    Need to get 234 kB of archives.
    After this operation, 774 kB of additional disk space will be used.
    Get:1 http://archive.ubuntu.com/ubuntu bionic/universe amd64 libomp5 amd64 5.0.1-1 [234 kB]
    Fetched 234 kB in 1s (359 kB/s)
    Selecting previously unselected package libomp5:amd64.
    (Reading database ... 144379 files and directories currently installed.)
    Preparing to unpack .../libomp5_5.0.1-1_amd64.deb ...
    Unpacking libomp5:amd64 (5.0.1-1) ...
    Setting up libomp5:amd64 (5.0.1-1) ...
    Processing triggers for libc-bin (2.27-3ubuntu1) ...
    /sbin/ldconfig.real: /usr/local/lib/python3.6/dist-packages/ideep4py/lib/libmkldnn.so.0 is not a symbolic link
    


Install fastai2 and the fastai_xla_extensions packages

```
!pip install fastai2 > /dev/null
```

```
!pip install git+https://github.com/butchland/fastai_xla_extensions > /dev/null
```

      Running command git clone -q https://github.com/butchland/fastai_xla_extensions /tmp/pip-req-build-0ytj2cpq


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
path = untar_data(URLs.PETS)/'images'
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

    Downloading: "https://download.pytorch.org/models/resnet34-333f7ec4.pth" to /root/.cache/torch/checkpoints/resnet34-333f7ec4.pth


    


```
#colab
learner.fine_tune(1)
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
      <td>0.865359</td>
      <td>0.295098</td>
      <td>0.907984</td>
      <td>15:03</td>
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
      <td>1.659670</td>
      <td>2.236336</td>
      <td>0.488498</td>
      <td>14:13</td>
    </tr>
  </tbody>
</table>


## Status
The fastai XLA extensions library is still in very early development phase (not even alpha) which means there's still a lot of things not working. 

Use it at your own risk.

If you wish to contribute to the project, fork it and make pull request. 

This project uses [nbdev](https://nbdev.fast.ai/) -- a jupyter notebook first development environment and is being developed on [Colab](https://colab.research.google.com).

