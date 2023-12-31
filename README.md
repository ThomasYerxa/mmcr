# Efficient Coding of Natural Images using Maximum Manifold Capacity Representations
This is a pytorch implementation of the paper [Efficient Coding of Natural Images using Maximum Manifold Capacity Representations](https://openreview.net/pdf?id=og9V7NgOrQ).

## Environment
To install dependencies create a conda environment from the provided `environment.yml` file, and install thei project package by running `pip install -e .` in the base directory.
We utilized Pytorch 1.11 for all experiments and [Composer from MosaicML](https://docs.mosaicml.com/projects/composer/en/stable/index.html) for distributed pretraining on ImageNet datasets.

## Datasets
We provide code for pretraining and linear evaluation on CIFAR-10/100, STL-10, and ImageNet-100/1k.
The code expects all dataset files to be located in the `/datasets` directory.
For ImageNet datasets we also provide an implementation for reading images from a ZIP archive rather than opening each image file individually. 
This reduces the I/O overhead of dataloading, but requires zipping the datasets before training which can take up to several hours for ImageNet-1k.
The use of zipped dataloading can be toggled on/off via the parameter `use_zip` (see below).


## Pretraining
The code is setup to run on a SLURM cluster and uses [submitit](https://github.com/facebookincubator/submitit) for job submission.

### ImageNet
To pretrain on ImageNet with default settings run the command:  
```
python3 pretrain_imagenet.py
```
By default training uses 4 nodes each with 4 A100 GPUs (though 8-view training requires 8 nodes).
Hyperparameters can be adjusted in the command line, i.e. to run with 4 views rather than 2: 
```
python3 pretrain_imagenet.py --n_aug 4
```
See `pretrain_imagenet.py` for details.


### CIFAR/STL
To pretrain on either CIFAR or STL instead run
```
python3 pretrain_cifar_stl.py 
```
Use command line arguments to specify the pretraining dataset and other hyperparameters (see `pretrain_cifar_stl.py` for details).
Pretraining on these smaller datasets uses a single A100 GPU.

## Evaluation
We run frozen linear evaluation for all datasets on a single A100 GPU.

### ImageNet
To run frozen-linear evaluation on an ImageNet dataset run
```
python3 linear_classifier_imagenet.py --model_path /path/to/checkpoint_file
```
```checkpoint_file``` should contain a checkpoint that is generated during an ImageNet pretraining run.
Other hyperparameters can be adjusted via command line arguments similarly to above.


### CIFAR/STL
For CIFAR/STL we run frozen linear evaluations on a large number of checkpoints saved during pretraining to perform model selection.
To run model selection run the command:
```
python3 model_select_cifar_stl.py --checkpoint_dir /path/to/checkpoint_directory
```
where `checkpoint_directory` contains all checkpoints generated by running pretraining on CIFAR/STL as specified above.
