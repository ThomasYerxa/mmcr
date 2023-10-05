import sys

sys.path.append("..")

import submitit
from mmcr.train_linear_classifier import train_classifier
from torch import nn

# submitit stuff
slurm_folder = "./slurm/classifier/%j"


executor = submitit.AutoExecutor(folder=slurm_folder)
executor.update_parameters(mem_gb=128, timeout_min=4000)
executor.update_parameters(slurm_array_parallelism=1024)
executor.update_parameters(gpus_per_node=1)
executor.update_parameters(cpus_per_task=13)
executor.update_parameters(slurm_partition="gpu")
executor.update_parameters(constraint="a100")
executor.update_parameters(name="classifier_train")

jobs = []
c = 0
job = executor.submit(
    train_classifier,
    model_path="/mnt/ceph/users/tyerxa/results/MCMC/ssl_training/imagenet/p_bs_momentum_2048_naug_barlow_2_ep_10002_lr_2.0_objective_MCMC_Momentum_projector_512_lmbda_0.0/latest-rank0",
    batch_size=2048,
    lr=1.6,
    epochs=50,
    save_path=None,
)
