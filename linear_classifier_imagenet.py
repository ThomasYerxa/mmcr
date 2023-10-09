import sys

sys.path.append("..")

import submitit
from mmcr.imagenet.train_linear_classifier import train_classifier
from torch import nn

# submitit stuff
slurm_folder = "./slurm/classifier/%j"


executor = submitit.AutoExecutor(folder=slurm_folder)
executor.update_parameters(mem_gb=128, timeout_min=4000)
executor.update_parameters(slurm_array_parallelism=1024)
executor.update_parameters(gpus_per_node=1)
executor.update_parameters(cpus_per_task=13)
executor.update_parameters(slurm_partition="gpu")
executor.update_parameters(constraint="a100-80gb")
executor.update_parameters(name="classifier_train")
#executor.update_parameters(exclude="workergpu[032,038,042,048,049,072,039,040,045,046,047,068,069,070,050,051,052,053,054,057,058,059,060,061,062,063]")
jobs = []
c = 0
with executor.batch():
    for i in range(24):
        job = executor.submit(
            train_classifier,
            model_path="./training_checkpoints/imagenet/four_views/latest-rank0",
            batch_size=2048,
            lr=1.6,
            epochs=50,
            save_path='./training_checkpoints/imagenet/four_views/',
            save_name='classifier'
        )
        jobs.append(job)
