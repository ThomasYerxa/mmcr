import sys

sys.path.append("..")

import submitit
from mmcr.imagenet.train_linear_classifier import train_classifier
from torch import nn
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--lr", type=float, default=0.3)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--model_path", type=str)
parser.add_argument("--save_path", type=str, default="./training_checkpoints/imagenet/")
parser.add_argument("--save_name", type=str, default="classifier")
parser.add_argument('--use_zip', action="store_true")

args = parser.parse_args()

# submitit stuff
slurm_folder = "./slurm/classifier/%j"
executor = submitit.AutoExecutor(folder=slurm_folder)
executor.update_parameters(mem_gb=128, timeout_min=10000)
executor.update_parameters(slurm_array_parallelism=1024)
executor.update_parameters(gpus_per_node=1)
executor.update_parameters(cpus_per_task=13)
executor.update_parameters(slurm_partition="gpu")
executor.update_parameters(constraint="a100-80gb")
executor.update_parameters(name="classifier_train")

job = executor.submit(
    train_classifier,
    model_path=args.model_path,
    batch_size=args.batch_size,
    lr=args.lr,
    epochs=args.epochs,
    save_path=args.save_path,
    save_name=args.save_name,
    use_zip=args.use_zip
)