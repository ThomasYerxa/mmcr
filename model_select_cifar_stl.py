import sys

sys.path.append("..")

import submitit
from mmcr.cifar_stl.model_select import select_model
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--checkpoint_dir", type=str)
parser.add_argument(
    "--save_path", type=str, default="./training_checkpoints/cifar_stl/"
)

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
executor.update_parameters(name="model_select")

job = executor.submit(
    select_model,
    checkpoint_directory=args.checkpoint_dir,
    dataset=args.dataset,
    batch_size=args.batch_size,
    lr=args.lr,
    epochs=args.epochs,
    save_dir=args.save_path,
)
