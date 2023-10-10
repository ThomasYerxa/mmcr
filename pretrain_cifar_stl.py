from mmcr.cifar_stl.train import train

from argparse import ArgumentParser
import submitit

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_aug", type=int, default=40)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lmbda", type=float, default=0.0)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--save_freq", type=int, default=5)
parser.add_argument(
    "--save_folder",
    type=str,
    default="./training_checkpoints/cifar_stl",
)
args = parser.parse_args()

# submitit job management
executor = submitit.AutoExecutor(folder="./slurm/pretrain/%j", slurm_max_num_timeout=30)

executor.update_parameters(
    mem_gb=128,
    gpus_per_node=1,
    tasks_per_node=1,
    cpus_per_task=args.num_workers,
    nodes=1,
    name="MMCR",
    timeout_min=60 * 72,
    slurm_partition="gpu",
    constraint="a100-80gb",
    slurm_array_parallelism=512,
)

job = executor.submit(
    train,
    dataset=args.dataset,
    n_aug=args.n_aug,
    batch_size=args.batch_size,
    lr=args.lr,
    epochs=args.epochs,
    lmbda=args.lmbda,
    save_folder=args.save_folder,
    save_freq=args.save_freq,
)
