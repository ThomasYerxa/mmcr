from mmcr.imagenet.misc import LARS, MomentumUpdate, LogLR, collate_fn, get_num_samples_in_batch
from mmcr.imagenet.loss_mmcr_momentum import MMCR_Momentum_Loss
from mmcr.imagenet.data import get_datasets
from mmcr.imagenet.knn import KnnMonitor
from mmcr.imagenet.models import MomentumModel, MomentumComposerWrapper, Model, ComposerWrapper

import torch
import composer
from composer.optim.scheduler import CosineAnnealingWithWarmupScheduler
import submitit

import os


def train(gpu, args, **kwargs):
    # composer doesn't require init_dist_gpu() function call
    job_env = submitit.JobEnvironment()
    args.gpu = job_env.local_rank
    args.rank = job_env.global_rank

    # better port
    tmp_port = os.environ["SLURM_JOB_ID"]
    tmp_port = int(tmp_port[-4:]) + 50000
    args.port = tmp_port

    os.environ["RANK"] = str(job_env.global_rank)
    os.environ["WORLD_SIZE"] = str(args.n_gpus * args.n_nodes)
    os.environ["LOCAL_RANK"] = str(job_env.local_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(args.n_gpus)
    os.environ["NODE_RANK"] = str(int(os.getenv("SLURM_NODEID")))
    os.environ["MASTER_ADDR"] = args.host_name_
    os.environ["MASTER_PORT"] = str(args.port)
    os.environ["PYTHONUNBUFFERED"] = "1"

    args.torch_cuda_device_count = torch.cuda.device_count()
    args.slurm_nodeid = int(os.getenv("SLURM_NODEID"))
    args.slurm_nnodes = int(os.getenv("SLURM_NNODES"))

    print(args)

    # datasets
    train_data, memory_data, test_data = get_datasets(
        args.n_aug, args.imagenet_path, args.zip_path
    )

    # samplers
    train_sampler = torch.utils.data.DistributedSampler(
        train_data, num_replicas=args.world_size, rank=args.rank, seed=31
    )
    memory_sampler = torch.utils.data.DistributedSampler(
        memory_data, num_replicas=args.world_size, rank=args.rank, seed=31
    )
    test_sampler = torch.utils.data.DistributedSampler(
        test_data, num_replicas=args.world_size, rank=args.rank, seed=31
    )

    # dataloaders
    batch_size = int(args.batch_size / args.n_gpus / args.n_nodes)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=train_sampler,
    )

    memory_loader = torch.utils.data.DataLoader(
        dataset=memory_data,
        batch_size=512,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=memory_sampler,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=128,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=test_sampler,
    )

    # objective
    args.distributed = args.n_gpus * args.n_nodes > 1
    objective = MMCR_Momentum_Loss(args.lmbda, args.n_aug, args.distributed)
    projector_dims = [8192, 8192, 512]
    objective = torch.nn.SyncBatchNorm.convert_sync_batchnorm(objective)

    # model
    model = MomentumModel(projector_dims=projector_dims)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    wrapped_model = MomentumComposerWrapper(module=model, objective=objective)

    # optimizer
    lr = args.lr * args.batch_size / 256
    optimizer = LARS(
        model.parameters(),
        lr=lr,
        weight_decay=1e-6,
        weight_decay_filter=True,
        lars_adaptation_filter=True,
    )

    # scheduler
    scheduler = CosineAnnealingWithWarmupScheduler(t_warmup="10ep", alpha_f=0.001)

    # callbacks
    #callback_list = [KnnMonitor(memory_loader, test_loader), LogLR()]
    callback_list = [LogLR()]
    callback_list.append(MomentumUpdate(tau=args.tau))

    # dspec
    train_dspec = composer.DataSpec(
        train_loader, get_num_samples_in_batch=get_num_samples_in_batch
    )

    print(model)

    # trainer
    trainer = composer.Trainer(
        train_dataloader=train_dspec,
        optimizers=optimizer,
        model=wrapped_model,
        max_duration=args.epochs,
        precision="amp",
        algorithms=[
            composer.algorithms.ChannelsLast(),
        ],
        device="gpu",
        seed=31,
        callbacks=callback_list,
        schedulers=(scheduler),
        save_interval=args.save_freq,
        save_overwrite=True,
        save_folder=args.save_folder,
    )

    trainer.fit()
