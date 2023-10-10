import torch
from tqdm import tqdm
import einops

from mmcr.cifar_stl.data import get_datasets
from mmcr.cifar_stl..models import Model
from mmrc.cifar_stl.knn import test_one_epoch
from mmcr.cifar_stl.loss_mmcr import MMCR_Loss

from src.utils import checkpoint_model


def train(
    dataset: str,
    n_aug: int,
    batch_size: int,
    lr: float,
    epochs: int,
    lmbda: float,
    save_folder: str,
    save_freq: int,
):
    train_dataset, memory_dataset, test_dataset = get_datasets(dataset=dataset, n_aug=n_aug)
    model = Model(projector_dims=[512, 128], dataset=dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=12
    )
    memory_loader = torch.utils.data.DataLoader(
        memory_dataset, batch_size=128, shuffle=True, num_workers=12
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=12
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    loss_function = MMCR_Loss(lmbda=0.0, n_aug=n_aug, distributed=False)

    model = model.cuda()
    top_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss, total_num, train_bar = 0.0, 0, tqdm(train_loader)
        for step, data_tuple in enumerate(train_bar):
            optimizer.zero_grad()

            # forward pass
            img_batch, labels = data_tuple
            img_batch = einops.rearrange(img_batch, "B N C H W -> (B N) C H W")
            features, out = model(img_batch.cuda(non_blocking=True))
            loss, loss_dict = loss_function(out)

            # backward pass
            loss.backward()
            optimizer.step()

            # update the training bar
            total_num += data_tuple[0].size(0)
            total_loss += loss.item() * data_tuple[0].size(0)
            train_bar.set_description(
                "Train Epoch: [{}/{}] Loss: {:.4f}".format(
                    epoch, epochs, total_loss / total_num
                )
            )

        if epoch % 1 == 0:
            acc_1, acc_5 = test_one_epoch(
                model, memory_loader, test_loader, c=10, epoch=epoch, writer=None
            )
            if acc_1 > top_acc:
                top_acc = acc_1

            if (epoch % save_freq == 0 or acc_1 == top_acc) and False:
                torch.save(
                    model.state_dict(),
                    f"{save_folder}/{dataset}_{n_aug}_{epoch}_acc_{acc_1:0.2f}.pth",
                )
