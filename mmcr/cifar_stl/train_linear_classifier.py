import torch
from torch import nn, optim
from torchvision.models.resnet import resnet50
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

import numpy as np
from tqdm import tqdm
from typing import OrderedDict

from mmcr.cifar_stl.data import get_datasets
from mmcr.cifar_stl.models import Model


# train or test linear classifier for one epoch
def train_val(net, data_loader, train_optimizer, epoch):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = (
        0.0,
        0.0,
        0.0,
        0,
        tqdm(data_loader),
    )
    with torch.enable_grad() if is_train else torch.no_grad():
        loss_criterion = nn.CrossEntropyLoss()
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum(
                (prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()
            ).item()
            total_correct_5 += torch.sum(
                (prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()
            ).item()

            data_bar.set_description(
                "{} Epoch: [{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%".format(
                    "Train" if is_train else "Test",
                    epoch,
                    total_loss / total_num,
                    total_correct_1 / total_num * 100,
                    total_correct_5 / total_num * 100,
                )
            )

    return (
        total_loss / total_num,
        total_correct_1 / total_num * 100,
        total_correct_5 / total_num * 100,
    )


def train_classifier(
    model_path: str,
    dataset: str = "cifar10",
    batch_size: int = 512,
    epochs: int = 50,
    lr: float = 1e-2,
    save_path=None,
    save_name=None,
):
    top_acc = 0.0
    train_data, _, test_data = get_datasets(
        dataset, 1, "./datasets", batch_transform=False
    )

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=13, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=13, pin_memory=True
    )

    # load pretrained weights
    pretrained_model = Model(dataset=dataset)
    sd = torch.load(model_path, map_location="cpu")
    pretrained_model.load_state_dict(sd)
    dataset_num_classes = {"cifar10": 10, "stl10": 10, "cifar100": 100}
    model = Net(pretrained_model.f, dataset_num_classes[dataset])

    # only fully connected requires grad
    model.requires_grad_(False)
    model.fc.requires_grad_(True)
    model = model.cuda()

    optimizer = optim.Adam(model.fc.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if save_path is not None and save_name is None:
        save_name = str(np.random.rand() * 1e5)
    print(save_name)
    for epoch in range(1, epochs + 1):
        # train one epoch
        train_loss, train_acc_1, train_acc_5 = train_val(
            model, train_loader, optimizer, epoch
        )
        # test one epoch
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None, epoch)
        scheduler.step()

        if test_acc_1 > top_acc:
            top_acc = test_acc_1

        if test_acc_1 == top_acc and save_path is not None and save_name is not None:
            save_str = save_path + save_name + ".pt"
            torch.save(model.state_dict(), save_str)

    return model, top_acc


# a wrapper class for the resnet50 model
class Net(nn.Module):
    def __init__(self, f, num_classes):
        super().__init__()
        self.f = f
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        f = self.f(x)
        f = f.view(f.size(0), -1)
        return self.fc(f)
