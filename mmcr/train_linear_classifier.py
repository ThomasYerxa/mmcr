import torch
from torch import nn, optim
from torchvision.models.resnet import resnet50
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

import numpy as np
from tqdm import tqdm
from typing import OrderedDict

from mmcr.data import ZipImageNet, ImageNetValTransform


# train or test linear classifier for one epoch
def train_val(net, data_loader, train_optimizer, epoch):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    print("startin epoch")

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
            print("data loaded")
            out = net(data)
            print("forward complete")
            loss = loss_criterion(out, target)
            print("loss_calculted")

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                print("backward completed")
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
    batch_size: int = 512,
    epochs: int = 50,
    lr: float = 1e-2,
    save_path=None,
    save_name=None,
):
    top_acc = 0.0
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_data = ZipImageNet(
        zip_path="./datasets/ILSVRC_2012.zip",
        root="./datasets/ILSVRC_2012",
        split="train",
        transform=train_transform,
    )
    test_data = ZipImageNet(
        zip_path="./datasets/ILSVRC_2012.zip",
        root="./datasets/ILSVRC_2012",
        split="val",
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=13, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=13, pin_memory=True
    )

    # load pretrained weights (fully connected layer excluded)
    model = resnet50()

    sd = torch.load(model_path, map_location="cpu")["state"]["model"]
    new_sd = OrderedDict()
    for k, v in sd.items():
        # skip projector, momentum networks, and fully connected
        if "g." in k or "mom_" in k or "fc" in k:
            continue
        parts = k.split(".")
        idx = parts.index("f")
        new_k = ".".join(parts[idx + 1 :])
        new_sd[new_k] = v
    model.load_state_dict(new_sd, strict=False)

    # only fully connected requires grad
    model.requires_grad_(False)
    model.fc.requires_grad_(True)
    model = model.cuda()

    optimizer = optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)
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

        if test_acc_1 > top_acc and save_path is not None and save_name is not None:
            top_acc = test_acc_1
            save_str = save_path + save_name + ".pt"
            torch.save(model.state_dict(), save_str)

    return None
