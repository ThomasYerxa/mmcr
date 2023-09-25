from zipfile import ZipFile
import random
import torch

import torchvision
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter


class ZipImageNet(torchvision.datasets.ImageNet):
    """
    Loads imagenet files from a zip archive.
    """

    def __init__(self, zip_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zip_path = zip_path
        self.zip_archive = None

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        parts = path.split("/")
        idx = parts.index("ILSVRC_2012")
        _path = "/".join(parts[idx:])
        if self.zip_archive is None:
            self.zip_archive = ZipFile(self.zip_path)
        fh = self.zip_archive.open(_path)
        image = Image.open(fh)
        sample = image.convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


# augmentation pipeline adapted from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
class ImageNetValTransform:
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, x):
        return self.transform(x)


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Barlow_Transform:
    def __init__(self, n_transform):
        self.n_aug = n_transform
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.transform_prime = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, x):
        y1 = self.transform(x)
        y = torch.zeros(self.n_aug, y1.shape[0], y1.shape[1], y1.shape[2])

        for i in range(self.n_aug // 2):
            y1 = self.transform(x)
            y2 = self.transform_prime(x)
            y[2 * i, :, :, :] = y1
            y[2 * i + 1, :, :, :] = y2
        return y


def get_datasets(n_aug, imagenet_path, zip_path, **kwargs):
    train_data = ZipImageNet(
        zip_path=zip_path,
        root=imagenet_path,
        split="train",
        transform=Barlow_Transform(n_transform=n_aug),
    )
    memory_data = ZipImageNet(
        zip_path=zip_path,
        root=imagenet_path,
        split="train",
        transform=ImageNetValTransform(),
    )
    test_data = ZipImageNet(
        zip_path=zip_path,
        root=imagenet_path,
        split="val",
        transform=ImageNetValTransform(),
    )

    return train_data, memory_data, test_data
