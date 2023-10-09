import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

###  KNN based evaluation, for use during unsupervised pretraining to track progress ###
def test_one_epoch(
    net: nn.Module,
    memory_data_loader: DataLoader,
    test_data_loader: DataLoader,
    temperature: float = 0.5,
    k: int = 200,
):
    net.eval()
    total_top1, total_top5, total_num, feature_bank, target_bank = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank and target bank
        for data_tuple in tqdm(memory_data_loader):
            data, target = data_tuple
            target_bank.append(target)
            features, out = net(data.cuda(non_blocking=True))
            feature = F.normalize(features, dim=-1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = (
            torch.cat(target_bank, dim=0).contiguous().to(feature_bank.device)
        )
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data_tuple in test_bar:
            data, target = data_tuple
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            features, out = net(data)
            feature = F.normalize(features, dim=-1)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(
                feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices
            )
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(
                data.size(0) * k, 1000, device=sim_labels.device
            )
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(
                dim=-1, index=sim_labels.view(-1, 1), value=1.0
            )
            # weighted score ---> [B, C]
            pred_scores = torch.sum(
                one_hot_label.view(data.size(0), -1, 1000)
                * sim_weight.unsqueeze(dim=-1),
                dim=1,
            )

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum(
                (pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()
            ).item()
            total_top5 += torch.sum(
                (pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()
            ).item()

            test_bar.set_description(
                "Test Epoch: Acc@1:{:.2f}% Acc@5:{:.2f}%".format(
                    total_top1 / total_num * 100, total_top5 / total_num * 100
                )
            )

        if total_num == 0:
            total_num += 1
    net.train()

    if total_num == 0:
        total_num += 1
    return total_top1 / total_num * 100, total_top5 / total_num * 100