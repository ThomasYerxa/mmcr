import torch
from torch import nn, Tensor
import torch.nn.functional as F

from typing import List, Tuple
import einops


class MMCR_Momentum_Loss(nn.Module):
    def __init__(
        self,
        lmbda: float,
        n_aug: int,
        distributed: bool = True,
    ):
        super(MMCR_Momentum_Loss, self).__init__()
        self.lmbda = lmbda
        self.n_aug = n_aug
        self.distributed = distributed

    def forward(self, z_list: List[Tensor]) -> Tuple[Tensor, dict]:
        z, z_m = z_list[0], z_list[1]
        z = F.normalize(z, dim=-1)
        z_m = F.normalize(z_m, dim=-1)

        z_local_ = einops.rearrange(z, "(B N) C -> B C N", N=self.n_aug)
        z_local_m = einops.rearrange(z_m, "(B N) C -> B C N", N=self.n_aug)

        # gather across devices into list
        if self.distributed:
            z_list = [
                torch.zeros_like(z_local_)
                for i in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(z_list, z_local_, async_op=False)
            z_list[torch.distributed.get_rank()] = z_local_

            # gather momentum outputs
            z_m_list = [
                torch.zeros_like(z_local_m)
                for i in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(z_m_list, z_local_m, async_op=False)
            z_m_list[torch.distributed.get_rank()] = z_local_m

            # append all
            z_local = torch.cat(z_list)
            z_m_local = torch.cat(z_m_list)

        else:
            z_local = z_local_
            z_m_local = z_local_m

        if self.lmbda == 0:
            local_nuc = 0
        else:
            local_nuc = torch.linalg.svdvals(z_local).sum()

        centroids = (torch.mean(z_local, dim=-1) + torch.mean(z_m_local, dim=-1)) * 0.5

        # filter infs and nans
        selected = centroids.isfinite().all(dim=1)
        centroids = centroids[selected]

        if selected.sum() != centroids.shape[0]:
            print("filtered nan")

        global_nuc = torch.linalg.svdvals(centroids).sum()

        batch_size = z_local.shape[0]
        loss = -1.0 * global_nuc + self.lmbda * local_nuc / batch_size
        loss = loss * 1.00

        loss_dict = {
            "loss": loss.item(),
            "local_nuc": local_nuc,
            "global_nuc": global_nuc.item(),
        }

        return loss, loss_dict
