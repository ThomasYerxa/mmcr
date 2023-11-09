import torch
from torch import optim
from composer import State, Callback, Logger


## momentum update via callback
class MomentumUpdate(Callback):
    def __init__(self, tau=0.99) -> None:
        super(MomentumUpdate, self).__init__()
        self.tau = tau
        self.first_time = True

    def batch_end(self, state: State, logger: Logger) -> None:
        online_f, momentum_f = (
            state.model.module.module.f,
            state.model.module.module.mom_f,
        )
        online_g, momentum_g = (
            state.model.module.module.g,
            state.model.module.module.mom_g,
        )

        with torch.no_grad():
            for op, mp in zip(online_f.parameters(), momentum_f.parameters()):
                mp.data = self.tau * mp.data + (1 - self.tau) * op.data

            for op, mp in zip(online_g.parameters(), momentum_g.parameters()):
                mp.data = self.tau * mp.data + (1 - self.tau) * op.data

            if self.first_time:
                print("Performing momentum update")
                self.first_time = False

        return None


# modified from https://github.com/mosaicml/composer/blob/80d3293df833edfdb4249daee3f0ddcd25259fa2/composer/callbacks/lr_monitor.py#L11 to print to stdout
class LogLR(Callback):
    def __init__(self):
        pass

    def epoch_start(self, state: State, logger: Logger):
        for optimizer in state.optimizers:
            lrs = [group["lr"] for group in optimizer.param_groups]
            name = optimizer.__class__.__name__
            for lr in lrs:
                for idx, lr in enumerate(lrs):
                    print({f"lr-{name}/group{idx}": lr})


# modified from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=1e-6,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=True,
        lars_adaptation_filter=True,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if not g["weight_decay_filter"] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if not g["lars_adaptation_filter"] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def collate_fn(batch):
    new_img = [torch.as_tensor(img) for img, _ in batch]
    new_target = [torch.as_tensor(lbl) for _, lbl in batch]
    new_img = torch.cat(new_img, dim=0)
    new_target = torch.stack(new_target, dim=0)
    return new_img, new_img


def get_num_samples_in_batch(multicrop_batch):
    return multicrop_batch[0][0][0].shape[0]
