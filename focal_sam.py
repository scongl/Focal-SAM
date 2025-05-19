import torch


class FocalSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(FocalSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                self.state[p]["grad"] = p.grad.clone()

        self.zero_grad()

    @torch.no_grad()
    def second_step(self):
        # 计算epsilon
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)

        self.zero_grad()

    @torch.no_grad()
    def third_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue     
                
                p.data = self.state[p]["old_p"]
                p.grad.add_(self.state[p]["grad"])

        self.base_optimizer.step()
        self.zero_grad()


    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
