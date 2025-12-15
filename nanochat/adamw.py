"""
Borrowed from modded-nanogpt. By Keller, @vagrawal, et al.
Not a general optimizer! But works for our specific use.
"""
import torch
import torch.distributed as dist
from torch import Tensor


class DistAdamW(torch.optim.Optimizer):
    """
    Distributed AdamW optimizer.
    In the style of ZeRO-2, i.e. sharded optimizer states and gradient reduction
    """
    def __init__(self, param_groups, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        grad_slices = []
        use_all_reduce = []  # Track which params use all_reduce vs reduce_scatter
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for base_i in range(len(params)):
                grad = params[base_i].grad
                # Check if first dimension is divisible by world_size
                if grad.shape[0] % world_size == 0:
                    # Use reduce_scatter for memory efficiency
                    rank_size = grad.shape[0] // world_size
                    grad_slice = torch.empty_like(grad[:rank_size])
                    reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                    grad_slices.append(grad_slice)
                    use_all_reduce.append(False)
                else:
                    # Use all_reduce for parameters that don't divide evenly
                    # This is less memory efficient but necessary for correctness
                    all_reduce_futures.append(dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                    grad_slices.append(grad)  # Use full grad for all_reduce
                    use_all_reduce.append(True)

        idx = 0
        reduce_scatter_idx = 0
        all_reduce_grad_idx = 0
        all_gather_futures = []
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            for base in range(len(params)):
                p = params[base]
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                
                if use_all_reduce[idx]:
                    # Wait for all_reduce to complete
                    all_reduce_futures[all_reduce_grad_idx].wait()
                    all_reduce_grad_idx += 1
                    g_slice = grad_slices[idx]  # Full gradient after all_reduce
                    p_slice = p  # Use full parameter
                else:
                    # Wait for reduce_scatter to complete
                    reduce_scatter_futures[reduce_scatter_idx].wait()
                    reduce_scatter_idx += 1
                    rank_size = p.shape[0] // world_size
                    p_slice = p[rank * rank_size:(rank + 1) * rank_size]
                    g_slice = grad_slices[idx]  # Sharded gradient
                
                state = self.state[p]
                # State init
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p_slice)
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                # update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                # bias corrections
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                # compute step
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)
                
                # Gather parameters back if we used reduce_scatter
                if not use_all_reduce[idx]:
                    all_gather_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())
                
                idx += 1
        torch.futures.collect_all(all_gather_futures).wait()
