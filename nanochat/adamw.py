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
        param_info = []  # Track (use_all_reduce, grad_slice_idx) for each param
        
        # First pass: reduce gradients
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for base_i in range(len(params)):
                grad = params[base_i].grad
                if grad is None:
                    param_info.append((False, -1))  # No grad, skip
                    continue
                
                # For scalars or 0D tensors, use all_reduce
                if grad.ndim == 0:
                    future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                    all_reduce_futures.append(future)
                    grad_slices.append(grad)
                    param_info.append((True, len(grad_slices) - 1))
                    continue
                
                # Check if first dimension is divisible by world_size
                if grad.shape[0] % world_size == 0:
                    # Use reduce_scatter for memory efficiency
                    rank_size = grad.shape[0] // world_size
                    grad_slice = torch.empty_like(grad[:rank_size])
                    future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                    reduce_scatter_futures.append(future)
                    grad_slices.append(grad_slice)
                    param_info.append((False, len(grad_slices) - 1))  # use_all_reduce=False, grad_slice index
                else:
                    # Use all_reduce for parameters that don't divide evenly
                    # This is less memory efficient but necessary for correctness
                    future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                    all_reduce_futures.append(future)
                    grad_slices.append(grad)  # Use full grad for all_reduce
                    param_info.append((True, len(grad_slices) - 1))  # use_all_reduce=True, grad_slice index

        # Second pass: update parameters
        idx = 0
        reduce_scatter_idx = 0
        all_reduce_idx = 0
        all_gather_futures = []
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            
            for base in range(len(params)):
                p = params[base]
                use_all_reduce, grad_slice_idx = param_info[idx]
                
                # Skip parameters without gradients
                if grad_slice_idx == -1:
                    idx += 1
                    continue
                
                # Wait for gradient reduction to complete
                if use_all_reduce:
                    all_reduce_futures[all_reduce_idx].wait()
                    all_reduce_idx += 1
                    g_slice = grad_slices[grad_slice_idx]  # Full gradient after all_reduce
                    p_slice = p  # Use full parameter
                else:
                    reduce_scatter_futures[reduce_scatter_idx].wait()
                    reduce_scatter_idx += 1
                    rank_size = p.shape[0] // world_size
                    p_slice = p[rank * rank_size:(rank + 1) * rank_size]
                    g_slice = grad_slices[grad_slice_idx]  # Sharded gradient
                
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
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
                if not use_all_reduce:
                    all_gather_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())
                
                idx += 1
        
        # Wait for all gather operations to complete
        torch.futures.collect_all(all_gather_futures).wait()
        
        # For all_reduce parameters, synchronize them across ranks to ensure numerical consistency
        idx = 0
        for group in self.param_groups:
            params = group['params']
            for base in range(len(params)):
                use_all_reduce, grad_slice_idx = param_info[idx]
                if use_all_reduce and grad_slice_idx != -1:
                    p = params[base]
                    # All-reduce the parameter to ensure all ranks have the same values
                    dist.all_reduce(p, op=dist.ReduceOp.AVG)
                idx += 1
