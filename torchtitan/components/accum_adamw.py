# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any, overload

import torch
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

__all__ = ["AccumAdamW"]


class AccumAdamW(Optimizer):
    """AdamW variant with accumulated moments from Algorithm 4.

    The optimizer updates parameters on every ``step()``. The current gradient
    is combined with Adam moments formed from completed groups of
    ``accumulation_steps`` gradients; at each group boundary the grouped average
    updates the stored moments and the accumulator is cleared.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-1,
        *,
        accumulation_steps: int = 4,
        foreach: bool | None = True,
        fused: bool | None = None,
        maximize: bool = False,
    ) -> None:
        if isinstance(lr, torch.Tensor):
            raise ValueError("AccumAdamW does not support tensor learning rates.")
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be positive.")
        if fused:
            raise ValueError(
                "AccumAdamW does not support fused=True; use foreach or for-loop."
            )

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "accumulation_steps": accumulation_steps,
            "foreach": foreach,
            "fused": fused,
            "maximize": maximize,
        }
        super().__init__(params, defaults)
        self._validate_param_groups()

    def _validate_param_groups(self) -> None:
        for group in self.param_groups:
            if group["fused"]:
                raise ValueError(
                    "AccumAdamW does not support fused=True; use foreach or for-loop."
                )
            if group["foreach"] is None:
                group["foreach"] = True
            if not isinstance(group["foreach"], bool):
                raise ValueError("foreach must be a bool or None.")
            accumulation_steps = group["accumulation_steps"]
            if not isinstance(accumulation_steps, int) or accumulation_steps <= 0:
                raise ValueError("accumulation_steps must be a positive integer.")
            betas = group["betas"]
            if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
                raise ValueError(f"Invalid beta parameters: {betas}")
            if group["foreach"] and any(
                isinstance(param, DTensor) for param in group["params"]
            ):
                warnings.warn(
                    "AccumAdamW foreach updates do not support DTensor parameters; "
                    "falling back to for-loop for this parameter group.",
                    stacklevel=2,
                )
                group["foreach"] = False

    @staticmethod
    def _zeros_like(param: torch.Tensor) -> torch.Tensor:
        if isinstance(param, DTensor):
            return torch.zeros_like(param)
        return torch.zeros_like(param, memory_format=torch.preserve_format)

    @staticmethod
    def _init_state(state: dict[str, Any], param: torch.Tensor) -> None:
        if state:
            return
        state["step"] = torch.tensor(0.0, dtype=torch.float32)
        state["exp_avg_hat"] = AccumAdamW._zeros_like(param)
        state["exp_avg_sq_hat"] = AccumAdamW._zeros_like(param)
        state["grad_accum"] = AccumAdamW._zeros_like(param)

    @staticmethod
    def _group_float(group: dict[str, Any], name: str) -> float:
        value = group[name]
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"{name} must be a scalar.")
            return float(value.item())
        return float(value)

    @overload
    def step(self, closure: None = None) -> None:
        ...

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["foreach"]:
                self._foreach_step(group)
            else:
                self._single_tensor_step(group)

        return loss

    def _single_tensor_step(self, group: dict[str, Any]) -> None:
        beta1, beta2 = group["betas"]
        lr = self._group_float(group, "lr")
        eps = self._group_float(group, "eps")
        weight_decay = self._group_float(group, "weight_decay")
        accumulation_steps = group["accumulation_steps"]
        maximize = group["maximize"]

        for param in group["params"]:
            if param.grad is None:
                continue
            if param.grad.is_sparse:
                raise RuntimeError("AccumAdamW does not support sparse gradients.")

            grad = param.grad if not maximize else -param.grad
            state = self.state[param]
            self._init_state(state, param)
            state["step"].add_(1)
            step = int(state["step"].item())
            hat_step = math.ceil(step / accumulation_steps)

            exp_avg_hat = state["exp_avg_hat"]
            exp_avg_sq_hat = state["exp_avg_sq_hat"]
            grad_accum = state["grad_accum"]

            if weight_decay != 0:
                param.mul_(1 - lr * weight_decay)

            exp_avg = exp_avg_hat.mul(beta1).add(grad, alpha=1 - beta1)
            exp_avg_sq = exp_avg_sq_hat.mul(beta2).addcmul(grad, grad, value=1 - beta2)

            bias_correction1 = 1 - beta1**hat_step
            bias_correction2 = 1 - beta2**hat_step
            denom = exp_avg_sq.div(bias_correction2).sqrt().add_(eps)
            param.addcdiv_(exp_avg, denom, value=-(lr / bias_correction1))

            grad_accum.add_(grad, alpha=1.0 / accumulation_steps)
            if step % accumulation_steps == 0:
                exp_avg_hat.mul_(beta1).add_(grad_accum, alpha=1 - beta1)
                exp_avg_sq_hat.mul_(beta2).addcmul_(
                    grad_accum, grad_accum, value=1 - beta2
                )
                grad_accum.zero_()

    def _foreach_step(self, group: dict[str, Any]) -> None:
        beta1, beta2 = group["betas"]
        lr = self._group_float(group, "lr")
        eps = self._group_float(group, "eps")
        weight_decay = self._group_float(group, "weight_decay")
        accumulation_steps = group["accumulation_steps"]
        maximize = group["maximize"]

        buckets: dict[
            tuple[torch.device, torch.dtype, torch.device, torch.dtype, int, bool],
            dict[str, list[torch.Tensor]],
        ] = defaultdict(
            lambda: {
                "params": [],
                "grads": [],
                "exp_avg_hats": [],
                "exp_avg_sq_hats": [],
                "grad_accums": [],
            }
        )

        for param in group["params"]:
            if param.grad is None:
                continue
            if param.grad.is_sparse:
                raise RuntimeError("AccumAdamW does not support sparse gradients.")

            grad = param.grad if not maximize else -param.grad
            state = self.state[param]
            self._init_state(state, param)
            state["step"].add_(1)
            step = int(state["step"].item())
            hat_step = math.ceil(step / accumulation_steps)
            boundary = step % accumulation_steps == 0

            key = (
                param.device,
                param.dtype,
                grad.device,
                grad.dtype,
                hat_step,
                boundary,
            )
            bucket = buckets[key]
            bucket["params"].append(param)
            bucket["grads"].append(grad)
            bucket["exp_avg_hats"].append(state["exp_avg_hat"])
            bucket["exp_avg_sq_hats"].append(state["exp_avg_sq_hat"])
            bucket["grad_accums"].append(state["grad_accum"])

        for (*_, hat_step, boundary), bucket in buckets.items():
            params = bucket["params"]
            grads = bucket["grads"]
            exp_avg_hats = bucket["exp_avg_hats"]
            exp_avg_sq_hats = bucket["exp_avg_sq_hats"]
            grad_accums = bucket["grad_accums"]

            if weight_decay != 0:
                torch._foreach_mul_(params, 1 - lr * weight_decay)

            exp_avgs = torch._foreach_mul(exp_avg_hats, beta1)
            torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

            exp_avg_sqs = torch._foreach_mul(exp_avg_sq_hats, beta2)
            grad_sqs = torch._foreach_mul(grads, grads)
            torch._foreach_add_(exp_avg_sqs, grad_sqs, alpha=1 - beta2)

            bias_correction1 = 1 - beta1**hat_step
            bias_correction2 = 1 - beta2**hat_step
            torch._foreach_div_(exp_avgs, bias_correction1)
            torch._foreach_div_(exp_avg_sqs, bias_correction2)
            torch._foreach_sqrt_(exp_avg_sqs)
            torch._foreach_add_(exp_avg_sqs, eps)
            torch._foreach_div_(exp_avgs, exp_avg_sqs)
            torch._foreach_add_(params, exp_avgs, alpha=-lr)

            torch._foreach_add_(grad_accums, grads, alpha=1.0 / accumulation_steps)
            if boundary:
                torch._foreach_mul_(exp_avg_hats, beta1)
                torch._foreach_add_(exp_avg_hats, grad_accums, alpha=1 - beta1)

                grad_accum_sqs = torch._foreach_mul(grad_accums, grad_accums)
                torch._foreach_mul_(exp_avg_sq_hats, beta2)
                torch._foreach_add_(
                    exp_avg_sq_hats,
                    grad_accum_sqs,
                    alpha=1 - beta2,
                )
                torch._foreach_mul_(grad_accums, 0.0)
