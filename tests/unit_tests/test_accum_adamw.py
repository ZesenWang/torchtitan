# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.components.optimizer import (
    AccumAdamW,
    OptimizersContainer,
    ParamGroupConfig,
)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=True)


def _reference_accum_adamw(
    param: torch.Tensor,
    grads: list[torch.Tensor],
    *,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    accumulation_steps: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    beta1, beta2 = betas
    result = param.clone()
    exp_avg_hat = torch.zeros_like(result)
    exp_avg_sq_hat = torch.zeros_like(result)
    grad_accum = torch.zeros_like(result)
    step = torch.tensor(0.0)

    for grad in grads:
        step += 1
        step_int = int(step.item())
        hat_step = (step_int + accumulation_steps - 1) // accumulation_steps

        if weight_decay != 0:
            result = result * (1 - lr * weight_decay)

        exp_avg = beta1 * exp_avg_hat + (1 - beta1) * grad
        exp_avg_sq = beta2 * exp_avg_sq_hat + (1 - beta2) * grad.square()
        bias_correction1 = 1 - beta1**hat_step
        bias_correction2 = 1 - beta2**hat_step
        denom = (exp_avg_sq / bias_correction2).sqrt() + eps
        result = result - lr * (exp_avg / bias_correction1) / denom

        grad_accum = grad_accum + grad / accumulation_steps
        if step_int % accumulation_steps == 0:
            exp_avg_hat = beta1 * exp_avg_hat + (1 - beta1) * grad_accum
            exp_avg_sq_hat = beta2 * exp_avg_sq_hat + (1 - beta2) * grad_accum.square()
            grad_accum.zero_()

    return result, {
        "step": step,
        "exp_avg_hat": exp_avg_hat,
        "exp_avg_sq_hat": exp_avg_sq_hat,
        "grad_accum": grad_accum,
    }


class TestAccumAdamW(unittest.TestCase):
    def test_default_hyperparameters(self) -> None:
        param = nn.Parameter(torch.tensor([1.0]))
        optimizer = AccumAdamW([param])
        group = optimizer.param_groups[0]

        self.assertEqual(group["lr"], 1e-3)
        self.assertEqual(group["betas"], (0.9, 0.999))
        self.assertEqual(group["eps"], 1e-8)
        self.assertEqual(group["weight_decay"], 1e-1)
        self.assertEqual(group["accumulation_steps"], 4)
        self.assertTrue(group["foreach"])

    def test_for_loop_matches_reference_recurrence(self) -> None:
        param = nn.Parameter(torch.tensor([1.0, -2.0]))
        grads = [
            torch.tensor([0.2, -0.4]),
            torch.tensor([0.5, 0.1]),
            torch.tensor([-0.3, 0.7]),
        ]
        kwargs = {
            "lr": 0.01,
            "betas": (0.8, 0.9),
            "eps": 1e-6,
            "weight_decay": 0.05,
            "accumulation_steps": 2,
        }
        optimizer = AccumAdamW([param], foreach=False, **kwargs)

        for grad in grads:
            param.grad = grad.clone()
            optimizer.step()

        expected_param, expected_state = _reference_accum_adamw(
            torch.tensor([1.0, -2.0]),
            grads,
            **kwargs,
        )
        state = optimizer.state[param]

        self.assertTrue(torch.allclose(param, expected_param))
        for name, expected in expected_state.items():
            self.assertTrue(
                torch.allclose(state[name], expected),
                f"state mismatch for {name}",
            )

    def test_accumulation_boundary_updates_both_moments_and_clears_grad_accum(
        self,
    ) -> None:
        param = nn.Parameter(torch.tensor([1.0, 2.0]))
        optimizer = AccumAdamW(
            [param],
            lr=0.1,
            betas=(0.5, 0.75),
            eps=1e-8,
            weight_decay=0.0,
            accumulation_steps=2,
            foreach=False,
        )
        grads = [torch.tensor([0.2, 0.4]), torch.tensor([0.6, -0.2])]

        param.grad = grads[0].clone()
        optimizer.step()
        state = optimizer.state[param]
        self.assertTrue(torch.equal(state["exp_avg_hat"], torch.zeros_like(param)))
        self.assertTrue(torch.equal(state["exp_avg_sq_hat"], torch.zeros_like(param)))
        self.assertTrue(torch.allclose(state["grad_accum"], grads[0] / 2))

        param.grad = grads[1].clone()
        optimizer.step()

        grad_accum = (grads[0] + grads[1]) / 2
        expected_exp_avg_hat = (1 - 0.5) * grad_accum
        expected_exp_avg_sq_hat = (1 - 0.75) * grad_accum.square()
        self.assertTrue(torch.allclose(state["exp_avg_hat"], expected_exp_avg_hat))
        self.assertTrue(
            torch.allclose(state["exp_avg_sq_hat"], expected_exp_avg_sq_hat)
        )
        self.assertTrue(torch.equal(state["grad_accum"], torch.zeros_like(param)))

    def test_foreach_matches_for_loop(self) -> None:
        params_for_loop = [
            nn.Parameter(torch.tensor([1.0, -1.0])),
            nn.Parameter(torch.tensor([0.5, 2.0])),
        ]
        params_foreach = [nn.Parameter(p.detach().clone()) for p in params_for_loop]
        opt_for_loop = AccumAdamW(
            params_for_loop,
            lr=0.02,
            betas=(0.7, 0.9),
            eps=1e-6,
            weight_decay=0.01,
            accumulation_steps=3,
            foreach=False,
        )
        opt_foreach = AccumAdamW(
            params_foreach,
            lr=0.02,
            betas=(0.7, 0.9),
            eps=1e-6,
            weight_decay=0.01,
            accumulation_steps=3,
            foreach=True,
        )

        grads_by_step = [
            [torch.tensor([0.1, 0.3]), torch.tensor([-0.2, 0.4])],
            [torch.tensor([0.5, -0.1]), torch.tensor([0.6, 0.2])],
            [torch.tensor([-0.3, 0.8]), torch.tensor([0.1, -0.7])],
            [torch.tensor([0.2, 0.2]), torch.tensor([-0.4, 0.5])],
        ]
        for grads in grads_by_step:
            for param, grad in zip(params_for_loop, grads, strict=True):
                param.grad = grad.clone()
            for param, grad in zip(params_foreach, grads, strict=True):
                param.grad = grad.clone()
            opt_for_loop.step()
            opt_foreach.step()

        for expected, actual in zip(params_for_loop, params_foreach, strict=True):
            self.assertTrue(torch.allclose(actual, expected))
            expected_state = opt_for_loop.state[expected]
            actual_state = opt_foreach.state[actual]
            for name in ("step", "exp_avg_hat", "exp_avg_sq_hat", "grad_accum"):
                self.assertTrue(
                    torch.allclose(actual_state[name], expected_state[name]),
                    f"state mismatch for {name}",
                )

    def test_accumulation_steps_one_matches_adamw(self) -> None:
        param_accum = nn.Parameter(torch.tensor([1.0, -2.0]))
        param_adamw = nn.Parameter(param_accum.detach().clone())
        kwargs = {
            "lr": 0.01,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.1,
        }
        accum_adamw = AccumAdamW(
            [param_accum],
            accumulation_steps=1,
            foreach=False,
            **kwargs,
        )
        adamw = torch.optim.AdamW(
            [param_adamw],
            foreach=False,
            fused=False,
            **kwargs,
        )
        grads = [
            torch.tensor([0.2, -0.5]),
            torch.tensor([-0.1, 0.3]),
            torch.tensor([0.4, 0.7]),
        ]

        for grad in grads:
            param_accum.grad = grad.clone()
            param_adamw.grad = grad.clone()
            accum_adamw.step()
            adamw.step()

        self.assertTrue(torch.allclose(param_accum, param_adamw))

    def test_container_builds_accum_adamw_and_rejects_fused(self) -> None:
        model = _TinyModel()
        config = OptimizersContainer.Config(
            implementation="foreach",
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*bias$",
                    optimizer_name="AccumAdamW",
                    optimizer_kwargs={
                        "lr": 1e-3,
                        "weight_decay": 0.0,
                        "accumulation_steps": 2,
                    },
                ),
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name="AccumAdamW",
                    optimizer_kwargs={
                        "lr": 1e-3,
                        "weight_decay": 0.1,
                        "accumulation_steps": 2,
                    },
                ),
            ],
        )
        container = config.build(model_parts=[model])

        self.assertEqual(len(container.optimizers), 1)
        self.assertIsInstance(container.optimizers[0], AccumAdamW)
        self.assertEqual(len(container.optimizers[0].param_groups), 2)
        self.assertTrue(container.optimizers[0].param_groups[0]["foreach"])

        fused_config = OptimizersContainer.Config(
            implementation="fused",
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name="AccumAdamW",
                    optimizer_kwargs={"lr": 1e-3, "accumulation_steps": 2},
                )
            ],
        )
        with self.assertRaisesRegex(ValueError, "does not support fused"):
            fused_config.build(model_parts=[_TinyModel()])

    def test_container_state_dict_roundtrip(self) -> None:
        config = OptimizersContainer.Config(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name="AccumAdamW",
                    optimizer_kwargs={
                        "lr": 1e-3,
                        "betas": (0.8, 0.9),
                        "eps": 1e-8,
                        "weight_decay": 0.1,
                        "accumulation_steps": 2,
                    },
                )
            ],
        )
        model = _TinyModel()
        container = config.build(model_parts=[model])
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        container.step()

        state_dict = container.state_dict()
        expected_state_suffixes = {
            "step",
            "exp_avg_hat",
            "exp_avg_sq_hat",
            "grad_accum",
        }
        state_suffixes = {
            key.rsplit(".", 1)[-1] for key in state_dict if key.startswith("state.")
        }
        self.assertEqual(state_suffixes, expected_state_suffixes)

        model2 = _TinyModel()
        container2 = config.build(model_parts=[model2])
        container2.load_state_dict(state_dict)
        state_dict2 = container2.state_dict()

        self.assertEqual(set(state_dict), set(state_dict2))
        for key, value in state_dict.items():
            loaded = state_dict2[key]
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(value, loaded), f"mismatch for {key}")
            else:
                self.assertEqual(value, loaded, f"mismatch for {key}")


if __name__ == "__main__":
    unittest.main()
