# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config.configs import DecentralizedConfig, ParallelismConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.decentralized import DecentralizedManager


class _SingleParameterModel(nn.Module):
    def __init__(self, value: float, device: str) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor([value], device=device, dtype=torch.float32)
        )


class TestDecentralizedManager(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def _parallel_dims(self) -> ParallelDims:
        return ParallelDims(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=4,
            decent_dp=2,
        )

    @with_comms
    def test_validation_average_parameters_reports_mean_l2_distance(self) -> None:
        rank = dist.get_rank()
        original_values = [0.0, 0.0, 3.0, 4.0]
        averaged_values = [1.5, 2.0, 1.5, 2.0]
        model = _SingleParameterModel(original_values[rank], self.device_type)

        manager = DecentralizedManager(
            DecentralizedConfig(enable=True, bucket_size_mb=1),
            parallel_dims=self._parallel_dims(),
            parallelism=ParallelismConfig(decent_dp_degree=2),
        )

        with manager.validation_average_parameters([model]) as extra_metrics:
            assert extra_metrics is not None
            self.assertEqual(
                set(extra_metrics),
                {
                    "validation_metrics/decent/mean_l2_distance_to_global_average",
                },
            )
            self.assertAlmostEqual(
                extra_metrics[
                    "validation_metrics/decent/mean_l2_distance_to_global_average"
                ],
                2.5,
            )
            self.assertEqual(float(model.weight.item()), averaged_values[rank])

        self.assertEqual(float(model.weight.item()), original_values[rank])
        manager.close()

    @with_comms
    def test_validation_average_parameters_omits_metric_when_disabled(self) -> None:
        rank = dist.get_rank()
        model = _SingleParameterModel(float(rank), self.device_type)
        manager = DecentralizedManager(
            DecentralizedConfig(enable=False),
            parallel_dims=self._parallel_dims(),
            parallelism=ParallelismConfig(decent_dp_degree=2),
        )

        with manager.validation_average_parameters([model]) as extra_metrics:
            self.assertIsNone(extra_metrics)
            self.assertEqual(float(model.weight.item()), float(rank))

        self.assertEqual(float(model.weight.item()), float(rank))


if __name__ == "__main__":
    unittest.main()
