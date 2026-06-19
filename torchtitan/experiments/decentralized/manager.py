# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor

from torchtitan.config.configs import DecentralizedConfig, ParallelismConfig
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger


@dataclass(frozen=True)
class _CommBucket:
    buffer: torch.Tensor
    views: list[torch.Tensor]


class DecentralizedManager:
    """Manage experimental decentralized model-parameter mixing."""

    def __init__(
        self,
        config: DecentralizedConfig,
        *,
        parallel_dims: ParallelDims,
        parallelism: ParallelismConfig,
    ) -> None:
        self.config = config
        self.parallel_dims = parallel_dims
        self.enabled = config.enable
        self._pending_works: list[Any] = []
        self._comm_buckets: list[_CommBucket] = []
        self._comm_buffer_views: list[torch.Tensor] = []
        self._pair_groups: dict[int, dist.ProcessGroup] = {}
        self._decent_group: dist.ProcessGroup | None = None

        if not self.enabled:
            if parallel_dims.decent_dp_enabled:
                logger.warning(
                    "parallelism.decent_dp_degree > 1 creates the decentralized "
                    "mesh axis, but decentralized communication is disabled. "
                    "Set decent.enable=true to run decentralized model mixing."
                )
            return

        self._validate_config(parallelism)
        self._create_pair_groups()
        logger.info(
            "Enabled decentralized model mixing with one-peer ring topology "
            f"(decent_dp={parallel_dims.decent_dp})"
        )

    def _validate_config(self, parallelism: ParallelismConfig) -> None:
        if self.config.algorithm != "model_mixing":
            raise ValueError(
                "decent.algorithm only supports 'model_mixing' in this "
                "experimental path."
            )
        if self.config.topology != "one_peer_ring":
            raise ValueError(
                "decent.topology only supports 'one_peer_ring' in this "
                "experimental path."
            )
        if self.config.bucket_size_mb <= 0:
            raise ValueError("decent.bucket_size_mb must be positive.")
        if not self.config.overlap:
            raise ValueError("decent.overlap=False is not implemented.")
        if not self.parallel_dims.decent_dp_enabled:
            raise ValueError("decent.enable requires parallelism.decent_dp_degree > 1.")
        if self.parallel_dims.decent_dp % 2 != 0:
            raise ValueError("decent_dp_degree must be even for one-peer ring mixing.")
        if parallelism.spmd_backend != "default":
            raise ValueError(
                "decentralized training currently supports only default SPMD backend."
            )
        if self.parallel_dims.dp_replicate_enabled:
            raise ValueError(
                "decentralized training does not support "
                "data_parallel_replicate_degree > 1 yet."
            )
        if self.parallel_dims.pp_enabled:
            raise ValueError(
                "decentralized training does not support pipeline parallelism yet."
            )
        if self.parallel_dims.ep_enabled:
            raise ValueError(
                "decentralized training does not support expert parallelism yet."
            )

    def _create_pair_groups(self) -> None:
        decent_dp = self.parallel_dims.decent_dp
        inner_size = self.parallel_dims.dp_replicate * self.parallel_dims.dp_shard
        inner_size *= self.parallel_dims.cp * self.parallel_dims.tp
        world_rank = dist.get_rank()

        for pp_idx in range(self.parallel_dims.pp):
            pp_base = pp_idx * decent_dp * inner_size
            for inner_idx in range(inner_size):
                peers = [
                    pp_base + decent_rank * inner_size + inner_idx
                    for decent_rank in range(decent_dp)
                ]
                decent_group = dist.new_group(ranks=peers)
                if world_rank in peers:
                    self._decent_group = decent_group
                for round_idx in (0, 1):
                    for start in range(round_idx, decent_dp, 2):
                        ranks = [peers[start], peers[(start + 1) % decent_dp]]
                        group = dist.new_group(ranks=ranks)
                        if world_rank in ranks:
                            self._pair_groups[round_idx] = group

        assert set(self._pair_groups) == {0, 1}, (
            "Current rank must belong to one decentralized pair group per round."
        )
        assert self._decent_group is not None, (
            "Current rank must belong to one decentralized global group."
        )

    @staticmethod
    def _local_tensor(param: torch.Tensor) -> torch.Tensor:
        value = param.detach()
        if isinstance(value, DTensor):
            return value.to_local()
        return value

    @classmethod
    def _local_tensors(cls, params: list[torch.Tensor]) -> list[torch.Tensor]:
        return [cls._local_tensor(param) for param in params]

    @staticmethod
    def _foreach_copy_tensors_(
        dst_tensors: list[torch.Tensor],
        src_tensors: list[torch.Tensor],
    ) -> None:
        groups: dict[
            tuple[torch.device, torch.dtype, torch.device, torch.dtype],
            tuple[list[torch.Tensor], list[torch.Tensor]],
        ] = {}
        for dst_tensor, src_tensor in zip(dst_tensors, src_tensors, strict=True):
            key = (
                dst_tensor.device,
                dst_tensor.dtype,
                src_tensor.device,
                src_tensor.dtype,
            )
            group_dst_tensors, group_src_tensors = groups.setdefault(key, ([], []))
            group_dst_tensors.append(dst_tensor)
            group_src_tensors.append(src_tensor)

        for group_dst_tensors, group_src_tensors in groups.values():
            torch._foreach_copy_(group_dst_tensors, group_src_tensors)

    @classmethod
    def _foreach_copy_local_tensors_(
        cls,
        dst_params: list[torch.Tensor],
        src_tensors: list[torch.Tensor],
    ) -> None:
        cls._foreach_copy_tensors_(cls._local_tensors(dst_params), src_tensors)

    @staticmethod
    def _foreach_mul_tensors_(tensors: list[torch.Tensor], value: float) -> None:
        groups: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
        for tensor in tensors:
            groups.setdefault((tensor.device, tensor.dtype), []).append(tensor)

        for group_tensors in groups.values():
            torch._foreach_mul_(group_tensors, value)

    @staticmethod
    def _trainable_parameters(model_parts: list[nn.Module]) -> list[torch.Tensor]:
        return [
            param
            for model_part in model_parts
            for param in model_part.parameters()
            if param.requires_grad
        ]

    @staticmethod
    def _parameters(model_parts: list[nn.Module]) -> list[torch.Tensor]:
        return [
            param
            for model_part in model_parts
            for param in model_part.parameters()
        ]

    def _build_buckets_from_tensors(
        self,
        tensors: list[torch.Tensor],
    ) -> tuple[list[_CommBucket], list[torch.Tensor]]:
        bucket_size_bytes = self.config.bucket_size_mb * 1024 * 1024
        buckets: list[_CommBucket] = []
        bucket_views: list[torch.Tensor] = []

        bucket_tensors: list[torch.Tensor] = []
        bucket_numel = 0
        bucket_nbytes = 0
        bucket_device: torch.device | None = None
        bucket_dtype: torch.dtype | None = None

        def flush_bucket() -> None:
            nonlocal bucket_tensors, bucket_numel, bucket_nbytes
            nonlocal bucket_device, bucket_dtype

            if not bucket_tensors:
                return

            assert bucket_device is not None
            assert bucket_dtype is not None
            buffer = torch.empty(
                bucket_numel,
                device=bucket_device,
                dtype=bucket_dtype,
            )
            views = []
            offset = 0
            for tensor in bucket_tensors:
                view = buffer.narrow(0, offset, tensor.numel()).view_as(tensor)
                views.append(view)
                offset += tensor.numel()

            buckets.append(_CommBucket(buffer=buffer, views=views))
            bucket_views.extend(views)

            bucket_tensors = []
            bucket_numel = 0
            bucket_nbytes = 0
            bucket_device = None
            bucket_dtype = None

        for tensor in tensors:
            tensor_nbytes = tensor.numel() * tensor.element_size()
            starts_new_bucket = (
                bucket_tensors
                and (
                    tensor.device != bucket_device
                    or tensor.dtype != bucket_dtype
                    or bucket_nbytes + tensor_nbytes > bucket_size_bytes
                )
            )
            if starts_new_bucket:
                flush_bucket()

            bucket_tensors.append(tensor)
            bucket_numel += tensor.numel()
            bucket_nbytes += tensor_nbytes
            bucket_device = tensor.device
            bucket_dtype = tensor.dtype

        flush_bucket()
        return buckets, bucket_views

    def _build_comm_buckets(self, params: list[torch.Tensor]) -> None:
        local_tensors = self._local_tensors(params)
        self._comm_buckets, self._comm_buffer_views = self._build_buckets_from_tensors(
            local_tensors
        )
        logger.info(
            "Created %s decentralized communication buckets for %s trainable "
            "parameter tensors with bucket_size_mb=%s",
            len(self._comm_buckets),
            len(params),
            self.config.bucket_size_mb,
        )

    def bootstrap(self, model_parts: list[nn.Module], *, next_step: int) -> None:
        if not self.enabled:
            return
        if self._pending_works:
            raise RuntimeError("Decentralized communication is already pending.")
        with torch.profiler.record_function("decent.bootstrap"):
            self._launch(model_parts, step=next_step)

    @torch.no_grad()
    def before_optimizer_step(self, model_parts: list[nn.Module], *, step: int) -> None:
        if not self.enabled:
            return
        with torch.profiler.record_function("decent.before_optimizer_step"):
            if not self._pending_works:
                # This can happen in load-only or hand-written trainer tests. Fall back
                # to a synchronous launch/apply for the current step.
                self._launch(model_parts, step=step)

            with torch.profiler.record_function("decent.wait_model_mix"):
                for work in self._pending_works:
                    work.wait()

            params = self._trainable_parameters(model_parts)
            if len(params) != len(self._comm_buffer_views):
                raise RuntimeError(
                    "Trainable parameter set changed after decentralized buffers "
                    "were created."
                )

            with torch.profiler.record_function("decent.apply_model_mix"):
                self._foreach_mul_tensors_(
                    [bucket.buffer for bucket in self._comm_buckets],
                    0.5,
                )
                self._foreach_copy_local_tensors_(params, self._comm_buffer_views)

            self._pending_works = []

    @torch.no_grad()
    def after_optimizer_step(
        self,
        model_parts: list[nn.Module],
        *,
        next_step: int,
    ) -> None:
        if not self.enabled:
            return
        if self._pending_works:
            raise RuntimeError("Previous decentralized communication was not consumed.")
        with torch.profiler.record_function("decent.after_optimizer_step"):
            self._launch(model_parts, step=next_step)

    @contextmanager
    def validation_average_parameters(
        self,
        model_parts: list[nn.Module],
    ) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        assert self._decent_group is not None

        with torch.profiler.record_function("decent.validation_average_parameters"):
            with torch.no_grad():
                with torch.profiler.record_function("decent.validation_wait_model_mix"):
                    for work in self._pending_works:
                        work.wait()

                params = self._parameters(model_parts)
                backups = [
                    self._local_tensor(param).clone(memory_format=torch.preserve_format)
                    for param in params
                ]

            try:
                with torch.no_grad():
                    with torch.profiler.record_function(
                        "decent.validation_global_average"
                    ):
                        local_tensors = self._local_tensors(params)
                        validation_buckets, validation_bucket_views = (
                            self._build_buckets_from_tensors(local_tensors)
                        )
                        self._foreach_copy_tensors_(
                            validation_bucket_views,
                            local_tensors,
                        )
                        for bucket in validation_buckets:
                            dist.all_reduce(bucket.buffer, group=self._decent_group)
                        self._foreach_mul_tensors_(
                            [bucket.buffer for bucket in validation_buckets],
                            1.0 / self.parallel_dims.decent_dp,
                        )
                        self._foreach_copy_local_tensors_(
                            params,
                            validation_bucket_views,
                        )
                yield
            finally:
                with torch.no_grad():
                    with torch.profiler.record_function(
                        "decent.validation_restore_parameters"
                    ):
                        self._foreach_copy_local_tensors_(params, backups)

    @torch.no_grad()
    def _launch(self, model_parts: list[nn.Module], *, step: int) -> None:
        params = self._trainable_parameters(model_parts)
        if not self._comm_buckets:
            self._build_comm_buckets(params)
        elif len(params) != len(self._comm_buffer_views):
            raise RuntimeError(
                "Trainable parameter set changed after decentralized buffers "
                "were created."
            )

        group = self._pair_groups[step % 2]
        self._pending_works = []
        with torch.profiler.record_function("decent.copy_params_to_comm_buffers"):
            self._foreach_copy_tensors_(
                self._comm_buffer_views,
                self._local_tensors(params),
            )
        with torch.profiler.record_function("decent.launch_model_all_reduce"):
            for bucket in self._comm_buckets:
                self._pending_works.append(
                    dist.all_reduce(bucket.buffer, group=group, async_op=True)
                )

    def close(self) -> None:
        with torch.profiler.record_function("decent.close"):
            for work in self._pending_works:
                work.wait()
        self._pending_works = []
