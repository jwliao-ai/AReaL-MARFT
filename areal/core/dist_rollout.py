from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.engine_api import InferenceEngine, TrainEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.platforms import current_platform
from areal.utils.data import (
    all_gather_tensor_container,
    broadcast_tensor_container,
    concat_padded_tensors,
    get_batch_size,
    tensor_container_to,
    concat_padded_tensors_per_key,
)
from areal.utils.datapack import ffd_allocate


@dataclass
class RedistributedData:
    all_data: list[dict[str, Any]]
    data: dict[str, Any]
    rank: int
    group_indices: list[list[int]]


def _slice_tensor_dict(data: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    """
    Slices tensors AND lists in a dictionary along the first dimension.
    MODIFIED: Uses a multi-agent aware get_batch_size and also slices lists
    (like multi_modal_input) that are batch-aligned.
    """
    sliced_data = {}
    
    batch_size = get_batch_size(data)

    if batch_size == 0:
        return data

    for key, value in data.items():
        if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == batch_size:
            sliced_data[key] = value[start:end]
            
        elif isinstance(value, list) and len(value) == batch_size:
            sliced_data[key] = value[start:end]
            
        else:
            sliced_data[key] = value
            
    return sliced_data


def redistribute(
    data: dict[str, Any], granularity: int = 1, group=None
) -> RedistributedData:
    """Redistribute a batch across a process group.

    This function only accepts padded data which must have an "attention_mask" field,
    Each tensor should have shape [bs, seqlen, *] or [bs].

    This function will divide the global batch into segments each with consecutive
    `granularity` sequences, and then redistribute the segments (e.g., for GRPO).
    """
    all_gathered = all_gather_tensor_container(data, group=group)

    all_data = []
    for d in all_gathered:
        bs = get_batch_size(d)
        assert bs % granularity == 0
        all_data += [
            _slice_tensor_dict(d, i, i + granularity) for i in range(0, bs, granularity)
        ]

    seqlens = [d["attention_mask"].sum().item() for d in all_data]

    # Remove pad positions
    for d in all_data:
        max_sequence_length = d["attention_mask"].sum(-1).max().item()
        attn_mask_shape = d["attention_mask"].shape
        for k, v in d.items():
            if (
                torch.is_tensor(v)
                and len(v.shape) >= 2
                and v.shape[:2] == attn_mask_shape[:2]
            ):
                d[k] = v[:, :max_sequence_length]

    # No capacity limit leads to balanced partition across this group
    group_indices = ffd_allocate(
        seqlens, capacity=int(1e12), min_groups=dist.get_world_size(group)
    )
    local_indices = group_indices[dist.get_rank(group=group)]

    data = concat_padded_tensors([all_data[i] for i in local_indices])

    return RedistributedData(
        all_data=all_data,
        data=data,
        rank=dist.get_rank(group=group),
        group_indices=group_indices,
    )

def ma_redistribute(
    data: dict[str, Any], granularity: int = 1, group=None
) -> RedistributedData:
    """
    Redistribute a batch across a process group.
    MODIFIED: Accepts multi-agent data (e.g., 'agent0_attention_mask')
    and relies on a per-key padding-aware 'concat_padded_tensors'.

    This function accepts padded data which must have one or more
    '*_attention_mask' fields.
    Each tensor should have shape [bs, seqlen, *] or [bs].

    This function will divide the global batch into segments each with consecutive
    `granularity` sequences, and then redistribute the segments (e.g., for GRPO).
    """
    all_gathered = all_gather_tensor_container(data, group=group) # a list of data-like

    all_data = []
    for d in all_gathered:
        bs = get_batch_size(d)
        assert bs % granularity == 0
        all_data += [
            _slice_tensor_dict(d, i, i + granularity) for i in range(0, bs, granularity) # _slice_tensor_dict function modified to adapt to multi-agent settings
        ]

    seqlens = []
    for d in all_data:
        current_seqlen_sum = 0
        for k, v in d.items():
            if k.endswith("attention_mask") and torch.is_tensor(v):
                current_seqlen_sum += v.sum().item()
        seqlens.append(current_seqlen_sum)

    for d in all_data:
        mask_properties = {}
        for k, v in d.items():
            if k.endswith("_attention_mask") and torch.is_tensor(v):
                max_len = v.sum(-1).max().item()
                shape = v.shape
                
                prefix = k[:-len("attention_mask")] 
                
                mask_properties[prefix] = {
                    "max_len": max_len,
                    "shape": shape
                }

        sorted_prefixes = sorted(mask_properties.keys(), key=len, reverse=True)

        for k, v in d.items():
            if not torch.is_tensor(v) or len(v.shape) < 2:
                continue

            matching_prefix = None
            for prefix in sorted_prefixes:
                if k.startswith(prefix):
                    matching_prefix = prefix
                    break
            
            if matching_prefix is not None:
                props = mask_properties[matching_prefix]
                
                if v.shape[:2] == props["shape"][:2]:
                    d[k] = v[:, :props["max_len"]]

    # No capacity limit leads to balanced partition across this group
    group_indices = ffd_allocate(
        seqlens, capacity=int(1e12), min_groups=dist.get_world_size(group)
    )
    local_indices = group_indices[dist.get_rank(group=group)]

    data = concat_padded_tensors_per_key([all_data[i] for i in local_indices])

    return RedistributedData(
        all_data=all_data,
        data=data,
        rank=dist.get_rank(group=group),
        group_indices=group_indices,
    )
    
def redistribute_multi_agent(
    data: dict[str, Any],
    num_agents: int,
    granularity: int = 1,
    group=None,
) -> RedistributedData:
    """Redistribute a multi-agent batch across a process group.

    This function handles multi-agent data where keys are prefixed with "agent{idx}_".
    Each agent's data must have an "agent{idx}_attention_mask" field.
    Each tensor should have shape [bs, seqlen, *] or [bs].

    This function will divide the global batch into segments each with consecutive
    `granularity` sequences, and then redistribute the segments.

    Parameters
    ----------
    data : Dict[str, Any]
        Multi-agent data with keys prefixed by "agent{idx}_"
    num_agents : int
        Number of agents in the data
    granularity : int, default=1
        Number of consecutive sequences per segment
    group : optional
        Process group for redistribution

    Returns
    -------
    RedistributedData
        Redistributed multi-agent data with all_data, data, rank, and group_indices
    """
    all_gathered = all_gather_tensor_container(data, group=group)

    all_data = []
    for d in all_gathered:
        # Validate batch size consistency across agents
        batch_sizes = set()
        for agent_idx in range(num_agents):
            attn_mask_key = f"agent{agent_idx}_attention_mask"
            if attn_mask_key in d:
                batch_sizes.add(d[attn_mask_key].shape[0])
        
        assert len(batch_sizes) == 1, (
            f"Inconsistent batch sizes across agents: {batch_sizes}"
        )
        bs = batch_sizes.pop()
        assert bs % granularity == 0, (
            f"Batch size {bs} not divisible by granularity {granularity}"
        )

        # Slice data for each segment
        all_data += [
            _slice_tensor_dict(d, i, i + granularity) for i in range(0, bs, granularity)
        ]

    # Calculate sequence lengths (sum across all agents)
    seqlens = []
    for d in all_data:
        total_seqlen = 0
        for agent_idx in range(num_agents):
            attn_mask_key = f"agent{agent_idx}_attention_mask"
            if attn_mask_key in d:
                total_seqlen += d[attn_mask_key].sum().item()
        seqlens.append(total_seqlen)

    # Remove pad positions for each agent
    for d in all_data:
        for agent_idx in range(num_agents):
            attn_mask_key = f"agent{agent_idx}_attention_mask"
            if attn_mask_key not in d:
                continue

            max_sequence_length = d[attn_mask_key].sum(-1).max().item()
            attn_mask_shape = d[attn_mask_key].shape
            prefix = f"agent{agent_idx}_"

            # Trim all tensors with matching prefix and shape
            for k, v in list(d.items()):
                if (
                    k.startswith(prefix)
                    and torch.is_tensor(v)
                    and len(v.shape) >= 2
                    and v.shape[:2] == attn_mask_shape[:2]
                ):
                    d[k] = v[:, :max_sequence_length]

    # Balanced partition across this group
    group_indices = ffd_allocate(
        seqlens, capacity=int(1e12), min_groups=dist.get_world_size(group)
    )
    local_indices = group_indices[dist.get_rank(group=group)]

    data = concat_padded_tensors([all_data[i] for i in local_indices])
    return RedistributedData(
        all_data=all_data,
        data=data,
        rank=dist.get_rank(group=group),
        group_indices=group_indices,
    )

class DistRolloutCoordinator:
    def __init__(self, rollout_engine: InferenceEngine, train_engine: TrainEngine):
        self.rollout_engine = rollout_engine
        self.train_engine = train_engine

    def _broadcast_and_redistribute_batch(
        self,
        batch: dict[str, Any] | None,
        granularity: int = 1,
    ) -> dict[str, Any]:
        """Broadcast and redistribute batch across distributed workers.

        This helper encapsulates:
        1. Redistribution within data parallel group (for load balancing)
        2. Broadcasting to context and model parallel group
        3. Synchronization barriers

        Parameters
        ----------
        batch : Dict[str, Any] | None
            Batch data from data parallel head, None for other ranks
        granularity : int, default=1
            Granularity for redistribution within data parallel group.
            - For single-turn rollouts: Use actor.config.group_size (GRPO grouping)
            - For multi-turn rollouts: Use 1 (default, per-completion redistribution)
            - For custom scenarios: Use custom value (e.g., n_trajs for agent trajectories)

        Returns
        -------
        Dict[str, Any]
            Redistributed and broadcast batch available on all ranks
        """
        if batch is not None:
            redist = redistribute(
                batch,
                granularity=granularity,
                group=self.train_engine.data_parallel_group,
            )
            batch = redist.data

        dist.barrier(device_ids=[current_platform.current_device()])
        current_platform.synchronize()

        batch = broadcast_tensor_container(
            batch,
            src_rank=self.train_engine.current_data_parallel_head(),
            group=self.train_engine.context_and_model_parallel_group,
        )

        dist.barrier(device_ids=[current_platform.current_device()])
        current_platform.synchronize()

        return batch

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
        granularity: int = 1,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
    ) -> dict[str, Any]:
        """Generate rollout batch with distributed coordination (synchronous).

        This method orchestrates distributed rollout generation:
        - Only data parallel heads generate rollouts (avoid redundancy)
        - Results are transferred to device and redistributed
        - Batch is broadcast to all workers
        - Synchronization barriers ensure consistency

        Must call connect_engine() before using this method.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            Input data batch for rollout generation
        granularity : int, default=1
            Granularity for redistribution within data parallel group.
            - For single-turn rollouts: Set to actor.config.group_size (GRPO grouping)
            - For multi-turn rollouts: Use default value of 1 (per-completion redistribution)
            - For custom scenarios: Use custom value (e.g., n_trajs for agent trajectories)
        workflow : RolloutWorkflow | type[RolloutWorkflow] | str
            Workflow defining rollout logic
        workflow_kwargs : Dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor
        should_accept_fn : Callable[[Dict[str, Any]], bool] | str, optional
            Filter function for accepting samples

        Returns
        -------
        Dict[str, Any]
            Generated rollout batch on all ranks

        Raises
        ------
        RuntimeError
            If rollout engine not connected via connect_engine()
        """

        batch = None
        if self.train_engine.is_data_parallel_head():
            batch = self.rollout_engine.rollout_batch(
                data,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                should_accept_fn=should_accept_fn,
            )
            batch = tensor_container_to(batch, current_platform.current_device())

        return self._broadcast_and_redistribute_batch(batch, granularity=granularity)

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
        granularity: int = 1,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
    ) -> dict[str, Any]:
        """Prepare async rollout batch with distributed coordination.

        Similar to rollout_batch but uses prepare_batch for async training,
        where rollout generation happens concurrently with training.

        Must call connect_engine() before using this method.

        Parameters
        ----------
        dataloader : StatefulDataLoader
            Dataloader to pull samples from
        granularity : int, default=1
            Granularity for redistribution within data parallel group.
            - For single-turn rollouts: Set to actor.config.group_size (GRPO grouping)
            - For multi-turn rollouts: Use default value of 1 (per-completion redistribution)
            - For custom scenarios: Use custom value (e.g., n_trajs for agent trajectories)
        workflow : RolloutWorkflow | type[RolloutWorkflow] | str
            Workflow defining rollout logic
        workflow_kwargs : Dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor
        should_accept_fn : Callable[[Dict[str, Any]], bool] | str, optional
            Filter function for accepting samples based on staleness

        Returns
        -------
        Dict[str, Any]
            Prepared rollout batch on all ranks

        Raises
        ------
        RuntimeError
            If rollout engine not connected via connect_engine()
        """

        batch = None
        if self.train_engine.is_data_parallel_head():
            batch = self.rollout_engine.prepare_batch(
                dataloader,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                should_accept_fn=should_accept_fn,
            )
            batch = tensor_container_to(batch, current_platform.current_device())

        return self._broadcast_and_redistribute_batch(batch, granularity=granularity)
