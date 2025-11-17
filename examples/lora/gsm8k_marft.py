"""
Multi-agent PPO with joint rollout and coordinated training.
"""

import os
import sys
from copy import deepcopy
from typing import Any

import torch
import torch.distributed as dist

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import PPOConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.core.dist_rollout import ma_redistribute, redistribute_multi_agent
from areal.dataset import get_custom_dataset
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.ppo.mas_actor import FSDPMAPPOActor
from areal.engine.ppo.critic import FSDPPPOCritic
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.platforms import current_platform
from areal.utils import seeding, stats_tracker
from areal.utils.data import (
    broadcast_tensor_container,
    concat_padded_tensors,
    cycle_dataloader,
    get_batch_size,
    tensor_container_to,
    concat_padded_tensors_per_key,
)
from areal.utils.dataloader import create_dataloader
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.logging import getLogger
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.workflow.marlvr import MultiAgentRLVRWorkflow

logger = getLogger(__name__)


def gsm8k_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    """Individual agent reward."""
    from areal.reward.math_parser import process_results
    return int(process_results(completions, answer)[0])


def bcast_and_split_from_rank0(unified_batch: dict | None, granularity: int) -> dict:
    """Broadcast batch from rank 0 and split across ranks."""
    unified_batch = broadcast_tensor_container(unified_batch, src_rank=0)
    bs = get_batch_size(unified_batch)
    world_size = dist.get_world_size()
    
    if bs % world_size != 0:
        valid_bs = (bs // world_size) * world_size
        if valid_bs == 0:
            raise RuntimeError(
                f"Batch size {bs} < world size {world_size}. Cannot split."
            )
        unified_batch = {k: v[:valid_bs] for k, v in unified_batch.items()}
        bs = valid_bs
    
    bs_per_rank = bs // world_size
    local_batch = []
    for i in range(dist.get_rank() * bs_per_rank, (dist.get_rank() + 1) * bs_per_rank):
        local_batch.append({k: v[i : i + 1] for k, v in unified_batch.items()})
    # local_batch = concat_padded_tensors(local_batch)
    local_batch = concat_padded_tensors_per_key(local_batch)
    return ma_redistribute(local_batch, granularity=granularity).data

class MultiAgentSystem:
    """Manages multiple agents' joint rollout and training."""
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.n_agents = config.n_agents
        self.interaction_mode = config.agent_interaction_mode
        self.rank = int(os.getenv("RANK", "0"))
        
        # Shared components
        self.tokenizer = load_hf_tokenizer(config.tokenizer_path)
        self.allocation_mode = AllocationMode.from_str(config.allocation_mode)
        self.parallel_strategy = self.allocation_mode.train
        
        # Per-agent components
        self.agents = []
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all agents' actor, critic, rollout engine, workflow."""
        # ✅ Read SGLang configuration from environment
        base_port = int(os.getenv("SGLANG_BASE_PORT", "17987"))
        sglang_host = os.getenv("SGLANG_HOST", "localhost")
        
        logger.info(
            f"Initializing {self.n_agents} agents with SGLang servers at "
            f"{sglang_host}:{base_port}..{base_port + self.n_agents - 1}"
        )
        
        for agent_id in range(self.n_agents):
            agent_config = deepcopy(self.config)
            agent_config.actor.experiment_name = f"{self.config.experiment_name}_agent{agent_id}"
            agent_config.critic.experiment_name = f"{self.config.experiment_name}_agent{agent_id}"
            agent_config.rollout.experiment_name = f"{self.config.experiment_name}_agent{agent_id}"
            
            agent_config.saver.experiment_name = f"{self.config.experiment_name}_agent{agent_id}"
            agent_config.recover.experiment_name = f"{self.config.experiment_name}_agent{agent_id}"
            agent_config.evaluator.experiment_name = f"{self.config.experiment_name}_agent{agent_id}"
            
            # Actor
            actor = FSDPMAPPOActor(config=agent_config.actor)
            actor.create_process_group(parallel_strategy=self.parallel_strategy)
            
            # Critic
            critic = FSDPPPOCritic(config=agent_config.critic)
            critic.create_process_group(parallel_strategy=self.parallel_strategy)
            
            # Rollout engine - pass addr directly to initialize()
            rollout = RemoteSGLangEngine(agent_config.rollout)
            agent_addr = f"{sglang_host}:{base_port + agent_id}"
            rollout.initialize(
                addr=agent_addr,  # ← Direct address specification
                train_data_parallel_size=1,
            )
            
            # Eval rollout engine - separate instance for evaluation
            eval_rollout = RemoteSGLangEngine(deepcopy(agent_config.rollout))
            eval_rollout.config.max_head_offpolicyness = int(1e12)
            eval_rollout.initialize(addr=agent_addr)
            
            # Multi-agent workflow with context support
            workflow = MultiAgentRLVRWorkflow(
                agent_id=agent_id,
                reward_fn=gsm8k_reward_fn,
                gconfig=agent_config.gconfig,
                tokenizer=self.tokenizer,
                enable_thinking=False,
                interaction_mode=self.interaction_mode,
                dump_dir=os.path.join(
                    StatsLogger.get_log_path(agent_config.stats_logger),
                    f"agent_{agent_id}/generated"
                ),
            )
            
            # Eval workflow with lower temperature for more deterministic evaluation
            eval_workflow = MultiAgentRLVRWorkflow(
                agent_id=agent_id,
                reward_fn=gsm8k_reward_fn,
                gconfig=agent_config.gconfig.new(temperature=0.6),
                tokenizer=self.tokenizer,
                enable_thinking=False,
                interaction_mode=self.interaction_mode,
                rollout_stat_scope=f"agent{agent_id}-eval-rollout",
                dump_dir=os.path.join(
                    StatsLogger.get_log_path(agent_config.stats_logger),
                    f"agent_{agent_id}/generated-eval"
                ),
            )
            
            # Weight update meta：使用 agent-specific experiment_name
            weight_update_meta = WeightUpdateMeta.from_disk(
                agent_config.actor.experiment_name,  # agent-specific
                agent_config.actor.trial_name,
                agent_config.saver.fileroot,
                use_lora=True,
            )
            
            # Saver
            saver = Saver(agent_config.saver, None)  # ft_spec set later
            
            # Evaluator
            evaluator = Evaluator(agent_config.evaluator, None)  # ft_spec set later
            
            self.agents.append({
                'id': agent_id,
                'config': agent_config,
                'actor': actor,
                'critic': critic,
                'rollout': rollout,
                'eval_rollout': eval_rollout,
                'workflow': workflow,
                'eval_workflow': eval_workflow,
                'weight_update_meta': weight_update_meta,
                'saver': saver,
                'evaluator': evaluator,
            })
            
            logger.info(f"Initialized agent {agent_id}")
    
    def finalize_initialization(self, ft_spec: FinetuneSpec):
        """Complete initialization after obtaining ft_spec."""
        for agent in self.agents:
            agent['actor'].initialize(None, ft_spec)
            agent['actor'].connect_engine(agent['rollout'], agent['weight_update_meta'])
            agent['critic'].initialize(None, ft_spec)
            agent['saver'].ft_spec = ft_spec
            agent['evaluator'].ft_spec = ft_spec
        
    def joint_rollout(self, data_batch: list[dict]) -> list[dict]:
        """Joint rollout - only rank 0 does rollout, all ranks participate in broadcast."""
        if dist.get_rank() == 0 and data_batch is not None:
            for idx, sample in enumerate(data_batch):
                sample['__global_idx__'] = idx
        
        # because we broadcast only when all agents' rollout data are ready,
        # so we initialize the unified batch out of the loop.
        agent_batches = None
        unified_batch = {}
        
        if dist.get_rank() == 0:
            agent_batches = []
            for agent_idx, agent in enumerate(self.agents):
                with stats_tracker.scope(f"agent{agent_idx}"):
                    with stats_tracker.record_timing("rollout"):
                        batch = None
                        logger.info(f"Agent {agent_idx} rollout starting...")
                        batch = agent['rollout'].rollout_batch(
                            data_batch,
                            workflow=agent['workflow'],
                            should_accept_fn=lambda sample: True,
                        )
                        logger.info(f"✅ Agent {agent_idx} rollout done")
                        batch = tensor_container_to(batch, agent['actor'].device)
                        
                        # recover the order for action tuple collection
                        if '__global_idx__' in batch:
                            indices = batch['__global_idx__']
                            sorted_indices = torch.sort(indices)[0]
                            if not torch.equal(indices, sorted_indices):
                                sort_perm = torch.argsort(indices)
                                batch = {
                                    k: v[sort_perm] if torch.is_tensor(v) and v.dim() > 0 else v
                                    for k, v in batch.items()
                                }
                                logger.info(f"Agent {agent_idx} batch reordered from {indices[:3].tolist()} to {batch['__global_idx__'][:3].tolist()}")
                            # # we can keep the global_idx for verification
                            # del batch['__global_idx__']
                            
                        # # not working because redistribute will change the order still
                        # # so we need to get the unified data, redistribute and unpack the unified data batch
                        # batch = bcast_and_split_from_rank0(batch, granularity=self.config.actor.group_size)
                        agent_batches.append(batch) # agent0: [8,486]; agent1: [8,517]

            for agent_idx, batch in enumerate(agent_batches):
                for key, value in batch.items():
                    unified_batch[f"agent{agent_idx}_{key}"] = value
        
        unified_batch = bcast_and_split_from_rank0(unified_batch, granularity=self.config.actor.group_size)
        
        # now start to unpack
        rollout_batches = []
        for agent_idx in range(self.n_agents):
            agent_batch = {}
            prefix = f"agent{agent_idx}_"
            
            for key in list(unified_batch.keys()):
                if key.startswith(prefix):
                    original_key = key[len(prefix):]
                    agent_batch[original_key] = unified_batch[key]
            
            # # we can keep the global idx for verification
            # if '__global_idx__' in agent_batch:
            #     del agent_batch['__global_idx__']
                
            rollout_batches.append(agent_batch)
            logger.info(f"✅ Agent {agent_idx} batch unpacked with {len(agent_batch)} fields on rank {dist.get_rank()}")
        
        dist.barrier()
        return rollout_batches
    
    def compute_joint_rewards(self, batches: list[dict]):
        """
        Compute joint rewards based on all agents' trajectories.
        
        Example: Team bonus if all agents solve correctly.
        """
        if dist.get_rank() == 0:
            all_correct = all(
                batch.get('rewards', torch.zeros(1))[0].item() > 0.5
                for batch in batches
            )
            team_bonus = 0.2 if all_correct else 0.0
            
            if team_bonus > 0:
                logger.info(f"Team bonus: {team_bonus} (all agents correct)")
                for batch in batches:
                    if 'rewards' in batch:
                        batch['rewards'] = batch['rewards'] + team_bonus
        
        for batch in batches:
            batch = broadcast_tensor_container(batch, src_rank=0)
    
    def joint_training(self, batches: list[dict]):
        """
        Joint training phase: all agents update simultaneously.
        """
        # We must compute advantages in reverse order, starting from the last agent.
        successor_values = None
        for i in reversed(range(len(self.agents))):
            agent = self.agents[i]
            batch = batches[i]
            agent_id = agent['id']
            
            with stats_tracker.scope(f"agent{agent_id}"):
                with stats_tracker.record_timing(f"critic_values"):
                    values = agent['critic'].compute_values(batch)
                    batch['values'] = values
                    log_gpu_stats(f"agent{agent_id} critic values")

                if successor_values is not None:
                    batch['successor_values'] = successor_values
                
                if self.config.actor.recompute_logprob or self.config.actor.use_decoupled_loss:
                    with stats_tracker.record_timing(f"recompute_logp"):
                        logp = agent['actor'].compute_logp(batch)
                        batch['prox_logp'] = logp
                        log_gpu_stats(f"agent{agent_id} recompute logp")
                
                with stats_tracker.record_timing(f"compute_advantage"):
                    agent['actor'].compute_advantages(batch)
                    log_gpu_stats(f"agent{agent_id} compute advantages")
            
                successor_values = batch['values']
                
        # Synchronization barrier
        dist.barrier(device_ids=[self.agents[0]['actor'].device.index])
        current_platform.synchronize()
        
        # PPO updates for all agents
        for agent, batch in zip(self.agents, batches):
            agent_id = agent['id']
            
            # ✅ Use scope to prefix all stats from actor/critic
            with stats_tracker.scope(f"agent{agent_id}"):
                with stats_tracker.record_timing(f"ppo_update"):
                    agent['actor'].ppo_update(batch)
                    agent['actor'].step_lr_scheduler()
                    log_gpu_stats(f"agent{agent_id} ppo actor update")
                
                with stats_tracker.record_timing(f"critic_update"):
                    agent['critic'].ppo_update(batch)
                    agent['critic'].step_lr_scheduler()
                    log_gpu_stats(f"agent{agent_id} ppo critic update")
    
    def update_weights(self, global_step: int):
        """Synchronize all agents' weight versions."""
        for agent in self.agents:
            agent['rollout'].pause()
        
        for agent in self.agents:
            agent_id = agent['id']
            with stats_tracker.scope(f"agent{agent_id}"):
                with stats_tracker.record_timing(f"update_weights"):
                    try:
                        agent['actor'].update_weights(agent['weight_update_meta'])
                        agent['actor'].set_version(global_step + 1)
                        agent['critic'].set_version(global_step + 1)
                        agent['rollout'].set_version(global_step + 1)
                        agent['eval_rollout'].set_version(global_step + 1)
                    except Exception as e:
                        logger.error(
                            f"Agent {agent['id']} failed to update weights: {e}",
                            exc_info=True
                        )
                        raise
        
        for agent in self.agents:
            agent['rollout'].resume()
            
    def save_checkpoints(self, epoch: int, step: int, global_step: int):
        """Save all agents' checkpoints."""
        for agent in self.agents:
            agent_id = agent['id']
            with stats_tracker.scope(f"agent{agent_id}"):
                with stats_tracker.record_timing(f"save"):
                    agent['saver'].save(
                        agent['actor'], epoch, step, global_step,
                        tokenizer=self.tokenizer,
                        name=f"actor_agent{agent['id']}"
                    )
                    agent['saver'].save(
                        agent['critic'], epoch, step, global_step,
                        tokenizer=self.tokenizer,
                        name=f"critic_agent{agent['id']}"
                    )
    
    def evaluate(self, valid_dataloader, epoch: int, step: int, global_step: int, force_run: bool = False):
        """
        Run joint evaluation for all agents with sequential decision-making.
        
        This mirrors the training rollout behavior:
        - Agent 0 generates first
        - Agent 1 sees Agent 0's output and generates
        - And so on...
        
        Args:
            valid_dataloader: Validation data loader
            epoch: Current epoch
            step: Current step within epoch
            global_step: Global training step
            force_run: If True, bypass frequency control and always run evaluation
        """
        def evaluate_fn():
            if dist.get_rank() == 0:
                # ✅ Sequential evaluation: agents process data one after another
                for data_batch in valid_dataloader:
                    # Each data_batch is a list of samples
                    for agent_idx, agent in enumerate(self.agents):
                        # For sequential mode, each agent sees previous agents' outputs
                        # which are written into the data dict by the workflow
                        with stats_tracker.scope(f"agent{agent_idx}"):
                            for item in data_batch:
                                agent['eval_rollout'].submit(item, agent['eval_workflow'])
                        
                        # Wait for this agent to finish before next agent starts
                        agent['eval_rollout'].wait(len(data_batch), timeout=None)
                        logger.info(
                            f"✅ Agent {agent_idx} evaluation completed for batch "
                            f"(sequential mode)"
                        )
            
            dist.barrier(device_ids=[self.agents[0]['actor'].device.index])
            current_platform.synchronize()
        
        # ✅ Run evaluation once for all agents (not per-agent)
        with stats_tracker.record_timing("eval"):
            if force_run:
                # Bypass frequency control for initial evaluation
                evaluate_fn()
            else:
                self.agents[0]['evaluator'].evaluate(
                    evaluate_fn,
                    epoch,
                    step,
                    global_step,
                )
    
    def destroy(self):
        """Clean up resources and process groups."""
        for agent in self.agents:
            agent['eval_rollout'].destroy()
            agent['rollout'].destroy()
            agent['critic'].destroy()
            agent['actor'].destroy()
        
        # ✅ 显式销毁 NCCL 进程组
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("NCCL process group destroyed")


def main(args):
    config, _ = load_expr_config(args, PPOConfig)
    config: PPOConfig
    
    # rank = int(os.getenv("RANK", "0"))
    # if rank == 0 and os.getenv("DEBUG_CHILD_PROCESS") == "1":
    #     import debugpy
    #     debugpy.listen(("0.0.0.0", 5678))
    #     print(f"⚠️  Waiting for debugger to attach on port 5678...")
    #     debugpy.wait_for_client()
    #     print(f"✅ Debugger attached!")
        
    rank = int(os.getenv("RANK", "0"))
    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    
    # Initialize multi-agent system
    ma_system = MultiAgentSystem(config)
    
    # Prepare datasets
    tokenizer = ma_system.tokenizer
    train_dataset = get_custom_dataset(
        split="train", dataset_config=config.train_dataset, tokenizer=tokenizer
    )
    valid_dataset = get_custom_dataset(
        split="test", dataset_config=config.valid_dataset, tokenizer=tokenizer
    )
    
    train_dataloader = create_dataloader(
        train_dataset,
        rank=0,
        world_size=1,
        dataset_config=config.train_dataset,
    )
    valid_dataloader = create_dataloader(
        valid_dataset,
        rank=0,
        world_size=1,
        dataset_config=config.valid_dataset,
    )
    
    # FinetuneSpec
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )
    
    ma_system.finalize_initialization(ft_spec)
    
    # Shared utilities
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)
    
    # Training loop
    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch
    data_generator = cycle_dataloader(train_dataloader)
    
    # Initial evaluation at step 0 (before any training)
    logger.info("Running initial evaluation at step 0 before training...")
    ma_system.evaluate(valid_dataloader, epoch=0, step=0, global_step=0, force_run=True)
    
    dist.barrier(device_ids=[ma_system.agents[0]['actor'].device.index])
    current_platform.synchronize()
    
    # Export initial evaluation stats
    initial_stats = stats_tracker.export_all(
        reduce_group=ma_system.agents[0]['actor'].data_parallel_group
    )
    stats_logger.commit(epoch=0, step=0, global_step=0, data=initial_stats)
    
    logger.info("Starting training loop...")
    for global_step in range(max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        
        # Fetch data batch (rank 0)
        base_batch = None
        if dist.get_rank() == 0:
            base_batch = next(data_generator)
        
        batches = ma_system.joint_rollout(base_batch)
        
        dist.barrier(device_ids=[ma_system.agents[0]['actor'].device.index])
        current_platform.synchronize()
        
        with stats_tracker.record_timing("compute_joint_rewards"):
            ma_system.compute_joint_rewards(batches)
        
        ma_system.joint_training(batches)
        
        dist.barrier(device_ids=[ma_system.agents[0]['actor'].device.index])
        current_platform.synchronize()
        
        ma_system.update_weights(global_step)
        
        ma_system.save_checkpoints(epoch, step, global_step)
        
        dist.barrier(device_ids=[ma_system.agents[0]['actor'].device.index])
        current_platform.synchronize()
        
        # Evaluation
        ma_system.evaluate(valid_dataloader, epoch, step, global_step)
        
        dist.barrier(device_ids=[ma_system.agents[0]['actor'].device.index])
        current_platform.synchronize()
        
        # Export all statistics (with agent-specific scopes automatically included)
        stats = stats_tracker.export_all(
            reduce_group=ma_system.agents[0]['actor'].data_parallel_group
        )
        stats_logger.commit(epoch, step, global_step, data=stats)
        
        # Cleanup
        for batch in batches:
            del batch
        torch.cuda.empty_cache()
    
    stats_logger.close()
    ma_system.destroy()
    logger.info("Multi-agent training completed!")


if __name__ == "__main__":
    main(sys.argv[1:])