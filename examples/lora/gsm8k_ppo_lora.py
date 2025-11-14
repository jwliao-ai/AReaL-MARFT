import os
import sys
from copy import deepcopy

import torch.distributed as dist

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import PPOConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.core.dist_rollout import redistribute
from areal.dataset import get_custom_dataset
from areal.engine.ppo.actor import FSDPPPOActor
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
)
from areal.utils.dataloader import create_dataloader
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow


def gsm8k_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    from areal.reward.math_parser import process_results

    return int(process_results(completions, answer)[0])


def bcast_and_split_from_rank0(batch: dict | None, granularity: int) -> dict:
    batch = broadcast_tensor_container(batch, src_rank=0)
    bs = get_batch_size(batch)
    world_size = dist.get_world_size()
    
    if bs % world_size != 0:
        valid_bs = (bs // world_size) * world_size
        if valid_bs == 0:
            raise RuntimeError(
                f"Batch size {bs} is smaller than world size {world_size}. "
                "Cannot split across ranks."
            )
        batch = {k: v[:valid_bs] for k, v in batch.items()}
        bs = valid_bs
    
    bs_per_rank = bs // world_size
    local_batch = []
    for i in range(dist.get_rank() * bs_per_rank, (dist.get_rank() + 1) * bs_per_rank):
        local_batch.append({k: v[i : i + 1] for k, v in batch.items()})
    local_batch = concat_padded_tensors(local_batch)
    return redistribute(local_batch, granularity=granularity).data


def main(args):
    config, _ = load_expr_config(args, PPOConfig)
    
        # 👇 在子进程启动时初始化 debugpy（仅 rank 0）
    rank = int(os.getenv("RANK", "0"))
    if rank == 0 and os.getenv("DEBUG_CHILD_PROCESS") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))  # 监听端口 5678
        print(f"⚠️  Waiting for debugger to attach on port 5678...")
        debugpy.wait_for_client()  # 阻塞等待 VS Code 连接
        print(f"✅ Debugger attached!")
        
    config: PPOConfig

    rank = int(os.getenv("RANK"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None
    if parallel_strategy.data_parallel_size != parallel_strategy.world_size:
        raise ValueError("LoRA does not support parallelism other than FSDP.")

    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)
    critic = FSDPPPOCritic(config=config.critic)
    critic.create_process_group(parallel_strategy=parallel_strategy)

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

    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(train_data_parallel_size=1)
    eval_rollout = RemoteSGLangEngine(deepcopy(config.rollout))
    eval_rollout.config.max_head_offpolicyness = int(1e12)
    eval_rollout.initialize()

    weight_update_meta = WeightUpdateMeta.from_disk(
        config.saver.experiment_name,
        config.saver.trial_name,
        config.saver.fileroot,
        use_lora=True,
    )

    actor.initialize(None, ft_spec)
    actor.connect_engine(rollout, weight_update_meta)

    critic.initialize(None, ft_spec)
    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.create_process_group(parallel_strategy=parallel_strategy)
        ref.initialize(None, ft_spec)

    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    workflow = RLVRWorkflow(
        reward_fn=gsm8k_reward_fn,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        enable_thinking=False,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
    )
    eval_workflow = RLVRWorkflow(
        reward_fn=gsm8k_reward_fn,
        gconfig=config.gconfig.new(temperature=0.6),
        tokenizer=tokenizer,
        enable_thinking=False,
        rollout_stat_scope="eval-rollout",
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated-eval"
        ),
    )

    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    engines = {"default": actor, "critic": critic}
    recover_handler = RecoverHandler(config.recover, ft_spec)
    recover_info = recover_handler.load(
        engines,
        saver,
        evaluator,
        stats_logger,
        train_dataloader,
        inference_engine=rollout,
        weight_update_meta=weight_update_meta,
    )
    start_step = (
        recover_info.last_step_info.next().global_step
        if recover_info is not None
        else 0
    )

    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch

    data_generator = cycle_dataloader(train_dataloader)
    for global_step in range(start_step, max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=step,
            steps_per_epoch=steps_per_epoch,
        )

        with stats_tracker.record_timing("rollout"):
            batch = None
            if dist.get_rank() == 0:
                if config.async_training:
                    batch = rollout.prepare_batch(
                        train_dataloader,
                        workflow=workflow,
                        should_accept_fn=lambda sample: True,
                    )
                else:
                    batch = rollout.rollout_batch(
                        next(data_generator),
                        workflow=workflow,
                        should_accept_fn=lambda sample: True,
                    )
                batch = tensor_container_to(batch, actor.device)
            batch = bcast_and_split_from_rank0(
                batch, granularity=config.actor.group_size
            )

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        with stats_tracker.record_timing("critic_values"):
            values = critic.compute_values(batch)
            batch["values"] = values
            log_gpu_stats("critic values")

        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                batch["prox_logp"] = logp
                log_gpu_stats("recompute logp")

        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                batch["ref_logp"] = ref.compute_logp(batch)
                log_gpu_stats("ref logp")

        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)
            log_gpu_stats("compute advantages")

        with stats_tracker.record_timing("train_step"):
            actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("ppo actor update")

        with stats_tracker.record_timing("train_step"):
            critic.ppo_update(batch)
            critic.step_lr_scheduler()
            log_gpu_stats("ppo critic update")

        rollout.pause()

        with stats_tracker.record_timing("update_weights"):
            actor.update_weights(weight_update_meta)

            actor.set_version(global_step + 1)
            critic.set_version(global_step + 1)
            rollout.set_version(global_step + 1)
            eval_rollout.set_version(global_step + 1)

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step, tokenizer=tokenizer)
            saver.save(
                critic, epoch, step, global_step, tokenizer=tokenizer, name="critic"
            )

        with stats_tracker.record_timing("checkpoint_for_recover"):
            recover_handler.dump(
                engines,
                step_info,
                saver,
                evaluator,
                stats_logger,
                train_dataloader,
                tokenizer=tokenizer,
            )

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        with stats_tracker.record_timing("eval"):

            def evaluate_fn():
                cnt = 0
                if dist.get_rank() == 0:
                    for data in valid_dataloader:
                        for item in data:
                            eval_rollout.submit(item, eval_workflow)
                            cnt += 1
                    eval_rollout.wait(cnt, timeout=None)
                dist.barrier(device_ids=[actor.device.index])
                current_platform.synchronize()

            evaluator.evaluate(
                evaluate_fn,
                epoch,
                step,
                global_step,
            )

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        stats = stats_tracker.export_all(reduce_group=actor.data_parallel_group)
        stats_logger.commit(epoch, step, global_step, stats)

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        rollout.resume()

    stats_logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    critic.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
