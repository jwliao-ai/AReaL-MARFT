"""
Multi-Agent RLVR Workflow.

This workflow extends RLVRWorkflow to support multi-agent scenarios where:
- Agents can observe previous agents' outputs (sequential mode)
- Agents can run independently (parallel mode)
- Agents can communicate with each other (communication mode)

The workflow uses in-place data augmentation: each agent appends its output
to the input data dict with key "agent_{agent_id}_response", ensuring that
context is correctly aligned per prompt in async rollout scenarios.

Example:
    >>> workflow = MultiAgentRLVRWorkflow(
    ...     agent_id=0,
    ...     reward_fn=my_reward_fn,
    ...     gconfig=gconfig,
    ...     tokenizer=tokenizer,
    ...     interaction_mode="sequential",
    ... )
    >>> # Agent 0 adds "agent_0_response" to data
    >>> result = await workflow.arun_episode(engine, data)
    >>> # Agent 1 reads "agent_0_response" from data and adds "agent_1_response"
    >>> result = await workflow.arun_episode(engine, data)
"""

import asyncio
import os
import uuid
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import aiofiles
import colorama
import torch
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
from areal.utils import logging
from areal.utils.data import concat_padded_tensors
from areal.workflow.rlvr import RLVRWorkflow, default_get_input_ids_fn

logger = logging.getLogger("Multi-Agent RLVR workflow")


class MultiAgentRLVRWorkflow(RLVRWorkflow):
    """
    Multi-agent extension of RLVRWorkflow.
    ...
    """
    
    def __init__(
        self,
        agent_id: int,
        reward_fn: Callable[..., Any],
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        interaction_mode: str = "parallel",
        enable_thinking: bool = False,
        rollout_stat_scope: str | None = None,
        dump_dir: str | None = None,
        get_input_ids_fn: Callable[
            [Any, PreTrainedTokenizerFast, bool], list[int]
        ] = default_get_input_ids_fn,
        data_extract_prompt_fn: Callable[[dict[str, Any]], Any] | None = None,
        context_format_fn: Callable[[int, str, list[dict]], str] | None = None,
        agent_profile: dict[str, Any] | None = None,
    ):
        """
        Initialize multi-agent workflow.
        
        Args:
            agent_id: Unique identifier for this agent (0-indexed)
            reward_fn: Function to compute rewards
            gconfig: Generation hyperparameters
            tokenizer: Tokenizer for encoding/decoding
            interaction_mode: 'parallel', 'sequential', or 'communication'
            enable_thinking: Enable thinking token mode
            rollout_stat_scope: Scope for statistics tracking (default: f"agent{agent_id}_rollout")
            dump_dir: Directory to dump generated samples
            get_input_ids_fn: Custom function to extract input_ids from data
            data_extract_prompt_fn: Custom function to extract prompt from data
            context_format_fn: Custom function to format context from other agents
            agent_profile: Agent profile dict containing 'agent_name', 'system_prompt', etc.
        """
        if rollout_stat_scope is None:
            rollout_stat_scope = f"agent{agent_id}_rollout"
        
        if data_extract_prompt_fn is None:
            data_extract_prompt_fn = lambda data: data.get("messages", data)
        
        super().__init__(
            reward_fn=reward_fn,
            gconfig=gconfig,
            tokenizer=tokenizer,
            enable_thinking=enable_thinking,
            rollout_stat_scope=rollout_stat_scope,
            dump_dir=dump_dir,
            get_input_ids_fn=get_input_ids_fn,
            data_extract_prompt_fn=data_extract_prompt_fn,
        )
        
        self.agent_id = agent_id
        self.interaction_mode = interaction_mode
        self.response_key = f"agent_{agent_id}_response"
        self.context_format_fn = context_format_fn or self._default_context_format
        
        # Unpack agent profile
        agent_profile = agent_profile or {}
        self.agent_name = agent_profile.get("agent_name", f"Agent{agent_id}")
        self.system_prompt = agent_profile.get("system_prompt", None)
        self.post_system_prompt = agent_profile.get("post_system_prompt", None)
        
        self.agent_profile = agent_profile
        
        logger.info(
            f"Initialized Agent {agent_id} ('{self.agent_name}') with "
            f"interaction_mode='{interaction_mode}', "
            f"response_key='{self.response_key}', "
            f"system_prompt={'set' if self.system_prompt else 'not set'}, "
            f"post_system_prompt={'set' if self.post_system_prompt else 'not set'}"
        )

    def _extract_previous_agent_responses(self, data: dict[str, Any]) -> list[dict]:
        context = []
        for i in range(self.agent_id):
            key = f"agent_{i}_response"
            if key in data:
                response = data[key]
                context.append({
                    "agent_id": i,
                    **response,
                })
        return context

    def _default_context_format(self, interaction_mode, agent_id: int, agent_name: str, context: list[dict]) -> str:
        if interaction_mode == "parallel":
            return ""
        
        if interaction_mode == "sequential":
            if not context:
                return ""
            
            lines = ["\n[TEACHER'S RESPONSE]"]
            for ctx in context:
                prev_agent_name = ctx.get('agent_name', f"Agent{ctx['agent_id']}")
                completion = ctx.get('completion', '')
                lines.append(f"{prev_agent_name}: {completion}")
            
            lines.append(f"[END OF TEACHER'S RESPONSE]\nNow it's your turn.")
            return "\n".join(lines)
        
        if interaction_mode == "communication":
            logger.warning(
                f"Agent {agent_id}: 'communication' mode not yet implemented, "
                "falling back to sequential format"
            )
            return ""
        
        logger.warning(
            f"Agent {agent_id}: Unknown interaction_mode '{interaction_mode}', "
            "no context will be provided"
        )
        return ""

    def _augment_prompt_with_context(
        self, prompt_data: Any, context: list[dict]
    ) -> Any:
        """
        Augment the prompt with system prompt, context, and post_system_prompt.
        
        Structure:
        [System Prompt]
        [User Query]
        [Context (Previous Agents)]
        [Post System Prompt] -> 紧跟在 Context 之后，提醒 Agent 注意事项
        """
        # 1. 获取 Context 字符串
        context_str = self.context_format_fn(interaction_mode=self.interaction_mode, agent_id=self.agent_id, agent_name=self.agent_name, context=context)
        
        # 2. 构建要在 User 消息末尾追加的完整内容 (Context + Post Prompt)
        parts_to_append = []
        if context_str:
            parts_to_append.append(context_str)
            
        if self.post_system_prompt:
            # 确保与前面的内容有分隔（如果 context_str 存在，加换行；如果不存在，也加换行与 User query 分隔）
            parts_to_append.append("\n\n" + self.post_system_prompt)
            
        full_suffix_str = "".join(parts_to_append)
        
        if isinstance(prompt_data, list) and prompt_data:
            augmented_data = deepcopy(prompt_data)
            
            # Step A: Add standard system prompt at the beginning
            if self.system_prompt:
                system_msg = {'role': 'system', 'content': self.system_prompt}
                if augmented_data and augmented_data[0].get('role') == 'system':
                    augmented_data[0]['content'] = self.system_prompt + "\n\n" + augmented_data[0]['content']
                else:
                    augmented_data.insert(0, system_msg)
            
            # Step B: Append Context + Post Prompt after the last user message
            if full_suffix_str:
                for i in range(len(augmented_data) - 1, -1, -1):
                    if augmented_data[i].get('role') == 'user':
                        augmented_data[i]['content'] = augmented_data[i]['content'] + full_suffix_str
                        break
                else:
                    # No user message found, append as a new user message
                    augmented_data.append({'role': 'user', 'content': full_suffix_str})
            
            return augmented_data
        
        elif isinstance(prompt_data, str):
            parts = []
            if self.system_prompt:
                parts.append(self.system_prompt)
            parts.append(prompt_data)
            if full_suffix_str:
                parts.append(full_suffix_str)
            return "\n\n".join(parts)
        
        else:
            logger.warning(
                f"Agent {self.agent_id}: Unknown prompt_data type {type(prompt_data)}, "
                "cannot inject context or system prompts"
            )
            return prompt_data
        
    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor] | None:
        """
        Run one episode with context from previous agents in data dict.
        
        **Key behavior**: 
        1. Reads "agent_N_response" keys from data (N < agent_id)
        2. Augments prompt with previous agents' outputs
        3. Generates response
        4. **Writes "agent_{agent_id}_response" back to data** for next agent
        5. **Preserves global index for cross-agent alignment**
        
        Args:
            engine: Inference engine for generation
            data: Input data dict. Will be modified to add this agent's response.
                Expected to contain previous agents' responses as:
                {
                    "messages": [...],
                    "__global_idx__": 0,  # ← Global index from joint_rollout
                    "agent_0_response": {"completion": "...", "reward": ...},
                    "agent_1_response": {"completion": "...", "reward": ...},
                    ...
                }
        
        Returns:
            Dict with keys:
                - input_ids: Full sequence (prompt + completion)
                - response_ids: Completion tokens only
                - loss_mask: Mask for loss computation
                - logprobs: Log probabilities
                - versions: Weight versions
                - attention_mask: Attention mask
                - rewards: Reward values
                - __global_idx__: (preserved from input data)
            Returns None if episode should be rejected.
        """
        # 1. Extract context from previous agents in data
        context = self._extract_previous_agent_responses(data)
        
        if context:
            logger.debug(
                f"Agent {self.agent_id} found context from {len(context)} previous agent(s)"
            )
        
        # 2. Extract and augment prompt with context
        original_prompt_data = self.data_extract_prompt_fn(data)
        augmented_prompt_data = self._augment_prompt_with_context(
            original_prompt_data, context
        )
        
        # 3. Get input_ids from augmented prompt
        input_ids = self.get_input_ids_fn(
            augmented_prompt_data,
            self.tokenizer,
            self.enable_thinking,
        )
        
        n_samples = self.gconfig.n_samples
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
        )
        
        version = engine.get_version()
        prompt_str = self.tokenizer.decode(input_ids)
        prompt_strs = [prompt_str] * n_samples
        
        # 4. Generate responses and collect rewards
        sample_results = await asyncio.gather(
            *[
                self._collect_samples(engine, req, prompt_str, data)
                for _ in range(n_samples)
            ]
        )
        
        if sample_results:
            resps, rewards, completions_strs = map(list, zip(*sample_results))
        else:
            resps, rewards, completions_strs = [], [], []
        
        # 5. Build result tensors
        results = []
        for resp, reward in zip(resps, rewards):
            seq = resp.input_tokens + resp.output_tokens
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions
            
            res = {
                "input_ids": torch.tensor(seq, dtype=torch.int32),
                "response_ids": torch.tensor(resp.output_tokens, dtype=torch.int32),
                "completion_ids": torch.tensor(resp.output_tokens, dtype=torch.int32),
                "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
                "logprobs": torch.tensor(logprobs, dtype=torch.float32),
                "versions": torch.tensor(versions, dtype=torch.int32),
                "attention_mask": torch.ones(len(seq), dtype=torch.bool),
                "rewards": torch.tensor(reward, dtype=torch.float32),
                "__global_idx__": torch.tensor(data.get("__global_idx__", -1), dtype=torch.int32),  # ✅ 携带全局索引
            }
            res = {k: v.unsqueeze(0) for k, v in res.items()}
            results.append(res)
        
        # 6. Write this agent's response back to data for next agent
        if resps and completions_strs:
            data[self.response_key] = {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "completion": completions_strs[0],
                "reward": sum(rewards) / len(rewards),
                "prompt": prompt_str,
                "response_ids": resps[0].output_tokens,
            }
            logger.debug(
                f"Agent {self.agent_id} ('{self.agent_name}') wrote response to data['{self.response_key}']"
            )
        
        # 7. Dump to file if enabled
        if self.dump_dir is not None:
            dump_path = os.path.join(self.dump_dir, str(version))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)
            
            # Get unique identifier
            qid = None
            for key in ["query_id", "id", "qid"]:
                qid = data.get(key, None)
                if qid is not None:
                    break
            qid = qid or uuid.uuid4().hex
            
            # Dump with agent_id prefix
            file_path = os.path.join(dump_path, f"agent{self.agent_id}_{qid}.txt")
            seqlens = [
                len(resp.input_tokens) + len(resp.output_tokens) for resp in resps
            ]
            
            async with aiofiles.open(file_path, "a") as f:
                # Write context information
                if context:
                    await f.write(
                        f"{colorama.Fore.CYAN}=== Context from previous agents ==={colorama.Style.RESET_ALL}\n"
                    )
                    for ctx in context:
                        await f.write(
                            f"Agent {ctx['agent_id']}: {ctx.get('completion', '')}\n"
                        )
                    await f.write("\n")
                
                # Write samples
                for i, (prompt, completion, reward, seqlen) in enumerate(
                    zip(prompt_strs, completions_strs, rewards, seqlens)
                ):
                    info = "\n".join(
                        [
                            f"{colorama.Fore.GREEN}Agent {self.agent_id}{colorama.Style.RESET_ALL} - "
                            f"idx: {i + 1}/{n_samples}, seqlen: {seqlen}, reward: {reward}",
                            f"prompt: \n{colorama.Fore.YELLOW + colorama.Style.DIM}{prompt}{colorama.Style.RESET_ALL}",
                            f"completion: \n{colorama.Fore.YELLOW + colorama.Style.DIM}{completion}{colorama.Style.RESET_ALL}",
                        ]
                    )
                    await f.write(info + "\n")
        
        return concat_padded_tensors(results)