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
    
    Supports different interaction modes:
    - 'parallel': Agents operate independently without seeing each other's outputs
    - 'sequential': Agents observe outputs of previous agents in order
    - 'communication': Agents can send/receive messages to/from each other
    
    Key difference from base workflow: Instead of external context setting,
    this workflow reads/writes agent outputs directly in the input data dict
    using keys like "agent_0_response", "agent_1_response", etc.
    
    Attributes:
        agent_id: Unique identifier for this agent (0-indexed)
        interaction_mode: How agents interact ('parallel', 'sequential', 'communication')
        response_key: Key used to store this agent's response in data dict
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
        context_format_fn: Callable[[int, list[dict]], str] | None = None,
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
        """
        # Set default rollout_stat_scope if not provided
        if rollout_stat_scope is None:
            rollout_stat_scope = f"agent{agent_id}_rollout"
        
        # Set default data_extract_prompt_fn if not provided
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
        
        logger.info(
            f"Initialized Agent {agent_id} with interaction_mode='{interaction_mode}', "
            f"response_key='{self.response_key}'"
        )
    
    def _extract_previous_agent_responses(self, data: dict[str, Any]) -> list[dict]:
        """
        Extract responses from previous agents stored in data dict.
        
        Args:
            data: Input data dict that may contain "agent_N_response" keys
        
        Returns:
            List of dicts with agent_id and response info, sorted by agent_id
        
        Example:
            data = {
                "messages": [...],
                "agent_0_response": {"completion": "...", "reward": 1.0},
                "agent_1_response": {"completion": "...", "reward": 0.5},
            }
            Returns: [
                {"agent_id": 0, "completion": "...", "reward": 1.0},
                {"agent_id": 1, "completion": "...", "reward": 0.5},
            ]
        """
        context = []
        for i in range(self.agent_id):  # Only look at agents before current one
            key = f"agent_{i}_response"
            if key in data:
                response = data[key]
                context.append({
                    "agent_id": i,
                    **response,  # Include all fields (completion, reward, etc.)
                })
        return context
    
    def _default_context_format(self, agent_id: int, context: list[dict]) -> str:
        """
        Default formatting for context from other agents.
        
        Args:
            agent_id: Current agent's ID
            context: List of context dicts from other agents
        
        Returns:
            Formatted context string to prepend to prompt
        """
        if not context:
            return ""
        
        lines = [f"You are Agent {agent_id}."]
        
        if self.interaction_mode == "sequential":
            lines.append("Previous agents' responses:")
            for ctx in context:
                prev_agent_id = ctx['agent_id']
                completion = ctx.get('completion', '')
                reward_str = ""
                if 'reward' in ctx:
                    reward_str = f" (reward: {ctx['reward']:.2f})"
                lines.append(f"- Agent {prev_agent_id}: {completion}{reward_str}")
            lines.append("\nNow answer the following question:")
        
        elif self.interaction_mode == "communication":
            lines.append("Messages from other agents:")
            for ctx in context:
                prev_agent_id = ctx['agent_id']
                completion = ctx.get('completion', '')
                lines.append(f"[Agent {prev_agent_id}]: {completion}")
            lines.append("\nYour response:")
        
        return "\n".join(lines)
    
    def _augment_prompt_with_context(
        self, prompt_data: Any, context: list[dict]
    ) -> Any:
        """
        Augment the prompt with context from other agents.
        
        Args:
            prompt_data: Original prompt data (typically a list of message dicts)
            context: List of context dicts from previous agents
        
        Returns:
            Augmented prompt data with context injected
        """
        # Skip if no context or parallel mode
        if not context or self.interaction_mode == "parallel":
            return prompt_data
        
        # Format context string
        context_str = self.context_format_fn(self.agent_id, context)
        
        if not context_str:
            return prompt_data
        
        # Handle chat format (list of message dicts)
        if isinstance(prompt_data, list) and prompt_data:
            augmented_data = deepcopy(prompt_data)
            
            # Find the first user message and prepend context
            for i, msg in enumerate(augmented_data):
                if msg.get('role') == 'user':
                    augmented_data[i]['content'] = (
                        context_str + "\n\n" + msg['content']
                    )
                    break
            else:
                # No user message found, insert as system message
                augmented_data.insert(0, {
                    'role': 'system',
                    'content': context_str
                })
            
            return augmented_data
        
        # Handle raw string format
        elif isinstance(prompt_data, str):
            return context_str + "\n\n" + prompt_data
        
        # Unknown format, return as-is
        else:
            logger.warning(
                f"Agent {self.agent_id}: Unknown prompt_data type {type(prompt_data)}, "
                "cannot inject context"
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
        
        Args:
            engine: Inference engine for generation
            data: Input data dict. Will be modified to add this agent's response.
                  Expected to contain previous agents' responses as:
                  {
                      "messages": [...],
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
            Returns None if episode should be rejected.
        """
        # 1. Extract context from previous agents in data
        context = self._extract_previous_agent_responses(data)
        
        if context:
            logger.debug(
                f"Agent {self.agent_id} found context from {len(context)} previous agent(s)"
            )
            
        # TODO: format context
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
            }
            res = {k: v.unsqueeze(0) for k, v in res.items()}
            results.append(res)
        
        # 6. ✅ Write this agent's response back to data for next agent
        # Use the first sample's completion and average reward as representative
        if resps and completions_strs:
            data[self.response_key] = {
                "completion": completions_strs[0],  # First sample
                "reward": sum(rewards) / len(rewards),  # Average reward
                "prompt": prompt_str,
                "response_ids": resps[0].output_tokens,  # Token IDs for advanced use
            }
            logger.debug(
                f"Agent {self.agent_id} wrote response to data['{self.response_key}']"
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