# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect

import torch

from verl import DataProto
from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
from verl.utils.reward_score.toolrl import compute_score as compute_score_fn


@register("toolrl")
class ToolRLRewardLoopManager(RewardLoopManagerBase):
    """The reward manager for ToolRL."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer)
        # Use the custom compute_score if provided, otherwise use the default toolrl compute_score
        self.compute_score = compute_score or compute_score_fn
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    async def run_single(self, data: DataProto) -> dict:
        """We will expand this function gradually based on the available datasets"""
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data_item.batch.keys():
            return {
                "reward_score": data_item.batch["rm_scores"],
                "reward_extra_info": {},
            }
        
        # Decode prompt + response (following workers/reward_manager/toolrl.py logic)
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]
        
        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        
        # Decode full sequence (prompt + response) - same as workers version
        sequences = torch.cat((valid_prompt_ids, valid_response_ids))
        sequences_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(sequences, skip_special_tokens=True)
        )
        
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        
        # Get step from extra_info if available, otherwise default to 0
        step = extra_info.get("step", 0)
        
        # Call compute_score_fn with step parameter (same as workers version)
        if self.is_async_reward_score:
            result = await self.compute_score(
                solution_str=sequences_str,
                ground_truth=ground_truth,
                step=step,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    solution_str=sequences_str,
                    ground_truth=ground_truth,
                    step=step,
                ),
            )
        
        # Handle tuple return (score, format_score, correctness_score, length_score)
        # Same as workers version: score, fomrat_score, correctness_score, length_score = compute_score_fn(...)
        reward_extra_info = {}
        if isinstance(result, tuple) and len(result) == 4:
            score, format_score, correctness_score, length_score = result
            reward_extra_info["format_score"] = format_score
            reward_extra_info["correctness_score"] = correctness_score
            reward_extra_info["length_score"] = length_score
        elif isinstance(result, dict):
            score = result.get("score", result.get("reward", 0.0))
            reward_extra_info.update(result)
        else:
            score = result
            reward_extra_info["acc"] = score
        
        return {"reward_score": score, "reward_extra_info": reward_extra_info}

