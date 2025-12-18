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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.toolrl import compute_score as compute_score_fn
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("toolrl")
class ToolRLRewardManager(AbstractRewardManager):
    """The reward manager."""
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        # TODO: compute_score和reward_fn_key未被使用
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console

    def __call__(self, data: DataProto, return_dict: bool = False, step: int = None, **kwargs) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""
        
        # print("==========ToolRLRewardManager.__call__ called==========")
        # print("==========has rm_scores:", 'rm_scores' in data.batch.keys())
        # print("==========data length:", len(data))

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch['rm_scores'], "reward_extra_info": reward_extra_info}
            else:
                return data.batch['rm_scores'], None

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            try:
                ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            except Exception as e:
                print(f"==========ERROR getting ground_truth: {e}")
                print(f"==========non_tensor_batch keys: {list(data_item.non_tensor_batch.keys())}")
                raise

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            
            # Get step from kwargs, extra_info, or default to 0
            if step is None:
                extra_info = data_item.non_tensor_batch.get("extra_info", {})
                step = extra_info.get("step", 0)

            # print("==========data_source: ", data_source)
            # print("==========prompt: \n", self.tokenizer.decode(prompt_ids, skip_special_tokens=False))
            # print("==========sequences_str: \n", sequences_str)
            # print("==========ground_truth: \n", ground_truth)
            # print("==========step: ", step)

            score, format_score, correctness_score, length_score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, step=step)
            reward_tensor[i, valid_response_length - 1] = score
            
            # Store extra info
            reward_extra_info["format_score"].append(format_score)
            reward_extra_info["correctness_score"].append(correctness_score)
            reward_extra_info["length_score"].append(length_score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt+response]", sequences_str)
                print("[score]", score)
                print("[format_score]", format_score)
                print("[correctness_score]", correctness_score)
                print("[length_score]", length_score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor, None