# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multi-turn SFT dataset that supports training on conversation data with multiple turns
"""

import hashlib
import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from datasets import load_from_disk, concatenate_datasets
from omegaconf import ListConfig
from torch.utils import data
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from tqdm import tqdm

from verl.utils import hf_tokenizer
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.fs import copy_local_path_from_hdfs


def convert_nested_value_to_list_recursive(data_item):
    if isinstance(data_item, dict):
        return {k: convert_nested_value_to_list_recursive(v) for k, v in data_item.items()}
    elif isinstance(data_item, list):
        return [convert_nested_value_to_list_recursive(elem) for elem in data_item]
    elif isinstance(data_item, np.ndarray):
        # Convert to list, then recursively process the elements of the new list
        return convert_nested_value_to_list_recursive(data_item.tolist())
    else:
        # Base case: item is already a primitive type (int, str, float, bool, etc.)
        return data_item


def _process_single_sample_multiprocess_worker(args):
    """
    Worker function for multiprocessing tokenization.
    This function runs in subprocesses, so we need to reconstruct tokenizer.
    This is a module-level function to avoid serialization issues with self.

    Args:
        args: tuple of (idx, messages, tools, enable_thinking, tokenizer_config, apply_chat_template_kwargs)

    Returns:
        tuple of (idx, result_dict) or (idx, None) if failed
    """
    from multiprocessing import current_process

    idx, messages, tools, enable_thinking, tokenizer_config, apply_chat_template_kwargs = args

    # Validate inputs
    if messages is None or (isinstance(messages, float) and pd.isna(messages)):
        logging.warning(f"Process {current_process().name}: Sample {idx} has invalid messages (None or NaN)")
        return (idx, None)

    if not isinstance(messages, (list, tuple)):
        logging.warning(f"Process {current_process().name}: Sample {idx} has invalid messages type: {type(messages)}")
        return (idx, None)

    # Handle NaN values in tools and enable_thinking
    if tools is not None and isinstance(tools, float) and pd.isna(tools):
        tools = None
    if enable_thinking is not None and isinstance(enable_thinking, float) and pd.isna(enable_thinking):
        enable_thinking = None

    # Reconstruct tokenizer in subprocess
    tokenizer_name = tokenizer_config.get('name_or_path')
    if tokenizer_name:
        try:
            tokenizer = hf_tokenizer(tokenizer_name)
        except Exception as e:
            logging.warning(f"Process {current_process().name}: Failed to reconstruct tokenizer: {e}")
            return (idx, None)
    else:
        logging.warning(f"Process {current_process().name}: Tokenizer name not found in config")
        return (idx, None)

    # Process sample using the same logic as _process_single_sample_impl
    try:
        # Get the full conversation tokens
        full_tokens = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
            **apply_chat_template_kwargs,
        )

        # Track concatenated tokens
        concat_tokens = []
        concat_loss_mask = []
        concat_attention_mask = []

        i = 0
        while i < len(messages):
            cur_messages = messages[i]
            if cur_messages["role"] == "assistant":
                tokens, loss_mask, attention_mask = _process_message_tokens_standalone(
                    messages, i, i + 1, tokenizer, is_assistant=True, enable_thinking=enable_thinking, tools=tools, apply_chat_template_kwargs=apply_chat_template_kwargs
                )
                i += 1
            elif cur_messages["role"] == "tool":
                st = i
                ed = i + 1
                while ed < len(messages) and messages[ed]["role"] == "tool":
                    ed += 1
                tokens, loss_mask, attention_mask = _process_message_tokens_standalone(
                    messages, st, ed, tokenizer, enable_thinking=enable_thinking, tools=tools, apply_chat_template_kwargs=apply_chat_template_kwargs
                )
                i = ed
            elif cur_messages["role"] in ["user", "system"]:
                if cur_messages["role"] == "system" and i != 0:
                    raise ValueError("System message should be the first message")
                tokens, loss_mask, attention_mask = _process_message_tokens_standalone(
                    messages, i, i + 1, tokenizer, enable_thinking=enable_thinking, tools=tools, apply_chat_template_kwargs=apply_chat_template_kwargs
                )
                i += 1
            else:
                raise ValueError(f"Unknown role: {cur_messages['role']}")

            # Override loss mask if present
            override_loss_mask = cur_messages.get("loss_mask", None)
            if override_loss_mask is not None:
                if isinstance(override_loss_mask, np.ndarray):
                    override_loss_mask = override_loss_mask.item()
                assert isinstance(override_loss_mask, int), f"loss_mask should be int, got {type(override_loss_mask)}"
                assert override_loss_mask in [0, 1], f"loss_mask should be 0 or 1, got {override_loss_mask}"
                loss_mask = [override_loss_mask] * len(tokens)

            concat_tokens.extend(tokens)
            concat_loss_mask.extend(loss_mask)
            concat_attention_mask.extend(attention_mask)

        # Validate and convert tokens
        input_ids, loss_mask, attention_mask = _validate_and_convert_tokens_standalone(
            full_tokens[0], concat_tokens, concat_loss_mask, concat_attention_mask
        )

        valid_length = attention_mask.sum().item()

        # Return without padding for packing
        result = {
            'input_ids': input_ids[:valid_length].cpu().numpy(),
            'attention_mask': attention_mask[:valid_length].cpu().numpy(),
            'loss_mask': loss_mask[:valid_length].cpu().numpy(),
            'valid_length': valid_length,
        }

        return (idx, result)
    except Exception as e:
        logging.warning(f"Process {current_process().name}: Failed to process sample {idx}: {e}")
        return (idx, None)


def _process_message_tokens_standalone(messages, start_idx, end_idx, tokenizer, is_assistant=False, enable_thinking=None, tools=None, apply_chat_template_kwargs=None):
    """
    Standalone version of _process_message_tokens for multiprocessing.
    """
    if apply_chat_template_kwargs is None:
        apply_chat_template_kwargs = {}

    if start_idx > 0:
        prev_applied_text = tokenizer.apply_chat_template(
            messages[:start_idx],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
            tools=tools,
            **apply_chat_template_kwargs,
        )
        if is_assistant:
            prev_applied_text_w_generation_prompt = tokenizer.apply_chat_template(
                messages[:start_idx],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                tools=tools,
                **apply_chat_template_kwargs,
            )
    else:
        prev_applied_text = ""

    cur_applied_text = tokenizer.apply_chat_template(
        messages[:end_idx],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
        tools=tools,
        **apply_chat_template_kwargs,
    )
    # Get tokens for the current message only
    if is_assistant:
        generation_prompt_text = prev_applied_text_w_generation_prompt[len(prev_applied_text) :]
        generation_prompt_tokens = tokenizer.encode(
            generation_prompt_text,
            add_special_tokens=False,
        )
        _message_tokens = tokenizer.encode(
            cur_applied_text[len(prev_applied_text_w_generation_prompt) :],
            add_special_tokens=False,
        )
        message_tokens = generation_prompt_tokens + _message_tokens
        loss_mask = [0] * (len(generation_prompt_tokens)) + [1] * (
            len(message_tokens) - len(generation_prompt_tokens)
        )
    else:
        message_tokens = tokenizer.encode(
            cur_applied_text[len(prev_applied_text) :],
            add_special_tokens=False,
        )
        loss_mask = [0] * len(message_tokens)

    attention_mask = [1] * len(message_tokens)

    return message_tokens, loss_mask, attention_mask


def _validate_and_convert_tokens_standalone(full_tokens, concat_tokens, concat_loss_mask, concat_attention_mask):
    """
    Standalone version of _validate_and_convert_tokens for multiprocessing.
    """
    import torch

    full_tokens_list = full_tokens.tolist()

    if len(concat_tokens) != len(full_tokens_list) or not all(
        a == b for a, b in zip(concat_tokens, full_tokens_list, strict=True)
    ):
        logging.warning(
            f"Token mismatch detected! Full tokenization length: {len(full_tokens_list)}, Concatenated tokens "
            f"length: {len(concat_tokens)}. Using concatenated version."
        )
        return (
            torch.tensor(concat_tokens, dtype=torch.long),
            torch.tensor(concat_loss_mask, dtype=torch.bool),
            torch.tensor(concat_attention_mask, dtype=torch.bool),
        )

    return (
        full_tokens,
        torch.tensor(concat_loss_mask, dtype=torch.bool),
        torch.tensor(concat_attention_mask, dtype=torch.bool),
    )


class MultiTurnSFTDataset(Dataset):
    """
    Dataset for multi-turn conversations where each assistant response should be trained
    """

    def __init__(self, parquet_files: str | list[str], tokenizer, config=None, max_samples: int = -1):
        # Set defaults and extract parameters from config if provided
        config = config or {}
        self.pad_mode = config.get("pad_mode", "right")
        assert self.pad_mode in ["right", "no_padding"], (
            f"Expect pad_mode to be 'right' or 'no_padding'. Got {self.pad_mode}"
        )
        self.truncation = config.get("truncation", "error")
        # for right padding
        self.max_length = config.get("max_length", 1024)
        # Get messages_key from the new multiturn config structure
        multiturn_config = config.get("multiturn", {})
        self.messages_key = multiturn_config.get("messages_key", "messages")
        self.tools_key = multiturn_config.get("tools_key", "tools")
        self.enable_thinking_key = multiturn_config.get("enable_thinking_key", "enable_thinking")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")
        self.max_samples = max_samples
        # Sample packing configuration
        self.enable_sample_packing = config.get("enable_sample_packing", True)
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None else (self.tokenizer.pad_token_id if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None else 0)
        self.pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None else 0
        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self._download()
        self._read_files_and_process()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            if parquet_file.endswith(".parquet"):
                self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, pandas.core.series.Series | numpy.ndarray) and len(ls) == 1:
                ls = ls[0]
            return ls

        # Store individual dataframes for per-dataset processing
        self.dataset_dataframes = []
        self.dataset_ratios = []  # Store ratios for later sampling after packing
        for parquet_file in self.parquet_files:
            if "@" in parquet_file:
                dataset_ratio, parquet_file = float(parquet_file.split("@")[0]), parquet_file.split("@")[1]
            else:
                dataset_ratio = 1.0
            if parquet_file.endswith(".parquet"):
                dataframe = pd.read_parquet(parquet_file)
            elif parquet_file.endswith(".jsonl"):
                dataframe = pd.read_json(parquet_file, lines=True)

            self.dataset_ratios.append(dataset_ratio)
            # NOTE: Do NOT sample here! We load the full dataset first, 
            # then cache tokenization/packing results, and sample at the end.
            # This ensures cache can be reused regardless of sampling ratio.
            self.dataset_dataframes.append(dataframe)

        # Concatenate for backward compatibility
        self.dataframe = pd.concat(self.dataset_dataframes, ignore_index=True)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.iloc[indices.tolist()]
            print(f"selected {self.max_samples} random samples out of {total}")

        # Extract messages list from dataframe
        messages_list = self.dataframe[self.messages_key].apply(convert_nested_value_to_list_recursive).tolist()

        # Extract tools list from dataframe
        if self.tools_key in self.dataframe.columns:
            tools_list = self.dataframe[self.tools_key].apply(convert_nested_value_to_list_recursive).tolist()
        else:
            tools_list = [None] * len(messages_list)

        # Extract enable_thinking list from dataframe
        if self.enable_thinking_key in self.dataframe.columns:
            enable_thinking_list = self.dataframe[self.enable_thinking_key].tolist()
        else:
            enable_thinking_list = [None] * len(messages_list)

        # Filter out invalid samples (where messages is NaN or not a list)
        valid_indices = []
        for idx, messages in enumerate(messages_list):
            if messages is not None and not (isinstance(messages, float) and pd.isna(messages)) and isinstance(messages, (list, tuple)):
                valid_indices.append(idx)

        if len(valid_indices) < len(messages_list):
            print(f"Filtering out {len(messages_list) - len(valid_indices)} invalid samples (NaN or non-list messages)")
            messages_list = [messages_list[i] for i in valid_indices]
            tools_list = [tools_list[i] if i < len(tools_list) else None for i in valid_indices]
            enable_thinking_list = [enable_thinking_list[i] if i < len(enable_thinking_list) else None for i in valid_indices]

        # Pre-pack samples if enabled
        if self.enable_sample_packing:
            print(f"Pre-packing samples with max_length={self.max_length}...")
            self._pre_pack_samples_per_dataset()
            total_packed = sum(len(samples) for samples in self.packed_samples) if isinstance(self.packed_samples, list) else len(self.packed_samples) if self.packed_samples else 0
            print(f"Pre-packing completed. Original samples: {len(messages_list)}, Packed samples: {total_packed}")
        else:
            # Store original samples without packing
            self.messages = messages_list
            self.tools = tools_list if self.tools_key in self.dataframe.columns else None
            self.enable_thinking = enable_thinking_list if self.enable_thinking_key in self.dataframe.columns else None
            self.packed_samples = None

    def _pre_pack_samples_per_dataset(self):
        """
        Pre-pack samples per dataset separately.
        Each dataset is processed, tokenized, and packed independently.
        Saves separate caches for tokenized (before packing) and packed (after packing) results.
        """
        cache_dir = os.path.expanduser("~/.cache/verl/packed_samples")
        os.makedirs(cache_dir, exist_ok=True)

        all_packed_samples = []

        # Process each dataset separately
        for dataset_idx, dataframe in enumerate(self.dataset_dataframes):
            # Extract data for this dataset
            messages_list = dataframe[self.messages_key].apply(convert_nested_value_to_list_recursive).tolist()

            if self.tools_key in dataframe.columns:
                tools_list = dataframe[self.tools_key].apply(convert_nested_value_to_list_recursive).tolist()
            else:
                tools_list = [None] * len(messages_list)

            if self.enable_thinking_key in dataframe.columns:
                enable_thinking_list = dataframe[self.enable_thinking_key].tolist()
            else:
                enable_thinking_list = [None] * len(messages_list)

            # Filter out invalid samples
            valid_indices = []
            for idx, messages in enumerate(messages_list):
                if messages is not None and not (isinstance(messages, float) and pd.isna(messages)) and isinstance(messages, (list, tuple)):
                    valid_indices.append(idx)

            if len(valid_indices) < len(messages_list):
                print(f"Dataset {dataset_idx}: Filtering out {len(messages_list) - len(valid_indices)} invalid samples")
                messages_list = [messages_list[i] for i in valid_indices]
                tools_list = [tools_list[i] if i < len(tools_list) else None for i in valid_indices]
                enable_thinking_list = [enable_thinking_list[i] if i < len(enable_thinking_list) else None for i in valid_indices]

            # Generate cache keys for this dataset
            dataset_file = self.parquet_files[dataset_idx]
            # Extract file name (without path and extension) for cache directory name
            file_name = os.path.basename(str(dataset_file))
            # Remove extension and sanitize for use in directory name
            file_name_base = os.path.splitext(file_name)[0]
            # Replace any problematic characters with underscores
            file_name_safe = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in file_name_base)
            # Use file path hash as additional identifier for uniqueness
            file_hash = hashlib.md5(str(dataset_file.split("@")[-1]).encode()).hexdigest()[:16]

            tokenized_cache_key = self._generate_cache_key(messages_list, suffix="tokenized")
            packed_cache_key = self._generate_cache_key(messages_list, suffix="packed")

            tokenized_cache_dir = os.path.join(cache_dir, f"{file_name_safe}_{file_hash}_tokenized_{tokenized_cache_key}")
            packed_cache_dir = os.path.join(cache_dir, f"{file_name_safe}_{file_hash}_packed_{packed_cache_key}")

            import torch.distributed as dist
            # Singleton pattern: only rank 0 processes data
            if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
                process_data = True
            else:
                process_data = False
            # Check if packed cache exists (only rank 0 should process)
            if process_data and (not os.path.exists(packed_cache_dir) or not os.path.isdir(packed_cache_dir)):
                print(f"Rank 0: Dataset {dataset_idx}: Processing and packing samples...")

                # Check if tokenized cache exists
                if os.path.exists(tokenized_cache_dir) and os.path.isdir(tokenized_cache_dir):
                    print(f"Rank 0: Dataset {dataset_idx}: Loading cached tokenized samples from {tokenized_cache_dir}")
                    processed_samples = self._load_tokenized_samples_from_datasets(tokenized_cache_dir)
                    print(f"Rank 0: Dataset {dataset_idx}: Loaded {len(processed_samples)} tokenized samples from cache")
                else:
                    # Step 1: Tokenize samples
                    print(f"Rank 0: Dataset {dataset_idx}: Tokenizing {len(messages_list)} samples...")
                    processed_samples = self._process_samples_parallel(
                        messages_list, tools_list, enable_thinking_list
                    )

                    if len(processed_samples) == 0:
                        print(f"Rank 0: Dataset {dataset_idx}: Warning - No valid samples after processing")
                        continue

                    # Save tokenized cache
                    self._save_tokenized_samples_to_datasets(processed_samples, tokenized_cache_dir)
                    print(f"Rank 0: Dataset {dataset_idx}: Saved {len(processed_samples)} tokenized samples to {tokenized_cache_dir}")

                # Step 2: Greedy packing
                print(f"Rank 0: Dataset {dataset_idx}: Packing {len(processed_samples)} samples...")
                packed_samples = self._greedy_pack_samples(processed_samples)
                print(f"Rank 0: Dataset {dataset_idx}: Packed {len(processed_samples)} samples into {len(packed_samples)} packed sequences")

                # Step 4: Save packed cache
                self._save_packed_samples_to_datasets(packed_samples, packed_cache_dir)
                print(f"Rank 0: Dataset {dataset_idx}: Saved {len(packed_samples)} packed samples to {packed_cache_dir}")

            # Other ranks wait and load
            if dist.is_initialized():
                rank = dist.get_rank()
                dist.barrier()  # Wait for rank 0 to complete
            else:
                rank = 0

            print(f"Rank {rank}: Dataset {dataset_idx}: Loading packed samples from cache...")
            if not os.path.exists(packed_cache_dir) or not os.path.isdir(packed_cache_dir):
                raise FileNotFoundError(
                    f"Rank {rank}: Dataset {dataset_idx}: Cache directory {packed_cache_dir} not found. "
                    "Rank 0 should have created it. Check if rank 0 completed successfully."
                )
            packed_samples = self._load_packed_samples_from_datasets(packed_cache_dir)
            print(f"Rank {rank}: Dataset {dataset_idx}: Loaded {len(packed_samples)} packed sequences from full dataset")
            
            # Apply sampling ratio AFTER loading packed samples (not before tokenization)
            # This ensures cache can be reused regardless of sampling ratio
            dataset_ratio = self.dataset_ratios[dataset_idx]
            if dataset_ratio != 1.0:
                orig_len = len(packed_samples)
                packed_samples = self._sample_packed_dataset(packed_samples, dataset_ratio)
                print(f"Rank {rank}: Dataset {dataset_idx}: Sampled {len(packed_samples)} sequences from {orig_len} (ratio={dataset_ratio})")
            
            # all_packed_samples.extend(packed_samples) # NOTE: 这个会先把packed_samples转换为list，消耗大量时间
            all_packed_samples.append(packed_samples)

        # 快速合并（只合并元数据，不读取实际数据)
        if len(all_packed_samples) > 1:
            self.packed_samples = concatenate_datasets(all_packed_samples)
        else:
            self.packed_samples = all_packed_samples[0] if all_packed_samples else None

    def _save_tokenized_samples_to_datasets(self, processed_samples, cache_dir):
        """
        Save tokenized samples (before packing) to disk using datasets library.
        """
        # Convert processed samples to list of dicts with numpy arrays
        data_list = []
        for sample in processed_samples:
            data_dict = {
                'input_ids': sample['input_ids'].cpu().numpy(),
                'attention_mask': sample['attention_mask'].cpu().numpy(),
                'loss_mask': sample['loss_mask'].cpu().numpy(),
                'valid_length': sample['valid_length'],
            }
            data_list.append(data_dict)

        # Create HuggingFace Dataset from list
        hf_dataset = HFDataset.from_list(data_list)

        # Save to disk
        hf_dataset.save_to_disk(cache_dir)

    def _load_tokenized_samples_from_datasets(self, cache_dir):
        """
        Load tokenized samples (before packing) from disk using datasets library.
        """
        # Load from disk
        hf_dataset = load_from_disk(cache_dir)

        # Convert back to list of dicts with PyTorch tensors
        processed_samples = []
        for item in hf_dataset:
            input_ids = item['input_ids']
            loss_mask = item['loss_mask']

            # Convert to tensor
            if isinstance(input_ids, np.ndarray):
                processed_sample = {
                    'input_ids': torch.from_numpy(input_ids).long(),
                    'loss_mask': torch.from_numpy(loss_mask).bool(),
                }
            else:
                processed_sample = {
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'loss_mask': torch.tensor(loss_mask, dtype=torch.bool),
                }
            processed_samples.append(processed_sample)

        return processed_samples

    def _save_packed_samples_to_datasets(self, packed_samples, cache_dir):
        """
        Save packed samples to disk using datasets library.
        Converts PyTorch tensors to numpy arrays for storage.
        """
        # Convert packed samples to list of dicts with numpy arrays
        data_list = []
        for sample in tqdm(packed_samples):
            data_dict = {
                'input_ids': sample['input_ids'].cpu().numpy(),
                'position_ids': sample['position_ids'].cpu().numpy(),
                'loss_mask': sample['loss_mask'].cpu().numpy(),
            }
            data_list.append(data_dict)

        # Create HuggingFace Dataset from list
        hf_dataset = HFDataset.from_list(data_list)

        # Save to disk
        hf_dataset.save_to_disk(cache_dir)

    def _load_packed_samples_from_datasets(self, cache_dir):
        """
        Load packed samples from disk using datasets library.
        Returns list of dicts with numpy arrays (conversion to tensor happens in __getitem__).
        """
        # Load from disk with memory mapping for faster loading
        packed_samples = load_from_disk(cache_dir, keep_in_memory=False)
        return packed_samples

    def _sample_packed_dataset(self, packed_samples, dataset_ratio):
        """
        Sample packed dataset according to the given ratio.
        This is done AFTER loading the full packed samples to ensure cache reusability.
        
        Args:
            packed_samples: HuggingFace Dataset containing packed samples
            dataset_ratio: Sampling ratio (can be > 1.0 for replication)
        
        Returns:
            Sampled HuggingFace Dataset
        """
        full_count = int(dataset_ratio)
        frac_ratio = dataset_ratio - full_count
        orig_len = len(packed_samples)
        
        sampled_datasets = []
        
        # Integer part: replicate the full dataset n times if needed
        for _ in range(full_count):
            sampled_datasets.append(packed_samples)
        
        # Fractional part: sample a portion of the data without replacement
        if frac_ratio > 0 and orig_len > 0:
            sample_size = int(round(orig_len * frac_ratio))
            if sample_size > 0:
                sample_indices = np.random.choice(orig_len, size=sample_size, replace=False)
                # Use HuggingFace Dataset's select method for efficient indexing
                sampled_datasets.append(packed_samples.select(sample_indices.tolist()))
        
        # Combine datasets (if none of the above applied, just return original)
        if len(sampled_datasets) == 0:
            return packed_samples
        elif len(sampled_datasets) == 1:
            return sampled_datasets[0]
        else:
            return concatenate_datasets(sampled_datasets)

    def _generate_cache_key(self, messages_list, suffix=""):
        """Generate a unique cache key based on data content and configuration."""
        import json
        # Create a hash from data content and relevant config
        cache_data = {
            'messages_hash': hashlib.md5(str(messages_list).encode()).hexdigest()[:16],
            'max_length': self.max_length,
            'tokenizer_name': getattr(self.tokenizer, 'name_or_path', 'unknown'),
            'pad_mode': self.pad_mode,
            'truncation': self.truncation,
            'eos_token_id': self.eos_token_id,
            'pad_token_id': self.pad_token_id,
        }
        if suffix:
            cache_data['suffix'] = suffix
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _process_samples_parallel(self, messages_list, tools_list, enable_thinking_list):
        """
        Process samples in parallel using Python multiprocessing.
        Tokenizes samples first, then stores the tokenized results as datasets.
        """
        from multiprocessing import Pool

        # Get tokenizer info for subprocess reconstruction
        tokenizer_name = getattr(self.tokenizer, 'name_or_path', None)
        tokenizer_config = {
            'name_or_path': tokenizer_name,
        }

        # Prepare arguments for multiprocessing
        num_proc = min(128, len(messages_list), os.cpu_count() or 1)
        # num_proc = 0  # Set to 0 to disable multiprocessing (for debugging with pdb)
        print(f"Using {num_proc} processes for parallel tokenization...")

        # Prepare arguments list
        args_list = []
        for idx, (messages, tools, enable_thinking) in enumerate(zip(messages_list, tools_list, enable_thinking_list)):
            args_list.append((
                idx,
                messages,
                tools,
                enable_thinking,
                tokenizer_config,
                self.apply_chat_template_kwargs,
            ))

        # Process samples
        if num_proc == 0:
            # Process sequentially in main process (allows pdb debugging)
            results = []
            for args in tqdm(args_list, desc="Tokenizing samples", unit="sample"):
                results.append(_process_single_sample_multiprocess_worker(args))
        else:
            # Process in parallel using multiprocessing.Pool with tqdm progress bar
            with Pool(processes=num_proc) as pool:
                # Use imap_unordered for better performance, then sort by idx
                results_iter = pool.imap_unordered(_process_single_sample_multiprocess_worker, args_list)
                results = list(tqdm(results_iter, total=len(args_list), desc="Tokenizing samples", unit="sample"))

            # Sort results by idx to maintain order
            results.sort(key=lambda x: x[0])

        # Collect results and convert to list of processed samples
        processed_samples = []
        
        for idx, result in results:
            if result is not None:
                # Convert numpy arrays back to tensors
                processed_samples.append({
                    'input_ids': torch.from_numpy(result['input_ids']).long(),
                    'attention_mask': torch.from_numpy(result['attention_mask']).bool(),
                    'loss_mask': torch.from_numpy(result['loss_mask']).bool(),
                    'valid_length': result['valid_length'],
                })

        print(f"Successfully processed {len(processed_samples)} out of {len(messages_list)} samples")

        return processed_samples


    def _greedy_pack_samples(self, processed_samples):
        """Greedily pack processed samples into longer sequences."""
        packed_samples = []
        current_pack = []
        current_length = 0

        # Sort by length for better packing efficiency
        processed_samples.sort(key=lambda x: len(x['input_ids']), reverse=True)

        for sample in processed_samples:
            sample_len = len(sample['input_ids'])

            if current_length + sample_len <= self.max_length:
                current_pack.append(sample)
                current_length += sample_len
            else:
                # Pack current group and start new one
                if current_pack:
                    packed_samples.append(self._concat_samples_for_packing(current_pack))
                current_pack = [sample]
                current_length = sample_len

        # Handle last pack
        if current_pack:
            packed_samples.append(self._concat_samples_for_packing(current_pack))

        return packed_samples

    def _process_single_sample(self, messages, tools, enable_thinking, apply_padding=True):
        """
        Process a single sample to get tokenized data.
        Similar to __getitem__ but returns dict without padding (if apply_padding=False).
        """
        return self._process_single_sample_impl(
            messages, tools, enable_thinking, self.tokenizer, apply_padding
        )

    def _process_single_sample_impl(self, messages, tools, enable_thinking, tokenizer, apply_padding=True):
        """
        Internal implementation of single sample processing.
        Allows passing tokenizer explicitly for subprocess compatibility.
        """

        # Get the full conversation tokens
        try:
            full_tokens = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                **self.apply_chat_template_kwargs,
            )
        except Exception as e:
            logging.warning(f"Error applying chat template: {e}")
            return None

        # Track concatenated tokens
        concat_tokens = []
        concat_loss_mask = []
        concat_attention_mask = []

        i = 0
        while i < len(messages):
            cur_messages = messages[i]
            if cur_messages["role"] == "assistant":
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, i, i + 1, is_assistant=True, enable_thinking=enable_thinking, tools=tools
                )
                i += 1
            elif cur_messages["role"] == "tool":
                st = i
                ed = i + 1
                while ed < len(messages) and messages[ed]["role"] == "tool":
                    ed += 1
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, st, ed, enable_thinking=enable_thinking, tools=tools
                )
                i = ed
            elif cur_messages["role"] in ["user", "system"]:
                if cur_messages["role"] == "system" and i != 0:
                    raise ValueError("System message should be the first message")
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, i, i + 1, enable_thinking=enable_thinking, tools=tools
                )
                i += 1
            else:
                raise ValueError(f"Unknown role: {cur_messages['role']}")

            # Override loss mask if present
            override_loss_mask = cur_messages.get("loss_mask", None)
            if override_loss_mask is not None:
                if isinstance(override_loss_mask, np.ndarray):
                    override_loss_mask = override_loss_mask.item()
                assert isinstance(override_loss_mask, int), f"loss_mask should be int, got {type(override_loss_mask)}"
                assert override_loss_mask in [0, 1], f"loss_mask should be 0 or 1, got {override_loss_mask}"
                loss_mask = [override_loss_mask] * len(tokens)

            concat_tokens.extend(tokens)
            concat_loss_mask.extend(loss_mask)
            concat_attention_mask.extend(attention_mask)

        # Validate and convert tokens
        input_ids, loss_mask, attention_mask = self._validate_and_convert_tokens(
            full_tokens[0], concat_tokens, concat_loss_mask, concat_attention_mask
        )

        valid_length = attention_mask.sum().item()

        if not apply_padding:
            # Return without padding for packing
            return {
                'input_ids': input_ids[:valid_length],
                'attention_mask': attention_mask[:valid_length],
                'loss_mask': loss_mask[:valid_length],
                'valid_length': valid_length,
            }

        # Apply padding/truncation if needed
        sequence_length = input_ids.shape[0]
        if self.pad_mode == DatasetPadMode.RIGHT:
            if sequence_length < self.max_length:
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                padded_input_ids = torch.full((self.max_length - sequence_length,), pad_token_id, dtype=input_ids.dtype)
                padded_attention_mask = torch.zeros((self.max_length - sequence_length,), dtype=attention_mask.dtype)
                padded_loss_mask = torch.zeros((self.max_length - sequence_length,), dtype=loss_mask.dtype)

                input_ids = torch.cat((input_ids, padded_input_ids))
                attention_mask = torch.cat((attention_mask, padded_attention_mask))
                loss_mask = torch.cat((loss_mask, padded_loss_mask))
            elif sequence_length > self.max_length:
                if self.truncation == "left":
                    input_ids = input_ids[-self.max_length:]
                    attention_mask = attention_mask[-self.max_length:]
                    loss_mask = loss_mask[-self.max_length:]
                elif self.truncation == "right":
                    input_ids = input_ids[:self.max_length]
                    attention_mask = attention_mask[:self.max_length]
                    loss_mask = loss_mask[:self.max_length]
                elif self.truncation == "error":
                    raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")

            position_ids = torch.arange(len(input_ids), dtype=torch.long)
            position_ids = position_ids * attention_mask

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
        elif self.pad_mode == DatasetPadMode.NO_PADDING:
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                loss_mask = loss_mask[:self.max_length]
            position_ids = torch.arange(len(input_ids), dtype=torch.long)
            return {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
        else:
            raise ValueError(f"Unknown pad mode {self.pad_mode}")

    def _concat_samples_for_packing(self, samples):
        """
        Concatenate multiple samples for pre-packing.
        Uses position_ids to mark dialogue boundaries (position_id resets to 0 for each dialogue).
        Flash Attention will automatically use varlen mode to ensure dialogues don't see each other.
        """
        input_ids_list = []
        loss_mask_list = []
        position_ids_list = []

        for sample in samples:
            input_ids = sample['input_ids']
            loss_mask = sample['loss_mask']

            # Append sample content
            input_ids_list.append(input_ids)
            loss_mask_list.append(loss_mask)
            
            # Position IDs: reset to 0 for each dialogue (this marks dialogue boundaries)
            position_ids_list.append(torch.arange(len(input_ids), dtype=torch.long))

        # Concatenate all parts
        concat_input_ids = torch.cat(input_ids_list)
        concat_loss_mask = torch.cat(loss_mask_list)
        concat_position_ids = torch.cat(position_ids_list)

        total_len = len(concat_input_ids)

        # Pad to max_length if needed
        if total_len < self.max_length:
            pad_len = self.max_length - total_len
            concat_input_ids = torch.cat([
                concat_input_ids,
                torch.full((pad_len,), self.pad_token_id, dtype=concat_input_ids.dtype)
            ])
            concat_loss_mask = torch.cat([
                concat_loss_mask,
                torch.zeros(pad_len, dtype=concat_loss_mask.dtype)
            ])
            concat_position_ids = torch.cat([
                concat_position_ids,
                torch.range(0, pad_len - 1, dtype=torch.long)
            ]) # NOTE: 假装当成个序列(为了让flash attention varlen模式生效)，但实际loss mask是0

        # Truncate if exceeds max_length
        elif total_len > self.max_length:
            print(f"Truncating from {total_len} to {self.max_length}")
            # 截取最后self.max_length个token
            concat_input_ids = concat_input_ids[-self.max_length:]
            concat_loss_mask = concat_loss_mask[-self.max_length:]
            concat_position_ids = concat_position_ids[-self.max_length:]
            # 如果截断发生在首个样本内部，需要把该样本的position_id重新从0开始
            # 只调整首个样本（直到下一个position_id为0的位置），避免影响后续样本的对齐
            if concat_position_ids.numel() > 0:
                first_zeros = (concat_position_ids == 0).nonzero(as_tuple=True)[0]
                # 当截断落在样本中间时，首token的position_id会大于0
                # 需要仅对首个样本片段进行平移
                offset = concat_position_ids[0].item()
                if offset > 0:
                    if first_zeros.numel() == 0:
                        concat_position_ids = concat_position_ids - offset
                    else:
                        first_boundary = first_zeros[0].item()
                        if first_boundary > 0:
                            concat_position_ids[:first_boundary] = concat_position_ids[:first_boundary] - offset

        return {
            'input_ids': concat_input_ids,
            'position_ids': concat_position_ids,
            'loss_mask': concat_loss_mask,
        }

    def __len__(self):
        if self.enable_sample_packing and self.packed_samples is not None:
            return len(self.packed_samples)
        return len(self.messages)

    def _process_message_tokens(
        self,
        messages: list[dict[str, Any]],
        start_idx: int,
        end_idx: int,
        is_assistant: bool = False,
        enable_thinking: Optional[bool] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Process tokens for a single message or a group of messages.

        Args:
            messages: List of message dictionaries
            start_idx: Start index in messages list
            end_idx: End index in messages list
            is_assistant: Whether this is an assistant message
            enable_thinking: Whether to enable thinking mode

        Returns:
            Tuple of (tokens, loss_mask, attention_mask)
        """
        if start_idx > 0:
            prev_applied_text = self.tokenizer.apply_chat_template(
                messages[:start_idx],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                tools=tools,
                **self.apply_chat_template_kwargs,
            )
            if is_assistant:
                prev_applied_text_w_generation_prompt = self.tokenizer.apply_chat_template(
                    messages[:start_idx],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                    tools=tools,
                    **self.apply_chat_template_kwargs,
                )

        else:
            prev_applied_text = ""

        cur_applied_text = self.tokenizer.apply_chat_template(
            messages[:end_idx],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
            tools=tools,
            **self.apply_chat_template_kwargs,
        )
        # Get tokens for the current message only
        if is_assistant:
            generation_prompt_text = prev_applied_text_w_generation_prompt[len(prev_applied_text) :]
            generation_prompt_tokens = self.tokenizer.encode(
                generation_prompt_text,
                add_special_tokens=False,
            )
            _message_tokens = self.tokenizer.encode(
                cur_applied_text[len(prev_applied_text_w_generation_prompt) :],
                add_special_tokens=False,
            )
            message_tokens = generation_prompt_tokens + _message_tokens
            loss_mask = [0] * (len(generation_prompt_tokens)) + [1] * (
                len(message_tokens) - len(generation_prompt_tokens)
            )
        else:
            message_tokens = self.tokenizer.encode(
                cur_applied_text[len(prev_applied_text) :],
                add_special_tokens=False,
            )
            loss_mask = [0] * len(message_tokens)

        attention_mask = [1] * len(message_tokens)

        return message_tokens, loss_mask, attention_mask

    def _validate_and_convert_tokens(
        self,
        full_tokens: torch.Tensor,
        concat_tokens: list[int],
        concat_loss_mask: list[int],
        concat_attention_mask: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Validate tokenization and convert to tensors.

        Args:
            full_tokens: Full conversation tokens
            concat_tokens: Concatenated tokens
            concat_loss_mask: Concatenated loss mask
            concat_attention_mask: Concatenated attention mask

        Returns:
            Tuple of (input_ids, loss_mask, attention_mask) as tensors
        """
        full_tokens_list = full_tokens.tolist()

        if len(concat_tokens) != len(full_tokens_list) or not all(
            a == b for a, b in zip(concat_tokens, full_tokens_list, strict=True)
        ):
            logging.warning(
                f"Token mismatch detected! Full tokenization length: {len(full_tokens_list)}, Concatenated tokens "
                f"length: {len(concat_tokens)}. Using concatenated version."
                # f"full tokens text: {self.tokenizer.decode(full_tokens_list)}"
                # f"concat tokens text: {self.tokenizer.decode(concat_tokens)}"
            )
            return (
                torch.tensor(concat_tokens, dtype=torch.long),
                torch.tensor(concat_loss_mask, dtype=torch.bool),
                torch.tensor(concat_attention_mask, dtype=torch.bool),
            )

        return (
            full_tokens,
            torch.tensor(concat_loss_mask, dtype=torch.bool),
            torch.tensor(concat_attention_mask, dtype=torch.bool),
        )

    def __getitem__(self, item):
        # If pre-packed, convert numpy arrays to tensors on-the-fly
        if self.enable_sample_packing and self.packed_samples is not None:
            sample = self.packed_samples[item]

            # Convert numpy arrays to tensors when accessed
            if isinstance(sample['input_ids'], np.ndarray):
                return {
                    'input_ids': torch.from_numpy(sample['input_ids']).long(),
                    'position_ids': torch.from_numpy(sample['position_ids']).long(),
                    'loss_mask': torch.from_numpy(sample['loss_mask']).bool(),
                }
            else:
                # Fallback for lists
                return {
                    'input_ids': torch.tensor(sample['input_ids'], dtype=torch.long),
                    'position_ids': torch.tensor(sample['position_ids'], dtype=torch.long),
                    'loss_mask': torch.tensor(sample['loss_mask'], dtype=torch.bool),
                }

        # Otherwise, process on-the-fly (original behavior)
        tokenizer = self.tokenizer
        messages = self.messages[item]
        # Handle tools: treat NaN (scalar or in list like [nan]) as None
        tools = self.tools[item] if self.tools is not None and pd.isna(self.tools[item]) is False else None
        enable_thinking = self.enable_thinking[item] if self.enable_thinking is not None else None

        # First, get the full conversation tokens
        try:
            full_tokens = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                **self.apply_chat_template_kwargs,
            )
        except Exception as e:
            logging.error(
                f"Error applying chat template: {e}\nMessages: {messages}\nTools: {tools}\nEnable thinking: "
                f"{enable_thinking}"
            )
            raise

        # Track concatenated tokens for validation
        concat_tokens = []
        concat_loss_mask = []
        concat_attention_mask = []

        i = 0
        while i < len(messages):
            cur_messages = messages[i]
            if cur_messages["role"] == "assistant":
                # Process assistant message
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, i, i + 1, is_assistant=True, enable_thinking=enable_thinking, tools=tools
                )
                i += 1
            elif cur_messages["role"] == "tool":
                # Process consecutive tool messages
                st = i
                ed = i + 1
                while ed < len(messages) and messages[ed]["role"] == "tool":
                    ed += 1
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, st, ed, enable_thinking=enable_thinking, tools=tools
                )
                i = ed
            elif cur_messages["role"] in ["user", "system"]:
                # Process user or system message
                if cur_messages["role"] == "system" and i != 0:
                    raise ValueError("System message should be the first message")
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, i, i + 1, enable_thinking=enable_thinking, tools=tools
                )
                i += 1
            else:
                raise ValueError(f"Unknown role: {cur_messages['role']}")

            # override loss mask with mask in the dataset to handle multi-turn conversation
            override_loss_mask = cur_messages.get("loss_mask", None)
            if override_loss_mask is not None:
                if isinstance(override_loss_mask, np.ndarray):
                    override_loss_mask = override_loss_mask.item()
                assert isinstance(override_loss_mask, int), f"loss_mask should be int, got {type(override_loss_mask)}"
                assert override_loss_mask in [0, 1], f"loss_mask should be 0 or 1, got {override_loss_mask}"
                loss_mask = [override_loss_mask] * len(tokens)

            concat_tokens.extend(tokens)
            concat_loss_mask.extend(loss_mask)
            concat_attention_mask.extend(attention_mask)

        # Validate and convert tokens
        input_ids, loss_mask, attention_mask = self._validate_and_convert_tokens(
            full_tokens[0], concat_tokens, concat_loss_mask, concat_attention_mask
        )

        # encode prompt
        if messages[0]["role"] == "system":
            assert messages[1]["role"] == "user"
            assert messages[2]["role"] == "assistant"
        elif messages[0]["role"] == "user":
            assert messages[1]["role"] == "assistant"
        else:
            raise ValueError(f"Unknown role: {messages[0]['role']}")

        sequence_length = input_ids.shape[0]
        # Handle sequence length
        if self.pad_mode == DatasetPadMode.RIGHT:
            if sequence_length < self.max_length:
                # Pad sequences
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                padded_input_ids = torch.full((self.max_length - sequence_length,), pad_token_id, dtype=input_ids.dtype)
                padded_attention_mask = torch.zeros((self.max_length - sequence_length,), dtype=attention_mask.dtype)
                padded_loss_mask = torch.zeros((self.max_length - sequence_length,), dtype=loss_mask.dtype)

                input_ids = torch.cat((input_ids, padded_input_ids))
                attention_mask = torch.cat((attention_mask, padded_attention_mask))
                loss_mask = torch.cat((loss_mask, padded_loss_mask))
            elif sequence_length > self.max_length:
                if self.truncation == "left":
                    if len(input_ids) > self.max_length:
                        print(f"Truncating left from {len(input_ids)} to {self.max_length}")
                    input_ids = input_ids[-self.max_length :]
                    attention_mask = attention_mask[-self.max_length :]
                    loss_mask = loss_mask[-self.max_length :]
                elif self.truncation == "right":
                    input_ids = input_ids[: self.max_length]
                    attention_mask = attention_mask[: self.max_length]
                    loss_mask = loss_mask[: self.max_length]
                elif self.truncation == "error":
                    raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
                else:
                    raise ValueError(f"Unknown truncation method {self.truncation}")

            # Create position IDs
            position_ids = torch.arange(len(input_ids), dtype=torch.long)
            # Zero out position IDs for padding
            position_ids = position_ids * attention_mask

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
        elif self.pad_mode == DatasetPadMode.NO_PADDING:
            # truncate input_ids if it is longer than max_length
            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            # create position IDs
            position_ids = torch.arange(len(input_ids), dtype=torch.long)
            # return nested tensor with out padding
            return {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
        else:
            raise ValueError(f"Unknown pad mode {self.pad_mode}")
