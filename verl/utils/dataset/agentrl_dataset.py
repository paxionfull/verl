import copy
import logging
import os
import re
import traceback
from collections import defaultdict
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
import pandas as pd
import json


logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    assert len(data_list) > 1
    data_keys = list(data_list[0].keys())
    batch_dict = {}
    for data_key in data_keys:
        batch_values = []
        for data in data_list:
            batch_values.append(data[data_key])
        # batch_dict[data_key] = batch_values
        batch_dict[data_key] = np.array(batch_values, dtype=object)

    return batch_dict


class TauRLDataset(Dataset):
    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.config = config

        self._read_files()


    def _read_files(self):
        self.dataset_items = []
        self.dataset_ratios = []  # Store ratios for later sampling after packing

        for parquet_file in self.data_files:
            if "@" in parquet_file:
                dataset_ratio, parquet_file = float(parquet_file.split("@")[0]), parquet_file.split("@")[1]
            else:
                dataset_ratio = 1.0
            assert parquet_file.endswith(".jsonl")
            with open(parquet_file, "r") as f:
                items = [json.loads(line.strip()) for line in f]
            self.dataset_ratios.append(dataset_ratio)
            self.dataset_items.extend(items)


    def __len__(self):
        return len(self.dataset_items)


    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataset_items[item]
        output_dict = {
            "user_id": row_dict["user_id"],
            "instruction": row_dict["instruction"],
            "actions": row_dict["actions"],
            "outputs": row_dict["outputs"],
            "wiki": row_dict["wiki"],
            "task_index": row_dict["task_index"],
            "env_name": row_dict["env_name"],
        }

        return output_dict



if __name__ == "__main__":
    from omegaconf import OmegaConf

    data_files = [
        "1.0@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/public/rl/tau1_retail.jsonl",
        "1.0@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/public/rl/tau1_airline.jsonl",
    ]
    dataset = TauRLDataset(
        data_files=data_files,
        tokenizer=None,
        config=None,
    )

    from verl.trainer.main_ppo import create_rl_sampler
    from torchdata.stateful_dataloader import StatefulDataLoader

    data_config = OmegaConf.create({
        "sampler": None,
        "shuffle": True,
        "seed": 42,
    })
    train_sampler = create_rl_sampler(data_config, dataset)
    train_dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=4,  # 示例 batch_size
        num_workers=2,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=train_sampler,
    )

    for batch in train_dataloader:
        batch_dict = batch
        break

    from IPython import embed; embed()

