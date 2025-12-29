import json
import os
import sys

sys.path.append("/home/yzy/eval_repos/tau-bench")

import tau_bench

from tau_bench.envs.airline import AIRLINE_TASKS, AIRLINE_WIKI
from tau_bench.envs.retail import RETAIL_TASKS_TEST, RETAIL_WIKI, RETAIL_TASKS_TRAIN, RETAIL_TASKS_DEV

def write_jsonl(data, file_path):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def format_item(task_items, wiki, env_name):
    def format_action(actions):
        return [{"name": action.name, "kwargs": action.kwargs} for action in actions]
    output = []
    for idx, task_item in enumerate(task_items):
        output.append({
            "user_id": task_item.user_id,
            "instruction": task_item.instruction,
            "actions": format_action(task_item.actions),
            "outputs": task_item.outputs,
            "wiki": wiki,
            "task_index": idx,
            "env_name": env_name,
        })
    return output


airline_trainset = format_item(AIRLINE_TASKS, AIRLINE_WIKI, "airline")
retail_trainset = format_item(RETAIL_TASKS_TEST, RETAIL_WIKI, "retail")
retail_trainset_train = format_item(RETAIL_TASKS_TRAIN, RETAIL_WIKI, "retail")
retail_trainset_dev = format_item(RETAIL_TASKS_DEV, RETAIL_WIKI, "retail")

# write_jsonl(retail_trainset, "/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/public/rl/tau1_retail.jsonl")
# write_jsonl(airline_trainset, "/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/public/rl/tau1_airline.jsonl")
write_jsonl(retail_trainset_train, "/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/public/rl/tau1_retail_train.jsonl")
write_jsonl(retail_trainset_dev, "/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/public/rl/tau1_retail_dev.jsonl")