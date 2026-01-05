export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

# 修复终端设置
export TERM=xterm-256color
export LANG=en_US.UTF-8
stty sane 2>/dev/null || true

n_gpus_per_node=8
rollout_num_workers=8
max_prompt_length=5120
max_response_length=11264
ppo_max_token_len_per_gpu=16384

use_kl_loss=False
kl_loss_coef=0.001
kl_loss_type=low_var_kl
adv_estimator=grpo
n_rollout=4
use_dynamic_bsz=True
lr=1e-6
# model_path=/mnt/public/algm/models/Qwen2.5-3B-Instruct
model_path=/mnt/public/algm/models/Qwen3-4B-Instruct-2507
# model_path=/mnt/public/algm/models/Qwen2.5-0.5B-Instruct
gpu_memory_utilization=0.8
rollout_mode=async  # async rollout会使用agent loop manager
rollout_tp_size=1
rollout_backend=vllm

interaction_config_path=/mnt/public/algm/yzy/train_repos/verl/yzy/interaction_config/tau_interaction_config.yaml
agent_loop_type=tau_agent
tool_config_path=/mnt/public/algm/yzy/train_repos/verl/yzy/tool_config/tau_tool_config.yaml

export EXPERIMENT_NAME="qwen3_4b_inst_agentrl_tau1_retail_maxsteps30_rewardweightv1_fix"

export WITHLENGTH=0
export REFINEDREWARD=0
export COARSEREWARD=0
export STRICTMATCH=0
export CORRECTMAX1=0
export MAX1STEP30MAX3=0
export SCHEDULEREWARD=0
export SCHEDULELENGTH=0

train_files=(
    "1.0@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/public/rl/tau1_retail_train.jsonl"
)

val_files=(
    "1.0@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/public/rl/tau1_retail_dev.jsonl"
)

# 转换为Hydra需要的列表格式
train_files_str="[$(printf "'%s'," "${train_files[@]}" | sed 's/,$//')]"
val_files_str="[$(printf "'%s'," "${val_files[@]}" | sed 's/,$//')]"

# 是否保存rollout结果, 保存到./rollout_res目录下
save_rollout=true

PYTHONUNBUFFERED=1 SAVE_ROLLOUT=${save_rollout} python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=${adv_estimator} \
 reward_model.enable=False \
 data.train_files=${train_files_str} \
 data.val_files=${val_files_str} \
 data.train_batch_size=16 \
 data.max_prompt_length=${max_prompt_length} \
 data.max_response_length=${max_response_length} \
 data.filter_overlong_prompts=True \
 data.truncation='error' \
 data.custom_cls.path=/mnt/public/algm/yzy/train_repos/verl/verl/utils/dataset/agentrl_dataset.py \
 data.custom_cls.name=TauRLDataset \
 actor_rollout_ref.model.path=${model_path} \
 actor_rollout_ref.rollout.mode=${rollout_mode} \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
 actor_rollout_ref.actor.optim.lr=${lr} \
 actor_rollout_ref.actor.ppo_mini_batch_size=4 \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.rollout.name=${rollout_backend} \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp_size} \
 actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.rollout.n=${n_rollout} \
 actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
 actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
 actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
 actor_rollout_ref.rollout.agent.default_agent_loop=${agent_loop_type} \
 actor_rollout_ref.rollout.agent.num_workers=${rollout_num_workers} \
 actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
 actor_rollout_ref.rollout.response_length=${max_response_length} \
 +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=true \
 +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=hermes \
 trainer.logger='["console","wandb"]' \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=${n_gpus_per_node} \
 trainer.nnodes=1 \
 trainer.experiment_name=${EXPERIMENT_NAME} \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_epochs=5 2>&1 | tee verl_demo.log



#  actor_rollout_ref.rollout.multi_turn.tool_config_path=${tool_config_path} \
#  actor_rollout_ref.rollout.multi_turn.interaction_config_path=${interaction_config_path} \

#  actor_rollout_ref.rollout.trace.backend=weave \
#  actor_rollout_ref.rollout.trace.token2text=True \
#  actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker=8 \