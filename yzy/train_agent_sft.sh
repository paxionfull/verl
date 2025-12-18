nproc_per_node=8
# save_path="./logs/agents-sft-qwen2.5-3b-instruct-nxbdata-turnmean-samplepacking-3e-multiturnft3e"
# save_path="./logs/qwen2.5-3b-instruct-enhance"
# save_path="./logs/qwen2.5-3b-instruct-enhance-taubenchoverfitacebenchformat"
save_path="./logs/Qwen3-4B-Instruct-2507-funreason"
# save_path="./logs/agents-sft-qwen2.5-coder-3b-instruct-nxbdata-turnmean-samplepacking-multistepturnft1e"
# save_path="./logs/agents-sft-qwen2.5-1.5b-distill-multiturnft1e"

lr=3e-5

# 定义多个训练文件路径（使用数组，换行更清晰）
train_files=(
    # public - single turn
    # "1.0@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/public/single_turn/xlam-function-calling-60k_openaiformat.jsonl"
    # "0.5@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/public/single_turn/xlam-function-calling-60k-acebenchformat_parallel_31k.jsonl"
    # "0.5@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/public/single_turn/xlam-function-calling-60k-acebenchformat_simple_28k.jsonl"
    # # agent simulator - single turn
    # "1.0@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/agent_simulator/single_turn/singleturn_nxb_4.5k.jsonl"
    # "1.0@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/agent_simulator/single_turn/singleturn_nxb_acebenchformat_4.5k.jsonl"
    # # agent simulator - multi step
    # "1.0@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/agent_simulator/multi_step/multi_step_nxb_2.5k.jsonl"
    # "0.5@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/agent_simulator/multi_step/multi_step_nxb_acebenchformat_13k.jsonl"
    # agent simulator - multi turn
    # "1.0@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/agent_simulator/multi_turn/multi_turn_nxb_1.7k_filtered.jsonl"
    # "0.5@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/agent_simulator/multi_turn/multi_turn_nxb_acebenchformat_20k_filtered.jsonl"
    # "0.5@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/agent_simulator/multi_turn/multi_turn_wguideline_2_extra_valid_processed_acebenchformat_filtered_12k.jsonl"
    # "1.0@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/agent_simulator/multi_turn/multi_turn_wguideline_3_gptoss_valid_processed_acebenchformat.jsonl"
    # "1.0@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/agent_simulator/multi_turn/acebench_multiturn_gptoss_acebenchformat.jsonl" 
    # "1.0@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/agent_simulator/multi_turn/taubench_gptoss120b_retail_acebenchformat.jsonl"
    # "1.0@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/agent_simulator/multi_turn/taubench_gptoss120b_airline_acebenchformat.jsonl"
    # "1.0@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/agent_simulator/multi_turn/taubench_gptoss120b_retail_openaiformat.jsonl"
    # "1.0@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/agent_simulator/multi_turn/taubench_gptoss120b_airline_openaiformat.jsonl"
    # "1.0@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/nxb/agent_data_2/hotpotqa_1_to_1200_subtasks_conversations.jsonl"
    "1.0@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/public/multi_turn/funreason_mt_bfcl_multi_turn.jsonl"
)

# # enhance
# train_files=(
#     # public - single turn
#     "1.0@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/public/single_turn/xlam-function-calling-60k_openaiformat.jsonl"
#     "0.5@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/public/single_turn/xlam-function-calling-60k-acebenchformat_parallel_31k.jsonl"
#     "0.5@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/public/single_turn/xlam-function-calling-60k-acebenchformat_simple_28k.jsonl"
#     "1.0@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/public/single_turn/longalign_10k.jsonl"
#     "1.0@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/public/single_turn/qwen3_long_reasoning_24k.jsonl"
#     # public - multi turn
#     "1.0@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/public/multi_turn/systemchat_v1.1_20k.jsonl"
#     "1.0@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/public/multi_turn/hermes-function-calling-v1_11k.jsonl"
#     "1.0@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/public/multi_turn/hermes_reasoning_tool_use_51k.jsonl"
#     "1.0@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/public/multi_turn/toolace_11k.jsonl"
#     # agent simulator - single turn
#     "1.0@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/agent_simulator/single_turn/singleturn_nxb_4.5k.jsonl"
#     "0.5@/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/agent_simulator/single_turn/singleturn_nxb_acebenchformat_4.5k.jsonl"
#     # agent simulator - multi step
#     # agent simulator - multi turn
# )



val_files=(
    "/mnt/public/algm/yzy/dataset/agent/private/agent_data_2025/agent_simulator/single_turn/singleturn_nxb_4.5k.jsonl"
)

# 转换为Hydra需要的列表格式
train_files_str="[$(printf "'%s'," "${train_files[@]}" | sed 's/,$//')]"

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.loss_agg_mode=turn-mean \
    data.truncation=left \
    data.train_files="$train_files_str" \
    data.val_files="$train_files_str" \
    data.multiturn.enable=true \
    data.multiturn.messages_key=conversations \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=16384 \
    +data.enable_sample_packing=true \
    +data.enable_sample_packing_val=false \
    +model.tokenizer_path=/mnt/public/algm/models/Qwen/Qwen3-4B-Instruct-2507 \
    model.partial_pretrain=/mnt/public/algm/models/Qwen/Qwen3-4B-Instruct-2507 \
    model.strategy=fsdp \
    model.fsdp_config.model_dtype=bf16 \
    model.enable_gradient_checkpointing=True \
    model.fsdp_config.cpu_offload=True \
    model.fsdp_config.offload_params=True \
    optim.lr=$lr \
    trainer.default_local_dir=$save_path \
    trainer.project_name=agent-sft \
    trainer.experiment_name=agent-sft-qwen2.5-3b-instruct \
    trainer.total_epochs=3 \
    trainer.save_freq=50 \
    trainer.test_freq=100 \
    trainer.logger='["console","wandb"]' $@


    # model.partial_pretrain=/mnt/public/algm/yzy/train_repos/verl/logs/qwen2.5-3b-instruct-enhance/global_step_1290/hf \
    # model.partial_pretrain=/mnt/public/algm/models/Qwen2.5-3B-Instruct \
    # model.partial_pretrain=/mnt/public/algm/yzy/train_repos/verl/logs/agents-sft-qwen2.5-3b-instruct-nxbdata-nosingleturn-acebenchformatweight1.0-turnmean-samplepacking-3e/global_step_546/hf \
    # +model.tokenizer_path=/mnt/public/algm/models/Qwen/Qwen2.5-Coder-3B-Instruct \
    # model.partial_pretrain=/mnt/public/algm/models/Qwen/Qwen2.5-Coder-3B-Instruct \
    # +model.tokenizer_path=/mnt/public/algm/models/Qwen/Qwen3-4B-Instruct-2507 \
    # model.partial_pretrain=/mnt/public/algm/models/Qwen/Qwen3-4B-Instruct-2507 \