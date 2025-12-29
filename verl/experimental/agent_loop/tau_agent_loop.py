# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import asyncio
import copy
import json
import logging
import os
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import regex

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import add_generation_prompt_for_gpt_oss, format_gpt_oss_tool_response_manually
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.chat_template import initialize_system_prompt
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: Any,
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        # Extra fields for dynamic addition
        self.extra_fields: dict[str, Any] = {}


@ToolParser.register("qwen")
class QwenToolParser(ToolParser):
    """Tool parser specifically for Qwen models.

    Qwen models output tool calls in the format:
    <tool_call>
    {"name": "function_name", "arguments": {...}}
    </tool_call>

    Multiple tool calls can appear in sequence, and may end with <|im_end|>.
    """

    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)

        # Qwen uses <tool_call>...</tool_call> tags (similar to Hermes)
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"

        # Regex pattern to match tool calls
        # DOTALL flag allows . to match newlines
        self.tool_call_regex = regex.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", regex.DOTALL)

        # Qwen-specific tokens that might appear
        self.im_end_token = "<|im_end|>"

    @rollout_trace_op
    async def extract_tool_calls(self, responses_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        """Extract tool calls from Qwen model responses.

        Args:
            responses_ids: Token IDs from the model response

        Returns:
            Tuple of (remaining_text, list_of_function_calls)
        """
        loop = asyncio.get_running_loop()

        # Decode the response (keep special tokens to properly handle <|im_end|>)
        text = await loop.run_in_executor(
            None, lambda: self.tokenizer.decode(responses_ids, skip_special_tokens=True)
        )

        # Check if there are any tool calls
        if self.tool_call_start_token not in text or self.tool_call_end_token not in text:
            # No tool calls, return the text as-is (remove <|im_end|> if present)
            content = text.replace(self.im_end_token, "").strip()
            return content, []

        # Find all tool call matches
        matches = self.tool_call_regex.findall(text)
        function_calls = []

        for match in matches:
            try:
                # Parse the JSON inside the tool_call tags
                function_call = json.loads(match.strip())

                # Extract name and arguments
                name = function_call.get("name")
                arguments = function_call.get("arguments", {})

                if name:
                    # Convert arguments to JSON string if it's a dict
                    if isinstance(arguments, dict):
                        arguments_str = json.dumps(arguments, ensure_ascii=False)
                    else:
                        arguments_str = str(arguments)

                    function_calls.append(FunctionCall(name=name, arguments=arguments_str))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool call JSON: {match[:100]}... Error: {e}")
            except Exception as e:
                logger.error(f"Failed to decode tool call: {e}, match: {match[:100]}")

        # Remove tool calls from the text to get remaining content
        content = self.tool_call_regex.sub("", text)

        # Remove Qwen-specific tokens
        content = content.replace(self.im_end_token, "").strip()

        return content, function_calls


@register("tau_agent")
class TauAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level TauAgentLoop initialization")

        # Basic shared configs
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length

        # Initialize tool parser for extracting tool calls from model outputs
        # Determine which tool parser to use based on model type
        model_path = config.actor_rollout_ref.model.path.lower()
        if "qwen" in model_path:
            # Use Qwen-specific parser
            parser_format = "qwen"
            logger.info(f"Detected Qwen model ({model_path}), using QwenToolParser")
        else:
            # Use the configured format (default: hermes)
            parser_format = config.actor_rollout_ref.rollout.multi_turn.format

        cls.tool_parser = ToolParser.get_tool_parser(parser_format, cls.tokenizer)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """
        TauBench-style tool-calling loop integrated into VERL AgentLoop.

        - messages 初始化为 [system(wiki), user(obs)]
        - 主循环中使用 VERL 的 server_manager + tool_parser 调用 LLM 与 env 交互。
        """

        # 延迟导入 tau-bench 相关依赖，避免在未安装时影响其他功能
        from tau_bench.envs import get_env
        from tau_bench.types import RESPOND_ACTION_NAME
        from tau_bench.agents.tool_calling_agent import message_to_action

        # 1. 从样本中提取 tau-bench 所需的额外信息
        extra_info: dict[str, Any] = kwargs.get("extra_info", {}) or {}

        # 需要补的标签：env_name, task_index
        env_name = kwargs["env_name"]
        user_strategy = "llm"
        user_model = "random"  # 随机选择，避免rate limit
        user_provider = "openai"  # 环境变量中修改key和url
        task_split = "train"
        task_index = kwargs["task_index"]
        max_num_steps = 10
        wiki = kwargs["wiki"]

        if env_name is None or user_strategy is None or user_model is None or user_provider is None:
            raise ValueError(
                "TauAgentLoop.run expects extra_info to contain "
                "'env_name', 'user_strategy', 'user_model', 'user_provider'. "
                f"Got extra_info={extra_info}"
            )

        # 2. 创建 tau-bench 的 env，并初始化任务
        isolated_env = get_env(
            env_name,
            user_strategy=user_strategy,
            user_model=user_model,
            task_split=task_split,
            user_provider=user_provider,
            task_index=task_index,
        )
        tools = isolated_env.tools_info

        env_reset_res = isolated_env.reset(task_index=task_index)
        obs = env_reset_res.observation
        if hasattr(env_reset_res.info, "model_dump"):
            info = env_reset_res.info.model_dump()
        else:
            info = dict(getattr(env_reset_res, "info", {}))

        # 3. 初始化 messages：system(wiki) + user(observation)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": wiki},
            {"role": "user", "content": obs},
        ]

        # 4. Tau 风格的多步交互循环
        total_reward = 0.0

        # 为 VERL 构造 token 序列
        # prompt_ids: 只包含初始 prompt (system + 第一轮 user)
        initial_prompt_ids: list[int] = []
        all_response_ids: list[int] = []
        all_response_mask: list[int] = []
        all_response_logprobs: list[float] = []

        # metrics 可用于记录一些简单统计（总步数、总 reward 等）
        metrics: dict[str, Any] = {}

        for turn_idx in range(max_num_steps):
            # 4.1 将 messages 转成 prompt_ids（带 tools schema）
            if self.processor is not None:
                raw_prompt = await self.loop.run_in_executor(
                    None,
                    lambda: self.processor.apply_chat_template(
                        messages,
                        tools=tools,
                        add_generation_prompt=True,
                        tokenize=False,
                        **self.apply_chat_template_kwargs,
                    ),
                )
                model_inputs = self.processor(text=[raw_prompt], images=None, return_tensors="pt")
                prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
            else:
                prompt_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        messages,
                        tools=tools,
                        add_generation_prompt=True,
                        tokenize=True,
                        **self.apply_chat_template_kwargs,
                    ),
                )

            # 只在第一轮记录初始 prompt (system + 第一轮 user)
            if turn_idx == 0:
                initial_prompt_ids = prompt_ids

            # 4.2 调用 VERL 的 AsyncLLMServerManager 进行生成
            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=None,
                )

            # 添加 assistant 回复到 response_ids，mask 设为 1（需要监督）
            response_ids = output.token_ids
            all_response_ids.extend(response_ids)
            all_response_mask.extend([1] * len(response_ids))
            if output.log_probs:
                all_response_logprobs.extend(output.log_probs)

            # 4.3 使用 VERL 的 ToolParser 从 token_ids 中解析 tool_calls
            if hasattr(self, "tool_parser"):
                response_text, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
            else:
                # 退化为纯文本回复（不解析工具）
                response_text = await self.loop.run_in_executor(
                    None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
                )
                tool_calls = []
            
            # 构造 OpenAI 风格的 assistant message，兼容 tau-bench 的 message_to_action
            if tool_calls:
                tool_call_dicts = []
                for i, call in enumerate(tool_calls):
                    tool_call_dicts.append(
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            },
                        }
                    )
                assistant_message = {
                    "role": "assistant",
                    "content": response_text,
                    "tool_calls": tool_call_dicts,
                }
            else:
                assistant_message = {
                    "role": "assistant",
                    "content": response_text,
                }

            # 4.4 通过 tau-bench 的 message_to_action 将 message 映射为 Action
            action = message_to_action(assistant_message)

            # 4.5 与 tau-bench env 交互：env.step(action)
            with simple_timer("tool_calls", metrics):
                env_response = isolated_env.step(action)
            step_reward = getattr(env_response, "reward", 0.0)
            total_reward = step_reward  # 也可以按需要改为累加

            env_info = env_response.info
            if hasattr(env_info, "model_dump"):
                env_info = env_info.model_dump()
            info = {**info, **(env_info or {})}

            # 4.6 按 TauBench 方式更新 messages（tool 调用 vs 普通回复）
            # 同时将 user/tool 响应添加到 response_ids，mask 设为 0（不监督）
            if action.name != RESPOND_ACTION_NAME and tool_calls:
                # 只保留第一个 tool_call，仿照 tau-bench 的切片逻辑
                assistant_message["tool_calls"] = assistant_message["tool_calls"][:1]
                tc = assistant_message["tool_calls"][0]
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "content": env_response.observation,
                }
                messages.extend([assistant_message, tool_message])
                
                # 将 tool 响应 tokenize 并添加到 response_ids，mask 设为 0
                tool_messages = [tool_message]
                if self.processor is not None:
                    raw_tool_response = await self.loop.run_in_executor(
                        None,
                        lambda: self.processor.apply_chat_template(
                            tool_messages,
                            add_generation_prompt=False,
                            tokenize=False,
                            **self.apply_chat_template_kwargs,
                        ),
                    )
                    model_inputs = self.processor(text=[raw_tool_response], images=None, return_tensors="pt")
                    tool_response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
                else:
                    tool_response_ids = await self.loop.run_in_executor(
                        None,
                        lambda: self.tokenizer.apply_chat_template(
                            tool_messages,
                            add_generation_prompt=False,
                            tokenize=True,
                            **self.apply_chat_template_kwargs,
                        ),
                    )
                all_response_ids.extend(tool_response_ids)
                all_response_mask.extend([0] * len(tool_response_ids))
                if all_response_logprobs:
                    all_response_logprobs.extend([0.0] * len(tool_response_ids))
            else:
                user_message = {"role": "user", "content": env_response.observation}
                messages.extend([assistant_message, user_message])
                
                # 将 user 响应 tokenize 并添加到 response_ids，mask 设为 0
                user_messages = [user_message]
                if self.processor is not None:
                    raw_user_response = await self.loop.run_in_executor(
                        None,
                        lambda: self.processor.apply_chat_template(
                            user_messages,
                            add_generation_prompt=False,
                            tokenize=False,
                            **self.apply_chat_template_kwargs,
                        ),
                    )
                    model_inputs = self.processor(text=[raw_user_response], images=None, return_tensors="pt")
                    user_response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
                else:
                    user_response_ids = await self.loop.run_in_executor(
                        None,
                        lambda: self.tokenizer.apply_chat_template(
                            user_messages,
                            add_generation_prompt=False,
                            tokenize=True,
                            **self.apply_chat_template_kwargs,
                        ),
                    )
                all_response_ids.extend(user_response_ids)
                all_response_mask.extend([0] * len(user_response_ids))
                if all_response_logprobs:
                    all_response_logprobs.extend([0.0] * len(user_response_ids))

            # 4.7 终止条件：直接复用 env 的 done 标志
            if getattr(env_response, "done", False):
                break

        # 5. 将累积的 token 序列裁剪到配置长度，并构建 AgentLoopOutput
        # prompt_ids: 只包含初始 prompt (system + 第一轮 user)
        if self.prompt_length > 0:
            final_prompt_ids = initial_prompt_ids[: self.prompt_length]
        else:
            final_prompt_ids = initial_prompt_ids

        # response_ids: 包含所有交互内容 (assistant + user/tool)，按顺序
        # response_mask: 只对 assistant 回复设为 1，user/tool 设为 0
        if self.response_length > 0:
            final_response_ids = all_response_ids[: self.response_length]
            final_response_mask = all_response_mask[: self.response_length]
            final_response_logprobs = (
                all_response_logprobs[: self.response_length] if all_response_logprobs else None
            )
        else:
            final_response_ids = all_response_ids
            final_response_mask = all_response_mask
            final_response_logprobs = all_response_logprobs or None

        output = AgentLoopOutput(
            prompt_ids=final_prompt_ids,
            response_ids=final_response_ids,
            response_mask=final_response_mask,
            response_logprobs=final_response_logprobs,
            multi_modal_data={},
            reward_score=total_reward,
            metrics=metrics,
            extra_fields={"messages": messages, "task_id": task_index},
        )
        return output