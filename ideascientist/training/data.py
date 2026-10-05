"""Dataset and sample processors for per-role GRPO training.

NeMo-RL's stock dataset drops every column except the messages and the task
name, and its stock processor forwards only a ground-truth string into the
environment. Neither is enough here: the environment needs the whole per-sample
record — the assigned axis or gap, the problem definition, and the reference
artifacts it seeds and grades against. Losing it does not raise; the rollout
simply runs ungrounded and scores zero, so the loss here is explicit.

Importing this module registers one processor per role. Everything lives in
this overlay so the vendored NeMo-RL checkout stays unmodified.
"""
from __future__ import annotations

import json
from typing import Any

from nemo_rl.data.datasets.response_datasets.response_dataset import ResponseDataset
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, TaskDataSpec
from nemo_rl.data.processors import PROCESSOR_REGISTRY, register_processor
from transformers import PreTrainedTokenizerBase

TokenizerType = PreTrainedTokenizerBase


def _system_prompt(role: str) -> str:
    """The role's system turn exactly as deployment builds it.

    Assembled from the production role prompt and tool block rather than a
    copy, so a trained policy meets a byte-identical system message at
    deployment.
    """
    from ideascientist.harness import tools as T
    from ideascientist.harness.client import _render_tools_as_text

    role_prompt = T.role_prompt(role)
    tools = T.openai_specs(T.role_tools(role))
    tool_block = _render_tools_as_text(tools)
    return f"{role_prompt}\n\n{tool_block}" if role_prompt else tool_block


def make_dataset_class(class_name: str = "AgentDataset") -> type[ResponseDataset]:
    """A ``ResponseDataset`` subclass that preserves the ``metadata`` column.

    Role-agnostic: the role only matters at processor time.
    """

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": data[self.input_key]},
                {"role": "assistant", "content": data[self.output_key]},
            ],
            "task_name": self.task_name,
            "metadata_json": json.dumps(data.get("metadata", {}), ensure_ascii=False),
        }

    return type(class_name, (ResponseDataset,), {"format_data": format_data})


def make_processor(role: str):
    """Build the per-sample processor that routes the FULL metadata into
    ``extra_env_info`` and templates the first (system + user) turn deterministically
    for ``role``. Returns the closure; call ``register_agent_processor`` to register."""

    def _processor(
        datum_dict: dict[str, Any],
        task_data_spec: TaskDataSpec,
        tokenizer: TokenizerType,
        max_seq_length: int,
        idx: int,
    ) -> DatumSpec:
        user_message = datum_dict["messages"]
        problem = user_message[0]["content"]

        meta_raw = datum_dict.get("metadata_json", "{}")
        try:
            extra_env_info = (
                json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw)
            )
        except (json.JSONDecodeError, TypeError):
            extra_env_info = {}

        # Build the SYSTEM turn deterministically (role prompt + rendered tool block),
        # reproducing production's `_normalize_text_tools_messages` merge, instead of
        # relying on the null `system_prompt_file`. The USER turn is `problem` verbatim
        # (already problem-context + axis, byte-identical to production's user content).
        message_list = [
            {"role": "system", "content": _system_prompt(role)},
            {"role": "user", "content": problem},
        ]

        # enable_thinking=True: this is the ONLY place the first assistant turn is
        # templated; env observations re-prime with `<think>\n` (see _OBS_SUF) so all
        # turns are consistently thinking-ON.
        message: str = tokenizer.apply_chat_template(  # type: ignore
            message_list,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
            enable_thinking=True,
        )
        token_ids = tokenizer(
            message,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0]
        message_log: LLMMessageLogType = [
            {"role": "user", "content": message, "token_ids": token_ids}
        ]

        length = sum(len(m["token_ids"]) for m in message_log)

        loss_multiplier = 1.0
        if length >= max_seq_length:
            for chat_message in message_log:
                chat_message["token_ids"] = chat_message["token_ids"][
                    : min(4, max_seq_length // len(message_log))
                ]
            loss_multiplier = 0.0

        output: DatumSpec = {
            "message_log": message_log,
            "length": length,
            "extra_env_info": extra_env_info,
            "loss_multiplier": loss_multiplier,
            "idx": idx,
            "task_name": datum_dict["task_name"],
        }
        return output

    return _processor


def register_agent_processor(role: str, processor_name: str) -> None:
    """Register the role's processor under the name its config references."""
    if processor_name not in PROCESSOR_REGISTRY:
        register_processor(processor_name, make_processor(role))


ROLES = ("gap_finder", "innovator", "report_writer")

# The config resolves ``dataset_name`` by dotted path, so the class has to exist
# as a module attribute rather than only as a factory result.
AgentDataset = make_dataset_class("AgentDataset")

for _role in ROLES:
    register_agent_processor(_role, f"{_role}_data_processor")
