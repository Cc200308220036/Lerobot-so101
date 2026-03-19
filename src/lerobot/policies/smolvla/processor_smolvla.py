#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def make_smolvla_pre_post_processors(
    config: SmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the SmolVLA policy.

    The pre-processing pipeline prepares input data for the model by:
    1.  Renaming features to match pretrained configurations.
    2.  Normalizing input and output features based on dataset statistics.
    3.  Adding a batch dimension.
    4.  Ensuring the language task description ends with a newline character.
    5.  Tokenizing the language task description.
    6.  Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1.  Moving data to the CPU.
    2.  Unnormalizing the output actions to their original scale.

    Args:
        config: The configuration object for the SmolVLA policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    input_steps = [
    # 1. 重命名 (Renaming)
    # 把数据集里的奇怪名字（如 obs.joint_pos）统一映射成标准名字
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
    # 2. 加 Batch 维度 (Batching)
    # 比如一张图是 [3, 224, 224]，变成 [1, 3, 224, 224] 以适应模型输入
        AddBatchDimensionProcessorStep(),
    # 3. 换行符处理 (Newline Processing) —— ★ VLM 特有的细节
    # 专门为了讨好 SmolVLM/PaliGemma 这种模型
        SmolVLANewLineProcessor(),
    # 4. 文本分词 (Tokenization)
    # 把 "Grab the apple" 变成 [101, 345, 221...]
        TokenizerProcessorStep(
            tokenizer_name=config.vlm_model_name,
            padding=config.pad_language_to,
            padding_side="right",
            max_length=config.tokenizer_max_length,
        ),
        # 5. 搬运到 GPU (Device)
        DeviceProcessorStep(device=config.device),

        # 6. 归一化 (Normalization)
        # 这一步非常关键！
        # 所有的 State 和 Action 都会减去均值、除以标准差 (Mean-Std)
        # 所有的 Image 可能会做 0-1 到 -1-1 的转换 (取决于具体实现)
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]

    output_steps = [
    # 1. 反归一化 (Unnormalization)
    # 把 [-1, 1] 还原回 [0.5, 1.5] (弧度/米)
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
         # 2. 搬回 CPU
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


@ProcessorStepRegistry.register(name="smolvla_new_line_processor")
class SmolVLANewLineProcessor(ComplementaryDataProcessorStep):
    """
    A processor step that ensures the 'task' description ends with a newline character.

    This step is necessary for certain tokenizers (e.g., PaliGemma) that expect a
    newline at the end of the prompt. It handles both single string tasks and lists
    of string tasks.
    """

    # def complementary_data(self, complementary_data):
    #     if "task" not in complementary_data:
    #         # ★★★ 插入调试点 1：如果没有 task 字段，说明有问题
    #         print("[DEBUG] WARNING: No 'task' key found in batch!") 
    #         return complementary_data

    #     task = complementary_data["task"]
    #     # 为了防止刷屏，只打印第一条，或者随机打印
    #     if isinstance(task, list) and len(task) > 0:
    #          print(f"[DEBUG] Current Prompt: {task[0]}")
    #     elif isinstance(task, str):
    #          print(f"[DEBUG] Current Prompt: {task}")

    #     if task is None:
    #         return complementary_data

    #     new_complementary_data = dict(complementary_data)

    #     # Handle both string and list of strings
    #     if isinstance(task, str):
    #         # Single string: add newline if not present
    #         # 强制给所有指令加一个 "\n" (换行符)
    #         if not task.endswith("\n"):
    #             new_complementary_data["task"] = f"{task}\n"
    #     elif isinstance(task, list) and all(isinstance(t, str) for t in task):
    #         # List of strings: add newline to each if not present
    #         new_complementary_data["task"] = [t if t.endswith("\n") else f"{t}\n" for t in task]
    #     # If task is neither string nor list of strings, leave unchanged

    #     return new_complementary_data

    def complementary_data(self, complementary_data):
            # 1. 基础检查
            if "task" not in complementary_data:
                return complementary_data

            task = complementary_data["task"]
            if task is None:
                return complementary_data

            # ================= [防刷屏逻辑开始] =================
            
            # 提取当前任务的文本内容（用于比较）
            current_task_str = ""
            if isinstance(task, str):
                current_task_str = task
            elif isinstance(task, list) and len(task) > 0:
                current_task_str = str(task[0])

            # 初始化记忆变量（如果是第一次运行，该属性不存在）
            if not hasattr(self, "_last_printed_task"):
                self._last_printed_task = None

            # 比较：只有当“当前指令”和“上一次指令”不同时，才打印
            # 注意：这里我们比较的是没有加 \n 之前的原始文本，这样更直观
            if current_task_str != self._last_printed_task:
                print(f"[DEBUG] 🟢 Processor 已接收到新指令: {current_task_str}")
                # 更新记忆，下次如果一样就不打印了
                self._last_printed_task = current_task_str
                
            # ================= [防刷屏逻辑结束] =================

            new_complementary_data = dict(complementary_data)

            # 2. 原始核心逻辑：处理换行符
            # Handle both string and list of strings
            if isinstance(task, str):
                # Single string: add newline if not present
                # 强制给所有指令加一个 "\n" (换行符)，这是 SmolVLM/PaliGemma 的要求
                if not task.endswith("\n"):
                    new_complementary_data["task"] = f"{task}\n"
            elif isinstance(task, list) and all(isinstance(t, str) for t in task):
                # List of strings: add newline to each if not present
                new_complementary_data["task"] = [t if t.endswith("\n") else f"{t}\n" for t in task]
            # If task is neither string nor list of strings, leave unchanged

            return new_complementary_data


    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
