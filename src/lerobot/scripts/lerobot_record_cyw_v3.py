#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import logging
import time
import torch
import numpy as np
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

# ==================== Imports ====================
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig

# [Fix NameError]
from lerobot.robots.config import RobotConfig

from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import (
    Robot,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import (
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
)
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


# SO-101 关节名称列表
SO101_JOINT_KEYS = [
    "shoulder_pan.pos", 
    "shoulder_lift.pos", 
    "elbow_flex.pos", 
    "wrist_flex.pos", 
    "wrist_roll.pos", 
    "gripper.pos"
]

@dataclass
class DatasetRecordConfig:
    repo_id: str
    single_task: str
    root: str | Path | None = None
    fps: int = 30
    episode_time_s: int | float = 60
    reset_time_s: int | float = 60
    num_episodes: int = 50
    video: bool = True
    push_to_hub: bool = True
    private: bool = False
    tags: list[str] | None = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    teleop: Any | None = None
    policy: PreTrainedConfig | None = None
    display_data: bool = False
    play_sounds: bool = True
    resume: bool = False

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


# =============================================================================
# 强制物理复位函数
# =============================================================================
def reset_robot_to_home(robot, home_state, fps, duration_s):
    if home_state is None:
        print("[Warning] 未记录初始位置，跳过复位。")
        return

    raw_obs = robot.get_observation()
    
    current_vals = []
    try:
        if "observation.state" in raw_obs:
            current_vals = raw_obs["observation.state"]
        else:
            for key in SO101_JOINT_KEYS:
                if key in raw_obs:
                    current_vals.append(raw_obs[key])
                elif key.replace(".pos", "") in raw_obs:
                    current_vals.append(raw_obs[key.replace(".pos", "")])
                else:
                    if "present_position" in raw_obs:
                        current_vals = raw_obs["present_position"]
                        break
                    print(f"[Error] 找不到关节数据: {key}")
                    return
    except Exception as e:
        print(f"[Error] 读取关节数据失败: {e}")
        return

    if isinstance(current_vals, torch.Tensor):
        start_pos = current_vals.detach().cpu().numpy().flatten()
    else:
        start_pos = np.array(current_vals).flatten()
        
    if isinstance(home_state, torch.Tensor):
        target_pos = home_state.detach().cpu().numpy().flatten()
    else:
        target_pos = np.array(home_state).flatten()

    dim = min(len(start_pos), len(target_pos))
    start_pos = start_pos[:dim]
    target_pos = target_pos[:dim]

    total_steps = int(duration_s * fps)
    if total_steps <= 0: return
    
    device = robot.device if hasattr(robot, 'device') else 'cpu'
    
    for step in range(total_steps):
        loop_start = time.perf_counter()
        
        alpha = (step + 1) / total_steps
        # 线性插值
        interp_pos = start_pos + alpha * (target_pos - start_pos)
        
        # [关键修复] 将插值结果打包成字典，而不是直接发 Tensor
        action_dict = {}
        for idx, key in enumerate(SO101_JOINT_KEYS):
            if idx < len(interp_pos):
                # 机器人只接受 tensor 或 float
                val = torch.tensor(interp_pos[idx], dtype=torch.float32, device=device)
                action_dict[key] = val
        
        # 发送字典格式的动作
        robot.send_action(action_dict)
        
        dt = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt)
        
    print("✅ 机械臂已强制复位。")


# =============================================================================
# 核心录制循环
# =============================================================================
@safe_stop_image_writer
def record_loop(
    robot, events, fps, teleop_action_processor, robot_action_processor, robot_observation_processor,
    dataset=None, teleop=None, policy=None, preprocessor=None, postprocessor=None,
    control_time_s=None, single_task=None, display_data=False, write_to_dataset=True,
):
    timestamp = 0
    start_episode_t = time.perf_counter()
    
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # 1. 获取观测
        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
        else:
            continue 

        # 2. 推理
        if policy is not None:
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )
            # 转换为字典 (Dict[str, Tensor])
            act_processed_policy = make_robot_action(action_values, dataset.features)
            
            # 发送给机器人前需要处理
            robot_action_to_send = robot_action_processor((act_processed_policy, obs))
        else:
            continue

        # 3. 执行
        robot.send_action(robot_action_to_send)

        # 4. 写入
        if write_to_dataset:
            # 必须传字典 act_processed_policy
            action_frame = build_dataset_frame(dataset.features, act_processed_policy, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            # [Fix RuntimeError] 传入字典 act_processed_policy
            log_rerun_data(observation=obs_processed, action=act_processed_policy)

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)
        timestamp = time.perf_counter() - start_episode_t


# =============================================================================
# 主程序
# =============================================================================
@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    
    # 1. 初始化
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    
    # 2. 初始化处理器
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # 3. 获取 HOME 点
    print(">>> 正在读取当前姿态作为 HOME 点...")
    time.sleep(1)
    
    raw_obs = robot.get_observation()
    home_state = None
    
    try:
        processed_obs = robot_observation_processor(raw_obs)
        if "observation.state" in processed_obs:
            home_state = processed_obs["observation.state"].clone()
            print(f"🏠 Home State (Processed) 已记录: {home_state.cpu().numpy()}")
        
        if home_state is None:
            home_vals = []
            if "present_position" in raw_obs:
                 home_state = torch.tensor(raw_obs["present_position"], dtype=torch.float32)
            else:
                for key in SO101_JOINT_KEYS:
                    if key in raw_obs:
                        home_vals.append(raw_obs[key])
                if len(home_vals) > 0:
                    home_state = torch.tensor(home_vals, dtype=torch.float32)

            if home_state is not None:
                print(f"🏠 Home State (Manual) 已记录: {home_state.numpy()}")

        if home_state is None:
            print("⚠️ 警告: 无法获取有效的 Home State，复位功能将失效。")

    except Exception as e:
        print(f"❌ 获取 Home State 异常: {e}")

    # 4. 初始化 Dataset
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(pipeline=teleop_action_processor, initial_features=create_initial_features(action=robot.action_features), use_videos=cfg.dataset.video),
        aggregate_pipeline_dataset_features(pipeline=robot_observation_processor, initial_features=create_initial_features(observation=robot.observation_features), use_videos=cfg.dataset.video),
    )
    sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
    dataset = LeRobotDataset.create(
        cfg.dataset.repo_id, cfg.dataset.fps, root=cfg.dataset.root,
        robot_type=robot.name, features=dataset_features,
        use_videos=cfg.dataset.video,
        image_writer_processes=cfg.dataset.num_image_writer_processes,
        image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
        batch_encoding_size=cfg.dataset.video_encoding_batch_size,
    )
    
    # 5. 加载 Policy
    policy = make_policy(cfg.policy, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
        },
    )

    listener, events = init_keyboard_listener()

    # 主循环
    with VideoEncodingManager(dataset):
        recorded_episodes = 0
        while not events["stop_recording"]:
            
            # --- 交互 ---
            print("\n" + "="*60)
            print(f"当前指令: [{cfg.dataset.single_task}]")
            print(">>> 请操作：")
            print("    1. 机械臂已复位，请更换物体。")
            print("    2. 输入新指令并回车 (直接回车 = 保持不变)。")
            print("    3. 输入 'q' 退出。")
            
            try:
                user_input = input(">> ")
                if user_input.strip().lower() == 'q':
                    events["stop_recording"] = True
                    break
                elif user_input.strip() != "":
                    cfg.dataset.single_task = user_input.strip()
                    print(f"✅ 指令已更新为: {cfg.dataset.single_task}")
            except KeyboardInterrupt:
                break

            print(f"🚀 Episode {recorded_episodes + 1} 开始...")
            time.sleep(1)

            # --- 执行 ---
            log_say("Action", cfg.play_sounds)
            record_loop(
                robot=robot, events=events, fps=cfg.dataset.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                dataset=dataset, policy=policy,
                preprocessor=preprocessor, postprocessor=postprocessor,
                control_time_s=cfg.dataset.episode_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
                write_to_dataset=True
            )

            # --- 复位 ---
            if not events["stop_recording"]:
                log_say("Reset", cfg.play_sounds)
                print("🔄 正在复位...")
                reset_robot_to_home(
                    robot=robot,
                    home_state=home_state,
                    fps=cfg.dataset.fps,
                    duration_s=cfg.dataset.reset_time_s
                )

            dataset.save_episode()
            recorded_episodes += 1

    robot.disconnect()
    if not is_headless() and listener is not None: listener.stop()
    return dataset


def main():
    register_third_party_devices()
    record()


if __name__ == "__main__":
    main()