"""Utils for evaluating policies in Isaac Lab simulation environments."""

import os
import math
import time
import platform
from pathlib import Path

import imageio
import numpy as np
import torch

from verl.utils.robot_utils import (
    DATE,
    DATE_TIME,
)

import gymnasium as gym
from isaaclab.app import AppLauncher


ASSET_BASE_PATH = Path("/data/ceph_hdd/main/artifactory/isaac_robocasa_assets/robocasa/new/robocasa/models/assets")
os.environ["ROBOCASA_ASSETS_ROOT"] = str(ASSET_BASE_PATH)
ASSET_PATH = Path("/home/zimu.gong/assets")


def get_isaac_env(task):
    """Initializes and returns the Isaac Lab environment"""

    app_launcher = AppLauncher(dict(enable_cameras=True, headless=True))  # Adjust headless as needed
    simulation_app = app_launcher.app

    from isaaclab_tasks.utils import parse_env_cfg, ExecuteMode

    task_name = "LiftObj"
    robot_name = "PandaOmron-Rel"
    scene_name = "robocasakitchen-1-8"
    robot_scale = 1.0
    num_envs = 1
    algorithm = "ppo"
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"

    # import_all_inits(os.path.join(ISAAC_ROBOCASA_ROOT, './tasks/_APIs'))
    from isaaclab_tasks.utils import import_packages
    # The blacklist is used to prevent importing configs from sub-packages
    _BLACKLIST_PKGS = ["utils", ".mdp"]
    # Import all configs in this package
    import_packages("tasks", _BLACKLIST_PKGS)

    env_cfg = parse_env_cfg(
        task_name=task_name,
        robot_name=robot_name,
        scene_name=scene_name,
        robot_scale=robot_scale,
        asset_base_path=ASSET_BASE_PATH,
        device=f"cuda:{app_launcher.local_rank}",
        num_envs=num_envs,
        use_fabric=True,
        first_person_view=False,
        enable_cameras=app_launcher._enable_cameras,
        execute_mode=ExecuteMode.TRAIN
    )
    task_name = f"Robocasa-{task}-{robot_name}-v0"

    gym.register(
        id=task_name,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={},
        disable_env_checker=True,
    )

    env: gym.make(task, cfg=env_cfg)

    return env, task_name


def get_isaac_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return torch.zeros((1, 11))


def get_isaac_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["policy"]["global_camera"][0].to('cpu').numpy()
    # img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_isaac_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["policy"]["eye_in_hand_camera"][0].to('cpu').numpy()
    # img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--openvla_oft--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
