"""Utils for evaluating policies in Isaac Lab simulation environments."""

import os
import platform
from pathlib import Path

import gymnasium as gym

from isaaclab.app import AppLauncher


ASSET_BASE_PATH = Path("/data/nas/AI/dev/zhengj/isaac_robocasa_assets_old/robocasa/models/assets")
# ASSET_BASE_PATH = Path("/data/nas/AI/dev/zhengj/assets")
os.environ["ROBOCASA_ASSETS_ROOT"] = str(ASSET_BASE_PATH)
ASSET_PATH = Path("/home/zimu.gong/IsaacLab/assets")


def get_isaac_env(task):
    """Initializes and returns the Isaac Lab environment"""

    app_launcher = AppLauncher()
    simulation_app = app_launcher.app

    from isaaclab.envs import (
        DirectMARLEnv, 
        ManagerBasedRLEnv, 
        multi_agent_to_single_agent
    )
    from isaaclab_tasks.utils import parse_env_cfg, ExecuteMode

    # parse configuration
    if platform.system() == "Windows":
        os.environ["MUJOCO_GL"] = "wgl"
    elif platform.system() == "Darwin":
        os.environ["MUJOCO_GL"] = "cgl"
    elif platform.system() == "Linux":
        os.environ["MUJOCO_GL"] = "egl"

    task_name = "PnPCounterToCab"
    robot_name = "PandaOmron"
    scene_name = "robocasakitchen-0-8"
    robot_scale = 1.0
    num_envs = 1
    algorithm = "ppo"
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
    
    env_cfg = parse_env_cfg(
        task_name=task_name,
        robot_name=robot_name,
        scene_name=scene_name,
        robot_scale=robot_scale,
        asset_base_path=ASSET_BASE_PATH,
        export_base_path=ASSET_PATH,
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

    from isaaclab_tasks.robocasa.utils.env import load_robocasa_cfg_cls_from_registry
    agent_cfg = None
    if agent_cfg_entry_point:
        agent_cfg = load_robocasa_cfg_cls_from_registry('task',task_name, agent_cfg_entry_point)

    # modify configuration
    env_cfg.terminations.time_out = None
    # create environment
    env: ManagerBasedRLEnv = gym.make(task, cfg=env_cfg)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    return env, task_name


def get_isaac_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]
