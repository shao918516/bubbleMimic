from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


# ===================== 1. 根部位姿跟踪奖励（大幅提升保真，权重最高） =====================
def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    # std缩小 → 误差轻微下降奖励大幅提升，强化根部对齐
    return torch.exp(-error / (std * 0.6) ** 2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / (std * 0.6) ** 2)


# ===================== 2. 全身末端位置/姿态奖励（提升肢体复刻精度） =====================
def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    # 缩小std，对肢体误差更敏感
    return torch.exp(-error.mean(-1) / (std * 0.7) ** 2)

# 四肢关节速度专项奖励（解决 error_joint_vel 平台期）
def motion_limb_joint_vel_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, joint_names: list[str]):
    command: MotionCommand = env.command_manager.get_term(command_name)
    # 正确获取机器人实体，机器人资产名称为 "robot"
    robot: Articulation = env.scene["robot"]
    robot_joint_vel = robot.data.joint_vel
    ref_joint_vel = command.joint_vel
    j_idx = [robot.joint_names.index(j) for j in joint_names]
    err = torch.sum(torch.square(ref_joint_vel[:, j_idx] - robot_joint_vel[:, j_idx]), dim=-1)
    return torch.exp(-err / std**2)
    
def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / (std * 0.7) ** 2)


# ===================== 3. 身体速度跟踪奖励（抑制动作滞后、抖动） =====================
def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


# 【新增】关节空间速度追踪：上面两个 body_*_velocity 追的是各部件在世界坐标系下的
# 线速度/角速度，跟 error_joint_vel（关节角速度）是不同的量，目前没有任何 reward
# 项直接给 error_joint_vel 提供梯度。
def motion_joint_velocity_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.joint_vel - command.robot_joint_vel), dim=-1)
    return torch.exp(-error / std**2)


# ===================== 4. 足底接触奖励（物理合理性：减少悬空、滑动） =====================
def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    # 取负值，变成惩罚项：脚离地太久会扣分
    return -reward


# 【新增】足底滑动惩罚（物理指标核心）
def feet_slide_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    slide_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """足底接触时水平滑动过大施加惩罚，抑制地面滑移。

    注意：ContactSensorData 本身没有 body_lin_vel / contact 这两个属性
    （那是别的仿真框架的写法）。IsaacLab 里正确的做法是：
    - 接触状态：用接触力大小是否超过阈值判断（net_forces_w 是真实存在的字段）
    - 足底速度：要从机器人本体（Articulation）的 body_lin_vel_w 里取，
      而不是从 contact sensor 上取。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    # 接触力大小 -> 接触掩码，形状 (N, B)
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    contact_mask = torch.norm(contact_forces, dim=-1) > contact_force_threshold

    # 足底水平（xy）线速度，从机器人本体取；注意 asset_cfg.body_ids 要和
    # sensor_cfg.body_ids 对应同一组物理 body（同样的 body_names）
    foot_lin_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    slide_mag = torch.norm(foot_lin_vel, dim=-1)

    slide_pen = torch.sum(torch.clamp(slide_mag - slide_threshold, min=0.0) * contact_mask, dim=-1)
    return -slide_pen


# ===================== 6. 关节空间速度追踪（补上目前完全没有奖励覆盖的 error_joint_vel）=====================
def motion_joint_velocity_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """关节空间（而不是笛卡尔空间）速度跟踪。motion_global_body_*_velocity_error_exp 追的是
    各身体部件在世界坐标系下的线速度/角速度，跟 error_joint_vel（关节角速度）是两个不同的量，
    目前没有任何 reward 项直接给 error_joint_vel 提供梯度。"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.joint_vel - command.robot_joint_vel), dim=-1)
    return torch.exp(-error / std**2)
def torque_sum_excess(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """超过力矩阈值返回正损失，在总奖励中减去"""
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is None:
        asset_cfg.joint_ids = slice(None)
    torque_sum = torch.sum(torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
    excess = torch.clamp(torque_sum - threshold, min=0.0)
    # 转为惩罚，总奖励 -= excess
    return -excess


# 【新增】关节动作平滑奖励（提升流畅度、降低抖动/跌倒）
def joint_torque_smooth_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), smooth_std: float = 0.2
) -> torch.Tensor:
    """相邻两步力矩变化过大扣分，动作更顺滑，提升稳定性与表现力"""
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is None:
        asset_cfg.joint_ids = slice(None)
    # 存储上一步力矩，计算差分
    if not hasattr(asset, "_last_torque"):
        asset._last_torque = torch.zeros_like(asset.data.applied_torque)
    delta_torque = asset.data.applied_torque[:, asset_cfg.joint_ids] - asset._last_torque[:, asset_cfg.joint_ids]
    smooth_error = torch.sum(torch.square(delta_torque), dim=-1)
    # 更新缓存
    asset._last_torque = asset.data.applied_torque.clone()
    return torch.exp(-smooth_error / smooth_std**2)
    
def motion_anchor_linear_velocity_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """骨盆(anchor)自身世界系线速度跟踪。motion_body_lin_vel 是全身平均，骨盆的速度误差
    被稀释在里面——跟当初 motion_body_pos → motion_ankle_pair_pos 是同一类拆分。"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w), dim=-1)
    return torch.exp(-error / std**2)

def motion_anchor_angular_velocity_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_ang_vel_w - command.robot_anchor_ang_vel_w), dim=-1)
    return torch.exp(-error / std**2)
    
def motion_global_anchor_position_xy_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """骨盆水平(xy)位置跟踪——单独拆分出来，避免被z轴误差稀释。
    对应 anchor_pos_xy 这个当前最主要的终止原因，给它一条专属梯度，
    跟当初脚踝/手肘从 motion_body_pos 里拆出来是同一个逻辑。"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w[:, :2] - command.robot_anchor_pos_w[:, :2]), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_position_z_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """骨盆垂直(z)位置跟踪——同样单独拆分，为高动态(深蹲/倒立类)动作预留精度。"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.square(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2])
    return torch.exp(-error / std**2)
