from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def align_stairs_with_envs(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    # stairs_low_offset: tuple[float, float, float],
    stairs_high_offset: tuple[float, float, float],
    # stairs_low_name: str = "stairs_low",
    stairs_high_name: str = "stairs_high",
    stairs_rot: tuple[float, float, float, float] = (1 ,0 ,0, 0),
):
    """Place stair rigid objects at each env origin so they follow terrain generator offsets."""
    origins = env.scene.env_origins
    if env_ids is None:
        env_ids = torch.arange(origins.shape[0], device=origins.device)
    else:
        env_ids = env_ids.to(origins.device)

    def _place(name: str, offset: tuple[float, float, float]):
        if name is None:
            return
        try:
            stairs = env.scene[name]
        except KeyError:
            return
        root_state = torch.zeros((len(env_ids), 13), device=origins.device)
        root_state[:, 0:3] = origins[env_ids] + torch.tensor(offset, device=origins.device, dtype=origins.dtype)
        root_state[:, 3:7] = torch.tensor(stairs_rot, device=origins.device, dtype=origins.dtype)
        # velocities stay zero
        stairs.write_root_state_to_sim(root_state, env_ids=env_ids)

    # _place(stairs_low_name, stairs_low_offset)
    _place(stairs_high_name, stairs_high_offset)


def align_chair_with_envs(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    chair_offset: tuple[float, float, float],
    chair_name: str = "chair",
    chair_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
):
    """Place chair rigid objects at each env origin so they follow terrain generator offsets."""
    origins = env.scene.env_origins
    if env_ids is None:
        env_ids = torch.arange(origins.shape[0], device=origins.device)
    else:
        env_ids = env_ids.to(origins.device)

    try:
        chair = env.scene[chair_name]
    except KeyError:
        return

    root_state = torch.zeros((len(env_ids), 13), device=origins.device)
    root_state[:, 0:3] = origins[env_ids] + torch.tensor(chair_offset, device=origins.device, dtype=origins.dtype)
    root_state[:, 3:7] = torch.tensor(chair_rot, device=origins.device, dtype=origins.dtype)
    chair.write_root_state_to_sim(root_state, env_ids=env_ids)


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)
    
def reset_last_torque(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """在每个环境的 episode 边界，把上一步力矩缓存清零，
    避免 joint_torque_smooth_reward 把"跨 episode 的力矩跳变"
    误判成同一段动作里的真实抖动。"""
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(asset, "_last_torque"):
        return
    asset._last_torque[env_ids] = 0.0


class anneal_termination_threshold(ManagerTermBase):
    """课程学习：让某个 DoneTerm 的 threshold 参数，随 env.common_step_counter
    从 initial_threshold 线性过渡到 final_threshold。

    新增 delay_steps（默认0，向后兼容旧用法）：在 delay_steps 之前，阈值保持在
    final_threshold（严格）不放宽；到 delay_steps 之后才开始从 initial_threshold
    向 final_threshold 线性收紧，用时 num_steps。整体形状是"先严格→跳到宽松→
    再逐步收紧回严格"，用于让某个已经能在严格阈值下自己学好的技能（比如根部追踪）
    先有一段不被干扰的窗口，再对另一个更难的技能（比如脚踝时机）单独给宽松期。

    实现上直接照抄 isaaclab.envs.mdp.curriculums.modify_reward_weight 的写法
    （只是把 reward_manager 换成 termination_manager，把"到点跳变"换成"线性插值"），
    这是官方文档里演示过的、确认可用的模式，不是自己臆造的接口。
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        term_name = cfg.params["term_name"]
        self._term_cfg = env.termination_manager.get_term_cfg(term_name)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        initial_threshold: float,
        final_threshold: float,
        num_steps: int,
        delay_steps: int = 0,
    ) -> float:
        if env.common_step_counter < delay_steps:
            new_threshold = final_threshold
        else:
            progress = min(float(env.common_step_counter - delay_steps) / num_steps, 1.0)
            new_threshold = initial_threshold + (final_threshold - initial_threshold) * progress
        self._term_cfg.params["threshold"] = new_threshold
        env.termination_manager.set_term_cfg(term_name, self._term_cfg)
        return new_threshold
