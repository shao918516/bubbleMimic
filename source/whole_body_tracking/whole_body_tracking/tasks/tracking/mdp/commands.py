from __future__ import annotations

import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu", z_offset: float = 0.0):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        # 参考数据整体 z 平移:npz 由 no_contact(无碰撞)URDF 回放录制,参考脚踝最低
        # 到 z=0.029,而机器人物理站立时脚踝实测稳定在 z=0.058——参考的站姿帧脚部
        # 实际"插在地面以下"约 3cm,策略越认真跟踪就越用力把脚往地里踩,导致接触
        # 混乱、绊倒(表现为"跟踪越准 episode 越短")。整体抬高让参考最低点对齐
        # 物理站立高度。只平移位置,速度/姿态不受影响。
        if z_offset != 0.0:
            self._body_pos_w[..., 2] += z_offset
            print(f"[MotionLoader] 参考动作整体 z 平移 {z_offset:+.3f} m")
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(
            self.cfg.motion_file, self.body_indexes, device=self.device, z_offset=self.cfg.motion_z_offset
        )
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)

        # 【诊断，一次性打印，不影响训练】直接从参考动作数据本身算脚踝高度、
        # 相邻帧高度变化量的实际范围，对照 ee_body_pos 终止用的 0.4m 这个阈值看是否合理。
        # 只在 cfg.body_names 里能找到这两个 body 名字时才打印，找不到就跳过，不报错。
        if "ankle_roll_l_link" in self.cfg.body_names and "ankle_roll_r_link" in self.cfg.body_names:
            ankle_l_idx = self.cfg.body_names.index("ankle_roll_l_link")
            ankle_r_idx = self.cfg.body_names.index("ankle_roll_r_link")
            ankle_l_z = self.motion.body_pos_w[:, ankle_l_idx, 2]
            ankle_r_z = self.motion.body_pos_w[:, ankle_r_idx, 2]
            ankle_l_dz = (ankle_l_z[1:] - ankle_l_z[:-1]).abs()
            ankle_r_dz = (ankle_r_z[1:] - ankle_r_z[:-1]).abs()
            print(
                f"[DEBUG] 参考动作全程 左脚踝高度范围 z={ankle_l_z.min().item():.3f}~{ankle_l_z.max().item():.3f} m, "
                f"相邻帧最大高度变化 {ankle_l_dz.max().item():.4f} m"
            )
            print(
                f"[DEBUG] 参考动作全程 右脚踝高度范围 z={ankle_r_z.min().item():.3f}~{ankle_r_z.max().item():.3f} m, "
                f"相邻帧最大高度变化 {ankle_r_dz.max().item():.4f} m"
            )

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        # Include the raw (unaligned) reference root pose so policies can directly consume the
        # commanded root state in addition to the DOF targets.
        # root_pos = self.anchor_pos_w
        # root_rot_mat = matrix_from_quat(self.anchor_quat_w)
        # root_rot = root_rot_mat[..., :2].reshape(root_rot_mat.shape[0], -1)
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        # body_pos_w 维度 [T, N_BODIES, 3]
        # 取出单帧、单个身体部件的XYZ坐标，shape [3]
        # pos_single = self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index, :]
        # num_envs = self._env.scene.env_origins.shape[0]
        # 扩展为 [num_envs, 3]
        # pos = pos_single.unsqueeze(0).repeat(num_envs, 1)
        # env_origin = self._env.scene.env_origins
        # # print(pos_single.shape)
        # print(pos.shape)
        # print(env_origin.shape)
        # return pos + env_origin
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)
        
        # 单独脚踝误差
        ankle_l_idx = self.cfg.body_names.index("ankle_roll_l_link")
        ankle_r_idx = self.cfg.body_names.index("ankle_roll_r_link")
        self.metrics["error_left_ankle_pos"] = torch.norm(
            self.body_pos_relative_w[:, ankle_l_idx] - self.robot_body_pos_w[:, ankle_l_idx], dim=-1
        )
        self.metrics["error_right_ankle_pos"] = torch.norm(
            self.body_pos_relative_w[:, ankle_r_idx] - self.robot_body_pos_w[:, ankle_r_idx], dim=-1
        )
        # 单独手肘误差
        elbow_l_idx = self.cfg.body_names.index("elbow_pitch_l_link")
        elbow_r_idx = self.cfg.body_names.index("elbow_pitch_r_link")
        self.metrics["error_left_elbow_pos"] = torch.norm(
            self.body_pos_relative_w[:, elbow_l_idx] - self.robot_body_pos_w[:, elbow_l_idx], dim=-1
        )
        self.metrics["error_right_elbow_pos"] = torch.norm(
            self.body_pos_relative_w[:, elbow_r_idx] - self.robot_body_pos_w[:, elbow_r_idx], dim=-1
        )
        # 四肢平均误差
        all_limb_idx = [ankle_l_idx, ankle_r_idx, elbow_l_idx]
        self.metrics["error_limb_avg_pos"] = torch.norm(
            self.body_pos_relative_w[:, all_limb_idx] - self.robot_body_pos_w[:, all_limb_idx], dim=-1
        ).mean(-1)
        # 腰部误差
        waist_joint_idx = [self.robot.joint_names.index(j) for j in ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]]
        self.metrics["error_waist_vel"] = torch.norm(
            self.joint_vel[:, waist_joint_idx] - self.robot_joint_vel[:, waist_joint_idx], dim=-1
        )
        
        self.metrics["error_anchor_pos_xy"] = torch.norm(self.anchor_pos_w[:, :2] - self.robot_anchor_pos_w[:, :2], dim=-1)
        self.metrics["error_anchor_pos_z"] = torch.abs(self.anchor_pos_w[:, 2] - self.robot_anchor_pos_w[:, 2])

    def _resample_command(self, env_ids: Sequence[int]):
        if self.cfg.eval_start_frame >= 0:
            # 评估/渲染模式:固定从指定帧开始
            start = min(self.cfg.eval_start_frame, self.motion.time_step_total - 1)
            self.time_steps[env_ids] = start
        else:
            phase = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
            self.time_steps[env_ids] = (phase * (self.motion.time_step_total - 1)).long()
            # 起步过采样:以 start_oversample_prob 的概率,把出生点改到动作开头
            # start_oversample_window 帧以内(均匀)。理由:均匀 RSI 下,"从静止起步"
            # 这个转换只对应开头约1秒的出生点(<10%的采样量),但评估/交付恰恰只考
            # 这一个场景——12000轮训练后评估仍然"左脚不动、起步就摔",正是这个
            # 训练/评估分布错配的直接表现。过采样让起步成为高频必修课。
            if self.cfg.start_oversample_prob > 0.0:
                pick = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device) < self.cfg.start_oversample_prob
                if pick.any():
                    window = min(self.cfg.start_oversample_window, self.motion.time_step_total - 1)
                    early = (sample_uniform(0.0, 1.0, (int(pick.sum()),), device=self.device) * window).long()
                    ids = torch.as_tensor(env_ids, device=self.device)[pick] if not isinstance(env_ids, torch.Tensor) else env_ids[pick]
                    self.time_steps[ids] = early
                    
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]
        # if len(env_ids) > 0:
        #     print(f"[DEBUG RSI] 本次reset采样到的time_steps: {self.time_steps[env_ids][:5].tolist()} / 总帧数{self.motion.time_step_total}")
        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )
        # if len(env_ids) > 0 and not hasattr(self, "_dbg_joint_once"):
        #     self._dbg_joint_once = True
        #     print("[DEBUG] Isaac 关节顺序 vs 写入的参考值 vs 写入后实际值:")
        #     jw = joint_pos[env_ids][0]
        #     ja = self.robot.data.joint_pos[env_ids][0]
        #     for i, n in enumerate(self.robot.joint_names):
        #         print(f"  {i:2d} {n:26s} 写入={jw[i].item():+.3f} 实际={ja[i].item():+.3f}")
        # ==== 诊断:root写入是否生效 ====
        # if len(env_ids) > 0:
        #     written = root_pos[env_ids]
        #     actual = self.robot.data.root_pos_w[env_ids]
        #     print(f"[DEBUG reset] 写入root(xyz)={written[0].tolist()}  "
        #           f"写入后实际root(xyz)={actual[0].tolist()}")

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    # 评估/渲染专用:>=0 时每次 reset 都从指定帧开始(不随机采样相位);
    # -1(默认)为训练行为:随机相位。用途:设 0 从头播;设约 4~5 秒对应的帧数
    # 可跳过 npz 开头人为拼接的 start_frames 过渡段,验证过渡段是否是摔倒元凶。
    eval_start_frame: int = -1

    # 起步过采样(训练用):以此概率把 RSI 出生点改到动作开头 window 帧以内,
    # 专项加练"从静止起步"这个评估必考、但均匀采样下占比很低的转换。
    # 每次 episode 重置，20% 概率强制把起始帧拉到动作开头区间
    # 强制起步区间为动作前 N帧，window=135 @90fps ≈ 前1.5秒(约覆盖过渡段末尾+起步第一步)。设 prob=0 关闭。
    start_oversample_prob: float = 0.10
    start_oversample_window: int = 270
    
    # 参考动作整体 z 平移量(米)。npz 用无碰撞 URDF 录制,参考脚踝最低 z=0.029,
    # 机器人物理站立脚踝实测 z=0.058,差约 3cm——参考站姿脚部插在地面以下,
    # 需整体抬高对齐。若换新动作文件,按"物理站立脚踝高度 - 该文件参考脚踝最低值"重估。
    motion_z_offset: float = 0.03
    
    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
