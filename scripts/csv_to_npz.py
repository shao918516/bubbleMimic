"""This script replay a motion from a csv file and output it to a npz file

.. code-block:: bash

    # Usage
    python csv_to_npz.py --input_file LAFAN/dance1_subject2.csv --input_fps 30 --frame_range 122 722 \
    --output_file ./motions/dance1_subject2.npz --output_fps 50
"""

"""Launch Isaac Sim Simulator first."""

# ====================== LAFAN1 Retarget Dataset 官方关节重映射配置（根源修复核心） ======================
# 1. 你的DEX_EVT仿真机器人固定23个关节顺序（输出npz的joint_pos顺序）
ROBOT_23_JOINTS = [
    "hip_pitch_l_joint",
    "hip_roll_l_joint",
    "hip_yaw_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
    "hip_pitch_r_joint",
    "hip_roll_r_joint",
    "hip_yaw_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
]
ROBOT_23_NUM = len(ROBOT_23_JOINTS)

# ============ 关节正负号校正表（修复"两个机器人对同名关节正方向定义不同"的问题）============
# 背景: 本脚本是"直接把G1 CSV某关节的数值原样复制给DEX_EVT同名关节"，这只在两者对
# 该关节"正方向"的物理定义一致时才成立。实测(test_single_joint.py)发现:
# DEX_EVT的 shoulder_pitch_l_joint 设为正值时，手臂是向后甩的，
# 与常规人体/G1的"正=手臂向前"约定相反，因此这里需要在复制数值时乘以-1做修正。
# 若后续测出其他关节(如shoulder_pitch_r / shoulder_yaw_l/r / elbow_pitch_l/r)也有
# 同样问题，把对应关节改成 -1 即可；已确认方向正常的关节保持 1，不用改。
JOINT_SIGN_MAP = {name: 1 for name in ROBOT_23_JOINTS}
JOINT_SIGN_MAP["shoulder_pitch_l_joint"] = -1
JOINT_SIGN_MAP["shoulder_pitch_r_joint"] = -1
# JOINT_SIGN_MAP["shoulder_yaw_l_joint"] = -1
# JOINT_SIGN_MAP["shoulder_yaw_r_joint"] = -1
# JOINT_SIGN_MAP["elbow_pitch_l_joint"] = -1
# JOINT_SIGN_MAP["elbow_pitch_r_joint"] = -1
# 以上6项已通过 test_single_joint.py 实测确认方向反转，其余17个关节(腿部/腰部/shoulder_roll)未发现问题，保持1
# ==========================================================================

# 新增：DEX_EVT关节名 → LAFAN CSV标准名称映射表（关键修复）
JOINT_ALIAS_MAP = {
    # 左腿
    "hip_pitch_l_joint": "left_hip_pitch_joint",
    "hip_roll_l_joint": "left_hip_roll_joint",
    "hip_yaw_l_joint": "left_hip_yaw_joint",
    "knee_pitch_l_joint": "left_knee_joint",
    "ankle_pitch_l_joint": "left_ankle_pitch_joint",
    "ankle_roll_l_joint": "left_ankle_roll_joint",
    # 右腿
    "hip_pitch_r_joint": "right_hip_pitch_joint",
    "hip_roll_r_joint": "right_hip_roll_joint",
    "hip_yaw_r_joint": "right_hip_yaw_joint",
    "knee_pitch_r_joint": "right_knee_joint",
    "ankle_pitch_r_joint": "right_ankle_pitch_joint",
    "ankle_roll_r_joint": "right_ankle_roll_joint",
    # 躯干腰部
    "waist_yaw_joint": "waist_yaw_joint",
    "waist_roll_joint": "waist_roll_joint",
    "waist_pitch_joint": "waist_pitch_joint",
    # 左臂
    "shoulder_pitch_l_joint": "left_shoulder_pitch_joint",
    "shoulder_roll_l_joint": "left_shoulder_roll_joint",
    "shoulder_yaw_l_joint": "left_shoulder_yaw_joint",
    "elbow_pitch_l_joint": "left_elbow_joint",
    # 右臂
    "shoulder_pitch_r_joint": "right_shoulder_pitch_joint",
    "shoulder_roll_r_joint": "right_shoulder_roll_joint",
    "shoulder_yaw_r_joint": "right_shoulder_yaw_joint",
    "elbow_pitch_r_joint": "right_elbow_joint",
}

# 2. 官方CSV各型号root后关节完整列表
# G1 CSV root后29列关节
G1_CSV_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
]
# H1 CSV root后19列关节
H1_CSV_JOINTS = [
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_joint",
    "torso_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint"
]
# H1_2 CSV root后27列关节
H1_2_CSV_JOINTS = [
    "left_hip_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
]

# 3. 重写映射函数：先翻译别名，再查找CSV索引（彻底解决名称不匹配）
def get_joint_mapping(robot_type: str):
    if robot_type == "g1":
        csv_joint_list = G1_CSV_JOINTS
    elif robot_type == "h1":
        csv_joint_list = H1_CSV_JOINTS
    elif robot_type == "h1_2":
        csv_joint_list = H1_2_CSV_JOINTS
    else:
        raise ValueError(f"robot_type仅支持 g1 / h1 / h1_2，输入：{robot_type}")
    
    map_idx = []
    for sim_joint_name in ROBOT_23_JOINTS:
        # 步骤1：把DEX_EVT名称翻译成CSV标准名称
        csv_joint_name = JOINT_ALIAS_MAP[sim_joint_name]
        # 步骤2：查找该名称在CSV数组中的索引
        if csv_joint_name in csv_joint_list:
            idx = csv_joint_list.index(csv_joint_name)
            # G1原始csv有29列，idx=22/23/24/25 可以正常读取，无需丢弃
            map_idx.append(idx)
        else:
            # 当前CSV无该关节，填充0中立姿态
            map_idx.append(-1)
            print(f"[WARN] {robot_type} CSV 不存在关节 {csv_joint_name}，将使用0值填充")
    return map_idx
# ==========================================================================

import argparse
import os
import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
parser.add_argument("--input_file", type=str, required=True, help="The path to the input motion csv file.")
parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded."
    ),
)
parser.add_argument("--output_name", type=str, required=True, help="The name of the motion npz file.")
parser.add_argument(
    "--output_dir",
    type=str,
    default="motion_data",
    help="Directory where the converted motion npz file will be stored.",
)
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
parser.add_argument("--robot_type", type=str, required=True, choices=["g1", "h1", "h1_2"],
    help="LAFAN1数据集机器人型号：g1 / h1 / h1_2，匹配CSV来源")
parser.add_argument(
    "--z_offset",
    type=float,
    default=0.0,
    help=(
        "对整条motion的root高度(z)施加的全局垂直校准偏移(单位:米)。"
        "用于修正mocap/CSV原始root高度与当前机器人实际腿长/落地高度不匹配导致的脚部穿地问题。"
        "建议先用 check_floor_clipping.py 检查生成的npz, 得到脚部最低点后, 用 -该最低点+少量安全余量 作为此参数重新生成。"
    ),
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

from whole_body_tracking.robots.dex_evt import DEX_EVT_CFG


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = DEX_EVT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
        robot_type: str,  # 新增形参
        z_offset: float = 0.0,  # 全局落地高度校准偏移
    ):
        self.robot_type = robot_type
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self.z_offset = z_offset
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the csv file."""
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        if self.z_offset != 0.0:
            # 全局落地高度校准: 修正mocap原始root高度与当前机器人实际腿长/落地高度的系统性偏差
            self.motion_base_poss_input = self.motion_base_poss_input.clone()
            self.motion_base_poss_input[:, 2] += self.z_offset
            print(f"[INFO] 已对root高度施加全局z偏移: +{self.z_offset:.4f} m")
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # convert to wxyz
        self.motion_dof_poss_input = motion[:, 7:]
        # 原始CSV全部29维关节（插值前）
        self._raw_dof_input = motion[:, 7:]
        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        # 第一步：先插值【完整29维原始关节】，存入临时变量
        self._interp_full_dof = self._lerp(
            self._raw_dof_input[index_0],
            self._raw_dof_input[index_1],
            blend.unsqueeze(1),
        )

        # 第二步：29维 → 23维映射，赋值给原有self.motion_dof_poss（业务不变）
        # ============ 官方标准动作重映射：任意型号CSV → 固定23DOF仿真机器人 ============
        # 读取全局映射索引
        mapping_idx = get_joint_mapping(self.robot_type)
        T = self._interp_full_dof.shape[0]
        # 初始化23维全0中立姿态
        remapped_dof = torch.zeros((T, ROBOT_23_NUM), device=self._interp_full_dof.device)
        for target_joint_idx, csv_col_idx in enumerate(mapping_idx):
            if csv_col_idx != -1:
                # CSV存在对应关节，赋值；同时乘以符号校正系数，修正两机器人正方向定义不一致的问题
                sign = JOINT_SIGN_MAP[ROBOT_23_JOINTS[target_joint_idx]]
                remapped_dof[:, target_joint_idx] = sign * self._interp_full_dof[:, csv_col_idx]
            # csv_col_idx=-1无对应关节，保持0中立姿态
        self.motion_dof_poss = remapped_dof
        print(f"[Remap Done] Robot Type:{self.robot_type}, mapping_idx={mapping_idx}, After remap shape: {self.motion_dof_poss.shape}")
        # ==========================================================================
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        # 关键：使用插值后完整29维数据计算梯度，不再用23维的motion_dof_poss
        raw_dof_vels = torch.gradient(self._interp_full_dof, spacing=self.output_dt, dim=0)[0]
        # 速度同步重映射
        mapping_idx = get_joint_mapping(self.robot_type)
        T = raw_dof_vels.shape[0]
        remapped_vels = torch.zeros((T, ROBOT_23_NUM), device=raw_dof_vels.device)
        for target_joint_idx, csv_col_idx in enumerate(mapping_idx):
            if csv_col_idx != -1:
                # 速度的符号校正必须和位置一致(同一个关节，同一个sign)，否则位置反了但速度没反，二者会自相矛盾
                sign = JOINT_SIGN_MAP[ROBOT_23_JOINTS[target_joint_idx]]
                remapped_vels[:, target_joint_idx] = sign * raw_dof_vels[:, csv_col_idx]
        self.motion_dof_vels = remapped_vels
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: list[str]):
    """Runs the simulation loop."""
    # Load motion
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
        robot_type=args_cli.robot_type,
        z_offset=args_cli.z_offset,
    )

    # Extract scene entities
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    # ------- data logger -------------------------------------------------------
    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False
    # --------------------------------------------------------------------------

    # Simulation loop
    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        # set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)

            output_dir = os.path.abspath(args_cli.output_dir)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{args_cli.output_name}.npz")
            np.savez(output_path, **log)
            print(f"[INFO]: Motion saved to {output_path}")
            # 保存完成直接退出进程，绕过Isaac Sim阻塞
            import sys
            sys.exit(0)

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(
        sim,
        scene,
        joint_names=[
            "hip_pitch_l_joint",
            "hip_roll_l_joint",
            "hip_yaw_l_joint",
            "knee_pitch_l_joint",
            "ankle_pitch_l_joint",
            "ankle_roll_l_joint",
            "hip_pitch_r_joint",
            "hip_roll_r_joint",
            "hip_yaw_r_joint",
            "knee_pitch_r_joint",
            "ankle_pitch_r_joint",
            "ankle_roll_r_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "shoulder_pitch_l_joint",
            "shoulder_roll_l_joint",
            "shoulder_yaw_l_joint",
            "elbow_pitch_l_joint",
            "shoulder_pitch_r_joint",
            "shoulder_roll_r_joint",
            "shoulder_yaw_r_joint",
            "elbow_pitch_r_joint",
        ],
    )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
