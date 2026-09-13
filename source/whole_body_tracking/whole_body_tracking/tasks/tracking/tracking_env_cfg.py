from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.tracking.mdp as mdp

##r
# Scene definition
##
VELOCITY_RANGE_ZERO = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

# 轻微斜坡 + 随机起伏地形：用于生成器模式
GENTLE_SLOPE_NOISE_TERRAIN = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    num_rows=5,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    sub_terrains={
        # 平地：半数子地形近似全平
        "flat": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0,  # 原 0.5 → 1.0，训练暂时全部用平地
            slope_range=(0.0, 0.0),  # 0 斜率，等价于平面
            platform_width=8.0,      # 接近子地形尺寸，保证大平面
            border_width=0.3,
        ),
        # "mild_noise" / "gentle_slopes" 暂时注释掉：这两种地形的高度假设
        # 跟参考动作(z0平地校准)、anchor_pos 的 0.35m 阈值都不兼容，除非以后
        # 让参考动作/终止阈值感知地形实际高度，否则留着只会持续制造这次这种
        # "出生即死"的假摔倒。
        # 小幅度起伏噪声：1/4
        # "mild_noise": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.25,
        #     noise_range=(0.00, 0.05),
        #     noise_step=0.01,
        #     border_width=0.3,
        # ),
        # # 轻微斜坡：1/4
        # "gentle_slopes": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.25,
        #     slope_range=(0.05, 0.2),
        #     platform_width=3.0,
        #     border_width=0.3,
        # ),
    },
)

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    def __post_init__(self):
        super().__post_init__()
        # robot_cfg = getattr(self, "robot", None)
        # if robot_cfg in (None, MISSING) or not hasattr(robot_cfg, "init_state"):
        #     px, py, pz = (0.0, 0.0, 1.0)
        # else:
        #     px, py, pz = getattr(robot_cfg.init_state, "pos", (0.0, 0.0, 1.0))
        # seat_height = 0.45 - self.stool.spawn.size[2] * 0.5
        # self.stool.init_state.pos = (px - 0.45, py , seat_height)

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",

        # terrain_type="plane",
        terrain_type="generator",
        terrain_generator=GENTLE_SLOPE_NOISE_TERRAIN,
        
        use_terrain_origins=True, # 必须开启，读取地形格子坐标
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robots
    robot: ArticulationCfg = MISSING
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
        # 显式写死 -1（随机相位/RSI，训练必需），不依赖 MotionCommandCfg 的类默认值——
        # 那个默认值曾经被改成过 0（大概率是调评估渲染时顺手改的，忘了改回来），
        # 一旦默认值再被谁动过，这里不受影响，训练侧永远是 -1。
        # 评估/渲染时想固定起始帧，去 play.py 里对 env_cfg.commands.motion.eval_start_frame
        # 单独赋值覆盖，不要来改这里的默认值。
        eval_start_frame=-1,
        # 新增两行
        # 每次 episode 重置，20% 概率强制把起始帧拉到动作开头区间
        start_oversample_prob=0.1,
        # 强制起步区间为动作前 200 帧
        start_oversample_window=270,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.25, n_max=0.25)
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            
        
        # def motion_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
        #     """当前在参考动作里的相位,sin/cos编码。给逐帧无记忆的MLP一个"现在是动作
                   #        哪个阶段"的显式信号,帮它区分瞬时姿态相近、但前后文不同的时刻。"""
        #     command: MotionCommand = env.command_manager.get_term(command_name)
        #     phase = command.time_steps.float() / command.motion.time_step_total
        #     return torch.stack([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    randomize_joint_params = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "friction_distribution_params": (0.001, 0.6),
            "armature_distribution_params": (0.002, 0.060),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    randomize_rigid_body_mass_others = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 5.0),
        params={"velocity_range": VELOCITY_RANGE},
    )
    
    # reset
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.75, 1.25),
            "damping_distribution_params": (0.75, 1.25),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    # 新增：joint_torque_smooth_reward 用到的 _last_torque 缓存必须在每个环境的
    # episode 边界清零，否则会把"上一段 episode 结束时的力矩"和"这一段刚开始时的
    # 力矩"当成同一段连续动作里的突变来算，产生虚假的大误差，把这个奖励项长期压到 0。
    # 依赖 events.py 里新增的 reset_last_torque 函数（需要一并添加）。
    reset_last_torque = EventTerm(
        func=mdp.reset_last_torque,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class RewardsCfg:
    # ========== 动作保真（总分权重50%） ==========
    # std 相比原来大幅放宽：原来的数值代入你训练日志里实际达到过的最好误差
    # （error_anchor_pos 最优约 0.08）算出来，reward 仍然是 exp(-20.5) ≈ 1.25e-9，
    # 训练早期误差更大时 exp(-100)~exp(-560)，在 float32 下直接算成 0——
    # 这四项 reward 从训练一开始到目前为止，实际上从没提供过真正有梯度的信号，
    # 这也是 Episode_Reward/motion_global_anchor_pos 等四项日志里长期显示 ~0.0002
    # 的直接原因。这里给的新值是按"std_eff(=std乘0.6或0.7) 大致等于日志里
    # 实际误差量级"这个思路估的起点，不是精调过的最终值，练起来之后建议持续盯着
    # 这四个 Episode_Reward 是否明显离开了 0 附近、再按需要继续调（可以配合
    # CurriculumCfg 后续做 std 随训练进度逐步收紧，目前 CurriculumCfg 是空的）。
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.22, # 原 0.15
        params={"command_name": "motion", "std": 0.25},  # 原 0.03
    )
    # 给全局位置奖励加一个"粗调"项——精细项(std=0.25)保留管近距离精度,再加一个大 std 的同函数项,让 0.5~1m 范围内也有把机器人往参考拉的梯度
    motion_global_anchor_pos_coarse = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.20,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.18,  # 原0.15上调
        params={"command_name": "motion", "std": 0.6},  # 原 0.04
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=0.06,  # 原 0.12，减少躯干主导
        params={"command_name": "motion", "std": 0.25},  # 原 0.05
    )
    # 【脚踝专项追踪】motion_body_pos 是把所有 body 的位置误差平均后给 reward，
    # 脚踝只是其中两个，误差信号被平均稀释——这是训练里反复出现"根部/整体学得好、
    # 脚踝一直是 98%+ 终止主因"的一个直接可疑原因。这里复用同一个函数、把
    # body_names 限定为两个脚踝，给它们单独的、不被稀释的追踪信号。
    # std 给 0.2（比整体 body_pos 的 0.25 稍紧，因为 ee_body_pos 终止判的就是
    # 脚踝 Z 向 0.4m，希望它在这个范围内有足够梯度）；weight 0.1 起步，
    # 若脚踝终止占比仍长期 90%+ 可再上调。
    # 【第三步实验:0.10 已验证"不伤根部但也没帮到脚踝"(iter200 根部 0.1227 vs
    # 基准 0.1186,ee_body_pos 终止 98.75% vs 98.86%,全在噪声范围内)。
    # 按预定加码路径升到 0.20——现在它是最大的正向项(超过 anchor_pos 的 0.15),
    # 这是有意为之:脚踝是唯一瓶颈,值得给最大权重。仍然只改这一个变量(std 不动)。
    # 决策线同前:根部停 0.5+ → 立刻停,回退基准 checkpoint;
    # 根部健康 → 看 ee_body_pos 终止占比和 episode length 有没有实质变化。
    # motion_ankle_pos = RewTerm(
    #     func=mdp.motion_relative_body_position_error_exp,
    #     weight=0.20,  # 0.10 -> 0.20
    #     params={
    #         "command_name": "motion",
    #         "std": 0.2,
    #         "body_names": ["ankle_roll_l_link", "ankle_roll_r_link"],
    #     },
    # )
    # 左右脚踝独立位置奖励（原 motion_ankle_pos 基础上调权重）
    # 双脚踝位置专项（最高权重，解决脚部跟踪瓶颈）
    motion_ankle_pair_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=0.24,  # 全局最高权重，优先约束脚部
        params={
            "command_name": "motion",
            "std": 0.20,
            "body_names": ["ankle_roll_l_link", "ankle_roll_r_link"],
        },
    )
    # 左脚单独奖励（便于日志拆分诊断）
    motion_left_ankle_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=0.13, # 左脚踝单独加码（日志左踝误差显著高于右踝）
        params={
            "command_name": "motion",
            "std": 0.18,
            "body_names": ["ankle_roll_l_link"],
        },
    )
    # 右脚单独奖励
    motion_right_ankle_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=0.11,
        params={
            "command_name": "motion",
            "std": 0.18,
            "body_names": ["ankle_roll_r_link"],
        },
    )
    
    # 手臂末端专项追踪——原计划用 wrist_roll，但 joint_pos 是23维、body_pos_w只有
    # 24个body，反推出这条训练线用的很可能是23自由度的v1 dex_evt（不是31自由度的
    # dex_evt2），这具机器人物理上没有 wrist_pitch/wrist_roll 关节，用 wrist 会报错
    # 找不到 body。改用手臂链条上实际存在、且已经在 body_names 追踪列表里的
    # elbow_pitch_l_link / elbow_pitch_r_link（这个机器人上手臂链条的最后一个关节），
    # 不需要重新转换数据，joint_pos/body_pos_w 里本来就有现成数值。
    # motion_elbow_pos = RewTerm(
    #     func=mdp.motion_relative_body_position_error_exp,
    #     weight=0.20,
    #     params={
    #         "command_name": "motion",
    #         "std": 0.2,
    #         "body_names": ["elbow_pitch_l_link", "elbow_pitch_r_link"],
    #     },
    # )
    # 双手肘整体位置奖励
    motion_elbow_pair_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=0.22,
        params={
            "command_name": "motion",
            "std": 0.20,
            "body_names": ["elbow_pitch_l_link", "elbow_pitch_r_link"],
        },
    )
    # 左肘单独
    motion_left_elbow_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=0.11,
        params={
            "command_name": "motion",
            "std": 0.18,
            "body_names": ["elbow_pitch_l_link"],
        },
    )
    # 右肘单独
    motion_right_elbow_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=0.11,
        params={
            "command_name": "motion",
            "std": 0.18,
            "body_names": ["elbow_pitch_r_link"],
        },
    )

    # 四肢末端姿态（脚踝+手肘旋转对齐）
    motion_limb_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=0.16,
        params={
            "command_name": "motion",
            "std": 0.55,
            "body_names": [
                "ankle_roll_l_link", "ankle_roll_r_link",
                "elbow_pitch_l_link", "elbow_pitch_r_link"
            ],
        },
    )
    # 下肢关节速度（髋/膝/踝）
    motion_lower_limb_vel = RewTerm(
        func=mdp.motion_limb_joint_vel_error_exp,
        weight=0.20,  # 原0.14 ↑
        params={
            "command_name": "motion",
            "std": 10.0, # 原18 ↓，误差敏感度大幅提升
            "joint_names": [
                "hip_pitch_l_joint","hip_roll_l_joint","hip_yaw_l_joint","knee_pitch_l_joint","ankle_pitch_l_joint","ankle_roll_l_joint",
                "hip_pitch_r_joint","hip_roll_r_joint","hip_yaw_r_joint","knee_pitch_r_joint","ankle_pitch_r_joint","ankle_roll_r_joint"
            ],
        },
    )
    # 上肢关节速度（肩/肘，不含手腕）
    motion_upper_limb_vel = RewTerm(
        func=mdp.motion_limb_joint_vel_error_exp,
        weight=0.26, # 原0.20小幅上调，匹配新增刚体速度奖励
        params={
            "command_name": "motion",
            "std": 10.0, # 原18 ↓，误差敏感度大幅提升
            "joint_names": [
                "shoulder_pitch_l_joint","shoulder_roll_l_joint","shoulder_yaw_l_joint","elbow_pitch_l_joint",
                "shoulder_pitch_r_joint","shoulder_roll_r_joint","shoulder_yaw_r_joint","elbow_pitch_r_joint"
            ],
        },
    )
    # ====================== 新增：整条上肢（肩+肘）线速度跟踪 =====================
    motion_upper_limb_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.13,
        params={
            "command_name": "motion",
            "std": 2, # 手臂挥舞速度比躯干更快，std略放大
            "body_names": [
                "shoulder_pitch_l_link","shoulder_roll_l_link","shoulder_yaw_l_link","elbow_pitch_l_link",
                "shoulder_pitch_r_link","shoulder_roll_r_link","shoulder_yaw_r_link","elbow_pitch_r_link"
            ],
        }
    )
    # 左臂单独线速度
    motion_left_arm_lin_vel = RewTerm(
       func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.09,
        params={
            "command_name": "motion",
            "std": 2,
            "body_names": ["shoulder_pitch_l_link","shoulder_roll_l_link","shoulder_yaw_l_link","elbow_pitch_l_link"],
        }
    )
    # 右臂单独线速度
    motion_right_arm_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.09,
        params={
            "command_name": "motion",
            "std": 2,
            "body_names": ["shoulder_pitch_r_link","shoulder_roll_r_link","shoulder_yaw_r_link","elbow_pitch_r_link"],
        }
    )
    # ====================== 新增：整条上肢（肩+肘）角速度跟踪 =====================
    motion_upper_limb_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=0.15, # 旋转变形更致命，权重略高于线速度
        params={
            "command_name": "motion",
            "std": 4.8, # 手臂快速绕环角速度大，放宽std避免奖励直接归零
            "body_names": [
                "shoulder_pitch_l_link","shoulder_roll_l_link","shoulder_yaw_l_link","elbow_pitch_l_link",
                "shoulder_pitch_r_link","shoulder_roll_r_link","shoulder_yaw_r_link","elbow_pitch_r_link"
            ],
        }
    )
    # 左臂单独角速度
    motion_left_arm_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=0.10,
        params={
            "command_name": "motion",
            "std": 4.8,
            "body_names": ["shoulder_pitch_l_link","shoulder_roll_l_link","shoulder_yaw_l_link","elbow_pitch_l_link"],
        }
    )
    # 右臂单独角速度
    motion_right_arm_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=0.10,
        params={
            "command_name": "motion",
            "std": 4.8,
            "body_names": ["shoulder_pitch_r_link","shoulder_roll_r_link","shoulder_yaw_r_link","elbow_pitch_r_link"],
        }
    )

    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=0.08,
        params={"command_name": "motion", "std": 0.6},  # 原 0.06
    )
    
    # 骨盆水平/垂直位置分轴专项——anchor_pos_xy 长期是终止占比最高的一项（65~68%），
    # 但目前唯一的位置奖励（motion_global_anchor_pos/_coarse）是xyz合并算误差，
    # xy这块的信号被z轴稀释。拆开后xy这条不用等z也贴合才有梯度。
    motion_anchor_pos_xy = RewTerm(
        func=mdp.motion_global_anchor_position_xy_error_exp,
        weight=0.16,
        params={"command_name": "motion", "std": 0.4},  # 略低于anchor_pos_xy终止阈值0.5，起点，练出error_anchor_pos_xy后再校准
    )
    motion_anchor_pos_z = RewTerm(
        func=mdp.motion_global_anchor_position_z_error_exp,
        weight=0.12,
        params={"command_name": "motion", "std": 0.25},  # 略低于anchor_pos(z-only)终止阈值0.35，起点
    )
    
    # 腰部关节速度（此前完全没有专属覆盖——12+8=20个关节已被
    # lower/upper_limb_vel覆盖，剩下的3个腰部关节只混在motion_joint_vel
    # 这个23维兜底项里被稀释，是error_joint_vel目前唯一还没对症处理过的缺口）
    motion_waist_vel = RewTerm(
        func=mdp.motion_limb_joint_vel_error_exp,
        weight=0.10,  # 先给一个比lower/upper_limb_vel(0.20/0.26)更保守的起点，只有3个关节，避免过度加权
        params={
            "command_name": "motion",
            "std": 10.0,  # 跟lower/upper_limb_vel当前值对齐作为起点，练出数据后再按实测调整
            "joint_names": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
        },
    )

    # ========== 物理合理性（总分权重20%） ==========
    feet_contact = RewTerm(
        func=mdp.feet_contact_time,
        # 【禁用,原0.06】这个项的实现是"脚离地(且落地时间<0.3s)就扣分",本意防原地
        # 快速跺脚,但对正在学起步的策略,它的实际梯度方向是"抬脚=挨罚,不抬=安全"——
        # 和 action_rate_l2/torque_smooth 一起构成三重反迈步压制,是"左脚不敢抬"的
        # 直接嫌疑之一。先归零观察;若后期出现跺脚问题再以更小权重(如0.01)恢复。
        weight=0.005, # 原0.08再上调，鼓励双脚落地
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["ankle_roll_l_link","ankle_roll_r_link"]), "threshold":0.3},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide_penalty,
        weight=-0.08, # 轻微减弱滑动惩罚
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["ankle_roll_l_link", "ankle_roll_r_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["ankle_roll_l_link", "ankle_roll_r_link"]),
            "slide_threshold": 0.02,
        },
    )
    # 力矩超限惩罚降低，避免策略不敢发力做大幅度舞蹈
    torque_over_limit = RewTerm(
        func=mdp.torque_sum_excess,
        weight=0.0002,  # 原 0.002小幅下降
        params={"threshold": 120},
    )
    # torque_sum_excess 是没有上限的线性惩罚（超多少扣多少），跟其它 exp() 类奖励天然
    # 压缩在 [0,1] 不是同一种量级——实测原始超限量常年在 15~75 这个区间，权重 0.06 时
    # 单这一项就到 -1~-4.5，比全部六个追踪 reward 加起来还大出百倍，把 50% 权重设计
    # 意图的追踪信号完全盖过去了。这里降到 0.002 只是把量级重新拉回可比范围（大约
    # -0.03~-0.15），不是说这项不重要，如果后续发现力矩超限的情况又变得普遍，
    # 可以再往上调，但不建议直接调回 0.06 这种数量级不匹配的值。

    # 【暂时去掉】ankle_torque_saturation_diag / ankle_velocity_saturation_diag 这两个
    # 纯诊断项（权重 1e-8，本来就不影响训练）依赖 rewards.py 里对应的新函数定义，
    # 上次报错就是因为磁盘上的 rewards.py 版本没同步这两个函数导致 AttributeError。
    # 现在时间紧，先去掉、不纠结这两个诊断指标；如果之后有空、想恢复这两个诊断，
    # 确认 rewards.py 里有 ankle_torque_saturation_diag/ankle_velocity_saturation_diag
    # 这两个函数定义之后，把下面这段取消注释即可：
    #
    # ankle_torque_saturation_diag = RewTerm(
    #     func=mdp.ankle_torque_saturation_diag,
    #     weight=1e-8,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[r".*ankle_pitch.*joint", r".*ankle_roll.*joint"])},
    # )
    # ankle_velocity_saturation_diag = RewTerm(
    #     func=mdp.ankle_velocity_saturation_diag,
    #     weight=1e-8,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[r".*ankle_pitch.*joint", r".*ankle_roll.*joint"])},
    # )

    # ===================== 6. 关节空间速度追踪（新增） =====================
    # 补上 error_joint_vel 目前完全没有 reward 覆盖的问题：iter253->iter500 这段训练里，
    # 位置/姿态类误差全线改善，但 error_joint_vel（16.89->21.91）和 error_body_ang_vel
    # （5.75->6.70）在变差，跟这两项一直没有直接的奖励信号对得上。std 是按
    # error_joint_vel≈22（先按范数处理，平方后约484）这个量级估的起点，同样需要后续跟着
    # 训练进展调整。
    motion_joint_vel = RewTerm(
        func=mdp.motion_joint_velocity_error_exp,
        weight=0.15, # 原0.10小幅上调
        params={"command_name": "motion", "std": 20.0},
    )

    # ========== 稳定性（总分权重10%） ==========
    # smooth_std 原来是 0.2，实测调试打印出来的 smooth_error（31个关节力矩变化量平方和）
    # 常年在 8.5万~12.4万这个量级（max 甚至到 30~70 万），跟 0.2 假设的量级差了近6个数量级，
    # exp(-124000/0.04) 直接下溢成 0.0，这是 torque_smooth 长期精确等于 0.0000 的真实原因
    # （不是 _last_torque 没重置，那部分已经用调试打印确认过是正常工作的）。
    # 这个 std 同样是按当前实测量级估的起点，注意这个量级本身也偏大（意味着策略输出的
    # 力矩帧间变化本身就很剧烈），值得回头查一下 rsl_rl_ppo_cfg.py 里 clip_action=100
    # 是不是让动作空间管得太松，这属于另一条排查线，不是这里能靠改 std 解决的。
    torque_smooth = RewTerm(
        func=mdp.joint_torque_smooth_reward,
        weight=0.10,
        params={"smooth_std": 300.0},  # 原 0.2
    )

    # ========== 动作流畅表现力（总分权重20%） ==========
    # 同样是 std 过紧导致长期饱和在 0 的问题，上一轮只改了上面四个位置/姿态类的，
    # 这两个速度类的漏掉了：这两个函数没有 ×0.6/0.7 的调整，直接是 exp(-error/std²)，
    # 代入日志里 error_body_lin_vel≈1.93、error_body_ang_vel≈5.7~5.8 算出来，
    # exp() 指数是 -759 和 -5170 这个量级，直接 float 下溢成 0.0，
    # 跟 Episode_Reward/motion_body_lin_vel、motion_body_ang_vel 长期精确等于 0.0000
    # 完全对得上。同样是先给一个大致贴近实际误差量级的起点，需要之后再调。
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.10,
        params={"command_name": "motion", "std": 1.5},  # 原 0.07
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=0.16,
        params={"command_name": "motion", "std": 4.0},  # 原 0.08
    )
    
    motion_anchor_lin_vel = RewTerm(
        func=mdp.motion_anchor_linear_velocity_error_exp,
        weight=0.16,
        params={"command_name": "motion", "std": 1.0},  # 起点贴近当前 error_anchor_lin_vel≈0.91
    )
    motion_anchor_ang_vel = RewTerm(
        func=mdp.motion_anchor_angular_velocity_error_exp,
        weight=0.16,
        params={"command_name": "motion", "std": 3.0},  # 起点贴近当前 error_anchor_ang_vel≈2.65~2.83
    )
    motion_ankle_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.14,
        params={
            "command_name": "motion",
            "std": 2.0,  # 脚踝快速动作时线速度通常比全身平均(≈1.35)更大，先给宽一档避免下溢
            "body_names": ["ankle_roll_l_link", "ankle_roll_r_link"],
        },
    )
    motion_ankle_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=0.14,
        params={
            "command_name": "motion",
            "std": 5.5,  # 同上，比全身平均(4.0)宽一档
            "body_names": ["ankle_roll_l_link", "ankle_roll_r_link"],
        },
    )
    motion_left_ankle_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.08,
        params={"command_name": "motion", "std": 2.0, "body_names": ["ankle_roll_l_link"]},
    )
    motion_right_ankle_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.08,
        params={"command_name": "motion", "std": 2.0, "body_names": ["ankle_roll_r_link"]},
    )

    # 原有基础约束惩罚保留
    # 四肢动作突变惩罚，平衡跟踪精度与稳定性
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    joint_torque_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_vel_limit = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-0.1,
        params={"soft_ratio": 0.95, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                # 原来排除的是 "wrist_yaw_l_link"/"wrist_yaw_r_link"，但 dex_evt2 的
                # 手腕关节叫 wrist_pitch/wrist_roll，压根没有 wrist_yaw 这个 body，
                # 排除条件从来没匹配到过任何东西——大概率是从别的机器人配置抄过来的、
                # 忘了改成 dex_evt2 实际的命名。这里改成实际存在的 wrist_roll_l/r_link
                # （跟 TerminationsCfg 里注释掉的那行用的是同一套正确命名）。
                body_names=[
                    r"^(?!ankle_roll_l_link$)(?!ankle_roll_r_link$)(?!wrist_roll_l_link$)(?!wrist_roll_r_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.35},  # 同步到你确认过的实际值，原来这里还是 0.25
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    # 超出2.3m自动重置，不会中途锁死位置
    anchor_pos_xy = DoneTerm(
        func=mdp.bad_anchor_pos_xy,
        params={"command_name": "motion", "threshold": 0.5},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.4,  # 同步到你确认过的实际值，原来这里还是 0.25
            "body_names": [
                "ankle_roll_l_link",
                "ankle_roll_r_link",
                # "wrist_roll_l_link",
                # "wrist_roll_r_link",
                # "left_ankle_roll_link",
                # "right_ankle_roll_link",
                # "left_wrist_yaw_link",
                # "right_wrist_yaw_link",
            ],
        },
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP。

    【延迟启动版：先让 anchor 在严格阈值下自己收敛，再单独给 ee_body_pos 放宽】

    这份文件之前的历史结论（"ee_body_pos 阈值课程试过4版都比零课程差5倍"）曾经怀疑
    是 commands.py 里 eval_start_frame 类默认值被误改成 0（RSI 失效）这个 bug 污染的
    实验——RSI 修复后重新验证过一版"从第0轮就放宽 ee_body_pos"的课程，结果
    error_anchor_pos 在 iter198 仍然停在 0.48，跟同样 RSI 已修复、但零课程那次
    iter200 就到 0.13 相比差了近4倍。这次是干净的对照实验，证明"放宽脚踝阈值拖累
    根部"是真实的物理耦合（脚踩得歪，骨盆的支撑基础本身就是错的，跟阈值判定严不严
    没关系），不是 RSI bug 的假象——之前的怀疑被推翻，原来的结论是对的。

    但零课程本身也有明确的老问题：根部学得好之后，几乎全员死于 ee_body_pos（98%），
    进不了后续训练。这一版用"延迟启动"来同时保留两边的好处：
    - delay_steps=7200（约 iter300，比零课程验证过 iter200 到 0.13 的窗口多留一点
      余量）之前，ee_body_pos 阈值保持在 final_threshold=0.4（严格），跟零课程完全
      一样，让根部有不受干扰的窗口先自己收敛好；
    - delay_steps 之后，阈值从 initial_threshold=0.55 开始（原0.8——上一轮验证过
      0.8 这个放宽幅度会让根部"回不去"，阈值收紧之后 error_anchor_pos 长期停在
      0.47 附近，不是暂时不稳定，是学出了一套不同的应对方式。这次把峰值砍到 0.55，
      测试"伤害是否跟放宽力度成正比"这个还没验证过的假设），用 num_steps=8000
      线性收紧回 0.4——收紧完成大约在 iter633 左右。

    anchor_pos / anchor_pos_xy 依旧不加课程，全程固定 0.4/0.5，这条结论没有变化。
    """

    ee_body_pos_threshold = CurrTerm(
        func=mdp.anneal_termination_threshold,
        params={
            "term_name": "ee_body_pos",
            "initial_threshold": 0.65,  # 原 0.8，Plan 1：降低放宽幅度，测试伤害是否跟力度成正比
            "final_threshold": 0.4,
            "num_steps": 40000,
            "delay_steps": 8000,
        },
    )

##
# Environment configuration
##


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "world"
        # self.viewer.asset_name = "robot"
