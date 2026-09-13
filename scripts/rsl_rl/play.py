"""Script to play a checkpoint if an RL agent from RSL-RL."""
 
"""Launch Isaac Sim Simulator first."""
 
import argparse
import sys
 
from isaaclab.app import AppLauncher
 
# local imports
import cli_args  # isort: skip
 
# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--play_full_motion",
    action="store_true",
    default=False,
    help="Start the reference motion at phase 0 and stop playback after one full trajectory.",
)
parser.add_argument(
    "--play_env_id",
    type=int,
    default=0,
    help="Environment index to monitor for --play_full_motion stopping condition.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument(
    "--keep_running",
    action="store_true",
    default=True,
    help="Prevent automatic exit after video capture or one full motion playback.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
 
# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
 
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
 
"""Rest everything follows."""
 
import gymnasium as gym
import os
import pathlib
import numpy as np
import torch
 
from rsl_rl.runners import OnPolicyRunner
 
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
 
# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
 
# Reward terms that were recently introduced and should be logged during playbacks.
_NEW_REWARD_TERMS = ("joint_torque_l2", "joint_vel_limit", "joint_torque_limit")
 
 
class ClipToLimit(gym.ActionWrapper):
    """Clip raw actions to a fixed scalar limit before env processing."""
 
    def __init__(self, env, limit: float):
        super().__init__(env)
        self.limit = float(limit)
 
    def action(self, action):
        if isinstance(action, torch.Tensor):
            return torch.clamp(action, -self.limit, self.limit)
        return np.clip(action, -self.limit, self.limit)
 
 
def _prepare_full_motion_play(vec_env: RslRlVecEnvWrapper):
    """Align the motion command with its first frame for deterministic playback."""
    base_env = getattr(vec_env, "unwrapped", vec_env)
    command_manager = getattr(base_env, "command_manager", None)
    if command_manager is None:
        return None, None
    try:
        motion_term = command_manager.get_term("motion")
    except KeyError:
        return None, None
 
    env_ids = torch.arange(motion_term.num_envs, device=motion_term.device, dtype=torch.long)
    motion_term.time_steps.zero_()
    horizon_s = float(motion_term.motion.time_step_total) * base_env.step_dt
    motion_term.time_left[env_ids] = horizon_s
 
    if horizon_s > base_env.cfg.episode_length_s:
        base_env.cfg.episode_length_s = horizon_s
        if hasattr(base_env, "episode_length_buf"):
            base_env.episode_length_buf.zero_()
 
    joint_pos = motion_term.joint_pos.clone()
    joint_vel = motion_term.joint_vel.clone()
    motion_term.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
 
    # 锁定原点出生，硬写死根位置，强制全部机器人在原点
    # root_state = torch.cat(
    #     [
    #         motion_term.body_pos_w[:, 0],
    #         motion_term.body_quat_w[:, 0],
    #         motion_term.body_lin_vel_w[:, 0],
    #         motion_term.body_ang_vel_w[:, 0],
    #     ],
    #     dim=-1,
    # )
    # motion_term.robot.write_root_state_to_sim(root_state[env_ids], env_ids=env_ids)
    return motion_term, int(motion_term.motion.time_step_total)
 
 
def _log_new_reward_terms(vec_env: RslRlVecEnvWrapper, env_idx: int = 0):
    """Prints the contribution of the newly added reward terms for a representative environment."""
    reward_manager = getattr(vec_env.unwrapped, "reward_manager", None)
    if reward_manager is None:
        return
 
    log_values = []
    for name, values in reward_manager.get_active_iterable_terms(env_idx=env_idx):
        if name in _NEW_REWARD_TERMS and len(values) > 0:
            log_values.append(f"{name}: {values[0]:.4f}")
 
    # if log_values:
    #     print(f"[REWARD] env {env_idx} | " + ", ".join(log_values))
 
 
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
 
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
 
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
 
    if args_cli.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
        env_cfg.commands.motion.motion_file = args_cli.motion_file

    # ========== 评估/渲染专用设置(不影响训练,训练用的是 train.py) ==========
    # 1) ee_body_pos(脚踝偏差判定)关掉——它是训练筛选工具,评估时瞬时超 0.4m
    #    不代表摔倒,不该打断;anchor_ori 同理关掉。
    env_cfg.terminations.ee_body_pos = None
    env_cfg.terminations.anchor_ori = None
    # 上面这行删掉了 ee_body_pos 这个终止判定，但 CurriculumCfg 里的
    # ee_body_pos_threshold 课程项初始化时需要去 termination_manager 里查到它，
    # 找不到会直接报 ValueError（"Termination term 'ee_body_pos' not found"）。
    # 评估本来就不需要课程学习这套机制，一并去掉。
    if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "ee_body_pos_threshold"):
        env_cfg.curriculum.ee_body_pos_threshold = None
    # 2) anchor_pos 不关,改为放大阈值到 0.7,让它充当"真实摔倒检测器":
    #    正常跟踪时根部 Z 偏差通常在 0.1~0.4 之间,不会触发;真摔倒躺地时根部
    #    比参考低 0.6m 以上,会触发重置——机器人摔了就自动重开,不用等 time_out。
    #    若发现还没摔就被误杀,把 0.7 再调大(如 0.85);若摔了迟迟不重置,调小(如 0.6)。
    env_cfg.terminations.anchor_pos.params["threshold"] = 0.7
    # 3) 评估时从动作开头、以干净的参考姿态起步,去掉训练用的随机化:
    env_cfg.commands.motion.pose_range = {k: (0.0, 0.0) for k in env_cfg.commands.motion.pose_range}
    env_cfg.commands.motion.velocity_range = {k: (0.0, 0.0) for k in env_cfg.commands.motion.velocity_range}
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    # 3.5) 固定起始帧(依赖 commands.py 的 eval_start_frame;训练不受影响,训练时该值为-1即随机相位)
    #      450 帧 ≈ 5.0 秒(fps=90),特意跳过 npz 开头 --start_frames 拼接的约 3.3 秒
    #      "默认姿态→动作首帧"过渡段——该过渡段未开 --knee_modify,脚可能拖地/穿插,
    #      是当前"开局甩腿摔倒"的头号嫌疑。想从头播就改成 0。
    env_cfg.commands.motion.eval_start_frame = 0
    # print("[INFO] 评估模式: ee_body_pos/anchor_ori 已关闭; anchor_pos=0.7 摔倒检测; 初始化随机已去除; 从第450帧(约5s)起播")
    # ================================================================
    # 4) 评估关掉所有随机化/扰动事件
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "randomize_actuator_gains"):
        env_cfg.events.randomize_actuator_gains = None
    
    # 关掉新加的 anchor_pos_xy 终止，XY 偏差到 0.5m 就会被判定重置,画面上和"摔倒"很难区分。关掉后渲染只剩"真摔"(anchor_pos z-only 0.7)
    env_cfg.terminations.anchor_pos_xy = None
    env_cfg.episode_length_s = 60.0   # 大于整段动作时长,让渲染不被 time_out 切断

    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (-3.2, -3.2, 1.0)     # 相对骨盆:右前方各3.2m、高1m(绝对高度约2m)
    env_cfg.viewer.lookat = (0.0, 0.0, -0.2) # 盯骨盆略偏下,全身居中、脚在画面内
    
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # 评估专用：不管 num_envs 多少，强制把 env_origins 挪到地形网格的中心格，
    # 避免默认分配(按 terrain_origins 展开顺序取前 N 个)总是从角落格子开始分配。
    terrain = env.unwrapped.scene.terrain
    if hasattr(terrain, "terrain_origins") and terrain.terrain_origins is not None:
        n_rows, n_cols = terrain.terrain_origins.shape[0], terrain.terrain_origins.shape[1]
        center_origin = terrain.terrain_origins[n_rows // 2, n_cols // 2].clone()
        env.unwrapped.scene.env_origins[:] = center_origin
        print(f"[DEBUG] 地形网格 {n_rows}x{n_cols}，取中心格 ({n_rows//2},{n_cols//2}) 坐标 = {center_origin}")
    else:
        print("[WARN] 没找到 terrain.terrain_origins，请检查属性名(可能因 IsaacLab 版本不同而不同)")
    
    print("[DEBUG] env_origins =", env.unwrapped.scene.env_origins)

    log_dir = os.path.dirname(resume_path)
 
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
 
    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
 
    # clip actions to keep inference consistent with training limits from PPO cfg
    env = ClipToLimit(env, limit=getattr(agent_cfg, "clip_action", np.inf))
 
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
 
    motion_term = None
    motion_max_steps = None
    prev_motion_step = None
    play_env_id = args_cli.play_env_id
    if args_cli.play_full_motion:
        motion_term, motion_max_steps = _prepare_full_motion_play(env)
        if motion_term is not None:
            play_env_id = max(0, min(play_env_id, motion_term.num_envs - 1))
            prev_motion_step = motion_term.time_steps[play_env_id].item()
 
    # --- 桥接旧版 policy= 配置到 rsl_rl 5.x 的 actor/critic 字典 ---
    # 与 train.py 里的转换逻辑保持一致，否则 OnPolicyRunner 构造会因为
    # 缺少 "actor"/"critic" 字段而报 KeyError: 'class_name'
    train_cfg = agent_cfg.to_dict()
    policy_cfg = train_cfg["policy"]
    train_cfg["actor"] = {
        "class_name": "rsl_rl.models.mlp_model:MLPModel",
        "hidden_dims": policy_cfg["actor_hidden_dims"],
        "activation": policy_cfg["activation"],
        "distribution_cfg": {
            "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
            "init_std": policy_cfg["init_noise_std"],
            "std_type": "scalar",
        },
    }
    train_cfg["critic"] = {
        "class_name": "rsl_rl.models.mlp_model:MLPModel",
        "hidden_dims": policy_cfg["critic_hidden_dims"],
        "activation": policy_cfg["activation"],
    }
 
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    # resolve observation normalizer from the actor model (rsl_rl >= 5.0: actor/critic 分离结构)
    actor_obs_normalizer = getattr(ppo_runner.alg.actor, "obs_normalizer", None)
 
    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
 
    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
 
    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.actor,
        normalizer=actor_obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, "local", export_model_dir)
    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # ==== 临时诊断:出生后前30步的高度轨迹 ====
            # if timestep < 30:
                # robot = env.unwrapped.scene["robot"]
                # root_z = robot.data.root_pos_w[0, 2].item()
                # 用 body_names 找两个脚踝的索引(只找一次可以缓存,临时诊断就每步找了)
                # ids = [robot.body_names.index(n) for n in ("ankle_roll_l_link", "ankle_roll_r_link")]
                # lz = robot.data.body_pos_w[0, ids[0], 2].item()
                # rz = robot.data.body_pos_w[0, ids[1], 2].item()
                # print(f"[DEBUG step {timestep}] root_z={root_z:.3f} 左脚z={lz:.3f} 右脚z={rz:.3f}")
            
            if dones.any():
                term_manager = env.unwrapped.termination_manager
                for name, value in term_manager.get_active_iterable_terms(env_idx=0):
                    print(name, value)
        _log_new_reward_terms(env)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            timestep += 1
            if args_cli.video and timestep >= args_cli.video_length:
                break
            # if timestep == args_cli.video_length and not args_cli.keep_running:
            #     break
        if args_cli.play_full_motion and motion_term is not None:
            current_step = motion_term.time_steps[play_env_id].item()
            if current_step < prev_motion_step and not args_cli.keep_running:
                break
            prev_motion_step = current_step
            if motion_max_steps is not None and current_step >= motion_max_steps - 1 and not args_cli.keep_running:
                break
 
    # close the simulator
    env.close()
 
 
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
