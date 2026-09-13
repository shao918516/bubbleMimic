#!/usr/bin/env python3
"""Offline ONNX exporter, launch Isaac Sim first to load pxr"""
import argparse
from isaaclab.app import AppLauncher

# 1. 分两步解析：先提取AppLauncher参数，剩余自定义参数手动解析
parser = argparse.ArgumentParser()
# 先添加IsaacSim启动参数
AppLauncher.add_app_launcher_args(parser)
# 先 parse 只拿 sim 相关参数，剩下给自定义解析
args_cli, rest_args = parser.parse_known_args()

# 2. 新建parser处理自定义参数
custom_parser = argparse.ArgumentParser()
custom_parser.add_argument("--ckpt", type=str, required=True, help="Path to model_xxx.pt")
custom_parser.add_argument("--out_dir", type=str, default="./export_onnx")
custom_parser.add_argument("--task", type=str, required=True)
custom_parser.add_argument("--motion_file", type=str, required=True)
custom_args, _ = custom_parser.parse_known_args(rest_args)

# Launch sim to inject pxr
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 3. 现在再导入所有 IsaacLab 模块，不会报 pxr 缺失
import os
import torch
import gymnasium as gym
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution
from whole_body_tracking.utils.exporter import export_motion_policy_as_onnx, attach_onnx_metadata

# -------------------------- 固定网络配置 --------------------------
ACTOR_HIDDEN = [512, 256, 128]
ACTOR_ACTIVATION = "elu"
INIT_STD = 1.0
STD_TYPE = "scalar"
OBS_GROUPS = {"actor": ["policy"], "critic": ["policy", "critic"]}
# -----------------------------------------------------------------

def remove_all_weight_norm(module: torch.nn.Module):
    for sub in module.children():
        try:
            torch.nn.utils.remove_weight_norm(sub)
        except ValueError:
            pass
        remove_all_weight_norm(sub)

class TempActorCriticWrapper(torch.nn.Module):
    def __init__(self, actor_model):
        super().__init__()
        self.actor = actor_model
        self.actor_obs_normalizer = getattr(actor_model, "obs_normalizer", None)
        self.is_recurrent = False
        self.num_actions = actor_model.mlp[-1].out_features
        self.in_features = actor_model.mlp[0].in_features

@hydra_task_config(custom_args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.commands.motion.motion_file = custom_args.motion_file
    env = gym.make(custom_args.task, cfg=env_cfg, render_mode=None).unwrapped

    ckpt_path = custom_args.ckpt
    export_dir = custom_args.out_dir
    os.makedirs(export_dir, exist_ok=True)

    # 加载权重
    data = torch.load(ckpt_path, map_location="cpu")
    num_actions = env.action_manager.action_term_dim
    obs_dim = env.observation_manager.group_obs_dim["policy"]

    # 重建Actor
    actor_cfg = {
        "hidden_dims": ACTOR_HIDDEN,
        "activation": ACTOR_ACTIVATION,
        "distribution_cfg": {
            "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
            "init_std": INIT_STD,
            "std_type": STD_TYPE
        }
    }
    actor = MLPModel(
        obs_dim=obs_dim,
        obs_groups=OBS_GROUPS,
        group_name="actor",
        num_actions=num_actions,
        **actor_cfg
    )
    actor.load_state_dict(data["actor_state_dict"])
    actor.eval()
    actor.to("cpu")
    remove_all_weight_norm(actor)

    wrapped = TempActorCriticWrapper(actor)
    out_name = "motion_policy.onnx"

    export_motion_policy_as_onnx(
        env=env,
        actor_critic=wrapped,
        normalizer=wrapped.actor_obs_normalizer,
        path=export_dir,
        filename=out_name,
        verbose=False
    )
    run_tag = os.path.basename(os.path.dirname(os.path.normpath(ckpt_path)))
    attach_onnx_metadata(env, run_tag, path=export_dir, filename=out_name)
    print(f"✅ 导出完成: {os.path.join(export_dir, out_name)}")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()

