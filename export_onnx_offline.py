import os
import torch
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution
from isaaclab.envs import ManagerBasedRLEnv
from whole_body_tracking.utils.exporter import export_motion_policy_as_onnx, attach_onnx_metadata
from whole_body_tracking.tasks.tracking.mdp import MotionCommand
import hydra
from isaaclab_tasks.utils.hydra import hydra_task_config

# ===================== 配置区（按需修改）=====================
CKPT_PATH = "/home/felix/isaac/xMimic/logs/rsl_rl/dex_evt_flat/2026-07-04_20-40-02/model_0.pt"
EXPORT_SAVE_DIR = "/home/felix/isaac/xMimic/export_onnx"
TASK_NAME = "Tracking-Flat-DexEVT-Wo-State-Estimation-v0"
MOTION_NPZ = "/home/felix/isaac/xMimic/motion_npz/dance1_subject1.npz"
ACTOR_HIDDEN = [512, 256, 128]
ACTOR_ACTIVATION = "elu"
INIT_STD = 1.0
STD_TYPE = "scalar"
# ============================================================

def remove_all_weight_norm(module: torch.nn.Module):
    """递归移除所有权重归一化，解决deepcopy报错"""
    for sub in module.children():
        try:
            torch.nn.utils.remove_weight_norm(sub)
        except ValueError:
            pass
        remove_all_weight_norm(sub)

# 临时包装类，兼容旧导出器需要的actor_critic结构
class TempActorCriticWrapper(torch.nn.Module):
    def __init__(self, actor_model):
        super().__init__()
        self.actor = actor_model
        self.actor_obs_normalizer = getattr(actor_model, "obs_normalizer", None)
        self.is_recurrent = False
        self.num_actions = actor_model.mlp[-1].out_features
        self.in_features = actor_model.mlp[0].in_features

@hydra_task_config(TASK_NAME, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    # 绑定motion文件
    env_cfg.commands.motion.motion_file = MOTION_NPZ
    # 构建环境
    env = torch.nn.Module()
    import gymnasium as gym
    env = gym.make(TASK_NAME, cfg=env_cfg, render_mode=None).unwrapped

    # 加载训练保存的权重
    ckpt_data = torch.load(CKPT_PATH, map_location="cpu")
    num_actions = env.action_manager.action_term_dim
    obs_groups = {"actor": ["policy"], "critic": ["policy", "critic"]}

    # 构造Actor模型（和训练时参数完全一致）
    actor_cfg = {
        "hidden_dims": ACTOR_HIDDEN,
        "activation": ACTOR_ACTIVATION,
        "distribution_cfg": {
            "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
            "init_std": INIT_STD,
            "std_type": STD_TYPE
        }
    }
    obs_dim = env.observation_manager.group_obs_dim["policy"]
    actor = MLPModel(
        obs_dim=obs_dim,
        obs_groups=obs_groups,
        group_name="actor",
        num_actions=num_actions,
        **actor_cfg
    )
    # 加载训练权重
    actor.load_state_dict(ckpt_data["actor_state_dict"])
    actor.eval()
    actor.to("cpu")
    # 关键：移除weight_norm，规避deepcopy崩溃
    remove_all_weight_norm(actor)

    # 包装为导出器需要的actor_critic对象
    wrapped_policy = TempActorCriticWrapper(actor)

    # 导出目录
    os.makedirs(EXPORT_SAVE_DIR, exist_ok=True)
    export_name = "motion_policy_export.onnx"

    # 执行motion专属导出
    export_motion_policy_as_onnx(
        env=env,
        actor_critic=wrapped_policy,
        normalizer=wrapped_policy.actor_obs_normalizer,
        path=EXPORT_SAVE_DIR,
        filename=export_name,
        verbose=False
    )
    # 附加机器人/动作元数据
    run_tag = os.path.basename(os.path.dirname(os.path.normpath(CKPT_PATH)))
    attach_onnx_metadata(
        env=env,
        run_path=run_tag,
        path=EXPORT_SAVE_DIR,
        filename=export_name
    )
    print(f"✅ ONNX导出完成，路径：{os.path.join(EXPORT_SAVE_DIR, export_name)}")

if __name__ == "__main__":
    main()

