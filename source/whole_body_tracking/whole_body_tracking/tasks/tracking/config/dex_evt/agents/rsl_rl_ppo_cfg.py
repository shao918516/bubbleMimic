from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class DexEVTFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env=24 # 原24，加长采样窗口，捕捉10s时序
    max_iterations = 20000
    save_interval = 200
    experiment_name = "dex_evt_flat"
    empirical_normalization = True
    clip_action = 6 # 原100，大幅缩小动作空间，减少力矩爆炸
    # 可选：消除obs_groups警告
    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy", "critic"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.5, # 1.0 → 1.5，提升初始探索
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001, # 0.001~0.02 → 提升熵激励，保持探索
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.012, # 0.01 略微放宽KL限制，避免过早压缩策略分布
        max_grad_norm=1.0,
    )


LOW_FREQ_SCALE = 0.5


@configclass
class DexEVTFlatLowFreqPPORunnerCfg(DexEVTFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)
