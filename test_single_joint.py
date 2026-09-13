"""
单关节方向测试脚本：只把 shoulder_pitch_l/r_joint 设成一个明显的正值，
其余关节保持默认姿态不动，直接肉眼观察：DEX_EVT的shoulder_pitch正方向,
到底是"手臂往前摆"还是"手臂往后摆"。

用法：
    python test_single_joint.py --joint shoulder_pitch_l_joint --value 0.8
    python test_single_joint.py --joint shoulder_pitch_r_joint --value 0.8

可以多次运行，换不同joint/value组合，逐一排查是哪个关节方向反了。
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--joint", type=str, required=True, help="要测试的关节名，比如 shoulder_pitch_l_joint")
parser.add_argument("--value", type=float, default=0.8, help="要设置的关节角度(rad)，默认0.8(约46度)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from whole_body_tracking.robots.dex_evt import DEX_EVT_CFG


@configclass
class TestSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = DEX_EVT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.01
    sim = SimulationContext(sim_cfg)
    scene_cfg = TestSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot: Articulation = scene["robot"]

    if args_cli.joint not in robot.joint_names:
        print(f"[ERROR] 关节名 '{args_cli.joint}' 不在机器人关节列表里。")
        print(f"可用关节名: {robot.joint_names}")
        simulation_app.close()
        return

    joint_idx = robot.joint_names.index(args_cli.joint)
    print(f"[INFO] 测试关节: {args_cli.joint} (index={joint_idx}), 设置角度: {args_cli.value} rad")
    print(f"[INFO] 请在viewer里观察: 该关节转动后, 手臂是往前摆(靠近正面/胸前方向)还是往后摆")

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, joint_idx] = args_cli.value

    # 持续渲染, 不跑物理, 方便你在viewer里从各个角度绕着看清楚方向
    while simulation_app.is_running():
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim.get_physics_dt())


if __name__ == "__main__":
    main()
    simulation_app.close()
