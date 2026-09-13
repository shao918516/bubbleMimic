"""
诊断脚本：检查 motion npz 文件中 joint_vel / body_ang_vel_w 是否存在异常尖峰。
用法：python check_motion_data.py /path/to/dance1_subject2dex_evt2_bvh.npz
"""
import sys
import numpy as np

def main(path):
    data = np.load(path)
    fps = data["fps"]
    joint_pos = data["joint_pos"]          # [T, num_joints]
    joint_vel = data["joint_vel"]          # [T, num_joints]
    body_pos_w = data["body_pos_w"]        # [T, num_bodies, 3]
    body_quat_w = data["body_quat_w"]      # [T, num_bodies, 4]
    body_lin_vel_w = data["body_lin_vel_w"]# [T, num_bodies, 3]
    body_ang_vel_w = data["body_ang_vel_w"]# [T, num_bodies, 3]

    print(f"fps = {fps}")
    print(f"joint_pos shape = {joint_pos.shape}")
    print(f"joint_vel shape = {joint_vel.shape}")
    print(f"body_pos_w shape = {body_pos_w.shape}")
    print(f"body_ang_vel_w shape = {body_ang_vel_w.shape}")
    print()

    T = joint_vel.shape[0]

    # ---------- 1. 关节速度尖峰检测 ----------
    jv_norm = np.linalg.norm(joint_vel, axis=-1) if joint_vel.ndim == 3 else np.abs(joint_vel)
    # 若 joint_vel 是 [T, num_joints]（标量角速度），直接看逐关节的绝对值
    if joint_vel.ndim == 2:
        per_joint_max = np.max(np.abs(joint_vel), axis=0)
        print("=== 每个关节的最大 |角速度| (rad/s) ===")
        for j, v in enumerate(per_joint_max):
            flag = "  <== 异常偏高" if v > 30 else ""
            print(f"  joint {j:2d}: max|vel| = {v:8.3f}{flag}")
        print()

        overall = np.abs(joint_vel)
        flat = overall.flatten()
        idx_sorted = np.argsort(flat)[::-1][:20]
        print("=== 全局 Top-20 关节速度绝对值 及 对应 (frame, joint) ===")
        for idx in idx_sorted:
            frame, joint = np.unravel_index(idx, overall.shape)
            print(f"  frame={frame:5d} joint={joint:2d}  |vel|={overall[frame, joint]:.3f}")
        print()

    # ---------- 2. 根部/body 角速度尖峰检测 ----------
    ang_norm = np.linalg.norm(body_ang_vel_w, axis=-1)  # [T, num_bodies]
    print("=== 每个 body 的最大 |角速度| (rad/s, world frame) ===")
    for b in range(ang_norm.shape[1]):
        v = np.max(ang_norm[:, b])
        flag = "  <== 异常偏高" if v > 20 else ""
        print(f"  body {b:2d}: max|ang_vel| = {v:8.3f}{flag}")
    print()

    flat_ang = ang_norm.flatten()
    idx_sorted = np.argsort(flat_ang)[::-1][:20]
    print("=== 全局 Top-20 body 角速度 及 对应 (frame, body) ===")
    for idx in idx_sorted:
        frame, body = np.unravel_index(idx, ang_norm.shape)
        print(f"  frame={frame:5d} body={body:2d}  |ang_vel|={ang_norm[frame, body]:.3f}")
    print()

    # ---------- 3. 帧间跳变检测（检查是否有不连续/丢帧） ----------
    pos_diff = np.linalg.norm(np.diff(body_pos_w, axis=0), axis=-1)  # [T-1, num_bodies]
    max_jump = np.max(pos_diff)
    max_jump_idx = np.unravel_index(np.argmax(pos_diff), pos_diff.shape)
    print(f"=== 帧间 body 位置跳变最大值 ===")
    print(f"  max jump = {max_jump:.4f} m, 发生在 frame={max_jump_idx[0]}, body={max_jump_idx[1]}")
    print()

    # ---------- 4. 画图（如果有 matplotlib） ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        if joint_vel.ndim == 2:
            axes[0].plot(np.abs(joint_vel))
        axes[0].set_title("Joint |velocity| over time (all joints overlaid)")
        axes[0].set_xlabel("frame")
        axes[0].set_ylabel("|joint_vel| (rad/s)")

        axes[1].plot(ang_norm)
        axes[1].set_title("Body |angular velocity| over time (all bodies overlaid, world frame)")
        axes[1].set_xlabel("frame")
        axes[1].set_ylabel("|ang_vel| (rad/s)")

        plt.tight_layout()
        out_path = "motion_velocity_check.png"
        plt.savefig(out_path, dpi=120)
        print(f"已保存速度曲线图到: {out_path}")
    except ImportError:
        print("未安装 matplotlib，跳过画图（可用 pip install matplotlib 后重跑）")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_motion_data.py /path/to/xxx.npz")
        sys.exit(1)
    main(sys.argv[1])
