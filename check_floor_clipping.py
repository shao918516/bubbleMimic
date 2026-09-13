"""
诊断脚本：检查 npz 动作数据里，身体各部位相对地面(z=0)的最低点，
判断是否存在系统性的"脚部穿地"标定偏差，还是只是零星轻微偏差。

用法：
  python check_floor_clipping.py /path/to/motion.npz
  python check_floor_clipping.py /path/to/motion.npz --foot_body_idx 5 11   # 如果已知脚对应的body索引
"""
import sys
import argparse
import numpy as np


def main(path, foot_body_idx=None):
    data = np.load(path)
    body_pos_w = data["body_pos_w"]  # [T, num_bodies, 3]
    T, num_bodies, _ = body_pos_w.shape
    print(f"total frames = {T}, num_bodies = {num_bodies}")

    z = body_pos_w[:, :, 2]  # [T, num_bodies]

    # ---------- 1. 全身所有部位里, 每一帧的最低点 (粗略判断是否有任何部位穿地) ----------
    global_min_z_per_frame = z.min(axis=1)  # [T]
    print()
    print("=== 全身(所有body)里, 每帧的最低点 z ===")
    print(f"  全程最低点: {global_min_z_per_frame.min():.4f} m")
    print(f"  最低点<0 的帧占比: {100*np.mean(global_min_z_per_frame < 0):.2f}%")
    print(f"  最低点<-0.01(1cm) 的帧占比: {100*np.mean(global_min_z_per_frame < -0.01):.2f}%")
    print(f"  最低点<-0.03(3cm) 的帧占比: {100*np.mean(global_min_z_per_frame < -0.03):.2f}%")
    print(f"  最低点<-0.05(5cm) 的帧占比: {100*np.mean(global_min_z_per_frame < -0.05):.2f}%")

    # 找出哪个body最常是"最低点" -> 大概率就是脚
    lowest_body_per_frame = z.argmin(axis=1)
    unique, counts = np.unique(lowest_body_per_frame, return_counts=True)
    order = np.argsort(-counts)
    print()
    print("=== 哪些body最常是全身最低点 (大概率是脚/脚踝) ===")
    for idx in order[:6]:
        b = unique[idx]
        print(f"  body {b:2d}: 是全身最低点的帧数占比 = {100*counts[idx]/T:.1f}%, "
              f"该body自身最低z = {z[:, b].min():.4f} m")

    # ---------- 2. 如果用户指定了脚的body索引, 专门看这些 ----------
    if foot_body_idx:
        print()
        print(f"=== 指定脚部 body 索引 {foot_body_idx} 的详细检查 ===")
        foot_z = z[:, foot_body_idx]  # [T, len(foot_body_idx)]
        foot_min_per_frame = foot_z.min(axis=1)
        print(f"  脚部全程最低点: {foot_min_per_frame.min():.4f} m")
        print(f"  脚部最低点<0 的帧占比: {100*np.mean(foot_min_per_frame < 0):.2f}%")
        print(f"  脚部最低点<-0.03(3cm) 的帧占比: {100*np.mean(foot_min_per_frame < -0.03):.2f}%")

        # 建议的全局垂直修正量: 让最低点恰好落在 z=0 (可留一点安全边距, 比如+0.005)
        suggested_shift = -foot_min_per_frame.min() + 0.005
        print(f"\n  建议的全局垂直修正量(把整条motion的root z统一上移): +{suggested_shift:.4f} m")
    else:
        suggested_shift = -global_min_z_per_frame.min() + 0.005
        print()
        print("=== 未指定脚部body索引, 用全身最低点粗略估算修正量 ===")
        print(f"  建议的全局垂直修正量(粗略, 建议确认脚部索引后重新计算): +{suggested_shift:.4f} m")

    # ---------- 3. 画图看分布 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(global_min_z_per_frame, label="全身最低点 z")
        if foot_body_idx:
            ax.plot(foot_z.min(axis=1), label="指定脚部最低点 z", alpha=0.7)
        ax.axhline(0, color="red", linestyle="--", linewidth=1, label="地面 z=0")
        ax.set_xlabel("frame")
        ax.set_ylabel("z (m)")
        ax.legend()
        ax.set_title("身体最低点相对地面的高度 (负值=穿地)")
        plt.tight_layout()
        plt.savefig("floor_clipping_check.png", dpi=120)
        print("\n已保存图像到: floor_clipping_check.png")
    except ImportError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--foot_body_idx", type=int, nargs="+", default=None,
                         help="脚部对应的body索引(可传多个,比如左右脚)")
    args = parser.parse_args()
    main(args.path, args.foot_body_idx)
