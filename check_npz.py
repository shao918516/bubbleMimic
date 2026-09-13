import numpy as np, sys
d = np.load(sys.argv[1], allow_pickle=True)
print("npz keys:", list(d.keys()))
fps = float(d["fps"]) if "fps" in d else 100.0
bp = d["body_pos_w"]          # [T, num_bodies, 3]
print("body_pos_w shape:", bp.shape, "fps:", fps)
# 不确定 body 顺序时,直接打所有 body 的 Z 范围,人肉认出两只脚(全程最低的两个)
for i in range(bp.shape[1]):
    z = bp[:, i, 2]
    print(f"body {i:2d}: z range {z.min():.3f}~{z.max():.3f}  "
          f"z>0.35m 的时刻(秒): {np.where(z>0.35)[0][:5]/fps if (z>0.35).any() else '无'}")

# 膝盖(body 10/11)下探到 0.25m 以下的时间段
for i in [10, 11]:
    z = bp[:, i, 2]
    low = np.where(z < 0.25)[0]
    if len(low):
        # 把连续帧段落合并显示
        splits = np.where(np.diff(low) > 1)[0]
        segs = np.split(low, splits + 1)
        print(f"body {i} 跪地段落(秒): " + ", ".join(f"{s[0]/fps:.1f}~{s[-1]/fps:.1f}" for s in segs))
        
# 前6秒诊断:root高度、双脚高度、关节最大帧间变化
T6 = int(6 * fps)
jp = d["joint_pos"][:T6]
djp = np.abs(np.diff(jp, axis=0)).max(axis=1)
print(f"前6秒: root_z {bp[:T6,0,2].min():.3f}~{bp[:T6,0,2].max():.3f}")
print(f"前6秒: 左脚z {bp[:T6,14,2].min():.3f}~{bp[:T6,14,2].max():.3f}  右脚z {bp[:T6,15,2].min():.3f}~{bp[:T6,15,2].max():.3f}")
print(f"前6秒: 单帧最大关节变化 {djp.max():.4f} rad @ 第{djp.argmax()}帧({djp.argmax()/fps:.2f}s)")
