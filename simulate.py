"""
simulate.py —— 没有相机也能验证算法：合成 D435 风格深度图并做误差实验

仿真的 D435 误差来源:
  1. 深度噪声 σz = z² · σ_sub / (fx · B)，σ_sub=0.08 px，B=50 mm（随距离平方增长）
  2. 遮挡阴影：右侧红外相机看不到的区域深度无效（物体一侧出现黑边）
  3. 边缘飞点：前景/背景交界像素取中间值
  4. 随机丢点；可选"黑色胶带"条带（大面积丢点）

用法:
  python simulate.py                 # 跑全部实验，结果存到 results/
  python simulate.py --show          # 只生成一张示例可视化
"""
import argparse, json, os, time
import numpy as np
import cv2
from volume_core import Intrinsics, calibrate_plane, measure, multi_frame_median, billing

K = Intrinsics(848, 480, 430.0, 430.0, 424.0, 240.0)
BASELINE = 0.050
SIGMA_SUB = 0.08


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def camera_pose(cam_h, pitch_deg):
    """世界系: 台面 Z=0，Z 向上。相机朝下看，pitch 为偏离竖直的倾角"""
    R_down = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)  # 相机 z 轴 → 世界 -Z
    R = R_down @ rot_x(np.deg2rad(pitch_deg))
    C = np.array([0.0, 0.0, cam_h])
    return R, C


def raycast(R, C, box, K=K):
    """返回 z-depth 图（米）。box=(cx,cy,L,W,H,yaw_deg) 或 None"""
    u, v = np.meshgrid(np.arange(K.width), np.arange(K.height))
    d_cam = np.dstack([(u - K.cx) / K.fx, (v - K.cy) / K.fy, np.ones_like(u, float)]).reshape(-1, 3)
    d_w = d_cam @ R.T
    t_tab = np.where(d_w[:, 2] < -1e-9, -C[2] / d_w[:, 2], np.inf)
    t = t_tab
    if box is not None:
        bx, by, L, W, H, yaw = box
        c, s = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
        Rb = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        o = (C - np.array([bx, by, H / 2])) @ Rb          # 世界 → 盒子坐标
        dl = d_w @ Rb
        half = np.array([L / 2, W / 2, H / 2])
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-half - o) / dl
            t2 = (half - o) / dl
        tn = np.nanmax(np.minimum(t1, t2), axis=1)
        tf = np.nanmin(np.maximum(t1, t2), axis=1)
        hit = (tn <= tf) & (tn > 0)
        t = np.where(hit & (tn < t), tn, t)
    depth = t.reshape(K.height, K.width)               # d_cam.z = 1 → t 就是 z-depth
    depth[~np.isfinite(depth)] = 0
    return depth


def smooth_field(rng, shape, cells=(6, 10)):
    """低频空间相关噪声（模拟 D435 深度图的"波纹"/系统性起伏），均值 0、标准差约 1"""
    f = rng.normal(0, 1, cells).astype(np.float32)
    return cv2.resize(f, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def d435_noise(depth, R, C, box, rng, tape=False, scale=1.0, wobble=None):
    """对理想深度加 D435 风格误差。
    scale : 本次开机的深度比例误差（系统误差，多帧平均也消不掉）
    wobble: 低频起伏场（同一次开机内固定）"""
    z = depth.copy()
    valid = z > 0
    # --- 遮挡阴影：从右相机(沿相机 x 轴平移 B)看不到的点置 0
    C2 = C + R[:, 0] * BASELINE
    d2 = raycast(R, C2, box)
    u, v = np.meshgrid(np.arange(K.width), np.arange(K.height))
    X = (u - K.cx) / K.fx * z
    Y = (v - K.cy) / K.fy * z
    Xr = X - BASELINE
    ur = np.round(Xr / np.maximum(z, 1e-6) * K.fx + K.cx).astype(int)
    inb = valid & (ur >= 0) & (ur < K.width)
    occl = np.zeros_like(valid)
    occl[inb] = d2[v[inb], ur[inb]] < z[inb] - 0.005
    z[occl] = 0
    # --- 边缘飞点
    gx = np.abs(cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3))
    edge = cv2.dilate(((gx + gy) > 0.08).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & (z > 0)
    fly = edge & (rng.random(z.shape) < 0.5)
    blur = cv2.blur(depth.astype(np.float32), (7, 7))
    z[fly] = blur[fly] + rng.normal(0, 0.01, fly.sum())
    # --- 高斯噪声（随 z² 增长）+ 低频起伏
    sigma = z ** 2 * SIGMA_SUB / (K.fx * BASELINE)
    z = z + rng.normal(0, 1, z.shape) * sigma
    if wobble is not None:
        z = z + wobble * sigma                # 低频起伏，幅度与 σz 同量级
    z = z * scale
    z[depth == 0] = 0
    # --- 随机丢点
    z[rng.random(z.shape) < 0.01] = 0
    # --- 黑色胶带：箱子顶面中间一条横贯的胶带，85% 丢点
    if tape and box is not None:
        Pw = (np.dstack([X, Y, depth]).reshape(-1, 3) @ R.T + C).reshape(depth.shape + (3,))
        bx, by, L, W, H, yaw = box
        c, s = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
        lx = (Pw[..., 0] - bx) * c + (Pw[..., 1] - by) * s        # 盒子坐标系 x（长边方向）
        top = Pw[..., 2] > H - 0.003
        band = top & (np.abs(lx) < 0.025)                          # 5 cm 宽胶带
        z[band & (rng.random(z.shape) < 0.85)] = 0
    z[z < 0] = 0
    return z.astype(np.float32)


def naive_measure(depth, plane):
    """对照组（基线）：同样的分割，但用全部物体点求外接矩形、高度取最大值、单帧"""
    from volume_core import deproject
    P = deproject(depth, K)
    h = (P.reshape(-1, 3) @ plane.n + plane.d).reshape(depth.shape)
    obj = ((h > 0.01) & (depth > 0)).astype(np.uint8)
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(obj)
    if n <= 1:
        return None
    obj = lbl == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    pts = P[obj]
    nrm = plane.n
    a = np.cross(nrm, [1.0, 0, 0]); a /= np.linalg.norm(a); b = np.cross(nrm, a)
    (_, _), (w, hh), _ = cv2.minAreaRect(np.c_[pts @ a, pts @ b].astype(np.float32))
    return max(w, hh) * 1000, min(w, hh) * 1000, h[obj].max() * 1000


def one_trial(cam_h, pitch, box, rng, frames=5, tape=False):
    R, C = camera_pose(cam_h, pitch)
    scale = 1 + rng.normal(0, 0.002)                 # 0.2% 深度比例误差（每次开机不同）
    wob = smooth_field(rng, (K.height, K.width))
    empty = [d435_noise(raycast(R, C, None), R, C, None, rng, scale=scale, wobble=wob) for _ in range(frames)]
    plane = calibrate_plane(multi_frame_median(empty), K)
    ideal = raycast(R, C, box)
    fr = [d435_noise(ideal, R, C, box, rng, tape, scale=scale, wobble=wob) for _ in range(frames)]
    t0 = time.perf_counter()
    m = measure(multi_frame_median(fr), K, plane)
    dt = (time.perf_counter() - t0) * 1000
    nv = naive_measure(fr[0], plane)
    return m, nv, dt, fr[0]


def random_box(rng, cam_h=1.0):
    """随机纸箱。顶面到相机至少留 0.4 m（D435 近距离精度差、视场小）"""
    L = rng.uniform(0.15, 0.45); W = rng.uniform(0.10, min(L, 0.35))
    H = rng.uniform(0.05, min(0.30, cam_h - 0.40))
    return (rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), L, W, H, rng.uniform(0, 90))


def run_experiments(out="results", n=20, seed=42):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for cam_h in [0.6, 0.8, 1.0, 1.2]:
        for i in range(n):
            box = random_box(rng, cam_h)
            m, nv, dt, _ = one_trial(cam_h, rng.uniform(0, 8), box, rng)
            gt = np.array([max(box[2], box[3]), min(box[2], box[3]), box[4]]) * 1000
            if not m.ok:
                rows.append(dict(cam_h=cam_h, ok=False)); continue
            est = np.array([m.L_mm, m.W_mm, m.H_mm])
            rows.append(dict(cam_h=cam_h, ok=True, gt=gt.tolist(), est=est.tolist(),
                             naive=[float(x) for x in nv] if nv else None, ms=dt,
                             vol_err=float((np.prod(est) / np.prod(gt) - 1) * 100),
                             vol_err_naive=float((np.prod(nv) / np.prod(gt) - 1) * 100) if nv else None))
        print(f"cam_h={cam_h} m 完成")
    # 重复性：同一个箱子测 20 次
    rep = []
    box = (0.0, 0.0, 0.30, 0.20, 0.15, 25.0)
    for _ in range(20):
        m, *_ = one_trial(0.8, 3.0, box, rng)
        rep.append([m.L_mm, m.W_mm, m.H_mm])
    # 黑色胶带
    m_tape, *_ = one_trial(0.8, 3.0, box, rng, tape=True)

    summ = {}
    for cam_h in [0.6, 0.8, 1.0, 1.2]:
        r = [x for x in rows if x["cam_h"] == cam_h and x["ok"]]
        e = np.array([np.array(x["est"]) - np.array(x["gt"]) for x in r])
        en = np.array([np.array(x["naive"]) - np.array(x["gt"]) for x in r if x["naive"]])
        summ[cam_h] = dict(n=len(r), fail=n - len(r),
                           mae_LWH=np.abs(e).mean(0).round(2).tolist(),
                           mae_LWH_naive=np.abs(en).mean(0).round(2).tolist(),
                           vol_mape=float(np.mean([abs(x["vol_err"]) for x in r])),
                           vol_mape_naive=float(np.mean([abs(x["vol_err_naive"]) for x in r if x["vol_err_naive"] is not None])),
                           ms=float(np.mean([x["ms"] for x in r])))
    rep = np.array(rep)
    result = dict(summary=summ, repeat_std_mm=rep.std(0, ddof=1).round(2).tolist(),
                  repeat_mean_mm=rep.mean(0).round(1).tolist(), repeat_gt=[300, 200, 150],
                  tape=dict(L=m_tape.L_mm, W=m_tape.W_mm, H=m_tape.H_mm, valid_ratio=m_tape.valid_ratio, msg=m_tape.msg))
    with open(os.path.join(out, "sim_results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out, "sim_trials.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    return result


def draw_result(depth, m, path):
    v = depth[depth > 0]
    lo, hi = np.percentile(v, 1), np.percentile(v, 99)
    norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    vis = cv2.applyColorMap((255 - norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)   # 近=红 远=蓝
    vis[depth == 0] = 0
    if m.ok:
        cnt, _ = cv2.findContours(m.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnt, -1, (255, 255, 255), 1)
        cv2.polylines(vis, [m.box_px.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        cv2.putText(vis, f"L{m.L_mm:.0f} W{m.W_mm:.0f} H{m.H_mm:.0f} mm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(path, vis)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(7)
    box = (0.02, -0.01, 0.30, 0.20, 0.15, 25.0)
    m, nv, dt, raw = one_trial(0.8, 4.0, box, rng)
    print(f"真值 300×200×150 mm → 测量 {m.L_mm:.1f}×{m.W_mm:.1f}×{m.H_mm:.1f} mm，"
          f"对照组 {nv[0]:.1f}×{nv[1]:.1f}×{nv[2]:.1f} mm，耗时 {dt:.1f} ms，{m.msg}")
    vk, ck, foam = billing(m, actual_kg=1.2, divisor=6000)
    print(f"抛比6000: 体积重 {vk:.2f} kg, 实重 1.20 kg → 计费 {ck:.0f} kg, {'泡货!' if foam else '按实重'}")
    draw_result(raw, m, "results/example.png")
    if not a.show:
        r = run_experiments(n=a.n)
        print(json.dumps(r, ensure_ascii=False, indent=1))
