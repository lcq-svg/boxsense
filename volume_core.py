"""
volume_core.py —— 包裹体积测量核心算法（只依赖 numpy + OpenCV，PC 和树莓派都能跑）

流程:
  深度图 ──反投影──> 有序点云 (H×W×3)
  空台面 ──RANSAC──> 台面平面 n·p + d = 0         (calibrate_plane)
  放上包裹 ──> 每个像素到台面的高度图 h(u,v)
          ──> 高度阈值 + 形态学 + 最大连通域 = 物体掩码
          ──> 高 H = 掩码内高度的 98% 分位数（抗边缘飞点）
          ──> 顶面点(h > 0.9H) 投影到台面坐标系 ──> cv2.minAreaRect ──> 长 L、宽 W
  体积重 = L×W×H(cm³) / 抛比；与实重比较 ──> 是否泡货
"""
from dataclasses import dataclass, field
import numpy as np
import cv2


# ----------------------------------------------------------------- 相机模型
@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @staticmethod
    def from_rs(intr):
        """从 pyrealsense2 的 rs.intrinsics 转换"""
        return Intrinsics(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)


def deproject(depth_m: np.ndarray, K: Intrinsics) -> np.ndarray:
    """深度图(米) -> 有序点云 H×W×3（相机坐标系，无效点为 0）"""
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth_m.astype(np.float32)
    x = (u - K.cx) / K.fx * z
    y = (v - K.cy) / K.fy * z
    return np.dstack([x, y, z])


# ----------------------------------------------------------------- 平面标定
@dataclass
class Plane:
    n: np.ndarray          # 单位法向量，指向相机一侧（离台面越远高度越大）
    d: float
    inlier_ratio: float = 0.0
    rms_mm: float = 0.0

    def height(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.n + self.d


def fit_plane_ransac(pts: np.ndarray, thresh=0.004, iters=500, rng=None) -> Plane:
    """pts: N×3。RANSAC 粗拟合 + 内点最小二乘精修"""
    rng = np.random.default_rng(0) if rng is None else rng
    N = len(pts)
    best_cnt, best = -1, None
    for _ in range(iters):
        p = pts[rng.choice(N, 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        d = -n @ p[0]
        cnt = np.count_nonzero(np.abs(pts @ n + d) < thresh)
        if cnt > best_cnt:
            best_cnt, best = cnt, (n, d)
    n, d = best
    inl = pts[np.abs(pts @ n + d) < thresh]
    c = inl.mean(0)                                   # SVD 精修
    n = np.linalg.svd(inl - c, full_matrices=False)[2][-1]
    d = -n @ c
    if d < 0:                                         # 让相机原点(0,0,0)在法向正侧
        n, d = -n, -d
    res = inl @ n + d
    return Plane(n, float(d), len(inl) / N, float(np.sqrt(np.mean(res ** 2)) * 1000))


def calibrate_plane(depth_m: np.ndarray, K: Intrinsics, roi_shrink=0.1) -> Plane:
    """空台面标定。roi_shrink: 去掉四周一圈，避免拍到台面以外"""
    P = deproject(depth_m, K)
    h, w = depth_m.shape
    my, mx = int(h * roi_shrink), int(w * roi_shrink)
    pts = P[my:h - my, mx:w - mx].reshape(-1, 3)
    pts = pts[pts[:, 2] > 0]
    if len(pts) > 40000:
        pts = pts[np.random.default_rng(1).choice(len(pts), 40000, replace=False)]
    return fit_plane_ransac(pts)


# ----------------------------------------------------------------- 测量
@dataclass
class Measurement:
    ok: bool
    L_mm: float = 0.0
    W_mm: float = 0.0
    H_mm: float = 0.0
    yaw_deg: float = 0.0
    valid_ratio: float = 0.0           # 物体区域内有效深度比例（低说明黑色/反光表面）
    msg: str = ""
    mask: np.ndarray = field(default=None, repr=False)
    box_px: np.ndarray = field(default=None, repr=False)   # 顶面矩形在图像里的 4 个角点，用于画图

    @property
    def volume_cm3(self):
        return self.L_mm * self.W_mm * self.H_mm / 1000.0


def measure(depth_m: np.ndarray, K: Intrinsics, plane: Plane,
            min_h=0.010, h_pct=98.0, top_ratio=0.90, min_area_px=800) -> Measurement:
    P = deproject(depth_m, K)
    valid = depth_m > 0
    hmap = np.where(valid, plane.height(P.reshape(-1, 3)).reshape(depth_m.shape), 0.0)

    # 1) 物体掩码：高于台面 min_h 的像素 → 开运算去孤立飞点
    #    → 闭运算把被深度空洞（黑胶带等）切开的区域连起来 → 最大连通域
    #    阈值随安装高度自适应：取 max(min_h, 4σz)，σz 按 D435 立体视觉误差模型估计
    sigma_z = plane.d ** 2 * 0.08 / (K.fx * 0.050)
    min_h = max(min_h, 4 * sigma_z)
    raw = ((hmap > min_h) & valid).astype(np.uint8)
    m = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    region = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    region = cv2.morphologyEx(region, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    if n_lbl <= 1:
        return Measurement(False, msg="台面上未检测到物体")
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    if stats[k, cv2.CC_STAT_AREA] < min_area_px:
        return Measurement(False, msg="物体太小或被噪声淹没")
    region = (lbl == k)
    obj = region & (m > 0)                       # 区域内真正有有效深度的像素
    x0, y0, bw, bh = stats[k, :4]
    if x0 <= 2 or y0 <= 2 or x0 + bw >= depth_m.shape[1] - 2 or y0 + bh >= depth_m.shape[0] - 2:
        return Measurement(False, msg="物体超出视场，请移到台面中央或升高相机", mask=obj)

    # 有效深度比例：物体凸包面积内，有效像素占多少（低 → 黑色/反光/透明表面）
    cnts, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = np.zeros(region.shape, np.uint8)
    cv2.fillPoly(hull, [cv2.convexHull(np.vstack(cnts))], 1)
    hull = cv2.erode(hull, np.ones((5, 5), np.uint8))   # 去掉轮廓线本身
    valid_ratio = (obj & (hull > 0)).sum() / max(hull.sum(), 1)

    # 2) 高度：腐蚀掉边缘（飞点集中在边缘），取分位数
    core = cv2.erode(obj.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    if core.sum() < 50:
        core = obj
    H0 = float(np.percentile(hmap[core], h_pct))        # 粗估（分位数抗飞点，但被噪声往上拉 ~2σ）
    top = obj & (hmap > top_ratio * H0)                 # 顶面点
    top_core = top & core
    H = float(np.median(hmap[top_core])) if top_core.sum() > 50 else H0   # 顶面高度中值：无偏

    # 3) 长宽：只用顶面点，投影到台面坐标系后求最小外接矩形
    pts = P[top]
    n = plane.n
    a = np.cross(n, [1.0, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    uv = np.c_[pts @ a, pts @ b].astype(np.float32)
    (cx, cy), (w, h), ang = cv2.minAreaRect(uv)
    L, W = max(w, h), min(w, h)

    # 顶面矩形 -> 图像坐标（画图用）
    corners = cv2.boxPoints(((cx, cy), (w, h), ang))
    # 平面坐标 (u,v) + 高度 h 的三维点: p = u·a + v·b + (h − d)·n
    c3 = np.array([c[0] * a + c[1] * b + (H - plane.d) * n for c in corners])
    px =np.c_[c3[:, 0] / c3[:, 2] * K.fx + K.cx, c3[:, 1] / c3[:, 2] * K.fy + K.cy]

    return Measurement(True, L * 1000, W * 1000, H * 1000, float(ang), float(valid_ratio),
                       "OK" if valid_ratio > 0.85 else "警告：物体表面深度缺失较多（黑色/反光/透明？）",
                       obj, px.astype(np.int32))


# ----------------------------------------------------------------- 计费
DIM_DIVISORS = {"顺丰特快": 6000, "通达系(中通/圆通/申通/韵达)": 8000}


def billing(m: Measurement, actual_kg: float, divisor=6000):
    """返回 (体积重kg, 计费重kg, 是否泡货)。计费重向上取整到 1kg（常见规则，各家可能不同）"""
    vol_kg = m.L_mm / 10 * m.W_mm / 10 * m.H_mm / 10 / divisor
    charge = max(vol_kg, actual_kg)
    return vol_kg, float(np.ceil(charge)), vol_kg > actual_kg


def multi_frame_median(frames):
    """多帧中值融合：0 视为无效，不参与统计"""
    st = np.stack(frames).astype(np.float32)
    st[st <= 0] = np.nan
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # 全 NaN 的像素
        med = np.nanmedian(st, axis=0)
    return np.nan_to_num(med, nan=0.0)
