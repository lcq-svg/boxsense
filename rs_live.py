"""
rs_live.py —— 接 RealSense 真机运行（PC 端）

  python rs_live.py                     # 只用相机
  python rs_live.py --pi 192.168.1.50   # 连接树莓派：读称重、下发泡货提示

按键:  c = 空台面标定（先把台面清空）   m = 测量   s = 保存当前帧   q = 退出
安装建议: 相机垂直朝下，镜头离台面 0.8~1.0 m（仿真里这个高度综合误差最小）
"""
import argparse, json, os, socket, time
import numpy as np
import cv2
import pyrealsense2 as rs
from volume_core import Intrinsics, calibrate_plane, measure, multi_frame_median, billing, DIM_DIVISORS


class PiLink:
    """和树莓派的 JSON-lines TCP 通信（一行一个 JSON）"""
    def __init__(self, host, port=9000):
        self.addr = (host, port); self.sock = None; self.buf = b""

    def _conn(self):
        if self.sock is None:
            self.sock = socket.create_connection(self.addr, timeout=2.0)

    def call(self, obj):
        try:
            self._conn()
            self.sock.sendall((json.dumps(obj, ensure_ascii=False) + "\n").encode())
            while b"\n" not in self.buf:
                chunk = self.sock.recv(4096)
                if not chunk:
                    raise ConnectionError("树莓派断开")
                self.buf += chunk
            line, self.buf = self.buf.split(b"\n", 1)
            return json.loads(line)
        except (OSError, ValueError) as e:
            print("[PiLink] 通信失败:", e); self.sock = None; self.buf = b""
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi", default=None, help="树莓派 IP")
    ap.add_argument("--frames", type=int, default=10, help="每次测量融合的帧数")
    ap.add_argument("--divisor", type=int, default=6000, help="抛比")
    args = ap.parse_args()

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    prof = pipe.start(cfg)
    dev = prof.get_device().first_depth_sensor()
    if dev.supports(rs.option.visual_preset):
        dev.set_option(rs.option.visual_preset, 3)          # High Accuracy 预设
    if dev.supports(rs.option.emitter_enabled):
        dev.set_option(rs.option.emitter_enabled, 1)        # 打开红外散斑投射器
    scale = dev.get_depth_scale()
    align = rs.align(rs.stream.color)                       # 深度对齐到彩色，方便叠加显示
    spatial, temporal, hole = rs.spatial_filter(), rs.temporal_filter(), rs.hole_filling_filter(1)

    for _ in range(30):                                     # 等自动曝光稳定
        pipe.wait_for_frames()
    fs = align.process(pipe.wait_for_frames())
    K = Intrinsics.from_rs(fs.get_depth_frame().profile.as_video_stream_profile().intrinsics)
    print("内参:", K)

    pi = PiLink(args.pi) if args.pi else None
    plane, last = None, None
    os.makedirs("captures", exist_ok=True)

    def grab(n):
        out, color = [], None
        for _ in range(n):
            f = align.process(pipe.wait_for_frames())
            d = f.get_depth_frame()
            d = temporal.process(spatial.process(d))
            out.append(np.asanyarray(d.get_data()).astype(np.float32) * scale)
            color = np.asanyarray(f.get_color_frame().get_data())
        return multi_frame_median(out), color

    while True:
        depth, color = grab(1)
        vis = color.copy()
        if last is not None and last.ok:
            cv2.polylines(vis, [last.box_px.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        status = "未标定: 清空台面后按 c" if plane is None else f"台面已标定 (RMS {plane.rms_mm:.1f} mm)  按 m 测量"
        cv2.putText(vis, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if last is not None:
            txt = (f"L {last.L_mm:.0f}  W {last.W_mm:.0f}  H {last.H_mm:.0f} mm" if last.ok else last.msg)
            cv2.putText(vis, txt, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        dvis = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255 / 1.5), cv2.COLORMAP_TURBO)
        cv2.imshow("RGB", vis); cv2.imshow("Depth", dvis)
        k = cv2.waitKey(1) & 0xFF

        if k == ord("q"):
            break
        if k == ord("c"):
            d, _ = grab(args.frames)
            plane = calibrate_plane(d, K)
            print(f"标定完成: n={plane.n.round(4)}, 相机到台面 {plane.d*1000:.0f} mm, 内点 {plane.inlier_ratio:.0%}, RMS {plane.rms_mm:.2f} mm")
            if pi: pi.call({"cmd": "tare"})
        if k == ord("m") and plane is not None:
            t0 = time.time()
            d, color = grab(args.frames)
            last = measure(d, K, plane)
            if not last.ok:
                print("测量失败:", last.msg); continue
            w = None
            if pi:
                r = pi.call({"cmd": "weight"})
                w = r.get("kg") if r else None
            print(f"\n长 {last.L_mm:.1f}  宽 {last.W_mm:.1f}  高 {last.H_mm:.1f} mm  体积 {last.volume_cm3:.0f} cm³"
                  f"  ({time.time()-t0:.2f}s, 有效深度 {last.valid_ratio:.0%})  {last.msg}")
            for name, div in DIM_DIVISORS.items():
                vk = last.volume_cm3 / div
                print(f"  {name} 抛比{div}: 体积重 {vk:.2f} kg")
            if w is not None:
                vk, ck, foam = billing(last, w, args.divisor)
                print(f"  实重 {w:.3f} kg → 计费重 {ck:.0f} kg  {'【泡货：按体积计费】' if foam else '按实重计费'}")
                pi.call({"cmd": "result", "foam": bool(foam), "L": last.L_mm, "W": last.W_mm, "H": last.H_mm,
                         "vol_kg": vk, "kg": w})
        if k == ord("s"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            np.save(f"captures/{ts}_depth.npy", depth); cv2.imwrite(f"captures/{ts}_color.png", color)
            print("已保存", ts)

    pipe.stop(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
