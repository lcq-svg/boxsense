# 包裹体积测量 · 验证版

| 文件 | 运行在 | 作用 |
|---|---|---|
| `volume_core.py` | PC / 树莓派 | 核心算法（只依赖 numpy + opencv） |
| `simulate.py` | PC | 不接相机：合成 D435 风格深度图，跑误差实验 |
| `rs_live.py` | PC + RealSense | 真机测量（c 标定、m 测量、s 保存、q 退出） |
| `pi_node.py` | 树莓派 4B | HX711 称重 + LED/蜂鸣器，TCP 9000 端口 |

## 1. 先跑仿真（不需要任何硬件）
```bash
pip install numpy opencv-python
python simulate.py            # 约 30 秒，结果在 results/
```

## 2. 接相机
```bash
pip install pyrealsense2 numpy opencv-python
python rs_live.py
```
相机**垂直朝下**，镜头离台面 0.8~1.0 m。先清空台面按 `c` 标定，再放箱子按 `m`。

## 3. 接树莓派
```bash
# 树莓派上
sudo apt install python3-rpi.gpio
python3 pi_node.py --mock          # 先不接硬件联调
python3 pi_node.py --calib 1.0     # 接好 HX711 后，用 1 kg 砝码标定
python3 pi_node.py
# PC 上
python rs_live.py --pi <树莓派IP>
```
接线（BCM）：HX711 DT→GPIO5，SCK→GPIO6；绿 LED→GPIO17，红 LED→GPIO27，蜂鸣器→GPIO22（LED 串 220 Ω 电阻）。

## 算法流程
深度图 → 反投影成点云 → **RANSAC 拟合台面** → 高度图 → 自适应阈值（4σz）+ 形态学 + 最大连通域 → 物体掩码
→ **顶面点**（h > 0.9H）→ 投影到台面坐标系 → `minAreaRect` 得长宽；顶面高度**中值**得高
→ 体积重 = L×W×H / 抛比，与实重比较判断泡货

## 仿真结果（每个安装高度 20 个随机箱子，每次测量 5 帧中值融合）
| 相机高度 | 超视场失败 | L/W/H 平均绝对误差 (mm) | 体积误差 | 基线方法体积误差 |
|---|---|---|---|---|
| 0.6 m | 5/20 | 3.4 / 1.6 / 0.4 | 2.3% | 64% |
| 0.8 m | 0 | 2.5 / 1.7 / 0.9 | 1.9% | 57% |
| 1.0 m | 0 | 2.4 / 1.1 / 1.2 | 1.5% | 58% |
| 1.2 m | 0 | 1.4 / 0.9 / 3.0 | 2.1% | 44% |

基线方法 = 同样的分割，但用全部物体点求外接矩形、高度取最大值。差距主要来自**边缘飞点**：飞点沿视线落在箱子外侧，把长宽撑大几厘米。

> ⚠️ 这是**仿真**结果，噪声模型是按 D435 立体视觉原理设定的（σz = z²·0.08/(fx·B)、遮挡阴影、飞点、0.2% 比例误差、低频起伏），比真机理想。真机误差预计会更大，需要用卷尺实测验证。
