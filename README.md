# boxsense

| 文件 | 运行在 | 作用 |
|---|---|---|
| `volume_core.py` | PC / 树莓派 | 核心算法（只依赖 numpy + opencv） |
| `simulate.py` | PC | 不接相机：合成 D435 风格深度图，跑误差实验 |
| `rs_live.py` | PC + RealSense | 真机测量（c 标定、m 测量、s 保存、q 退出） |
| `pi_node.py` | 树莓派 4B | HX711 称重 + LED/蜂鸣器，TCP 9000 端口 |

## 仿真
```bash
pip install numpy opencv-python
python simulate.py            # 约 30 秒，结果在 results/
```

## 接相机
```bash
pip install pyrealsense2 numpy opencv-python
python rs_live.py
```
相机**垂直朝下**，镜头离台面 0.8~1.0 m。先清空台面按 `c` 标定，再放箱子按 `m`。

## 树莓派
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
