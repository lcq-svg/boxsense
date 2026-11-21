"""
pi_node.py —— 树莓派 4B 下位机：HX711 称重 + LED/蜂鸣器提示，TCP 服务端（端口 9000）

  python3 pi_node.py --mock        # 没接硬件时先用模拟模式联调
  python3 pi_node.py               # 真机（需要 sudo apt install python3-rpi.gpio）
  python3 pi_node.py --calib 1.000 # 放 1.000 kg 砝码求比例系数，写到 hx711_cal.json

接线 (BCM 编号):  HX711 DT→GPIO5  SCK→GPIO6   绿LED→GPIO17  红LED→GPIO27  蜂鸣器→GPIO22
协议: 一行一个 JSON
  {"cmd":"weight"}            → {"ok":true,"kg":1.234}
  {"cmd":"tare"}              → {"ok":true}
  {"cmd":"result","foam":true,...} → 泡货亮红灯+响两声，否则亮绿灯  → {"ok":true}
  {"cmd":"ping"}              → {"ok":true,"t":...}       （心跳）
"""
import argparse, json, os, socket, socketserver, statistics, threading, time

CAL_FILE = "hx711_cal.json"
PIN_DT, PIN_SCK, PIN_GREEN, PIN_RED, PIN_BUZZ = 5, 6, 17, 27, 22


class HX711:
    """位操作读取 HX711（A 通道，128 增益）"""
    def __init__(self, GPIO):
        self.G = GPIO
        GPIO.setup(PIN_SCK, GPIO.OUT); GPIO.setup(PIN_DT, GPIO.IN)
        GPIO.output(PIN_SCK, 0)

    def read_raw(self):
        G = self.G
        t0 = time.time()
        while G.input(PIN_DT):                  # DT 拉低表示数据就绪
            if time.time() - t0 > 0.5:
                raise TimeoutError("HX711 无响应，检查接线")
        v = 0
        for _ in range(24):
            G.output(PIN_SCK, 1); G.output(PIN_SCK, 0)
            v = (v << 1) | G.input(PIN_DT)
        G.output(PIN_SCK, 1); G.output(PIN_SCK, 0)   # 第 25 个脉冲：下次仍选 A 通道 128 增益
        return v - (1 << 24) if v & 0x800000 else v


class MockScale:
    def __init__(self):
        self.k = 1.237
    def read_raw(self):
        import random
        return int(self.k * 420000 + 8000 + random.gauss(0, 150))


class Station:
    def __init__(self, mock):
        self.mock, self.lock = mock, threading.Lock()
        cal = json.load(open(CAL_FILE)) if os.path.exists(CAL_FILE) else {"offset": 8000, "per_kg": 420000}
        self.offset, self.per_kg = cal["offset"], cal["per_kg"]
        if mock:
            self.GPIO, self.adc = None, MockScale()
        else:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM); GPIO.setwarnings(False)
            for p in (PIN_GREEN, PIN_RED, PIN_BUZZ):
                GPIO.setup(p, GPIO.OUT, initial=0)
            self.GPIO, self.adc = GPIO, HX711(GPIO)

    def raw(self, n=15):
        with self.lock:
            vals = sorted(self.adc.read_raw() for _ in range(n))
        return statistics.mean(vals[n // 5: n - n // 5])     # 去掉首尾 20% 后求均值，抗毛刺

    def kg(self):
        return (self.raw() - self.offset) / self.per_kg

    def tare(self):
        self.offset = self.raw(30); self.save()

    def calibrate(self, ref_kg):
        input("清空秤盘后回车..."); self.tare()
        input(f"放上 {ref_kg} kg 砝码后回车..."); self.per_kg = (self.raw(30) - self.offset) / ref_kg
        self.save(); print("标定完成:", self.offset, self.per_kg)

    def save(self):
        json.dump({"offset": self.offset, "per_kg": self.per_kg}, open(CAL_FILE, "w"))

    def signal(self, foam):
        print(f"[提示] {'泡货 → 红灯+蜂鸣' if foam else '正常 → 绿灯'}")
        if self.mock:
            return
        G = self.GPIO
        G.output(PIN_GREEN, 0 if foam else 1); G.output(PIN_RED, 1 if foam else 0)
        if foam:
            for _ in range(2):
                G.output(PIN_BUZZ, 1); time.sleep(0.15); G.output(PIN_BUZZ, 0); time.sleep(0.1)


def make_handler(st):
    class H(socketserver.StreamRequestHandler):
        def handle(self):
            print("上位机已连接:", self.client_address)
            for line in self.rfile:
                try:
                    req = json.loads(line)
                    c = req.get("cmd")
                    if c == "weight":
                        resp = {"ok": True, "kg": round(st.kg(), 4)}
                    elif c == "tare":
                        st.tare(); resp = {"ok": True}
                    elif c == "result":
                        st.signal(bool(req.get("foam"))); resp = {"ok": True}
                    elif c == "ping":
                        resp = {"ok": True, "t": time.time()}
                    else:
                        resp = {"ok": False, "err": f"未知命令 {c}"}
                except Exception as e:
                    resp = {"ok": False, "err": str(e)}
                self.wfile.write((json.dumps(resp, ensure_ascii=False) + "\n").encode())
    return H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--calib", type=float, default=None)
    a = ap.parse_args()
    st = Station(a.mock)
    if a.calib:
        st.calibrate(a.calib)
    else:
        socketserver.ThreadingTCPServer.allow_reuse_address = True
        with socketserver.ThreadingTCPServer(("0.0.0.0", a.port), make_handler(st)) as srv:
            print(f"树莓派节点已启动 :{a.port}  ({'模拟' if a.mock else '真机'})")
            srv.serve_forever()
