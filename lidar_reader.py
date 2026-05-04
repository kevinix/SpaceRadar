"""
MS200P 激光雷达 3D 点云采集程序
硬件：ORADAR MS200P 单线激光雷达 + NEMA17 步进电机
接口：串口 COM6，波特率 230400

扫描方案：
  - 步进电机以 2.25°/s 顺时针旋转 180°，耗时 80 秒
  - 雷达自身在扫描平面内 360° 旋转，每秒约 10 圈
  - 二者叠加形成完整 3D 点云

坐标系定义：
  - 雷达平面角 φ（lidar_angle）：0° = 装置正下方，180° = 装置正上方
  - 步进电机方位角 θ（motor_angle）：顺时针，0° 为起始位置
  - 球坐标 → 直角坐标（单位 mm，Z 轴朝上）：
      elevation = φ - 90°
      X = r * cos(elevation) * cos(θ)
      Y = r * cos(elevation) * sin(θ)
      Z = r * sin(elevation)

MS200P 数据帧格式（每帧47字节）：
  [0]       帧头        0x54
  [1]       帧类型      0x2C（每包12点）
  [2-3]     转速        小端序，单位 0.01 °/s
  [4-5]     起始角度    小端序，单位 0.01°
  [6-41]    12个测量点  每点3字节：距离(2字节,mm,小端序) + 强度(1字节)
  [42-43]   终止角度    小端序，单位 0.01°
  [44-45]   时间戳      小端序，单位 ms
  [46]      CRC8 校验
"""

import serial
import struct
import time
import math
import os

# ─────────────── 串口配置 ───────────────
PORT     = "COM6"
BAUDRATE = 230400
TIMEOUT  = 1.0

# ─────────────── 协议常量 ───────────────
FRAME_HEADER   = 0x54
FRAME_TYPE     = 0x2C
POINTS_PER_PKT = 12
FRAME_SIZE     = 47

# ─────────────── 步进电机参数 ───────────────
MOTOR_SPEED_DEG_S = 2.25    # 电机旋转速度，°/s
TOTAL_DURATION    = 80.0    # 总采集时长，秒（电机转 180°）

# ─────────────── CRC8 查找表（多项式 0x4D）───────────────
CRC8_TABLE = [0] * 256

def _build_crc8_table():
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = ((crc << 1) ^ 0x4D) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
        CRC8_TABLE[i] = crc

_build_crc8_table()


def calc_crc8(data: bytes) -> int:
    """CRC8 校验，多项式 0x4D"""
    crc = 0
    for b in data:
        crc = CRC8_TABLE[(crc ^ b) & 0xFF]
    return crc


def parse_frame(raw: bytes):
    """
    解析单帧（47字节）。
    返回：list of (lidar_angle_deg, distance_mm, intensity) 或 None
    """
    if len(raw) < FRAME_SIZE:
        return None
    if raw[0] != FRAME_HEADER or raw[1] != FRAME_TYPE:
        return None

    if calc_crc8(raw[:FRAME_SIZE - 1]) != raw[FRAME_SIZE - 1]:
        return None

    start_angle = struct.unpack_from('<H', raw, 4)[0] / 100.0
    end_angle   = struct.unpack_from('<H', raw, 42)[0] / 100.0

    # 角度步长（处理跨 0° 情况）
    if end_angle >= start_angle:
        step = (end_angle - start_angle) / (POINTS_PER_PKT - 1)
    else:
        step = (end_angle + 360.0 - start_angle) / (POINTS_PER_PKT - 1)

    points = []
    for i in range(POINTS_PER_PKT):
        offset    = 6 + i * 3
        dist_mm   = struct.unpack_from('<H', raw, offset)[0]
        intensity = raw[offset + 2]
        lidar_angle = (start_angle + step * i) % 360.0
        if dist_mm > 0:
            points.append((lidar_angle, dist_mm, intensity))

    return points


def to_xyz(lidar_angle_deg: float, motor_angle_deg: float, dist_mm: float):
    """
    将雷达极坐标 + 电机方位角转换为三维直角坐标（单位 mm）。

    雷达角 φ：0° = 正下方，90° = 水平，180° = 正上方
    电机角 θ：顺时针方位角

    elevation = φ - 90°  （统一到仰角坐标系）
    X = r * cos(elev) * cos(θ)
    Y = r * cos(elev) * sin(θ)
    Z = r * sin(elev)
    """
    elev  = math.radians(lidar_angle_deg - 90.0)
    theta = math.radians(motor_angle_deg)
    r     = dist_mm

    x = r * math.cos(elev) * math.cos(theta)
    y = r * math.cos(elev) * math.sin(theta)
    z = r * math.sin(elev)
    return x, y, z


class LidarReader:
    """MS200P 激光雷达串口读取器"""

    def __init__(self, port: str = PORT, baudrate: int = BAUDRATE):
        self.port     = port
        self.baudrate = baudrate
        self.ser      = None
        self._buf     = bytearray()

    def open(self):
        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=TIMEOUT
        )
        print(f"[INFO] 串口已打开：{self.port} @ {self.baudrate} bps")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[INFO] 串口已关闭")

    def read_available_frames(self):
        """
        从串口缓冲区提取所有完整帧，返回原始点列表。
        每个点：(lidar_angle_deg, distance_mm, intensity)
        """
        if self.ser.in_waiting:
            self._buf.extend(self.ser.read(self.ser.in_waiting))

        points = []
        while True:
            idx = self._buf.find(FRAME_HEADER)
            if idx == -1:
                self._buf.clear()
                break
            if idx > 0:
                del self._buf[:idx]

            if len(self._buf) < FRAME_SIZE:
                break

            frame = bytes(self._buf[:FRAME_SIZE])
            pts   = parse_frame(frame)
            if pts is not None:
                points.extend(pts)
                del self._buf[:FRAME_SIZE]
            else:
                del self._buf[:1]

        return points


def save_ply(cloud_3d, filepath: str):
    """
    将三维点云保存为 PLY 文件（ASCII格式）。
    cloud_3d：list of (x, y, z, intensity)
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(cloud_3d)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar intensity\n")
        f.write("end_header\n")
        for x, y, z, intensity in cloud_3d:
            f.write(f"{x:.3f} {y:.3f} {z:.3f} {intensity}\n")


def save_csv(cloud_3d, filepath: str):
    """
    将三维点云保存为 CSV 文件。
    cloud_3d：list of (x, y, z, intensity)
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("x_mm,y_mm,z_mm,intensity\n")
        for x, y, z, intensity in cloud_3d:
            f.write(f"{x:.3f},{y:.3f},{z:.3f},{intensity}\n")


def main():
    """
    主流程：
      1. 记录采集开始时刻 t0
      2. 持续读取雷达帧，根据 (当前时刻 - t0) 计算电机方位角
      3. 将每个雷达点转换为 3D 坐标并累积
      4. 80 秒后自动结束，保存完整 3D 点云
    """
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    ply_path = os.path.join(output_dir, "pointcloud_3d.ply")
    csv_path = os.path.join(output_dir, "pointcloud_3d.csv")

    print("=" * 62)
    print("  MS200P 激光雷达 3D 点云采集程序")
    print(f"  端口: {PORT}  波特率: {BAUDRATE}")
    print(f"  采集时长: {TOTAL_DURATION:.0f} 秒  电机速度: {MOTOR_SPEED_DEG_S}°/s")
    print(f"  输出路径: {output_dir}")
    print("  按 Ctrl+C 提前停止并保存已采集数据")
    print("=" * 62)

    reader  = LidarReader(PORT, BAUDRATE)
    cloud_3d = []   # 最终 3D 点云：list of (x, y, z, intensity)

    try:
        reader.open()

        t0           = time.time()
        last_print   = t0
        last_second  = -1

        print(f"\n[开始采集] 请勿移动装置，电机正在旋转...")

        while True:
            now     = time.time()
            elapsed = now - t0

            # 超过总时长则结束
            if elapsed >= TOTAL_DURATION:
                break

            # 当前电机方位角（顺时针，0° 起始）
            motor_angle = elapsed * MOTOR_SPEED_DEG_S  # 0° ~ 180°

            # 读取当前缓冲区中的所有帧
            raw_points = reader.read_available_frames()

            # 转换为 3D 坐标
            for lidar_angle, dist_mm, intensity in raw_points:
                x, y, z = to_xyz(lidar_angle, motor_angle, dist_mm)
                cloud_3d.append((x, y, z, intensity))

            # 每秒打印一次进度
            cur_second = int(elapsed)
            if cur_second != last_second:
                last_second = cur_second
                remain = TOTAL_DURATION - elapsed
                print(f"  [{elapsed:5.1f}s / {TOTAL_DURATION:.0f}s]  "
                      f"电机角: {motor_angle:6.2f}°  "
                      f"已采集: {len(cloud_3d):,} 点  "
                      f"剩余: {remain:.0f}s")

            time.sleep(0.005)  # 5ms 轮询

    except serial.SerialException as e:
        print(f"\n[错误] 串口异常: {e}")
        print(f"  请确认：1) 设备已连接  2) 端口号 {PORT} 正确  3) 无其他程序占用")
    except KeyboardInterrupt:
        print(f"\n[中断] 用户提前停止，已采集 {len(cloud_3d):,} 个点")
    finally:
        reader.close()

    # ── 保存结果 ──
    if cloud_3d:
        print(f"\n[保存中] 共 {len(cloud_3d):,} 个 3D 点...")
        save_ply(cloud_3d, ply_path)
        save_csv(cloud_3d, csv_path)
        print(f"  PLY 文件: {ply_path}")
        print(f"  CSV 文件: {csv_path}")
        print(f"\n[完成] 可用 CloudCompare / MeshLab 打开 PLY 文件查看 3D 点云")
    else:
        print("\n[警告] 未采集到任何有效点，文件未保存")


if __name__ == "__main__":
    main()
