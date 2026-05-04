"""奥锐达MS200P dToF单线激光雷达串口客户端 — 垂直放置+步进电机旋转版

物理安装:
  雷达垂直放置，由步进电机带动在水平面旋转。
  MS200P 0° 标记朝下（装置正下方），180° 朝上（装置正上方）。

扫描机制:
  1. 点击「捕获」→ 开始接收 MS200P 串口数据
  2. 按下 MCU 按钮 → 步进电机以 2.25°/s 顺时针旋转 180°（80秒）
  3. MS200P 内部电机以 10Hz 旋转，每秒输出约 4500 个点
  4. 软件按时间推算电机方位角，将雷达球坐标转换为 3D 笛卡尔坐标

坐标系（与 lidar_reader.py 一致）:
  雷达角 φ：0°=正下方, 180°=正上方
  电机角 θ：顺时针方位角
  elevation = φ - 90°
  X = r × cos(elev) × cos(θ)
  Y = r × cos(elev) × sin(θ)
  Z = r × sin(elev)

协议参考: 《MS200p dToF单线激光雷达 用户手册》V1.0
CRC8 多项式: 0x4D
"""

import threading
import time
import math
import struct
import numpy as np

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None


class MS200PClient:
    """奥锐达MS200P 垂直安装 + 步进电机3D扫描客户端"""

    # ── 协议常量 ──
    PACKET_HEADER = 0x54
    PACKET_TYPE = 0x2C       # 帧类型：12点数据包
    POINTS_PER_PACKET = 12
    POINT_DATA_SIZE = 3
    PACKET_HEADER_SIZE = 6
    PACKET_FOOTER_SIZE = 5
    PACKET_MIN_SIZE = PACKET_HEADER_SIZE + POINTS_PER_PACKET * POINT_DATA_SIZE + PACKET_FOOTER_SIZE  # 47

    # ── 步进电机参数 ──
    MOTOR_SPEED_DEG_PER_SEC = 2.25   # 旋转速度 (°/s)
    MOTOR_TOTAL_ANGLE = 180.0        # 总旋转角度 (°)
    MOTOR_DIRECTION = 1              # 1=顺时针
    MOTOR_TOTAL_TIME = MOTOR_TOTAL_ANGLE / MOTOR_SPEED_DEG_PER_SEC  # 80秒

    _CRC8_TABLE = None

    @classmethod
    def _init_crc8_table(cls):
        if cls._CRC8_TABLE is not None:
            return
        cls._CRC8_TABLE = []
        for i in range(256):
            crc = i
            for _ in range(8):
                if crc & 0x80:
                    crc = ((crc << 1) ^ 0x4D) & 0xFF
                else:
                    crc = (crc << 1) & 0xFF
            cls._CRC8_TABLE.append(crc)

    @classmethod
    def _crc8(cls, data):
        cls._init_crc8_table()
        crc = 0x00
        for byte in data:
            crc = cls._CRC8_TABLE[(crc ^ byte) & 0xFF]
        return crc

    def __init__(self, port, baud_rate=230400, capture_interval=0.1, min_distance=0.0):
        self.port = port
        self.baud_rate = baud_rate
        self.capture_interval = capture_interval  # 采集间隔（秒），用于前端轮询频率
        self.min_distance = min_distance          # 最小距离过滤（米），小于此距离的点丢弃
        self.ser = None
        self.connected = False
        self.running = False
        self.capture_thread = None

        # 捕获开始时间（用于推算步进电机方位角）
        self.capture_start_time = None

        # 累积的 3D 点云数据
        self._lock = threading.Lock()
        self.points_x = []
        self.points_y = []
        self.points_z = []
        self.intensities = []

        # 统计
        self.packet_count = 0
        self.point_count = 0
        self.last_rotation_speed = 0.0
        self.last_azimuth = 0.0       # 当前推算的电机方位角
        self.last_elapsed = 0.0       # 当前已用时间
        self._stats_time = time.time()
        self._stats_packets = 0
        self.fps = 0.0

        # 诊断统计：追踪数据包被丢弃的原因
        self.diag_crc_fail = 0        # CRC8校验失败
        self.diag_n_mismatch = 0      # 点数不匹配
        self.diag_angle_filtered = 0  # 角度范围外被过滤
        self.diag_parsed_ok = 0       # 成功解析的数据包

    # ─── 连接管理 ────────────────────────────────────────

    def connect(self):
        if serial is None:
            raise ImportError("pyserial 未安装")
        self.ser = serial.Serial(
            port=self.port, baudrate=self.baud_rate,
            bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE, timeout=1.0,
        )
        # 增大操作系统串口接收缓冲区，防止高速数据溢出
        try:
            self.ser.set_buffer_size(rx_size=65536)
        except (AttributeError, ValueError):
            pass  # 某些平台/驱动不支持，静默忽略
        self.connected = True
        return True

    def start_capture(self):
        if not self.connected:
            self.connect()
        self.capture_start_time = time.time()  # 以此为步进电机 t=0
        with self._lock:
            self.points_x.clear()
            self.points_y.clear()
            self.points_z.clear()
            self.intensities.clear()
        self.point_count = 0
        self.packet_count = 0
        self.diag_crc_fail = 0
        self.diag_n_mismatch = 0
        self.diag_angle_filtered = 0
        self.diag_parsed_ok = 0
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        return True

    def stop_capture(self):
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=3)
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
        self.connected = False

    # ─── 数据捕获主循环 ──────────────────────────────────

    def _capture_loop(self):
        """MS200P数据捕获主循环

        与 lidar_reader.py 保持一致的轮询策略：
          1. 检查串口缓冲区是否有数据 (in_waiting)
          2. 一次性读取所有可用数据
          3. 处理缓冲区中的完整帧
          4. 休眠 5ms 后再次轮询
        """
        buf = bytearray()

        while self.running:
            try:
                # 与 lidar_reader.py 一致：检查 in_waiting 后读取全部可用数据
                if self.ser.in_waiting:
                    buf.extend(self.ser.read(self.ser.in_waiting))
            except (serial.SerialException, OSError):
                break

            # 处理缓冲区中的所有完整数据包
            while len(buf) >= self.PACKET_MIN_SIZE:
                header_pos = buf.find(b'\x54')
                if header_pos < 0:
                    buf.clear()
                    break
                if header_pos > 0:
                    del buf[:header_pos]

                if len(buf) < self.PACKET_MIN_SIZE:
                    break

                try:
                    packet_time = time.time()
                    frame = bytes(buf[:self.PACKET_MIN_SIZE])
                    parsed = self._parse_packet_to_3d(frame, packet_time)
                    if parsed:
                        xs, ys, zs, intensities, speed, azimuth = parsed
                        self.diag_parsed_ok += 1
                        with self._lock:
                            self.points_x.extend(xs)
                            self.points_y.extend(ys)
                            self.points_z.extend(zs)
                            self.intensities.extend(intensities)
                            self.point_count += len(xs)
                            self.last_rotation_speed = speed
                            self.last_azimuth = azimuth
                            if self.capture_start_time:
                                self.last_elapsed = packet_time - self.capture_start_time
                        self.packet_count += 1
                        self._stats_packets += 1

                        now = time.time()
                        if now - self._stats_time >= 1.0:
                            self.fps = self._stats_packets / (now - self._stats_time)
                            self._stats_packets = 0
                            self._stats_time = now

                        del buf[:self.PACKET_MIN_SIZE]
                    else:
                        del buf[:1]
                except Exception:
                    del buf[:1]

            # 与 lidar_reader.py 一致的轮询间隔：5ms
            time.sleep(0.005)

    def _parse_packet_to_3d(self, data, packet_time):
        """解析MS200P数据包并转换为3D坐标

        与 lidar_reader.py 保持一致的坐标转换逻辑：
          雷达角 φ（lidar_angle）：0° = 正下方，180° = 正上方
          电机角 θ（motor_angle）：顺时针方位角，0° 起始
          elevation = φ - 90°
          X = r × cos(elev) × cos(θ)
          Y = r × cos(elev) × sin(θ)
          Z = r × sin(elev)

        Args:
            data: 至少47字节的原始数据
            packet_time: 数据包接收时刻（用于计算电机方位角）

        Returns:
            (xs, ys, zs, intensities, speed, azimuth) 或 None（单位均为米/度）
        """
        if len(data) < self.PACKET_MIN_SIZE or data[0] != self.PACKET_HEADER:
            return None

        # CRC8 校验（多项式 0x4D）
        if self._crc8(data[:self.PACKET_MIN_SIZE - 1]) != data[self.PACKET_MIN_SIZE - 1]:
            self.diag_crc_fail += 1
            return None

        # 帧类型检查
        if data[1] != self.PACKET_TYPE:
            self.diag_n_mismatch += 1
            return None

        # 解析转速（°/s）
        speed = struct.unpack_from('<H', data, 2)[0]

        # 起始角度（单位 0.01°）
        start_angle = struct.unpack_from('<H', data, 4)[0] / 100.0
        # 结束角度（单位 0.01°）
        end_angle = struct.unpack_from('<H', data, self.PACKET_MIN_SIZE - 5)[0] / 100.0

        # 角度步长（与 lidar_reader.py 一致，处理跨 0° 情况）
        if end_angle >= start_angle:
            step = (end_angle - start_angle) / (self.POINTS_PER_PACKET - 1)
        else:
            step = (end_angle + 360.0 - start_angle) / (self.POINTS_PER_PACKET - 1)

        # ── 计算步进电机方位角 θ ──
        if self.capture_start_time:
            elapsed = packet_time - self.capture_start_time
            azimuth = self.MOTOR_SPEED_DEG_PER_SEC * elapsed
        else:
            elapsed = 0
            azimuth = 0

        # ── 坐标转换 ──
        theta_rad = math.radians(azimuth)
        xs, ys, zs, valid_ints = [], [], [], []

        for i in range(self.POINTS_PER_PACKET):
            # 距离（mm）和强度
            offset = self.PACKET_HEADER_SIZE + i * self.POINT_DATA_SIZE
            dist_mm = struct.unpack_from('<H', data, offset)[0]
            intensity = data[offset + 2]

            # 强度 0~15 为无效测距值，跳过
            if intensity <= 15 or dist_mm == 0:
                continue

            # 近距离反射点过滤：距离小于阈值则丢弃（用于滤除雷达自身反射/地面杂波）
            if self.min_distance > 0 and dist_mm < self.min_distance * 1000.0:
                continue

            # 雷达角度插值
            lidar_angle = (start_angle + step * i) % 360.0

            # 球坐标 → 直角坐标（与 lidar_reader.py to_xyz 一致）
            elev_rad = math.radians(lidar_angle - 90.0)
            r = dist_mm / 1000.0  # mm → m

            x = r * math.cos(elev_rad) * math.cos(theta_rad)
            y = r * math.cos(elev_rad) * math.sin(theta_rad)
            z = r * math.sin(elev_rad)

            xs.append(x)
            ys.append(y)
            zs.append(z)
            valid_ints.append(float(intensity))

        return xs, ys, zs, valid_ints, speed, azimuth

    # ─── 数据获取 ────────────────────────────────────────

    def get_accumulated_points_json(self, max_points=500000):
        """获取累积的3D点云为JSON格式"""
        with self._lock:
            n = len(self.points_x)
            if n == 0:
                return {'positions': [], 'num_points': 0, 'colors': []}
            xs = np.array(self.points_x, dtype=np.float32)
            ys = np.array(self.points_y, dtype=np.float32)
            zs = np.array(self.points_z, dtype=np.float32)
            its = np.array(self.intensities, dtype=np.float32)

        if n > max_points:
            indices = np.linspace(0, n - 1, max_points, dtype=np.int32)
            xs, ys, zs = xs[indices], ys[indices], zs[indices]
            its = its[indices]
            n = max_points

        positions = np.column_stack([xs, ys, zs]).flatten().tolist()

        i_min, i_max = its.min(), its.max()
        i_range = i_max - i_min if i_max > i_min else 1.0
        norm = (its - i_min) / i_range
        colors = np.column_stack([norm, norm * 0.6, 1.0 - norm * 0.5]).flatten().tolist()

        return {'positions': positions, 'num_points': n, 'colors': colors}

    def get_status(self):
        with self._lock:
            n = len(self.points_x)
        # 计算总处理的数据包数（成功+各类失败）
        total_diag = self.diag_parsed_ok + self.diag_crc_fail + self.diag_n_mismatch
        if total_diag > 0:
            crc_rate = round(self.diag_crc_fail / total_diag * 100, 1)
        else:
            crc_rate = 0.0
        return {
            'connected': self.connected,
            'capturing': self.running,
            'port': self.port,
            'baud_rate': self.baud_rate,
            'capture_interval': self.capture_interval,
            'min_distance': self.min_distance,
            'fps': round(self.fps, 1),
            'num_points': n,
            'packets_received': self.packet_count,
            'rotation_speed': self.last_rotation_speed,
            'azimuth': round(self.last_azimuth, 1),
            'elapsed': round(self.last_elapsed, 1),
            'total_time': self.MOTOR_TOTAL_TIME,
            # 诊断统计
            'diag_parsed_ok': self.diag_parsed_ok,
            'diag_crc_fail': self.diag_crc_fail,
            'diag_n_mismatch': self.diag_n_mismatch,
            'diag_angle_filtered': self.diag_angle_filtered,
            'diag_crc_fail_rate': crc_rate,
        }

    def get_open3d_cloud(self):
        import open3d as o3d
        with self._lock:
            n = len(self.points_x)
            if n == 0:
                return None
            pts = np.column_stack([
                np.array(self.points_x, dtype=np.float64),
                np.array(self.points_y, dtype=np.float64),
                np.array(self.points_z, dtype=np.float64),
            ])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd


# ─── 全局管理 ────────────────────────────────────────────

_ms200p_connections = {}
_ms200p_lock = threading.Lock()


def get_ms200p(session_id):
    with _ms200p_lock:
        return _ms200p_connections.get(session_id)


def create_ms200p(session_id, port, baud_rate=230400, capture_interval=0.1, min_distance=0.0):
    with _ms200p_lock:
        old = _ms200p_connections.get(session_id)
        if old and old.running:
            old.stop_capture()
        client = MS200PClient(port, baud_rate, capture_interval, min_distance)
        _ms200p_connections[session_id] = client
    return client


def remove_ms200p(session_id):
    with _ms200p_lock:
        client = _ms200p_connections.pop(session_id, None)
    if client:
        client.stop_capture()


def list_serial_ports():
    if serial is None:
        return []
    ports = serial.tools.list_ports.comports()
    return [
        {'port': p.device, 'description': p.description, 'hwid': p.hwid}
        for p in sorted(ports, key=lambda x: x.device)
    ]
