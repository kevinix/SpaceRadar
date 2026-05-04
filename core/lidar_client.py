"""镭神C16激光雷达UDP客户端，负责数据接收、解析和点云生成"""

import socket
import struct
import threading
import time
import math
import numpy as np


class LidarClient:
    """镭神C16激光雷达UDP客户端，支持实时数据捕获和解析"""

    BLOCK_HEADER = 0xEEFF
    BLOCK_SIZE = 100
    BLOCKS_PER_PACKET = 12
    CHANNELS_PER_BLOCK = 16
    SCAN_STRIDE = 6
    POINTS_PER_PACKET = BLOCKS_PER_PACKET * CHANNELS_PER_BLOCK
    RAW_PACKET_SIZE = 1212
    LIDAR_LINE_COUNT = 16

    # C16 垂直角度（按通道0-15顺序排列）
    # 通道对应角度: 0→15°, 1→11°, 2→7°, 3→3°, 4→-1°, 5→-5°, 6→-9°, 7→-13°
    #               8→13°,  9→9°, 10→5°, 11→1°, 12→-3°, 13→-7°, 14→-11°, 15→-15°
    C16_VERTICAL_ANGLES = [
        15.0, 11.0, 7.0, 3.0, -1.0, -5.0, -9.0, -13.0,
        13.0, 9.0, 5.0, 1.0, -3.0, -7.0, -11.0, -15.0
    ]

    def __init__(self, ip, data_port, device_port, return_mode=1):
        """初始化雷达客户端

        Args:
            ip: 雷达设备IP地址
            data_port: MSOP数据端口
            device_port: DIFOP设备端口
            return_mode: 回波模式，1=单回波，2=双回波
        """
        self.ip = ip
        self.data_port = data_port
        self.device_port = device_port
        self.return_mode = return_mode

        self.sock = None
        self.running = False
        self.connected = False
        self.capture_thread = None

        self.points_lock = threading.Lock()
        self.current_points = np.zeros((0, 3), dtype=np.float64)
        self.current_intensities = np.zeros(0, dtype=np.float64)

        self.frame_count = 0
        self.packet_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0

        # 预计算每个通道对应的 sin/cos 值，用于坐标转换
        self._sincos_table = {}
        for ch, angle in enumerate(self.C16_VERTICAL_ANGLES):
            rad = math.radians(angle)
            self._sincos_table[ch] = (math.sin(rad), math.cos(rad))

    def connect(self):
        """建立与雷达的UDP连接，创建并绑定socket

        Returns:
            bool: 连接是否成功

        Raises:
            ConnectionError: 连接失败时抛出
        """
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind(('0.0.0.0', self.data_port))
            self.sock.settimeout(3.0)
            self.connected = True
            return True
        except socket.error as e:
            self.connected = False
            raise ConnectionError(f"连接失败: {str(e)}")

    def start_capture(self):
        """启动数据捕获线程

        Returns:
            bool: 是否成功启动
        """
        if not self.connected:
            self.connect()

        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        return True

    def stop_capture(self):
        """停止数据捕获并释放socket资源"""
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5)
        self._close_socket()
        self.connected = False

    def _close_socket(self):
        """安全关闭socket连接"""
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def _capture_loop(self):
        """数据捕获主循环，持续接收并解析UDP数据包"""
        scan_points = []
        scan_intensities = []
        last_azimuth = -1.0
        packet_in_frame = 0

        while self.running:
            try:
                data, addr = self.sock.recvfrom(2048)
            except socket.timeout:
                if len(scan_points) > 0:
                    with self.points_lock:
                        self.current_points = np.array(scan_points, dtype=np.float64)
                        self.current_intensities = np.array(scan_intensities, dtype=np.float64)
                    self.frame_count += 1
                    scan_points = []
                    scan_intensities = []
                    last_azimuth = -1.0
                continue
            except OSError:
                break

            if len(data) < self.RAW_PACKET_SIZE:
                continue

            self.packet_count += 1

            try:
                points, intensities, azimuths = self._parse_packet(data)
            except Exception:
                continue

            if len(azimuths) == 0:
                continue

            for i in range(len(points)):
                block_idx = i // self.CHANNELS_PER_BLOCK
                azimuth = azimuths[block_idx]

                if azimuth < last_azimuth and last_azimuth > 100 and len(scan_points) > 100:
                    with self.points_lock:
                        self.current_points = np.array(scan_points, dtype=np.float64)
                        self.current_intensities = np.array(scan_intensities, dtype=np.float64)
                    self.frame_count += 1
                    scan_points = []
                    scan_intensities = []
                    packet_in_frame = 0

                scan_points.append(points[i])
                scan_intensities.append(intensities[i])
                last_azimuth = azimuth
                packet_in_frame += 1

            now = time.time()
            if now - self.last_fps_time >= 1.0:
                self.fps = self.frame_count / (now - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = now

    def _parse_packet(self, data):
        """解析单个MSOP数据包，提取点云坐标、强度和方位角

        C16 MSOP 数据包结构 (1212字节，6字节/通道版本):
          每包12个数据块 (BLOCKS_PER_PACKET=12)
          每块100字节: 2字节Flag(0xFFEE) + 2字节方位角 + 16通道×6字节
          每通道6字节: 2字节距离 + 2字节反射率 + 2字节保留

        Args:
            data: 原始UDP数据包字节流

        Returns:
            (points, intensities, azimuths) 元组
        """
        points = []
        intensities = []
        azimuths = []

        for block_idx in range(self.BLOCKS_PER_PACKET):
            offset = block_idx * self.BLOCK_SIZE

            # 验证数据块头部标识 0xFFEE
            header = struct.unpack_from('<H', data, offset)[0]
            if header != self.BLOCK_HEADER:
                continue

            # 读取方位角 (0.01°单位，little-endian)
            azimuth_raw = struct.unpack_from('<H', data, offset + 2)[0]
            azimuth = azimuth_raw % 36000
            azimuth_deg = azimuth / 100.0
            azimuths.append(azimuth_deg)

            azimuth_rad = math.radians(azimuth_deg)
            cos_azimuth = math.cos(azimuth_rad)
            sin_azimuth = math.sin(azimuth_rad)

            # 解析16个通道数据 (每通道6字节: 距离2B + 反射率2B + 保留2B)
            for ch in range(self.CHANNELS_PER_BLOCK):
                ch_offset = offset + 4 + ch * self.SCAN_STRIDE
                distance_raw = struct.unpack_from('<H', data, ch_offset)[0]
                intensity = struct.unpack_from('<H', data, ch_offset + 2)[0]

                # 距离单位转换: 原始值 × 0.004 = 米
                distance = distance_raw * 0.004

                # 过滤无效点 (距离<0.15米视为噪点)
                if distance < 0.15:
                    continue

                # 获取该通道对应的垂直角度 sin/cos (通道号直接对应角度索引)
                sin_vert, cos_vert = self._sincos_table[ch]

                # 球坐标 → 笛卡尔坐标转换
                # X轴: 水平投影 × sin(方位角) → 右侧
                # Y轴: 水平投影 × cos(方位角) → 前方(0°方向)
                # Z轴: 垂直分量 → 上方
                x = distance * cos_vert * sin_azimuth
                y = distance * cos_vert * cos_azimuth
                z = distance * sin_vert

                points.append([x, y, z])
                intensities.append(float(intensity))

        return points, intensities, azimuths

    def get_points_json(self, max_points=5000000):
        """获取当前帧的点云数据，格式化为JSON友好的字典

        Args:
            max_points: 最大返回点数，超出时随机采样

        Returns:
            包含点坐标和强度信息的字典，使用扁平数组提高传输效率
        """
        with self.points_lock:
            if len(self.current_points) == 0:
                return {'positions': [], 'num_points': 0, 'colors': []}

            pts = self.current_points.copy()
            ints = self.current_intensities.copy()

        if len(pts) > max_points:
            indices = np.random.choice(len(pts), max_points, replace=False)
            pts = pts[indices]
            ints = ints[indices]

        i_min = ints.min()
        i_max = ints.max()
        i_range = i_max - i_min if i_max > i_min else 1.0
        normalized = (ints - i_min) / i_range

        colors = np.zeros((len(pts), 3), dtype=np.float32)
        colors[:, 0] = normalized
        colors[:, 1] = normalized * 0.6
        colors[:, 2] = 1.0 - normalized * 0.5

        flat_positions = pts.astype(np.float32).flatten().tolist()
        flat_colors = colors.flatten().tolist()

        return {
            'positions': flat_positions,
            'num_points': len(pts),
            'colors': flat_colors,
        }

    def get_status(self):
        """获取当前连接和数据捕获状态

        Returns:
            包含连接状态、帧率、点数等信息的字典
        """
        with self.points_lock:
            num_points = len(self.current_points)

        return {
            'connected': self.connected,
            'capturing': self.running,
            'ip': self.ip,
            'data_port': self.data_port,
            'device_port': self.device_port,
            'fps': round(self.fps, 1),
            'num_points': num_points,
            'packets_received': self.packet_count,
        }

    def get_open3d_cloud(self):
        """获取当前帧的 Open3D PointCloud 对象

        Returns:
            open3d.geometry.PointCloud 对象，或 None（如果没有数据）
        """
        import open3d as o3d

        with self.points_lock:
            if len(self.current_points) == 0:
                return None
            pts = self.current_points.copy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd


lidar_connections = {}
lidar_lock = threading.Lock()


def get_lidar(session_id):
    """根据会话ID获取雷达客户端实例

    Args:
        session_id: 会话标识符

    Returns:
        LidarClient 实例，或 None
    """
    with lidar_lock:
        return lidar_connections.get(session_id)


def cleanup_conflicting_connections(ip, data_port):
    """清理与指定IP/端口冲突的旧连接，避免页面刷新后端口被占用

    遍历所有现有连接，停止并移除与目标IP和数据端口相同的连接

    Args:
        ip: 雷达IP地址
        data_port: MSOP数据端口

    Returns:
        被清理的连接数量
    """
    cleaned = 0
    with lidar_lock:
        to_remove = []
        for sid, client in lidar_connections.items():
            if client.ip == ip and client.data_port == data_port:
                to_remove.append(sid)
        for sid in to_remove:
            client = lidar_connections.pop(sid, None)
            if client:
                client.stop_capture()
                cleaned += 1
    return cleaned


def get_existing_lidar_by_port(ip, data_port):
    """查找指定IP/端口上是否存在活跃连接

    Args:
        ip: 雷达IP地址
        data_port: MSOP数据端口

    Returns:
        (session_id, LidarClient) 或 (None, None)
    """
    with lidar_lock:
        for sid, client in lidar_connections.items():
            if client.ip == ip and client.data_port == data_port:
                return sid, client
    return None, None


def create_lidar(session_id, ip, data_port, device_port):
    """创建新的雷达客户端并注册到全局管理器

    Args:
        session_id: 会话标识符
        ip: 雷达IP地址
        data_port: 数据端口
        device_port: 设备端口

    Returns:
        新创建的 LidarClient 实例
    """
    # 先清理同一端口上的旧连接，防止端口冲突
    cleanup_conflicting_connections(ip, data_port)

    client = LidarClient(ip, data_port, device_port)
    with lidar_lock:
        old = lidar_connections.get(session_id)
        if old and old.running:
            old.stop_capture()
        lidar_connections[session_id] = client
    return client


def remove_lidar(session_id):
    """停止并移除雷达客户端，释放资源

    Args:
        session_id: 会话标识符
    """
    with lidar_lock:
        client = lidar_connections.pop(session_id, None)
    if client:
        client.stop_capture()
