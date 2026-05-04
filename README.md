# SpaceRadar

激光雷达 3D 点云采集与空间建模平台。支持镭神 C16 和奥锐达 MS200P 两款雷达，提供从数据采集、预处理、三维重建到房间外框架提取的完整工作流。

## 功能

| 模块 | 说明 |
|------|------|
| **雷达连接** | 镭神 C16（UDP）、奥锐达 MS200P（串口），实时点云显示 |
| **MS200P 3D 扫描** | 垂直安装 + 步进电机旋转，球坐标 → 笛卡尔坐标转换 |
| **点云预处理** | 统计去噪、体素下采样、法线估计 |
| **三维重建** | 泊松表面重建 / Alpha Shapes / Ball Pivoting |
| **空间模型** | RANSAC 墙面检测 + 房间外框架提取 |
| **导入导出** | PLY / PCD / XYZ / STL / OBJ 格式支持 |

## 硬件要求

- **镭神 C16**：通过以太网 UDP 连接，默认 IP `192.168.99.14`
- **奥锐达 MS200P**：通过 USB 转串口连接，波特率 230400，配合 NEMA17 步进电机（2.25°/s 旋转）

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python app.py

# 浏览器打开
open http://localhost:5000
```

## 依赖

- Python 3.8+
- Flask
- Open3D
- NumPy
- PySerial
- Three.js（CDN 加载，无需本地安装）

## 目录结构

```
SpaceRadar/
├── app.py                  # Flask 主应用
├── config.py               # 配置文件
├── requirements.txt        # Python 依赖
├── lidar_reader.py         # MS200P 独立采集脚本（参考实现）
├── core/
│   ├── lidar_client.py     # 镭神 C16 UDP 客户端
│   ├── ms200p_client.py    # 奥锐达 MS200P 串口客户端
│   ├── preprocessor.py     # 点云预处理
│   ├── reconstruction.py   # 三维表面重建
│   ├── spatial_model.py    # 空间模型 / 房间框架提取
│   ├── exporter.py         # 格式导出
│   └── io_loader.py        # 文件加载器
├── static/
│   ├── css/style.css       # 样式
│   └── js/main.js          # Three.js 前端渲染逻辑
└── templates/
    └── index.html          # Web 界面
```

## MS200P 坐标系

雷达垂直安装，0° 标记朝下（正下方），180° 朝上（正上方）。步进电机带动雷达在水平面顺时针旋转。

```
雷达角 φ：0°=正下方, 180°=正上方
电机角 θ：顺时针方位角
elevation = φ - 90°
X = r × cos(elev) × cos(θ)
Y = r × cos(elev) × sin(θ)
Z = r × sin(elev)
```

## License

MIT
