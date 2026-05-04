"""Flask Web 应用主入口，提供点云处理和3D模型生成的Web界面"""

import os
import json
import uuid
import re
import time
import numpy as np
import open3d as o3d
from flask import Flask, render_template, request, jsonify, send_from_directory
from config import UPLOAD_FOLDER, OUTPUT_FOLDER, MAX_CONTENT_LENGTH, ALLOWED_EXTENSIONS
from core.io_loader import PointCloudLoader
from core.preprocessor import PointCloudPreprocessor
from core.reconstruction import MeshReconstructor
from core.exporter import ModelExporter
from core.lidar_client import create_lidar, get_lidar, remove_lidar, cleanup_conflicting_connections, get_existing_lidar_by_port
from core.spatial_model import SpatialModelProcessor
from core.ms200p_client import (
    get_ms200p, create_ms200p, remove_ms200p, list_serial_ports
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

sessions = {}


def allowed_file(filename):
    """检查文件扩展名是否在允许列表中"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def numpy_to_list(obj):
    """将 numpy 类型转换为 Python 原生类型，用于 JSON 序列化"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numpy_to_list(i) for i in obj]
    return obj


@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """上传点云文件"""
    if 'file' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'不支持的文件格式，支持: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    session_id = str(uuid.uuid4())
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{session_id}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        loader = PointCloudLoader()
        cloud = loader.load(filepath)
        info = loader.get_info()

        exporter = ModelExporter()
        pc_data = exporter.point_cloud_to_json(cloud, max_points=200000)

        sessions[session_id] = {
            'filepath': filepath,
            'loader': loader,
            'cloud': cloud,
            'preprocessor': None,
            'reconstructor': None,
            'mesh': None,
        }

        return jsonify({
            'session_id': session_id,
            'info': numpy_to_list(info),
            'point_cloud': pc_data,
        })
    except Exception as e:
        return jsonify({'error': f'加载文件失败: {str(e)}'}), 500


@app.route('/api/preprocess', methods=['POST'])
def preprocess():
    """对点云执行预处理操作"""
    data = request.json
    session_id = data.get('session_id')
    if session_id not in sessions:
        return jsonify({'error': '无效的会话'}), 400

    steps = data.get('steps', [])
    pipeline_steps = []
    for step in steps:
        pipeline_steps.append((step.get('name', ''), step.get('params', {})))

    try:
        session = sessions[session_id]
        cloud = session['cloud']

        preprocessor = PointCloudPreprocessor(cloud)
        processed_cloud = preprocessor.run_pipeline(pipeline_steps)
        stats = preprocessor.get_stats()

        session['cloud'] = processed_cloud
        session['preprocessor'] = preprocessor

        exporter = ModelExporter()
        pc_data = exporter.point_cloud_to_json(processed_cloud, max_points=200000)

        return jsonify({
            'stats': numpy_to_list(stats),
            'point_cloud': pc_data,
        })
    except Exception as e:
        return jsonify({'error': f'预处理失败: {str(e)}'}), 500


@app.route('/api/reconstruct', methods=['POST'])
def reconstruct():
    """执行三维重建"""
    data = request.json
    session_id = data.get('session_id')
    if session_id not in sessions:
        return jsonify({'error': '无效的会话'}), 400

    method = data.get('method', 'poisson')
    params = data.get('params', {})

    try:
        session = sessions[session_id]
        cloud = session['cloud']

        # 检查点云是否有法线（泊松重建和Ball Pivoting都需要法线）
        if method in ('poisson', 'ball_pivoting') and not cloud.has_normals():
            return jsonify({
                'error': '当前点云没有法线信息，请先在预处理中勾选"估计法线"并执行'
            }), 400

        # 泊松重建 depth 参数校验和自动下调
        if method == 'poisson':
            depth = params.get('depth', 9)
            if depth < 2:
                params['depth'] = 6
            # 点数超过5万时，点云密度可能不足，自动降低depth
            num_points = len(cloud.points)
            if num_points > 200000 and params['depth'] > 7:
                params['depth'] = 7
            elif num_points > 100000 and params['depth'] > 8:
                params['depth'] = 8

        # 点数超限自动下采样，确保重建可在合理时间内完成
        num_points = len(cloud.points)
        if num_points > 80000 and method in ('poisson', 'ball_pivoting'):
            voxel_size = params.pop('auto_voxel_size', None)
            if voxel_size is None:
                bbox = cloud.get_axis_aligned_bounding_box()
                extent = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
                target_points = 50000
                voxel_size = extent * (1.0 / (target_points ** (1/3))) * 0.5
                voxel_size = max(voxel_size, extent * 0.0005)
            cloud = cloud.voxel_down_sample(voxel_size=voxel_size)
            # 重新估计法线，因为下采样后法线失效
            cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=voxel_size * 3, max_nn=30
                )
            )
            cloud.orient_normals_consistent_tangent_plane(k=15)
            session['cloud'] = cloud

        reconstructor = MeshReconstructor(cloud)
        mesh = reconstructor.reconstruct(method, **params)

        # 检查mesh是否有效
        if mesh is None or len(mesh.triangles) == 0:
            return jsonify({'error': '重建结果为空，请调整参数后重试'}), 400

        # 平滑处理：单独捕获异常，失败时跳過平滑直接返回未平滑结果
        smooth_applied = False
        if data.get('smooth', False):
            try:
                smooth_iterations = data.get('smooth_iterations', 10)
                mesh = reconstructor.smooth_mesh(iterations=smooth_iterations)
                smooth_applied = True
            except Exception:
                pass  # 平滑失败，保持未平滑的mesh

        mesh_info = reconstructor.get_mesh_info()
        if smooth_applied:
            mesh_info['smoothed'] = True

        session['mesh'] = mesh
        session['reconstructor'] = reconstructor

        try:
            exporter = ModelExporter(mesh)
            mesh_data = exporter.mesh_to_json()
        except Exception:
            # mesh_to_json 序列化失败时，尝试清理mesh后重试
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
            exporter = ModelExporter(mesh)
            mesh_data = exporter.mesh_to_json()

        return jsonify({
            'mesh_info': numpy_to_list(mesh_info),
            'mesh': mesh_data,
        })
    except Exception as e:
        return jsonify({'error': f'重建失败: {str(e)}'}), 500


@app.route('/api/export', methods=['POST'])
def export_model():
    """导出3D模型文件"""
    data = request.json
    session_id = data.get('session_id')
    if session_id not in sessions:
        return jsonify({'error': '无效的会话'}), 400

    export_format = data.get('format', 'stl')

    try:
        session = sessions[session_id]
        mesh = session.get('mesh')
        if mesh is None:
            return jsonify({'error': '没有可导出的模型，请先执行三维重建'}), 400

        filename = f"{session_id}.{export_format}"
        output_path = os.path.join(OUTPUT_FOLDER, filename)

        exporter = ModelExporter(mesh)
        exporter.export(output_path)

        return jsonify({
            'download_url': f'/api/download/{filename}',
            'filename': filename,
        })
    except Exception as e:
        return jsonify({'error': f'导出失败: {str(e)}'}), 500


@app.route('/api/download/<filename>')
def download_file(filename):
    """下载导出的模型文件"""
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)


@app.route('/api/export_pointcloud', methods=['POST'])
def export_pointcloud():
    """导出点云数据文件"""
    data = request.json
    session_id = data.get('session_id')
    if session_id not in sessions:
        return jsonify({'error': '无效的会话'}), 400

    export_format = data.get('format', 'ply')

    try:
        session = sessions[session_id]
        cloud = session.get('cloud')
        if cloud is None:
            return jsonify({'error': '没有可导出的点云数据'}), 400

        filename = f"{session_id}_pointcloud.{export_format}"
        output_path = os.path.join(OUTPUT_FOLDER, filename)

        exporter = ModelExporter()
        exporter.export_point_cloud(cloud, output_path)

        return jsonify({
            'download_url': f'/api/download/{filename}',
            'filename': filename,
        })
    except Exception as e:
        return jsonify({'error': f'导出失败: {str(e)}'}), 500


@app.route('/api/lidar/connect', methods=['POST'])
def lidar_connect():
    """建立与镭神C16雷达的连接"""
    data = request.json
    ip = data.get('ip', '').strip()
    data_port = data.get('data_port')
    device_port = data.get('device_port')

    if not ip:
        return jsonify({'error': '请输入雷达IP地址'}), 400

    ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    if not re.match(ipv4_pattern, ip):
        return jsonify({'error': 'IP地址格式不正确，请输入标准IPv4地址'}), 400

    octets = ip.split('.')
    for octet in octets:
        if int(octet) > 255:
            return jsonify({'error': 'IP地址格式不正确，每段数值不能超过255'}), 400

    try:
        data_port = int(data_port)
        if data_port < 0 or data_port > 65535:
            raise ValueError()
    except (TypeError, ValueError):
        return jsonify({'error': '数据端口无效，请输入0-65535之间的数值'}), 400

    try:
        device_port = int(device_port)
        if device_port < 0 or device_port > 65535:
            raise ValueError()
    except (TypeError, ValueError):
        return jsonify({'error': '设备端口无效，请输入0-65535之间的数值'}), 400

    session_id = str(uuid.uuid4())

    try:
        client = create_lidar(session_id, ip, data_port, device_port)
        client.connect()

        return jsonify({
            'session_id': session_id,
            'status': client.get_status(),
        })
    except Exception as e:
        remove_lidar(session_id)
        return jsonify({'error': f'连接雷达失败: {str(e)}'}), 500


@app.route('/api/lidar/start', methods=['POST'])
def lidar_start():
    """开始捕获雷达数据"""
    data = request.json
    session_id = data.get('session_id')

    client = get_lidar(session_id)
    if client is None:
        return jsonify({'error': '雷达未连接，请先建立连接'}), 400

    try:
        if not client.connected:
            client.connect()
        client.start_capture()
        return jsonify({'status': client.get_status()})
    except Exception as e:
        return jsonify({'error': f'启动捕获失败: {str(e)}'}), 500


@app.route('/api/lidar/stop', methods=['POST'])
def lidar_stop():
    """停止捕获雷达数据并释放连接"""
    data = request.json
    session_id = data.get('session_id')

    client = get_lidar(session_id)
    if client is None:
        return jsonify({'error': '雷达未连接'}), 400

    try:
        client.stop_capture()
        return jsonify({'status': client.get_status()})
    except Exception as e:
        return jsonify({'error': f'停止捕获失败: {str(e)}'}), 500


@app.route('/api/lidar/data', methods=['POST'])
def lidar_data():
    """获取雷达实时点云数据"""
    data = request.json
    session_id = data.get('session_id')

    client = get_lidar(session_id)
    if client is None:
        return jsonify({'error': '雷达未连接'}), 400

    return jsonify({
        'point_cloud': client.get_points_json(max_points=5000000),
        'status': client.get_status(),
    })


@app.route('/api/lidar/status', methods=['POST'])
def lidar_status():
    """获取雷达连接状态"""
    data = request.json
    session_id = data.get('session_id')

    client = get_lidar(session_id)
    if client is None:
        return jsonify({'connected': False, 'capturing': False})

    return jsonify({'status': client.get_status()})


@app.route('/api/lidar/disconnect', methods=['POST'])
def lidar_disconnect():
    """断开雷达连接并释放所有资源"""
    data = request.json
    session_id = data.get('session_id')

    remove_lidar(session_id)
    return jsonify({'connected': False, 'capturing': False})


@app.route('/api/lidar/to_session', methods=['POST'])
def lidar_to_session():
    """将雷达当前帧数据导入为文件处理会话，以便后续预处理和重建"""
    data = request.json
    lidar_session_id = data.get('lidar_session_id')

    client = get_lidar(lidar_session_id)
    if client is None:
        return jsonify({'error': '雷达未连接'}), 400

    cloud = client.get_open3d_cloud()
    if cloud is None:
        return jsonify({'error': '没有可用的点云数据，请先捕获数据'}), 400

    session_id = str(uuid.uuid4())

    exporter = ModelExporter()
    pc_data = exporter.point_cloud_to_json(cloud, max_points=200000)

    info = {
        'num_points': len(cloud.points),
        'has_normals': cloud.has_normals(),
        'has_colors': cloud.has_colors(),
        'center': np.array(cloud.get_center()).tolist(),
        'min_bound': np.array(cloud.get_min_bound()).tolist(),
        'max_bound': np.array(cloud.get_max_bound()).tolist(),
    }

    sessions[session_id] = {
        'filepath': None,
        'loader': None,
        'cloud': cloud,
        'preprocessor': None,
        'reconstructor': None,
        'mesh': None,
    }

    return jsonify({
        'session_id': session_id,
        'info': numpy_to_list(info),
        'point_cloud': pc_data,
    })


@app.route('/api/lidar/check', methods=['POST'])
def lidar_check():
    """检查指定IP/端口上是否存在活跃的雷达连接，用于页面刷新后恢复状态"""
    data = request.json
    ip = data.get('ip', '').strip()
    data_port = data.get('data_port')

    try:
        data_port = int(data_port)
    except (TypeError, ValueError):
        return jsonify({'exists': False})

    sid, client = get_existing_lidar_by_port(ip, data_port)
    if client and client.connected:
        return jsonify({
            'exists': True,
            'session_id': sid,
            'capturing': client.running,
            'status': client.get_status(),
        })
    return jsonify({'exists': False})


@app.route('/api/lidar/export_pointcloud', methods=['POST'])
def lidar_export_pointcloud():
    """将雷达当前帧点云数据导出为文件

    支持的导出格式: PLY, PCD, XYZ
    """
    data = request.json
    lidar_session_id = data.get('lidar_session_id')
    export_format = data.get('format', 'ply')

    if export_format not in ('ply', 'pcd', 'xyz'):
        return jsonify({'error': f'不支持的导出格式: {export_format}，支持: ply, pcd, xyz'}), 400

    client = get_lidar(lidar_session_id)
    if client is None:
        return jsonify({'error': '雷达未连接'}), 400

    cloud = client.get_open3d_cloud()
    if cloud is None:
        return jsonify({'error': '没有可用的点云数据，请先捕获数据'}), 400

    filename = f"lidar_{lidar_session_id[:8]}_{int(time.time())}.{export_format}"
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    try:
        o3d.io.write_point_cloud(output_path, cloud)
        return jsonify({
            'download_url': f'/api/download/{filename}',
            'filename': filename,
            'num_points': len(cloud.points),
        })
    except Exception as e:
        return jsonify({'error': f'导出失败: {str(e)}'}), 500


# ═══════════════════════════════════════════════════════════
# MS200P 串口雷达端点
# ═══════════════════════════════════════════════════════════

@app.route('/api/serial/ports', methods=['GET'])
def serial_ports():
    """获取系统可用串口列表"""
    try:
        ports = list_serial_ports()
        return jsonify({'ports': ports})
    except Exception as e:
        return jsonify({'error': f'获取串口列表失败: {str(e)}'}), 500


@app.route('/api/ms200p/connect', methods=['POST'])
def ms200p_connect():
    """建立MS200P串口连接"""
    data = request.json
    port = data.get('port', '').strip()
    baud_rate = data.get('baud_rate', 230400)
    capture_interval = float(data.get('capture_interval', 0.1))
    min_distance = float(data.get('min_distance', 0.0))

    if not port:
        return jsonify({'error': '请选择串口设备'}), 400

    session_id = str(uuid.uuid4())

    try:
        baud_rate = int(baud_rate)
        client = create_ms200p(session_id, port, baud_rate, capture_interval, min_distance)
        client.connect()

        return jsonify({
            'session_id': session_id,
            'status': client.get_status(),
        })
    except Exception as e:
        remove_ms200p(session_id)
        return jsonify({'error': f'连接MS200P失败: {str(e)}'}), 500


@app.route('/api/ms200p/start', methods=['POST'])
def ms200p_start():
    """开始MS200P数据捕获"""
    data = request.json
    session_id = data.get('session_id')
    capture_interval = float(data.get('capture_interval', 0.1))
    min_distance = float(data.get('min_distance', 0.0))

    client = get_ms200p(session_id)
    if client is None:
        return jsonify({'error': 'MS200P未连接，请先建立连接'}), 400

    try:
        if not client.connected:
            client.connect()
        client.capture_interval = capture_interval
        client.min_distance = min_distance
        client.start_capture()
        return jsonify({'status': client.get_status()})
    except Exception as e:
        return jsonify({'error': f'启动MS200P捕获失败: {str(e)}'}), 500


@app.route('/api/ms200p/stop', methods=['POST'])
def ms200p_stop():
    """停止MS200P数据捕获"""
    data = request.json
    session_id = data.get('session_id')

    client = get_ms200p(session_id)
    if client is None:
        return jsonify({'error': 'MS200P未连接'}), 400

    try:
        client.stop_capture()
        return jsonify({'status': client.get_status()})
    except Exception as e:
        return jsonify({'error': f'停止MS200P捕获失败: {str(e)}'}), 500


@app.route('/api/ms200p/data', methods=['POST'])
def ms200p_data():
    """获取MS200P累积点云数据"""
    data = request.json
    session_id = data.get('session_id')

    client = get_ms200p(session_id)
    if client is None:
        return jsonify({'error': 'MS200P未连接'}), 400

    return jsonify({
        'point_cloud': client.get_accumulated_points_json(max_points=500000),
        'status': client.get_status(),
    })


@app.route('/api/ms200p/status', methods=['POST'])
def ms200p_status():
    """获取MS200P连接状态"""
    data = request.json
    session_id = data.get('session_id')

    client = get_ms200p(session_id)
    if client is None:
        return jsonify({'connected': False, 'capturing': False})

    return jsonify({'status': client.get_status()})


@app.route('/api/ms200p/disconnect', methods=['POST'])
def ms200p_disconnect():
    """断开MS200P连接"""
    data = request.json
    session_id = data.get('session_id')

    remove_ms200p(session_id)
    return jsonify({'connected': False, 'capturing': False})


@app.route('/api/ms200p/to_session', methods=['POST'])
def ms200p_to_session():
    """将MS200P累积点云导入为文件处理会话"""
    data = request.json
    ms200p_session_id = data.get('ms200p_session_id')

    client = get_ms200p(ms200p_session_id)
    if client is None:
        return jsonify({'error': 'MS200P未连接'}), 400

    cloud = client.get_open3d_cloud()
    if cloud is None:
        return jsonify({'error': '没有可用的点云数据，请先捕获数据'}), 400

    session_id = str(uuid.uuid4())

    exporter = ModelExporter()
    # MS200P 扫描一圈约 30 万点，采样到 5 万点足够可视化且不卡浏览器
    pc_data = exporter.point_cloud_to_json(cloud, max_points=50000)

    info = {
        'num_points': len(cloud.points),
        'has_normals': False,
        'has_colors': False,
        'center': np.array(cloud.get_center()).tolist(),
        'min_bound': np.array(cloud.get_min_bound()).tolist(),
        'max_bound': np.array(cloud.get_max_bound()).tolist(),
    }

    sessions[session_id] = {
        'filepath': None,
        'loader': None,
        'cloud': cloud,
        'preprocessor': None,
        'reconstructor': None,
        'mesh': None,
    }

    return jsonify({
        'session_id': session_id,
        'info': numpy_to_list(info),
        'point_cloud': pc_data,
    })


@app.route('/api/ms200p/export_pointcloud', methods=['POST'])
def ms200p_export_pointcloud():
    """导出MS200P累积点云数据为文件"""
    data = request.json
    ms200p_session_id = data.get('ms200p_session_id')
    export_format = data.get('format', 'ply')

    if export_format not in ('ply', 'pcd', 'xyz'):
        return jsonify({'error': f'不支持的导出格式: {export_format}，支持: ply, pcd, xyz'}), 400

    client = get_ms200p(ms200p_session_id)
    if client is None:
        return jsonify({'error': 'MS200P未连接'}), 400

    cloud = client.get_open3d_cloud()
    if cloud is None:
        return jsonify({'error': '没有可用的点云数据，请先捕获数据'}), 400

    filename = f"ms200p_{ms200p_session_id[:8]}_{int(time.time())}.{export_format}"
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    try:
        o3d.io.write_point_cloud(output_path, cloud)
        return jsonify({
            'download_url': f'/api/download/{filename}',
            'filename': filename,
            'num_points': len(cloud.points),
        })
    except Exception as e:
        return jsonify({'error': f'导出失败: {str(e)}'}), 500


@app.route('/api/spatial_model', methods=['POST'])
def spatial_model():
    """处理空间模型：从点云数据提取房间外框架

    请求参数:
        session_id: 文件上传会话ID
        num_layers: 高度分层数 (默认15)
        min_distance: 最小保留距离(米) (默认1.0，小于此距离的点被丢弃)
        grid_size: 2D栅格大小(米) (默认0.08，越小越精细但计算量大)
        ransac_threshold: RANSAC墙面拟合阈值(米) (默认0.12，越小墙面越精确但可能碎片化)
    """
    data = request.json
    session_id = data.get('session_id')
    if session_id not in sessions:
        return jsonify({'error': '无效的会话'}), 400

    num_layers = data.get('num_layers', 15)
    min_distance = data.get('min_distance', 1.0)
    grid_size = data.get('grid_size', 0.08)
    ransac_threshold = data.get('ransac_threshold', 0.12)

    try:
        session = sessions[session_id]
        cloud = session['cloud']

        processor = SpatialModelProcessor(cloud)
        result = processor.process(
            num_layers=num_layers,
            min_distance=min_distance,
            grid_size=grid_size,
            ransac_threshold=ransac_threshold,
        )

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'空间模型处理失败: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
