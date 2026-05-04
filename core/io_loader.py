"""点云数据读取模块，支持 PLY、PCD、XYZ 等多种格式"""

import os
import numpy as np
import open3d as o3d


class PointCloudLoader:
    """点云文件加载器，根据文件扩展名自动选择读取方式"""

    SUPPORTED_FORMATS = ('.ply', '.pcd', '.xyz', '.xyzn', '.xyzrgb', '.pts')

    def __init__(self):
        self.file_path = None
        self.point_cloud = None

    def load(self, file_path: str) -> o3d.geometry.PointCloud:
        """加载点云文件，根据扩展名自动选择读取方法

        Args:
            file_path: 点云文件路径

        Returns:
            Open3D PointCloud 对象

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的文件格式
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"不支持的文件格式: {ext}，"
                f"支持的格式: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        self.file_path = file_path

        if ext == '.ply':
            self.point_cloud = self._load_ply(file_path)
        elif ext == '.pcd':
            self.point_cloud = self._load_pcd(file_path)
        elif ext in ('.xyz', '.xyzn', '.xyzrgb', '.pts'):
            self.point_cloud = self._load_xyz(file_path)
        else:
            self.point_cloud = self._load_generic(file_path)

        return self.point_cloud

    def _load_ply(self, file_path: str) -> o3d.geometry.PointCloud:
        """读取 PLY 格式点云文件，支持 ASCII 和二进制模式"""
        pcd = o3d.io.read_point_cloud(file_path)
        if pcd.is_empty():
            mesh = o3d.io.read_triangle_mesh(file_path)
            if not mesh.is_empty():
                pcd = mesh.sample_points_uniformly(number_of_points=100000)
        return pcd

    def _load_pcd(self, file_path: str) -> o3d.geometry.PointCloud:
        """读取 PCD 格式点云文件"""
        pcd = o3d.io.read_point_cloud(file_path)
        return pcd

    def _load_xyz(self, file_path: str) -> o3d.geometry.PointCloud:
        """读取 XYZ 系列格式点云文件，自动检测列数和格式"""
        data = np.loadtxt(file_path)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, :3])

        if data.shape[1] >= 6:
            pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6])
        if data.shape[1] >= 9:
            pcd.normals = o3d.utility.Vector3dVector(data[:, 6:9])

        return pcd

    def _load_generic(self, file_path: str) -> o3d.geometry.PointCloud:
        """使用 Open3D 通用接口读取点云文件"""
        return o3d.io.read_point_cloud(file_path)

    def get_info(self) -> dict:
        """获取当前加载的点云文件信息

        Returns:
            包含点数、是否有法线/颜色等信息的字典
        """
        if self.point_cloud is None:
            return {}

        return {
            'file_path': self.file_path,
            'num_points': len(self.point_cloud.points),
            'has_normals': self.point_cloud.has_normals(),
            'has_colors': self.point_cloud.has_colors(),
            'center': np.array(self.point_cloud.get_center()).tolist(),
            'min_bound': np.array(self.point_cloud.get_min_bound()).tolist(),
            'max_bound': np.array(self.point_cloud.get_max_bound()).tolist(),
        }

    def to_numpy(self) -> tuple:
        """将点云数据转换为 numpy 数组

        Returns:
            (points, colors, normals) 元组，colors 和 normals 可能为 None
        """
        if self.point_cloud is None:
            return None, None, None

        points = np.asarray(self.point_cloud.points)
        colors = np.asarray(self.point_cloud.colors) if self.point_cloud.has_colors() else None
        normals = np.asarray(self.point_cloud.normals) if self.point_cloud.has_normals() else None

        return points, colors, normals
