"""模型导出模块，支持将网格模型导出为 STL、OBJ、PLY 等格式"""

import os
import numpy as np
import open3d as o3d


class ModelExporter:
    """模型导出器，将 Open3D TriangleMesh 导出为多种3D模型格式"""

    SUPPORTED_FORMATS = {
        '.stl': 'STL (Stereolithography)',
        '.obj': 'OBJ (Wavefront)',
        '.ply': 'PLY (Polygon)',
        '.off': 'OFF (Object File Format)',
        '.gltf': 'glTF (GL Transmission Format)',
        '.glb': 'GLB (Binary glTF)',
    }

    def __init__(self, mesh: o3d.geometry.TriangleMesh = None):
        self.mesh = mesh

    def set_mesh(self, mesh: o3d.geometry.TriangleMesh):
        """设置要导出的网格模型

        Args:
            mesh: Open3D TriangleMesh 对象
        """
        self.mesh = mesh

    def export(self, file_path: str, ascii_mode: bool = False) -> str:
        """将网格模型导出到文件

        根据文件扩展名自动选择导出格式

        Args:
            file_path: 输出文件路径
            ascii_mode: 是否使用 ASCII 模式（仅部分格式支持）

        Returns:
            导出文件的绝对路径

        Raises:
            ValueError: 没有网格模型或不支持的格式
        """
        if self.mesh is None:
            raise ValueError("没有可导出的网格模型")

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"不支持的导出格式: {ext}，"
                f"支持的格式: {', '.join(self.SUPPORTED_FORMATS.keys())}"
            )

        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)

        if ext == '.stl':
            return self._export_stl(file_path, ascii_mode)
        elif ext == '.obj':
            return self._export_obj(file_path)
        elif ext == '.ply':
            return self._export_ply(file_path, ascii_mode)
        elif ext == '.off':
            return self._export_off(file_path)
        elif ext in ('.gltf', '.glb'):
            return self._export_gltf(file_path)
        else:
            return self._export_generic(file_path)

    def _export_stl(self, file_path: str, ascii_mode: bool = False) -> str:
        """导出为 STL 格式"""
        if ascii_mode:
            o3d.io.write_triangle_mesh(file_path, self.mesh, write_ascii=True)
        else:
            o3d.io.write_triangle_mesh(file_path, self.mesh, write_ascii=False)
        return os.path.abspath(file_path)

    def _export_obj(self, file_path: str) -> str:
        """导出为 OBJ 格式"""
        o3d.io.write_triangle_mesh(file_path, self.mesh, write_ascii=True)
        return os.path.abspath(file_path)

    def _export_ply(self, file_path: str, ascii_mode: bool = False) -> str:
        """导出为 PLY 格式"""
        o3d.io.write_triangle_mesh(file_path, self.mesh, write_ascii=ascii_mode)
        return os.path.abspath(file_path)

    def _export_off(self, file_path: str) -> str:
        """导出为 OFF 格式"""
        o3d.io.write_triangle_mesh(file_path, self.mesh, write_ascii=True)
        return os.path.abspath(file_path)

    def _export_gltf(self, file_path: str) -> str:
        """导出为 glTF/GLB 格式"""
        o3d.io.write_triangle_mesh(file_path, self.mesh)
        return os.path.abspath(file_path)

    def _export_generic(self, file_path: str) -> str:
        """使用 Open3D 通用接口导出"""
        o3d.io.write_triangle_mesh(file_path, self.mesh)
        return os.path.abspath(file_path)

    def export_point_cloud(
        self, point_cloud: o3d.geometry.PointCloud, file_path: str
    ) -> str:
        """将点云数据导出到文件

        Args:
            point_cloud: Open3D PointCloud 对象
            file_path: 输出文件路径

        Returns:
            导出文件的绝对路径
        """
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        o3d.io.write_point_cloud(file_path, point_cloud)
        return os.path.abspath(file_path)

    def mesh_to_json(self) -> dict:
        """将网格模型转换为 JSON 格式，用于 Web 前端 Three.js 渲染

        Returns:
            包含顶点、面片、法线等数据的字典
        """
        if self.mesh is None:
            return {}

        vertices = np.asarray(self.mesh.vertices).tolist()
        triangles = np.asarray(self.mesh.triangles).tolist()

        result = {
            'vertices': vertices,
            'triangles': triangles,
            'num_vertices': len(vertices),
            'num_triangles': len(triangles),
        }

        if self.mesh.has_vertex_normals():
            result['vertex_normals'] = np.asarray(self.mesh.vertex_normals).tolist()

        if self.mesh.has_vertex_colors():
            result['vertex_colors'] = np.asarray(self.mesh.vertex_colors).tolist()

        if self.mesh.has_triangle_normals():
            result['triangle_normals'] = np.asarray(self.mesh.triangle_normals).tolist()

        return result

    def point_cloud_to_json(
        self, point_cloud: o3d.geometry.PointCloud, max_points: int = 500000
    ) -> dict:
        """将点云数据转换为 JSON 格式，用于 Web 前端渲染

        返回扁平数组格式（而非嵌套列表），前端可直接用于 Float32Array 构造，
        避免 JavaScript 侧逐点转换循环的开销。

        Args:
            point_cloud: Open3D PointCloud 对象
            max_points: 最大点数，超出时均匀采样

        Returns:
            {'num_points': N, 'positions': [x1,y1,z1,...], 'colors': [r1,g1,b1,...]}
        """
        points = np.asarray(point_cloud.points, dtype=np.float32)

        if len(points) > max_points:
            # 均匀采样替代随机采样，速度更快且分布更均匀
            indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int32)
            points = points[indices]
            if point_cloud.has_colors():
                colors = np.asarray(point_cloud.colors, dtype=np.float32)[indices]
            else:
                colors = None
        else:
            if point_cloud.has_colors():
                colors = np.asarray(point_cloud.colors, dtype=np.float32)
            else:
                colors = None

        # 扁平化：直接转为一维列表，前端可直接 new Float32Array(data.positions)
        result = {
            'num_points': int(len(points)),
            'positions': points.flatten().tolist(),
        }

        if colors is not None:
            result['colors'] = colors.flatten().tolist()

        return result

    def get_supported_formats(self) -> dict:
        """获取支持的导出格式列表"""
        return self.SUPPORTED_FORMATS.copy()
