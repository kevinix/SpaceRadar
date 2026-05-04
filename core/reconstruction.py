"""三维重建模块，提供泊松表面重建、Alpha Shapes 和 Ball Pivoting 算法"""

import numpy as np
import open3d as o3d


class MeshReconstructor:
    """网格重建器，将点云数据转换为三角网格模型"""

    def __init__(self, point_cloud: o3d.geometry.PointCloud):
        self.cloud = point_cloud
        self.mesh = None

    def poisson_reconstruction(
        self, depth: int = 9, width: int = 0, scale: float = 1.1, linear_fit: bool = False
    ) -> o3d.geometry.TriangleMesh:
        """使用泊松表面重建算法生成网格模型

        通过求解泊松方程从点云法线场中提取等值面，生成水密的三角网格

        Args:
            depth: 八叉树最大深度，控制重建精度，越大越精细
            width: 八叉树宽度，0 表示自适应
            scale: 密度值的缩放因子
            linear_fit: 是否使用线性插值

        Returns:
            重建的三角网格模型

        Raises:
            ValueError: 点云没有法线信息
        """
        if not self.cloud.has_normals():
            raise ValueError("泊松重建需要点云包含法线信息，请先估计法线")

        result, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd=self.cloud,
            depth=depth,
            width=width,
            scale=scale,
            linear_fit=linear_fit,
        )

        densities = np.asarray(densities)
        density_threshold = np.percentile(densities, 10)
        vertices_to_remove = densities < density_threshold
        result.remove_vertices_by_mask(vertices_to_remove)

        result.remove_degenerate_triangles()
        result.remove_duplicated_triangles()
        result.remove_duplicated_vertices()
        result.remove_non_manifold_edges()

        self.mesh = result
        return self.mesh

    def alpha_shapes(
        self, alpha: float = 0.5
    ) -> o3d.geometry.TriangleMesh:
        """使用 Alpha Shapes 算法生成网格模型

        通过控制 alpha 参数决定表面细节程度，alpha 越小表面越贴合点云

        Args:
            alpha: alpha 参数值，控制重建的松紧程度

        Returns:
            重建的三角网格模型
        """
        self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd=self.cloud, alpha=alpha
        )

        self.mesh.remove_degenerate_triangles()
        self.mesh.remove_duplicated_triangles()
        self.mesh.remove_duplicated_vertices()
        self.mesh.remove_non_manifold_edges()

        return self.mesh

    def ball_pivoting(
        self, radii: list = None
    ) -> o3d.geometry.TriangleMesh:
        """使用 Ball Pivoting 算法生成网格模型

        通过在点云表面滚动虚拟球体来构建三角面片

        Args:
            radii: 滚动球半径列表，默认根据点云密度自动计算

        Returns:
            重建的三角网格模型

        Raises:
            ValueError: 点云没有法线信息
        """
        if not self.cloud.has_normals():
            raise ValueError("Ball Pivoting 需要点云包含法线信息，请先估计法线")

        if radii is None:
            distances = self.cloud.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radii = [
                o3d.utility.DoubleVector([avg_dist * 2, avg_dist * 4, avg_dist * 8])
            ]
        else:
            radii = [o3d.utility.DoubleVector(radii)]

        self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd=self.cloud, radii=radii[0]
        )

        self.mesh.remove_degenerate_triangles()
        self.mesh.remove_duplicated_triangles()
        self.mesh.remove_duplicated_vertices()
        self.mesh.remove_non_manifold_edges()

        return self.mesh

    def reconstruct(self, method: str = 'poisson', **kwargs) -> o3d.geometry.TriangleMesh:
        """根据方法名称选择重建算法

        Args:
            method: 重建方法，支持 'poisson', 'alpha', 'ball_pivoting'
            **kwargs: 传递给具体重建方法的参数

        Returns:
            重建的三角网格模型

        Raises:
            ValueError: 不支持的重建方法
        """
        methods = {
            'poisson': self.poisson_reconstruction,
            'alpha': self.alpha_shapes,
            'ball_pivoting': self.ball_pivoting,
        }

        if method not in methods:
            raise ValueError(
                f"不支持的重建方法: {method}，"
                f"支持的方法: {', '.join(methods.keys())}"
            )

        return methods[method](**kwargs)

    def get_mesh_info(self) -> dict:
        """获取当前网格模型的信息

        Returns:
            包含顶点数、面片数、是否水密等信息的字典
        """
        if self.mesh is None:
            return {}

        return {
            'num_vertices': len(self.mesh.vertices),
            'num_triangles': len(self.mesh.triangles),
            'has_vertex_normals': self.mesh.has_vertex_normals(),
            'has_vertex_colors': self.mesh.has_vertex_colors(),
            'has_triangle_normals': self.mesh.has_triangle_normals(),
            'is_watertight': self.mesh.is_watertight(),
            'center': np.array(self.mesh.get_center()).tolist(),
            'min_bound': np.array(self.mesh.get_min_bound()).tolist(),
            'max_bound': np.array(self.mesh.get_max_bound()).tolist(),
        }

    def get_mesh(self) -> o3d.geometry.TriangleMesh:
        """获取当前网格模型"""
        return self.mesh

    def smooth_mesh(
        self, iterations: int = 10, lambda_filter: float = 0.5
    ) -> o3d.geometry.TriangleMesh:
        """对网格进行 Taubin 平滑处理

        Args:
            iterations: 平滑迭代次数
            lambda_filter: 平滑系数

        Returns:
            平滑后的网格
        """
        if self.mesh is None:
            raise ValueError("没有可平滑的网格模型")

        self.mesh = self.mesh.filter_smooth_taubin(
            number_of_iterations=iterations, lambda_filter=lambda_filter
        )
        self.mesh.compute_vertex_normals()
        return self.mesh

    def simplify_mesh(
        self, target_triangle_count: int = 10000
    ) -> o3d.geometry.TriangleMesh:
        """使用二次误差度量简化网格

        Args:
            target_triangle_count: 目标三角形数量

        Returns:
            简化后的网格
        """
        if self.mesh is None:
            raise ValueError("没有可简化的网格模型")

        self.mesh = self.mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_triangle_count
        )
        self.mesh.compute_vertex_normals()
        return self.mesh
