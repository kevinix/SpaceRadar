"""点云预处理模块，提供去噪、下采样、法线估计、配准等功能"""

import numpy as np
import open3d as o3d


class PointCloudPreprocessor:
    """点云预处理器，提供多种数据清洗和优化操作"""

    def __init__(self, point_cloud: o3d.geometry.PointCloud):
        self.original_cloud = point_cloud
        self.cloud = o3d.geometry.PointCloud(point_cloud)

    def remove_noise_statistical(
        self, nb_neighbors: int = 20, std_ratio: float = 2.0
    ) -> o3d.geometry.PointCloud:
        """使用统计离群点去除方法去除噪声点

        对每个点计算到其邻居的平均距离，然后剔除偏离全局统计的离群点

        Args:
            nb_neighbors: 用于计算平均距离的邻居数量
            std_ratio: 标准差倍数阈值，越小过滤越严格

        Returns:
            去噪后的点云
        """
        self.cloud, _ = self.cloud.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        return self.cloud

    def remove_noise_radius(
        self, nb_points: int = 16, radius: float = 1.0
    ) -> o3d.geometry.PointCloud:
        """使用半径离群点去除方法去除噪声点

        对每个点检查指定半径内的邻居数量，少于阈值的点被移除

        Args:
            nb_points: 半径内最少邻居数量
            radius: 搜索半径

        Returns:
            去噪后的点云
        """
        self.cloud, _ = self.cloud.remove_radius_outlier(
            nb_points=nb_points, radius=radius
        )
        return self.cloud

    def voxel_downsample(
        self, voxel_size: float = 0.02
    ) -> o3d.geometry.PointCloud:
        """使用体素下采样方法减少点云密度

        将空间划分为指定大小的体素网格，每个体素内的点取质心

        Args:
            voxel_size: 体素边长，越大下采样越激进

        Returns:
            下采样后的点云
        """
        self.cloud = self.cloud.voxel_down_sample(voxel_size=voxel_size)
        return self.cloud

    def uniform_downsample(self, every_k_points: int = 5) -> o3d.geometry.PointCloud:
        """使用均匀下采样方法减少点云数量

        每隔 k 个点保留一个

        Args:
            every_k_points: 采样间隔

        Returns:
            下采样后的点云
        """
        self.cloud = self.cloud.uniform_down_sample(every_k_points=every_k_points)
        return self.cloud

    def estimate_normals(
        self,
        radius: float = 0.1,
        max_nn: int = 30,
        camera_location: list = None,
    ) -> o3d.geometry.PointCloud:
        """估计点云法线向量

        使用混合搜索方法估计每个点的法线方向

        Args:
            radius: 搜索半径
            max_nn: 最大邻居数
            camera_location: 相机位置，用于统一法线朝向

        Returns:
            带法线的点云
        """
        self.cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=max_nn
            )
        )
        if camera_location is not None:
            self.cloud.orient_normals_towards_camera_location(
                camera_location=np.array(camera_location)
            )
        else:
            self.cloud.orient_normals_consistent_tangent_plane(k=15)

        return self.cloud

    def crop_by_bounding_box(
        self, min_bound: list, max_bound: list
    ) -> o3d.geometry.PointCloud:
        """使用轴对齐包围盒裁剪点云

        保留指定范围内的点

        Args:
            min_bound: 包围盒最小角 [x, y, z]
            max_bound: 包围盒最大角 [x, y, z]

        Returns:
            裁剪后的点云
        """
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=min_bound, max_bound=max_bound
        )
        self.cloud = self.cloud.crop(bbox)
        return self.cloud

    def filter_by_height(
        self, min_height: float = -float('inf'), max_height: float = float('inf')
    ) -> o3d.geometry.PointCloud:
        """按 Z 轴高度过滤点云

        Args:
            min_height: 最小高度
            max_height: 最大高度

        Returns:
            过滤后的点云
        """
        points = np.asarray(self.cloud.points)
        mask = (points[:, 2] >= min_height) & (points[:, 2] <= max_height)
        self.cloud = self.cloud.select_by_index(np.where(mask)[0])
        return self.cloud

    def icp_registration(
        self,
        target: o3d.geometry.PointCloud,
        max_correspondence_distance: float = 0.02,
        max_iteration: int = 100,
    ) -> dict:
        """使用 ICP 算法将当前点云配准到目标点云

        通过迭代最近点算法计算变换矩阵，使源点云对齐到目标点云

        Args:
            target: 目标点云
            max_correspondence_distance: 最大对应点距离
            max_iteration: 最大迭代次数

        Returns:
            包含变换矩阵、匹配度等信息的字典
        """
        source = o3d.geometry.PointCloud(self.cloud)

        if not source.has_normals():
            source.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
        if not target.has_normals():
            target_temp = o3d.geometry.PointCloud(target)
            target_temp.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            target = target_temp

        result = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration),
        )

        self.cloud.transform(result.transformation)

        return {
            'transformation': result.transformation.tolist(),
            'fitness': result.fitness,
            'inlier_rmse': result.inlier_rmse,
        }

    def run_pipeline(self, steps: list) -> o3d.geometry.PointCloud:
        """按顺序执行预处理流水线

        Args:
            steps: 预处理步骤列表，每个元素为 (方法名, 参数字典) 的元组

        Returns:
            处理后的点云

        Example:
            steps = [
                ('remove_noise_statistical', {'nb_neighbors': 20, 'std_ratio': 2.0}),
                ('voxel_downsample', {'voxel_size': 0.05}),
                ('estimate_normals', {'radius': 0.1, 'max_nn': 30}),
            ]
        """
        for step_name, params in steps:
            method = getattr(self, step_name, None)
            if method is None:
                raise ValueError(f"未知的预处理步骤: {step_name}")
            method(**params)

        return self.cloud

    def get_result(self) -> o3d.geometry.PointCloud:
        """获取当前处理结果"""
        return self.cloud

    def get_stats(self) -> dict:
        """获取当前点云的统计信息"""
        if len(self.cloud.points) == 0:
            return {'num_points': 0}

        points = np.asarray(self.cloud.points)
        stats = {
            'original_points': len(self.original_cloud.points),
            'current_points': len(self.cloud.points),
            'has_normals': self.cloud.has_normals(),
            'has_colors': self.cloud.has_colors(),
            'center': np.array(self.cloud.get_center()).tolist(),
            'min_bound': np.array(self.cloud.get_min_bound()).tolist(),
            'max_bound': np.array(self.cloud.get_max_bound()).tolist(),
            'bounding_box_size': (
                np.array(self.cloud.get_max_bound()) - np.array(self.cloud.get_min_bound())
            ).tolist(),
        }
        return stats
