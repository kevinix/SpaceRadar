"""空间模型处理器 — 从室内点云数据提取房间外框架房型图

处理流程（改进版）:
1. 距离过滤 → 丢弃近距离点（室内杂物、噪点）
2. 网格量化为2D occupancy map → 将点云投影到XY平面建立占据栅格
3. 形态学闭运算 → 填补墙体扫描缝隙
4. RANSAC 直线检测 → 从边界点中迭代提取墙面直线段
5. 墙面交点计算 → 直线两两求交得到房间角点
6. Douglas-Peucker 简化 → 角点排序形成房型轮廓多边形
7. 高度分层拉伸 → 将2D轮廓在Z轴方向拉伸为3D框架模型
"""

import numpy as np


class SpatialModelProcessor:
    """将室内激光雷达点云转换为房间外框架3D模型

    与旧版不同，新版使用 RANSAC 直线检测 + 墙面交点来计算房型轮廓，
    而非简单的角度扇区最远点，从而生成平直、规整的房间框架。
    """

    def __init__(self, cloud):
        """初始化处理器

        Args:
            cloud: open3d.geometry.PointCloud 对象
        """
        self.cloud = cloud
        self.vertices = []       # 所有顶点 [[x, y, z], ...]
        self.edges = []          # 线框边 [[v1_idx, v2_idx], ...]
        self.faces = []          # 三角面 [[v1, v2, v3], ...]
        self.result_info = {}    # 处理结果统计

    # ─── 公开接口 ──────────────────────────────────────────

    def process(self, num_layers=15, min_distance=0.5,
                grid_size=0.08, ransac_threshold=0.12):
        """执行完整的空间模型处理流程

        Args:
            num_layers: 高度分层数，决定垂直方向拉伸的精度 (推荐 10-30)
            min_distance: 最小保留距离(米)，小于此距离的点视为杂物被丢弃
            grid_size: 2D占据栅格的单元格大小(米)，越小越精细 (推荐 0.05-0.15)
            ransac_threshold: RANSAC直线拟合的距离阈值(米)，越小墙面越精确
                             但可能导致墙面碎片化 (推荐 0.08-0.20)

        Returns:
            dict: 包含 vertices / edges / faces / info 的模型数据
        """
        points = np.asarray(self.cloud.points)

        if len(points) < 20:
            raise ValueError("点云点数过少 (<20)，无法构建空间模型")

        # ── Step 1: 距离过滤，丢弃近距离杂点 ──
        h_dist = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        keep = h_dist >= min_distance
        far_pts = points[keep]

        if len(far_pts) < 10:
            raise ValueError(
                f"过滤后点数不足 ({len(far_pts)})，"
                f"请降低最小距离阈值 (当前: {min_distance}m)"
            )

        # ── Step 2: 取墙体高度范围的点（中间60%的Z范围） ──
        z_all = far_pts[:, 2]
        z_min, z_max = z_all.min(), z_all.max()
        z_mid_lo = z_min + (z_max - z_min) * 0.15
        z_mid_hi = z_min + (z_max - z_min) * 0.85
        wall_mask = (z_all >= z_mid_lo) & (z_all <= z_mid_hi)
        wall_pts = far_pts[wall_mask]

        if len(wall_pts) < 20:
            # 高度范围太窄，回退到使用全部点
            wall_pts = far_pts

        # ── Step 3: 2D占据栅格构建 + 闭运算填补缝隙 ──
        occupancy, grid_origin, grid_cellsize = self._build_occupancy_grid(
            wall_pts, grid_size
        )
        # 形态学闭运算: 先膨胀填小洞，再腐蚀恢复轮廓
        occupancy = self._morphological_close(occupancy, radius=2)

        # ── Step 4: 提取占据栅格的外边界点 ──
        boundary_pts = self._extract_boundary(occupancy, grid_origin, grid_cellsize)

        if len(boundary_pts) < 6:
            raise ValueError(
                "边界提取失败，无法检测到足够的墙面边界点。"
                "请尝试调整 grid_size 或 min_distance 参数"
            )

        # ── Step 5: RANSAC 迭代直线检测 ──
        wall_lines = self._detect_wall_lines(
            boundary_pts, ransac_threshold, max_walls=12
        )

        if len(wall_lines) < 3:
            raise ValueError(
                f"墙面检测不足 ({len(wall_lines)}条)，"
                "请调大 ransac_threshold 或减小 grid_size"
            )

        # ── Step 6: 墙面直线交点计算 → 房间角点 ──
        corners_2d = self._compute_corners(wall_lines)

        if len(corners_2d) < 3:
            raise ValueError("无法从墙面计算出足够的房间角点")

        # ── Step 7: Douglas-Peucker 简化 + 角点排序 ──
        outline_2d = self._sort_and_simplify(corners_2d)

        # ── Step 8: 高度分层 → 3D拉伸构建框架 ──
        self._build_extruded_frame(outline_2d, z_min, z_max, num_layers)

        if len(self.vertices) == 0:
            raise ValueError("未能生成有效的框架顶点")

        self.result_info = {
            'num_vertices': len(self.vertices),
            'num_edges': len(self.edges),
            'num_faces': len(self.faces),
            'num_wall_lines': len(wall_lines),
            'num_corners': len(outline_2d),
            'num_layers': num_layers,
            'z_min': float(z_min),
            'z_max': float(z_max),
        }

        return self._to_json()

    # ─── 内部方法 ──────────────────────────────────────────

    def _build_occupancy_grid(self, pts, cell_size):
        """将2D点云投影构建为占据栅格 (occupancy grid)

        Args:
            pts: (N, 3) 点云
            cell_size: 单元格大小（米）

        Returns:
            (occupancy, origin_xy, cell_size)
            occupancy: 布尔2D数组
            origin_xy: 栅格左下角世界坐标 (x, y)
        """
        x, y = pts[:, 0], pts[:, 1]
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        # 扩展2个cell的边距
        x_min -= cell_size * 2
        y_min -= cell_size * 2
        x_max += cell_size * 2
        y_max += cell_size * 2

        nx = int(np.ceil((x_max - x_min) / cell_size)) + 1
        ny = int(np.ceil((y_max - y_min) / cell_size)) + 1

        occupancy = np.zeros((nx, ny), dtype=bool)

        xi = np.floor((x - x_min) / cell_size).astype(np.int32)
        yi = np.floor((y - y_min) / cell_size).astype(np.int32)
        valid = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
        occupancy[xi[valid], yi[valid]] = True

        return occupancy, (x_min, y_min), cell_size

    def _morphological_close(self, grid, radius=2):
        """形态学闭运算: 先膨胀后腐蚀，用于填补墙体扫描缝隙

        对于稀疏扫描导致的墙体不连续，膨胀可以连接相邻的占据cell，
        腐蚀则将轮廓恢复为原始大小，整体效果是填充小缝隙。

        Args:
            grid: 布尔2D数组
            radius: 形态学操作半径（cell数）

        Returns:
            闭运算后的布尔2D数组
        """
        nx, ny = grid.shape
        result = grid.copy()

        # 创建圆形结构元素
        struct = np.zeros((radius * 2 + 1, radius * 2 + 1), dtype=bool)
        cy = cx = radius
        for i in range(radius * 2 + 1):
            for j in range(radius * 2 + 1):
                if (i - cy)**2 + (j - cx)**2 <= radius**2:
                    struct[i, j] = True

        # 膨胀
        dilated = np.zeros_like(result)
        occupied = np.argwhere(result)
        for ox, oy in occupied:
            si_min = max(0, ox - radius)
            si_max = min(nx, ox + radius + 1)
            sj_min = max(0, oy - radius)
            sj_max = min(ny, oy + radius + 1)
            dilated[si_min:si_max, sj_min:sj_max] = True

        # 腐蚀: 只有邻居都被占据才保留
        eroded = np.zeros_like(dilated)
        occ_dilated = np.argwhere(dilated)
        for ox, oy in occ_dilated:
            # 检查以该点为中心的窗口内是否有空隙
            si_min = max(0, ox - 1)
            si_max = min(nx, ox + 2)
            sj_min = max(0, oy - 1)
            sj_max = min(ny, oy + 2)
            if np.all(dilated[si_min:si_max, sj_min:sj_max]):
                eroded[ox, oy] = True

        return eroded

    def _extract_boundary(self, occupancy, origin, cell_size):
        """从占据栅格提取外边界点

        使用简单的边界追踪: 对于每个占据cell，如果其4邻域中
        有任何一个未被占据，则该cell是边界cell。

        Args:
            occupancy: 布尔2D数组
            origin: (x_min, y_min) 世界坐标
            cell_size: 单元格大小

        Returns:
            (M, 2) numpy数组，边界点的2D世界坐标
        """
        nx, ny = occupancy.shape
        boundary = []

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if occupancy[i, j]:
                    # 检查4邻域
                    neighbors = [
                        occupancy[i-1, j], occupancy[i+1, j],
                        occupancy[i, j-1], occupancy[i, j+1]
                    ]
                    if not all(neighbors):
                        # 边界点: 取cell中心的世界坐标
                        wx = origin[0] + (i + 0.5) * cell_size
                        wy = origin[1] + (j + 0.5) * cell_size
                        boundary.append([wx, wy])

        return np.array(boundary)

    def _detect_wall_lines(self, points_2d, threshold, max_walls):
        """迭代RANSAC直线检测：从2D边界点中提取墙面直线段

        每轮RANSAC拟合一条直线，提取其内点作为一条墙面，
        然后移除这些内点，继续检测下一条墙面。

        Args:
            points_2d: (N, 2) 边界点坐标
            threshold: RANSAC内点距离阈值
            max_walls: 最大检测墙面数

        Returns:
            list of dict: [{'start': [x,y], 'end': [x,y], 'normal': [nx,ny]}, ...]
        """
        remaining = points_2d.copy()
        remaining_indices = np.arange(len(remaining))
        wall_lines = []

        for _ in range(max_walls):
            if len(remaining) < 15:
                break

            line_info, inlier_local = self._fit_line_ransac(
                remaining, threshold, max_iter=200
            )

            if line_info is None or len(inlier_local) < 10:
                break

            # 提取内点的实际坐标
            inlier_pts = remaining[inlier_local]
            p1, direction, normal = line_info

            # 沿直线方向投影，取首尾端点作为墙体起止点
            dots = inlier_pts @ direction
            start_idx = np.argmin(dots)
            end_idx = np.argmax(dots)
            start = inlier_pts[start_idx].tolist()
            end = inlier_pts[end_idx].tolist()

            wall_len = np.linalg.norm(np.array(end) - np.array(start))
            if wall_len < 0.3:  # 忽略过短的线段（噪点）
                # 移除这些点但不算作有效墙面
                remaining = np.delete(remaining, inlier_local, axis=0)
                continue

            wall_lines.append({
                'start': start,
                'end': end,
                'normal': normal.tolist(),
                'length': float(wall_len),
            })

            # 移除外点（保留非墙面点用于后续检测）
            mask = np.ones(len(remaining), dtype=bool)
            mask[inlier_local] = False
            remaining = remaining[mask]

        return wall_lines

    def _fit_line_ransac(self, points, threshold, max_iter):
        """RANSAC 2D直线拟合

        随机采样两点确定一条直线，计算所有点到该直线的距离，
        选取内点最多的直线作为最佳拟合。

        Args:
            points: (N, 2) 点集
            threshold: 内点距离阈值
            max_iter: 最大迭代次数

        Returns:
            ((p1, direction, normal), inlier_indices) 或 (None, None)
        """
        n = len(points)
        if n < 2:
            return None, None

        best_inliers = np.array([], dtype=np.int64)
        best_line = None
        best_count = 0

        for _ in range(max_iter):
            # 随机采样两个点
            idx = np.random.choice(n, 2, replace=False)
            p1, p2 = points[idx[0]], points[idx[1]]
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length < 1e-8:
                continue
            direction = direction / length
            normal = np.array([-direction[1], direction[0]])

            # 计算所有点到直线的垂直距离
            distances = np.abs((points - p1) @ normal)
            inliers = np.where(distances < threshold)[0]

            if len(inliers) > best_count:
                best_count = len(inliers)
                best_inliers = inliers
                best_line = (p1, direction, normal)

        if best_line is None or best_count < 5:
            return None, None

        return best_line, best_inliers

    def _compute_corners(self, wall_lines):
        """计算墙面直线的交点，作为房间角点

        两两计算墙面直线段的交点。只有交点位于两条线段
        的延长线范围内（或足够接近端点）才保留。

        Args:
            wall_lines: 墙面直线列表

        Returns:
            list of [x, y] 角点坐标
        """
        corners = []

        for i in range(len(wall_lines)):
            for j in range(i + 1, len(wall_lines)):
                w1, w2 = wall_lines[i], wall_lines[j]

                # 计算两条直线的交点
                p1 = np.array(w1['start'])
                d1 = np.array(w1['end']) - np.array(w1['start'])
                p2 = np.array(w2['start'])
                d2 = np.array(w2['end']) - np.array(w2['start'])

                # 检查是否接近平行（夹角<15°则跳过）
                cos_angle = abs(np.dot(d1, d2) / (
                    np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-10
                ))
                if cos_angle > 0.96:  # cos(15°) ≈ 0.966
                    continue

                intersection = self._line_intersection_2d(p1, p1 + d1, p2, p2 + d2)
                if intersection is None:
                    continue

                # 验证交点是否在线段附近（在线段延长线2倍长度范围内）
                if self._point_near_segment(intersection, w1, extend=2.0) and \
                   self._point_near_segment(intersection, w2, extend=2.0):
                    corners.append(intersection.tolist())

        return corners

    def _line_intersection_2d(self, p1, p2, p3, p4):
        """计算两条二维线段的交点

        Args:
            p1, p2: 第一条线的两个端点
            p3, p4: 第二条线的两个端点

        Returns:
            交点坐标 (2,) numpy数组，或无交点时返回 None
        """
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        x3, y3 = p3[0], p3[1]
        x4, y4 = p4[0], p4[1]

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)

        return np.array([px, py])

    def _point_near_segment(self, pt, wall, extend=1.5):
        """判断点是否在线段附近（在延长线范围内）

        Args:
            pt: 待检查的点 (2,) numpy数组
            wall: 墙面字典，包含 start, end
            extend: 延长倍数，1.0 为线段本身，>1 允许在线段延长线上

        Returns:
            bool
        """
        start = np.array(wall['start'])
        end = np.array(wall['end'])
        direction = end - start
        seg_len = np.linalg.norm(direction)
        if seg_len < 1e-8:
            return False
        direction = direction / seg_len

        # 点在线段上的投影参数
        t = np.dot(pt - start, direction)
        # 允许在延长线范围内
        return -seg_len * (extend - 1) / 2 <= t <= seg_len * (extend + 1) / 2

    def _sort_and_simplify(self, corners):
        """对角点进行排序和简化，形成闭合的房型轮廓多边形

        1. 合并距离过近的角点（去重）
        2. 按角度排序形成多边形
        3. Douglas-Peucker 简化

        Args:
            corners: 角点列表 [[x, y], ...]

        Returns:
            简化后的轮廓点列表 [[x, y], ...]
        """
        if len(corners) < 3:
            return corners

        pts = np.array(corners)

        # 合并距离过近的角点（< 0.15m 视为同一个）
        merged = [pts[0]]
        for pt in pts[1:]:
            dists = [np.linalg.norm(pt - np.array(m)) for m in merged]
            if min(dists) > 0.15:
                merged.append(pt.tolist())

        if len(merged) < 3:
            return corners

        pts = np.array(merged)

        # 按角度排序
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 0] - center[0], pts[:, 1] - center[1])
        sorted_idx = np.argsort(angles)
        pts = pts[sorted_idx]

        # Douglas-Peucker 简化
        simplified = self._douglas_peucker(pts, epsilon=0.15)

        return simplified.tolist()

    def _douglas_peucker(self, points, epsilon):
        """Douglas-Peucker 折线简化算法

        递归地将折线中偏离直线较小的点移除，保留关键的拐角点。

        Args:
            points: (N, 2) 点序列
            epsilon: 简化容差（米），值越大保留的点越少

        Returns:
            简化后的 (M, 2) 点序列
        """
        if len(points) <= 2:
            return points

        # 找到距离首尾连线最远的点
        start, end = points[0], points[-1]
        line_vec = end - start
        line_len_sq = np.dot(line_vec, line_vec)

        if line_len_sq < 1e-10:
            return np.array([start, end])

        # 点到直线的垂直距离
        max_dist = 0
        max_idx = 0
        for i in range(1, len(points) - 1):
            # 使用叉积计算点到直线的距离
            cross = abs(
                (points[i][0] - start[0]) * (end[1] - start[1]) -
                (points[i][1] - start[1]) * (end[0] - start[0])
            )
            dist = cross / np.sqrt(line_len_sq)
            if dist > max_dist:
                max_dist = dist
                max_idx = i

        if max_dist > epsilon:
            left = self._douglas_peucker(points[:max_idx + 1], epsilon)
            right = self._douglas_peucker(points[max_idx:], epsilon)
            return np.vstack([left[:-1], right])
        else:
            return np.array([start, end])

    def _build_extruded_frame(self, outline_2d, z_min, z_max, num_layers):
        """从2D房型轮廓拉伸为3D框架模型

        在 Z 轴方向分 num_layers 层，每层复制2D轮廓。
        相邻层之间创建三角面片形成墙面，顶部和底部封盖。

        Args:
            outline_2d: 2D轮廓点列表 [[x, y], ...]
            z_min: 模型底部Z值
            z_max: 模型顶部Z值
            num_layers: 层数
        """
        n_pts = len(outline_2d)
        if n_pts < 3:
            return

        # 如果 Z 范围太小，扩展一定高度
        if z_max - z_min < 0.5:
            z_mid = (z_min + z_max) / 2
            z_min = z_mid - 0.5
            z_max = z_mid + 0.5

        # ── 为每层生成3D轮廓顶点 ──
        layer_zs = np.linspace(z_min, z_max, num_layers)
        self.vertices = []
        layer_starts = []

        for lz in layer_zs:
            layer_starts.append(len(self.vertices))
            for pt in outline_2d:
                self.vertices.append([float(pt[0]), float(pt[1]), float(lz)])

        # ── 层间三角面片（垂直墙面） ──
        for li in range(num_layers - 1):
            sa = layer_starts[li]
            sb = layer_starts[li + 1]
            for i in range(n_pts):
                j = (i + 1) % n_pts
                self.faces.append([sa + i, sa + j, sb + j])
                self.faces.append([sa + i, sb + j, sb + i])

        # ── 底部封盖 ──
        bottom_center = self._mean_xy(outline_2d, layer_zs[0])
        bc_idx = len(self.vertices)
        self.vertices.append(bottom_center)
        for i in range(n_pts):
            j = (i + 1) % n_pts
            self.faces.append([layer_starts[0] + i, layer_starts[0] + j, bc_idx])

        # ── 顶部封盖 ──
        top_center = self._mean_xy(outline_2d, layer_zs[-1])
        tc_idx = len(self.vertices)
        self.vertices.append(top_center)
        for i in range(n_pts):
            j = (i + 1) % n_pts
            self.faces.append([layer_starts[-1] + j, layer_starts[-1] + i, tc_idx])

        # ── 线框边：水平轮廓 + 垂直棱线 ──
        for li in range(num_layers):
            s = layer_starts[li]
            for i in range(n_pts):
                self.edges.append([s + i, s + (i + 1) % n_pts])

        for li in range(num_layers - 1):
            sa = layer_starts[li]
            sb = layer_starts[li + 1]
            for i in range(n_pts):
                self.edges.append([sa + i, sb + i])

    def _mean_xy(self, outline_2d, z):
        """计算2D轮廓的几何中心，返回3D坐标

        Args:
            outline_2d: 2D点列表
            z: Z坐标值

        Returns:
            [cx, cy, z] 中心坐标
        """
        cx = sum(pt[0] for pt in outline_2d) / len(outline_2d)
        cy = sum(pt[1] for pt in outline_2d) / len(outline_2d)
        return [float(cx), float(cy), float(z)]

    def _to_json(self):
        """转换为前端可用JSON格式"""
        return {
            'vertices': self.vertices,
            'edges': self.edges,
            'faces': self.faces,
            'info': self.result_info,
        }
