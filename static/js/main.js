/**
 * SpaceDisplay 前端主脚本
 * 负责文件上传、参数配置、API交互和 Three.js 3D渲染
 */

let sessionId = null;
let currentPointCloud = null;
let currentMesh = null;
let scene, camera, renderer, controls;
let pointCloudObj = null;
let meshObj = null;
let lidarPointObj = null;
let frameObj = null;
let activeView = 'pointcloud';

let lidarSessionId = null;
let lidarCapturing = false;
let lidarPollingTimer = null;

// MS200P串口雷达状态
let ms200pSessionId = null;
let ms200pCapturing = false;
let ms200pPollingTimer = null;
let radarType = 'c16';  // 'c16' | 'ms200p'

document.addEventListener('DOMContentLoaded', function () {
    initThreeJS();
    bindEvents();
    animate();
    checkExistingConnection();
});

/**
 * 初始化 Three.js 场景、相机、渲染器和控制器
 */
function initThreeJS() {
    var container = document.getElementById('viewer');

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x010409);

    camera = new THREE.PerspectiveCamera(
        60, container.clientWidth / container.clientHeight, 0.01, 10000
    );
    camera.position.set(3, 3, 3);
    camera.up.set(0, 0, 1);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;

    scene.add(new THREE.AmbientLight(0x404040, 2));
    var dirLight = new THREE.DirectionalLight(0xffffff, 1);
    dirLight.position.set(5, 7, 10);
    scene.add(dirLight);

    var gridHelper = new THREE.GridHelper(10, 20, 0x30363d, 0x21262d);
    gridHelper.rotation.x = Math.PI / 2;
    scene.add(gridHelper);

    var axesHelper = new THREE.AxesHelper(2);
    scene.add(axesHelper);

    window.addEventListener('resize', function () {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    });
}

/**
 * Three.js 动画循环
 */
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

/**
 * 绑定所有 UI 事件
 */
function bindEvents() {
    var uploadArea = document.getElementById('uploadArea');
    var fileInput = document.getElementById('fileInput');

    uploadArea.addEventListener('click', function () { fileInput.click(); });
    uploadArea.addEventListener('dragover', function (e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    uploadArea.addEventListener('dragleave', function () {
        uploadArea.classList.remove('dragover');
    });
    uploadArea.addEventListener('drop', function (e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', function () {
        if (fileInput.files.length > 0) uploadFile(fileInput.files[0]);
    });

    document.getElementById('btnPreprocess').addEventListener('click', preprocess);
    document.getElementById('btnReconstruct').addEventListener('click', reconstruct);
    document.getElementById('btnExport').addEventListener('click', exportModel);
    document.getElementById('btnCapture').addEventListener('click', toggleCapture);
    document.getElementById('btnUseData').addEventListener('click', useLidarData);
    document.getElementById('btnExportPC').addEventListener('click', exportLidarPointcloud);
    document.getElementById('btnSpatialModel').addEventListener('click', processSpatialModel);

    // 雷达型号切换
    document.getElementById('radarType').addEventListener('change', switchRadarType);

    // MS200P事件绑定
    document.getElementById('btnRefreshPorts').addEventListener('click', refreshSerialPorts);
    document.getElementById('btnMS200PCapture').addEventListener('click', toggleMS200PCapture);
    document.getElementById('btnMS200PUseData').addEventListener('click', useMS200PData);
    document.getElementById('btnMS200PExport').addEventListener('click', exportMS200PPointcloud);
    // 近距离过滤开关
    document.getElementById('chkMS200PMinDist').addEventListener('change', function () {
        document.getElementById('ms200pMinDistControls').style.display = this.checked ? 'block' : 'none';
    });

    // 初始加载串口列表
    refreshSerialPorts();

    var reconMethod = document.getElementById('reconMethod');
    reconMethod.addEventListener('change', function () {
        document.getElementById('poissonParams').style.display =
            reconMethod.value === 'poisson' ? 'block' : 'none';
        document.getElementById('alphaParams').style.display =
            reconMethod.value === 'alpha' ? 'block' : 'none';
        document.getElementById('bpParams').style.display =
            reconMethod.value === 'ball_pivoting' ? 'block' : 'none';
    });

    document.querySelectorAll('.tab').forEach(function (tab) {
        tab.addEventListener('click', function () {
            document.querySelectorAll('.tab').forEach(function (t) { t.classList.remove('active'); });
            tab.classList.add('active');
            activeView = tab.dataset.view;
            updateVisibility();
        });
    });
}

/**
 * 更新3D场景中点云和网格的可见性
 */
function updateVisibility() {
    if (pointCloudObj) pointCloudObj.visible = (activeView === 'pointcloud');
    if (meshObj) meshObj.visible = (activeView === 'mesh');
    if (lidarPointObj) lidarPointObj.visible = (activeView === 'lidar');
    if (frameObj) frameObj.visible = (activeView === 'frame');
}

/**
 * 上传点云文件到服务器
 */
function uploadFile(file) {
    var formData = new FormData();
    formData.append('file', file);

    showLoading('上传并加载点云数据...');
    fetch('/api/upload', { method: 'POST', body: formData })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            hideLoading();
            if (data.error) { showError(data.error); return; }
            sessionId = data.session_id;
            currentPointCloud = data.point_cloud;

            var info = data.info;
            document.getElementById('fileInfo').style.display = 'block';
            document.getElementById('fileInfo').innerHTML =
                '点数: ' + info.num_points + '<br>' +
                '中心: [' + info.center.map(function (v) { return v.toFixed(3); }).join(', ') + ']<br>' +
                '法线: ' + (info.has_normals ? '有' : '无') + ' | 颜色: ' + (info.has_colors ? '有' : '无');

            renderPointCloud(currentPointCloud);
            fitCamera();

            document.getElementById('btnPreprocess').disabled = false;
            document.getElementById('btnSpatialModel').disabled = false;
            // 清除旧的空间模型
            if (frameObj) { scene.remove(frameObj); frameObj = null; }
            document.getElementById('spatialModelInfo').style.display = 'none';
            setStatus('点云加载完成: ' + info.num_points + ' 个点');
        })
        .catch(function (err) { hideLoading(); showError('上传失败: ' + err.message); });
}

/**
 * 执行点云预处理
 */
function preprocess() {
    if (!sessionId) return;

    var steps = [];

    if (document.getElementById('chkDenoise').checked) {
        steps.push({
            name: 'remove_noise_statistical',
            params: {
                nb_neighbors: parseInt(document.getElementById('nbNeighbors').value),
                std_ratio: parseFloat(document.getElementById('stdRatio').value),
            }
        });
    }

    if (document.getElementById('chkDownsample').checked) {
        steps.push({
            name: 'voxel_downsample',
            params: { voxel_size: parseFloat(document.getElementById('voxelSize').value) }
        });
    }

    if (document.getElementById('chkNormals').checked) {
        steps.push({
            name: 'estimate_normals',
            params: {
                radius: parseFloat(document.getElementById('normalRadius').value),
                max_nn: 30
            }
        });
    }

    showLoading('预处理中...');
    fetch('/api/preprocess', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, steps: steps })
    })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            hideLoading();
            if (data.error) { showError(data.error); return; }

            currentPointCloud = data.point_cloud;
            var stats = data.stats;

            document.getElementById('preprocessInfo').style.display = 'block';
            document.getElementById('preprocessInfo').innerHTML =
                '原始点数: ' + stats.original_points + '<br>' +
                '处理后点数: ' + stats.current_points + '<br>' +
                '法线: ' + (stats.has_normals ? '有' : '无');

            renderPointCloud(currentPointCloud);
            fitCamera();

            document.getElementById('btnReconstruct').disabled = false;
            document.getElementById('btnSpatialModel').disabled = false;
            // 清除旧的空间模型
            if (frameObj) { scene.remove(frameObj); frameObj = null; }
            document.getElementById('spatialModelInfo').style.display = 'none';
            setStatus('预处理完成: ' + stats.current_points + ' 个点');
        })
        .catch(function (err) { hideLoading(); showError('预处理失败: ' + err.message); });
}

/**
 * 执行三维重建
 */
function reconstruct() {
    if (!sessionId) return;

    var method = document.getElementById('reconMethod').value;
    var params = {};

    if (method === 'poisson') {
        params.depth = parseInt(document.getElementById('poissonDepth').value);
    } else if (method === 'alpha') {
        params.alpha = parseFloat(document.getElementById('alphaValue').value);
    }

    var smooth = document.getElementById('chkSmooth').checked;
    var smoothIter = parseInt(document.getElementById('smoothIter').value);

    showLoading('三维重建中，请耐心等待...');
    fetch('/api/reconstruct', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            session_id: sessionId,
            method: method,
            params: params,
            smooth: smooth,
            smooth_iterations: smoothIter
        })
    })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            hideLoading();
            if (data.error) { showError(data.error); return; }

            currentMesh = data.mesh;
            var info = data.mesh_info;

            document.getElementById('reconInfo').style.display = 'block';
            document.getElementById('reconInfo').innerHTML =
                '顶点数: ' + info.num_vertices + '<br>' +
                '面片数: ' + info.num_triangles + '<br>' +
                '水密: ' + (info.is_watertight ? '是' : '否');

            renderMesh(currentMesh);
            fitCamera();

            document.getElementById('btnExport').disabled = false;

            document.querySelectorAll('.tab').forEach(function (t) { t.classList.remove('active'); });
            document.querySelector('.tab[data-view="mesh"]').classList.add('active');
            activeView = 'mesh';
            updateVisibility();

            setStatus('重建完成: ' + info.num_vertices + ' 顶点, ' + info.num_triangles + ' 面片');
        })
        .catch(function (err) { hideLoading(); showError('重建失败: ' + err.message); });
}

/**
 * 导出模型文件
 */
function exportModel() {
    if (!sessionId) return;

    var format = document.getElementById('exportFormat').value;

    showLoading('导出模型...');
    fetch('/api/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, format: format })
    })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            hideLoading();
            if (data.error) { showError(data.error); return; }

            var link = document.getElementById('downloadLink');
            link.href = data.download_url;
            link.style.display = 'block';
            link.textContent = '下载 ' + format.toUpperCase() + ' 文件';
            setStatus('模型已导出: ' + data.filename);
        })
        .catch(function (err) { hideLoading(); showError('导出失败: ' + err.message); });
}

/**
 * 使用 Three.js 渲染点云数据
 */
function renderPointCloud(pcData) {
    if (pointCloudObj) scene.remove(pointCloudObj);

    var geometry = new THREE.BufferGeometry();
    var n = pcData.num_points;

    // 支持两种格式：扁平数组（新）和嵌套数组（旧）
    var positions;
    if (pcData.positions) {
        // 新格式：扁平数组 [x1,y1,z1,x2,y2,z2,...]，直接构造 Float32Array
        positions = new Float32Array(pcData.positions);
    } else if (pcData.points) {
        // 旧格式：嵌套数组 [[x,y,z],...]，逐点转换
        positions = new Float32Array(n * 3);
        var pts = pcData.points;
        for (var i = 0; i < n; i++) {
            var j = i * 3;
            positions[j] = pts[i][0];
            positions[j + 1] = pts[i][1];
            positions[j + 2] = pts[i][2];
        }
    } else {
        return;
    }
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    if (pcData.colors) {
        var colors;
        if (typeof pcData.colors[0] === 'number') {
            // 新格式：扁平数组 [r1,g1,b1,r2,g2,b2,...]
            colors = new Float32Array(pcData.colors);
        } else {
            // 旧格式：嵌套数组
            colors = new Float32Array(n * 3);
            for (var i = 0; i < n; i++) {
                var j = i * 3;
                colors[j] = pcData.colors[i][0];
                colors[j + 1] = pcData.colors[i][1];
                colors[j + 2] = pcData.colors[i][2];
            }
        }
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    }

    // 计算包围盒供后续相机定位使用
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();

    var material = new THREE.PointsMaterial({
        size: 0.02,
        vertexColors: pcData.colors && pcData.colors.length > 0,
        color: pcData.colors && pcData.colors.length > 0 ? 0xffffff : 0x58a6ff,
    });

    pointCloudObj = new THREE.Points(geometry, material);
    scene.add(pointCloudObj);
    pointCloudObj.visible = (activeView === 'pointcloud');

    // 保存包围盒数据，fitCamera 优先使用
    pointCloudObj._bboxMin = bboxMin;
    pointCloudObj._bboxMax = bboxMax;
}

/**
 * 使用 Three.js 渲染三角网格模型
 */
function renderMesh(meshData) {
    if (meshObj) scene.remove(meshObj);

    var geometry = new THREE.BufferGeometry();
    var vertices = meshData.vertices;
    var triangles = meshData.triangles;

    var positions = new Float32Array(triangles.length * 9);
    var idx = 0;
    for (var i = 0; i < triangles.length; i++) {
        var tri = triangles[i];
        for (var j = 0; j < 3; j++) {
            positions[idx++] = vertices[tri[j]][0];
            positions[idx++] = vertices[tri[j]][1];
            positions[idx++] = vertices[tri[j]][2];
        }
    }
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.computeVertexNormals();

    var material = new THREE.MeshPhongMaterial({
        color: 0x58a6ff,
        shininess: 50,
        side: THREE.DoubleSide,
        wireframe: false,
    });

    meshObj = new THREE.Mesh(geometry, material);
    scene.add(meshObj);
    meshObj.visible = (activeView === 'mesh');

    var wireframe = new THREE.WireframeGeometry(geometry);
    var wireMat = new THREE.LineBasicMaterial({ color: 0x30363d, transparent: true, opacity: 0.3 });
    var wireObj = new THREE.LineSegments(wireframe, wireMat);
    meshObj.add(wireObj);
}

/**
 * 自动调整相机位置以适配场景内容
 * 优先使用点云对象预计算的包围盒，确保大型点云也能正确定位相机
 */
function fitCamera() {
    var bboxMin = [Infinity, Infinity, Infinity];
    var bboxMax = [-Infinity, -Infinity, -Infinity];
    var hasData = false;

    // 优先从点云对象获取预计算的包围盒数据
    if (pointCloudObj && pointCloudObj._bboxMin) {
        bboxMin = pointCloudObj._bboxMin;
        bboxMax = pointCloudObj._bboxMax;
        hasData = true;
    }
    if (meshObj) {
        if (meshObj.geometry) meshObj.geometry.computeBoundingBox();
        var meshBox = meshObj.geometry ? meshObj.geometry.boundingBox : null;
        if (meshBox) {
            bboxMin[0] = Math.min(bboxMin[0], meshBox.min.x);
            bboxMin[1] = Math.min(bboxMin[1], meshBox.min.y);
            bboxMin[2] = Math.min(bboxMin[2], meshBox.min.z);
            bboxMax[0] = Math.max(bboxMax[0], meshBox.max.x);
            bboxMax[1] = Math.max(bboxMax[1], meshBox.max.y);
            bboxMax[2] = Math.max(bboxMax[2], meshBox.max.z);
            hasData = true;
        }
    }
    // 也考虑空间模型的包围盒
    if (frameObj && frameObj._bboxMin) {
        bboxMin[0] = Math.min(bboxMin[0], frameObj._bboxMin[0]);
        bboxMin[1] = Math.min(bboxMin[1], frameObj._bboxMin[1]);
        bboxMin[2] = Math.min(bboxMin[2], frameObj._bboxMin[2]);
        bboxMax[0] = Math.max(bboxMax[0], frameObj._bboxMax[0]);
        bboxMax[1] = Math.max(bboxMax[1], frameObj._bboxMax[1]);
        bboxMax[2] = Math.max(bboxMax[2], frameObj._bboxMax[2]);
        hasData = true;
    }

    // 如果没有点云也没有网格，回退到场景遍历
    if (!hasData) {
        var box = new THREE.Box3();
        scene.traverse(function (obj) {
            if (obj.geometry) {
                if (obj.geometry.boundingBox === null) {
                    obj.geometry.computeBoundingBox();
                }
                box.expandByObject(obj);
            }
        });
        if (box.isEmpty()) return;
        bboxMin = [box.min.x, box.min.y, box.min.z];
        bboxMax = [box.max.x, box.max.y, box.max.z];
    }

    // 检查包围盒是否有效
    if (!isFinite(bboxMin[0])) return;

    var center = new THREE.Vector3(
        (bboxMin[0] + bboxMax[0]) / 2,
        (bboxMin[1] + bboxMax[1]) / 2,
        (bboxMin[2] + bboxMax[2]) / 2
    );
    var sizeX = bboxMax[0] - bboxMin[0];
    var sizeY = bboxMax[1] - bboxMin[1];
    var sizeZ = bboxMax[2] - bboxMin[2];
    var maxDim = Math.max(sizeX, sizeY, sizeZ, 0.01);

    // 点云是扁平的（Z范围很小）时，增大Z方向距离以免看到一条线
    if (sizeZ < maxDim * 0.05) {
        sizeZ = maxDim * 0.3;
    }

    var distance = maxDim * 1.8;

    controls.target.copy(center);
    camera.position.set(
        center.x + distance * 0.6,
        center.y + distance * 0.6,
        center.z + distance
    );
    camera.lookAt(center);
    controls.update();

    // 根据数据尺度自动调整点大小
    var pointSize = maxDim * 0.002;
    if (pointCloudObj && pointCloudObj.material) {
        pointCloudObj.material.size = Math.max(pointSize, 0.005);
    }
}

/**
 * 切换雷达捕获状态（开始/停止）
 */
function toggleCapture() {
    if (lidarCapturing) {
        stopLidarCapture();
    } else {
        startLidarCapture();
    }
}

/**
 * 验证雷达配置参数
 */
function validateLidarConfig() {
    var ip = document.getElementById('lidarIp').value.trim();
    var dataPort = document.getElementById('lidarDataPort').value;
    var devPort = document.getElementById('lidarDevPort').value;
    var valid = true;

    var ipv4Regex = /^(\d{1,3}\.){3}\d{1,3}$/;
    var ipError = document.getElementById('ipError');
    if (!ip) {
        ipError.textContent = '请输入IP地址';
        valid = false;
    } else if (!ipv4Regex.test(ip)) {
        ipError.textContent = 'IP格式不正确';
        valid = false;
    } else {
        var parts = ip.split('.');
        for (var i = 0; i < parts.length; i++) {
            if (parseInt(parts[i]) > 255) {
                ipError.textContent = '每段不能超过255';
                valid = false;
                break;
            }
        }
        if (valid) ipError.textContent = '';
    }

    var dpError = document.getElementById('dataPortError');
    var dp = parseInt(dataPort);
    if (isNaN(dp) || dp < 0 || dp > 65535) {
        dpError.textContent = '端口范围: 0-65535';
        valid = false;
    } else {
        dpError.textContent = '';
    }

    var vpError = document.getElementById('devPortError');
    var vp = parseInt(devPort);
    if (isNaN(vp) || vp < 0 || vp > 65535) {
        vpError.textContent = '端口范围: 0-65535';
        valid = false;
    } else {
        vpError.textContent = '';
    }

    return valid;
}

/**
 * 连接雷达并开始捕获数据
 */
function startLidarCapture() {
    if (!validateLidarConfig()) return;

    var ip = document.getElementById('lidarIp').value.trim();
    var dataPort = parseInt(document.getElementById('lidarDataPort').value);
    var devPort = parseInt(document.getElementById('lidarDevPort').value);

    var btn = document.getElementById('btnCapture');
    btn.disabled = true;
    btn.textContent = '连接中...';

    fetch('/api/lidar/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ip: ip, data_port: dataPort, device_port: devPort })
    })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (data.error) {
                btn.disabled = false;
                btn.textContent = '捕获';
                showLidarStatus('error', data.error);
                return;
            }

            lidarSessionId = data.session_id;

            fetch('/api/lidar/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: lidarSessionId })
            })
                .then(function (r) { return r.json(); })
                .then(function (startData) {
                    btn.disabled = false;
                    if (startData.error) {
                        btn.textContent = '捕获';
                        showLidarStatus('error', startData.error);
                        return;
                    }

                    lidarCapturing = true;
                    btn.textContent = '停止捕获';
                    btn.classList.add('capturing');

                    document.getElementById('btnUseData').style.display = 'block';
                    document.getElementById('btnUseData').disabled = false;
                    document.getElementById('btnExportPC').style.display = 'block';
                    document.getElementById('btnExportPC').disabled = false;

                    document.querySelectorAll('.tab').forEach(function (t) { t.classList.remove('active'); });
                    document.querySelector('.tab[data-view="lidar"]').classList.add('active');
                    activeView = 'lidar';
                    updateVisibility();

                    showLidarStatus('success', '雷达已连接 ' + ip + ':' + dataPort + '，正在捕获数据...');
                    startLidarPolling();
                    setStatus('雷达数据捕获中...');
                });
        })
        .catch(function (err) {
            btn.disabled = false;
            btn.textContent = '捕获';
            showLidarStatus('error', '连接失败: ' + err.message);
        });
}

/**
 * 停止雷达数据捕获
 */
function stopLidarCapture() {
    if (lidarPollingTimer) {
        clearInterval(lidarPollingTimer);
        lidarPollingTimer = null;
    }

    var btn = document.getElementById('btnCapture');
    btn.disabled = true;
    btn.textContent = '停止中...';

    fetch('/api/lidar/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: lidarSessionId })
    })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            lidarCapturing = false;
            btn.disabled = false;
            btn.textContent = '捕获';
            btn.classList.remove('capturing');
            showLidarStatus('info', '数据捕获已停止');
            setStatus('雷达捕获已停止');
        })
        .catch(function (err) {
            lidarCapturing = false;
            btn.disabled = false;
            btn.textContent = '捕获';
            btn.classList.remove('capturing');
        });
}

/**
 * 启动雷达数据轮询（每200ms获取一帧）
 */
function startLidarPolling() {
    if (lidarPollingTimer) clearInterval(lidarPollingTimer);

    lidarPollingTimer = setInterval(function () {
        if (!lidarCapturing || !lidarSessionId) return;

        fetch('/api/lidar/data', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: lidarSessionId })
        })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.point_cloud && data.point_cloud.num_points > 0) {
                    renderLidarPoints(data.point_cloud);
                    updateLidarStatusBar(data.status);
                }
            })
            .catch(function () {});
    }, 300);
}

/**
 * 使用 Three.js 渲染雷达实时点云数据（优化：使用扁平数组直接填充Buffer）
 * 坐标系已在后端转换：X/Y水平，Z垂直（朝上）
 */
function renderLidarPoints(pcData) {
    if (!pcData || pcData.num_points === 0) return;

    var geometry;
    if (lidarPointObj) {
        geometry = lidarPointObj.geometry;
    } else {
        geometry = new THREE.BufferGeometry();
        var material = new THREE.PointsMaterial({
            size: 0.02,
            vertexColors: true,
            sizeAttenuation: true,
        });
        lidarPointObj = new THREE.Points(geometry, material);
        scene.add(lidarPointObj);
        lidarPointObj.visible = (activeView === 'lidar');
    }

    var positions = pcData.positions;
    var colors = pcData.colors;

    geometry.setAttribute('position',
        new THREE.BufferAttribute(new Float32Array(positions), 3));

    if (colors && colors.length > 0) {
        geometry.setAttribute('color',
            new THREE.BufferAttribute(new Float32Array(colors), 3));
    }

    geometry.attributes.position.needsUpdate = true;
    if (geometry.attributes.color) geometry.attributes.color.needsUpdate = true;
}

/**
 * 更新雷达状态信息显示
 */
function updateLidarStatusBar(status) {
    if (!status) return;
    var el = document.getElementById('lidarStatus');
    if (el.style.display === 'none') return;
    el.innerHTML =
        '<span class="status-line">帧率: ' + status.fps + ' fps</span>' +
        '<span class="status-line">点数: ' + status.num_points.toLocaleString() + '</span>' +
        '<span class="status-line">数据包: ' + status.packets_received.toLocaleString() + '</span>';
}

/**
 * 显示雷达状态信息
 */
function showLidarStatus(type, message) {
    var el = document.getElementById('lidarStatus');
    el.style.display = 'block';
    el.className = 'info-box';
    if (type === 'error') el.classList.add('error-box');
    else if (type === 'success') el.classList.add('success-box');
    el.textContent = message;
}

/**
 * 将雷达当前帧数据导入为文件处理会话
 */
function useLidarData() {
    if (!lidarSessionId) return;

    showLoading('导入雷达数据...');
    fetch('/api/lidar/to_session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lidar_session_id: lidarSessionId })
    })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            hideLoading();
            if (data.error) { showError(data.error); return; }

            sessionId = data.session_id;
            currentPointCloud = data.point_cloud;

            var info = data.info;
            document.getElementById('fileInfo').style.display = 'block';
            document.getElementById('fileInfo').innerHTML =
                '来源: 雷达实时捕获<br>' +
                '点数: ' + info.num_points + '<br>' +
                '中心: [' + info.center.map(function (v) { return v.toFixed(3); }).join(', ') + ']';

            renderPointCloud(currentPointCloud);
            fitCamera();

            document.getElementById('btnPreprocess').disabled = false;
            document.getElementById('btnSpatialModel').disabled = false;
            // 清除旧的空间模型
            if (frameObj) { scene.remove(frameObj); frameObj = null; }
            document.getElementById('spatialModelInfo').style.display = 'none';

            document.querySelectorAll('.tab').forEach(function (t) { t.classList.remove('active'); });
            document.querySelector('.tab[data-view="pointcloud"]').classList.add('active');
            activeView = 'pointcloud';
            updateVisibility();

            setStatus('雷达数据已导入，可进行预处理和重建');
        })
        .catch(function (err) { hideLoading(); showError('导入失败: ' + err.message); });
}

/**
 * 执行空间模型处理：从点云数据提取房间外框架房型图
 *
 * 新算法流程:
 * 1. 距离过滤 → 丢弃室内杂物点
 * 2. 2D占据栅格 + 形态学闭运算 → 填补墙体扫描缝隙
 * 3. RANSAC迭代直线检测 → 自动识别所有墙面
 * 4. 墙面交点计算 → 得到房间角点
 * 5. Douglas-Peucker多边形简化 → 生成平整的房型轮廓
 * 6. 高度分层拉伸 → 将2D轮廓在Z轴拉伸为3D框架
 */
function processSpatialModel() {
    if (!sessionId) return;

    var minDistance = parseFloat(document.getElementById('spatialMinDist').value);
    var gridSize = parseFloat(document.getElementById('spatialGridSize').value);
    var ransacTh = parseFloat(document.getElementById('spatialRansacTh').value);
    var numLayers = parseInt(document.getElementById('spatialLayers').value);

    showLoading('空间模型处理中，正在检测墙面...');
    fetch('/api/spatial_model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            session_id: sessionId,
            min_distance: minDistance,
            grid_size: gridSize,
            ransac_threshold: ransacTh,
            num_layers: numLayers,
        })
    })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            hideLoading();
            if (data.error) { showError(data.error); return; }

            var info = data.info;
            document.getElementById('spatialModelInfo').style.display = 'block';
            document.getElementById('spatialModelInfo').innerHTML =
                '检测墙面: ' + info.num_wall_lines + ' 条<br>' +
                '房间角点: ' + info.num_corners + ' 个<br>' +
                '模型顶点: ' + info.num_vertices.toLocaleString() + '<br>' +
                '框架边: ' + info.num_edges.toLocaleString() + '<br>' +
                '墙面片: ' + info.num_faces.toLocaleString() + '<br>' +
                '高度范围: ' + info.z_min.toFixed(2) + 'm ~ ' + info.z_max.toFixed(2) + 'm';

            renderFrameModel(data);
            fitCamera();

            // 切换到空间模型视图
            document.querySelectorAll('.tab').forEach(function (t) { t.classList.remove('active'); });
            document.querySelector('.tab[data-view="frame"]').classList.add('active');
            activeView = 'frame';
            updateVisibility();

            setStatus('空间模型完成: ' + info.num_vertices.toLocaleString() + ' 顶点, ' + info.num_faces.toLocaleString() + ' 面片');
        })
        .catch(function (err) { hideLoading(); showError('空间模型处理失败: ' + err.message); });
}

/**
 * 使用 Three.js 渲染空间模型（房间外框架）
 *
 * 渲染两层:
 * - 半透明三角面片（显示墙面结构）
 * - 白色线框边（突出框架轮廓）
 */
function renderFrameModel(data) {
    if (frameObj) scene.remove(frameObj);

    var verts = data.vertices;
    var faces = data.faces;
    var edges = data.edges;
    var group = new THREE.Group();

    // 计算包围盒用于后续 camera 适配
    var bboxMin = [Infinity, Infinity, Infinity];
    var bboxMax = [-Infinity, -Infinity, -Infinity];

    // ── 渲染三角面片（半透明墙面） ──
    if (faces && faces.length > 0) {
        var faceGeo = new THREE.BufferGeometry();
        var facePositions = new Float32Array(faces.length * 9);
        for (var i = 0; i < faces.length; i++) {
            var f = faces[i];
            for (var j = 0; j < 3; j++) {
                var v = verts[f[j]];
                facePositions[i * 9 + j * 3] = v[0];
                facePositions[i * 9 + j * 3 + 1] = v[1];
                facePositions[i * 9 + j * 3 + 2] = v[2];

                // 同时计算包围盒
                if (v[0] < bboxMin[0]) bboxMin[0] = v[0];
                if (v[1] < bboxMin[1]) bboxMin[1] = v[1];
                if (v[2] < bboxMin[2]) bboxMin[2] = v[2];
                if (v[0] > bboxMax[0]) bboxMax[0] = v[0];
                if (v[1] > bboxMax[1]) bboxMax[1] = v[1];
                if (v[2] > bboxMax[2]) bboxMax[2] = v[2];
            }
        }
        faceGeo.setAttribute('position', new THREE.BufferAttribute(facePositions, 3));
        faceGeo.computeVertexNormals();

        var faceMat = new THREE.MeshPhongMaterial({
            color: 0x3fb950,
            transparent: true,
            opacity: 0.35,
            side: THREE.DoubleSide,
            depthWrite: false,
        });
        var faceMesh = new THREE.Mesh(faceGeo, faceMat);
        group.add(faceMesh);
    }

    // ── 渲染线框边（白色框架线） ──
    if (edges && edges.length > 0) {
        var edgeGeo = new THREE.BufferGeometry();
        var edgePositions = new Float32Array(edges.length * 6);
        for (var k = 0; k < edges.length; k++) {
            var e = edges[k];
            var v0 = verts[e[0]];
            var v1 = verts[e[1]];
            edgePositions[k * 6] = v0[0];
            edgePositions[k * 6 + 1] = v0[1];
            edgePositions[k * 6 + 2] = v0[2];
            edgePositions[k * 6 + 3] = v1[0];
            edgePositions[k * 6 + 4] = v1[1];
            edgePositions[k * 6 + 5] = v1[2];
        }
        edgeGeo.setAttribute('position', new THREE.BufferAttribute(edgePositions, 3));

        var edgeMat = new THREE.LineBasicMaterial({
            color: 0xe6edf3,
        });
        var edgeLines = new THREE.LineSegments(edgeGeo, edgeMat);
        group.add(edgeLines);
    }

    frameObj = group;
    scene.add(frameObj);
    frameObj.visible = (activeView === 'frame');

    // 保存包围盒数据供 fitCamera 使用
    frameObj._bboxMin = bboxMin;
    frameObj._bboxMax = bboxMax;
}

function showLoading(text) {
    document.getElementById('loadingText').textContent = text;
    document.getElementById('loadingOverlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

function showError(msg) {
    setStatus('错误: ' + msg);
    alert(msg);
}

function setStatus(text) {
    document.getElementById('statusBar').textContent = text;
}

/**
 * 页面加载时检查是否存在活跃的雷达连接，用于刷新后恢复状态
 */
function checkExistingConnection() {
    // 仅对C16检查现有连接
    if (radarType !== 'c16') return;

    var ip = document.getElementById('lidarIp').value.trim();
    var dataPort = parseInt(document.getElementById('lidarDataPort').value);

    if (!ip || isNaN(dataPort)) return;

    fetch('/api/lidar/check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ip: ip, data_port: dataPort })
    })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (data.exists) {
                lidarSessionId = data.session_id;
                if (data.capturing) {
                    // 恢复捕获状态
                    lidarCapturing = true;
                    var btn = document.getElementById('btnCapture');
                    btn.textContent = '停止捕获';
                    btn.classList.add('capturing');
                    btn.disabled = false;

                    document.getElementById('btnUseData').style.display = 'block';
                    document.getElementById('btnUseData').disabled = false;
                    document.getElementById('btnExportPC').style.display = 'block';
                    document.getElementById('btnExportPC').disabled = false;

                    showLidarStatus('success', '已恢复雷达连接 ' + ip + ':' + dataPort + '，正在捕获...');
                    startLidarPolling();
                } else {
                    showLidarStatus('info', '检测到现有连接，但未在捕获状态。点击"捕获"继续。');
                }
            }
        })
        .catch(function () {});
}

/**
 * 导出雷达实时点云数据为文件
 */
function exportLidarPointcloud() {
    if (!lidarSessionId) return;

    var format = prompt('请输入导出格式 (ply / pcd / xyz):', 'ply');
    if (!format) return;
    format = format.trim().toLowerCase();
    if (['ply', 'pcd', 'xyz'].indexOf(format) === -1) {
        alert('不支持的格式，请选择: ply, pcd, xyz');
        return;
    }

    showLoading('导出现场点云...');
    fetch('/api/lidar/export_pointcloud', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lidar_session_id: lidarSessionId, format: format })
    })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            hideLoading();
            if (data.error) { showError(data.error); return; }

            // 触发浏览器下载
            var link = document.createElement('a');
            link.href = data.download_url;
            link.download = data.filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            setStatus('点云已导出: ' + data.filename + ' (' + data.num_points.toLocaleString() + ' 点)');
        })
        .catch(function (err) { hideLoading(); showError('导出失败: ' + err.message); });
}

// ═══════════════════════════════════════════════════════════
// MS200P 串口雷达前端逻辑
// ═══════════════════════════════════════════════════════════

/**
 * 切换雷达型号，显示对应的连接面板
 */
function switchRadarType() {
    radarType = document.getElementById('radarType').value;

    // 清除雷达实时视图中的旧数据
    if (lidarPointObj) {
        scene.remove(lidarPointObj);
        lidarPointObj = null;
    }

    if (radarType === 'c16') {
        document.getElementById('c16Panel').style.display = 'block';
        document.getElementById('ms200pPanel').style.display = 'none';
        // 停止MS200P
        if (ms200pCapturing) stopMS200PCapture();
    } else {
        document.getElementById('c16Panel').style.display = 'none';
        document.getElementById('ms200pPanel').style.display = 'block';
        // 停止C16
        if (lidarCapturing) stopLidarCapture();
        // 刷新串口列表
        refreshSerialPorts();
    }
}

/**
 * 刷新系统串口列表
 */
function refreshSerialPorts() {
    fetch('/api/serial/ports')
        .then(function (r) { return r.json(); })
        .then(function (data) {
            var sel = document.getElementById('ms200pPort');
            sel.innerHTML = '';
            if (data.ports && data.ports.length > 0) {
                data.ports.forEach(function (p) {
                    var opt = document.createElement('option');
                    opt.value = p.port;
                    opt.textContent = p.port + ' - ' + (p.description || '');
                    sel.appendChild(opt);
                });
            } else {
                sel.innerHTML = '<option value="">未检测到串口设备</option>';
            }
        })
        .catch(function () {
            document.getElementById('ms200pPort').innerHTML =
                '<option value="">获取失败，请重试</option>';
        });
}

/**
 * 切换MS200P捕获状态
 */
function toggleMS200PCapture() {
    if (ms200pCapturing) {
        stopMS200PCapture();
    } else {
        startMS200PCapture();
    }
}

/**
 * 连接MS200P串口并开始累积数据捕获
 */
function startMS200PCapture() {
    var port = document.getElementById('ms200pPort').value;
    if (!port) {
        showMS200PStatus('error', '请选择串口设备');
        return;
    }

    var baud = parseInt(document.getElementById('ms200pBaud').value);
    var interval = parseFloat(document.getElementById('ms200pInterval').value);

    // 读取最小距离过滤参数
    var minDist = 0.0;
    var chkMinDist = document.getElementById('chkMS200PMinDist');
    if (chkMinDist.checked) {
        minDist = parseFloat(document.getElementById('ms200pMinDist').value) || 0.0;
    }

    var btn = document.getElementById('btnMS200PCapture');
    btn.disabled = true;
    btn.textContent = '连接中...';

    fetch('/api/ms200p/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ port: port, baud_rate: baud, capture_interval: interval, min_distance: minDist })
    })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (data.error) {
                btn.disabled = false;
                btn.textContent = '捕获';
                showMS200PStatus('error', data.error);
                return;
            }

            ms200pSessionId = data.session_id;

            fetch('/api/ms200p/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: ms200pSessionId, capture_interval: interval, min_distance: minDist })
            })
                .then(function (r) { return r.json(); })
                .then(function (startData) {
                    btn.disabled = false;
                    if (startData.error) {
                        btn.textContent = '捕获';
                        showMS200PStatus('error', startData.error);
                        return;
                    }

                    ms200pCapturing = true;
                    btn.textContent = '停止捕获';
                    btn.classList.add('capturing');

                    document.getElementById('btnMS200PUseData').style.display = 'block';
                    document.getElementById('btnMS200PUseData').disabled = false;
                    document.getElementById('btnMS200PExport').style.display = 'block';
                    document.getElementById('btnMS200PExport').disabled = false;

                    document.querySelectorAll('.tab').forEach(function (t) { t.classList.remove('active'); });
                    document.querySelector('.tab[data-view="lidar"]').classList.add('active');
                    activeView = 'lidar';
                    updateVisibility();

                    showMS200PStatus('success', 'MS200P已连接 ' + port + '，正在累积数据...');
                    startMS200PPolling();
                    setStatus('MS200P数据采集中（累积模式）...');
                });
        })
        .catch(function (err) {
            btn.disabled = false;
            btn.textContent = '捕获';
            showMS200PStatus('error', '连接失败: ' + err.message);
        });
}

/**
 * 停止MS200P数据捕获
 */
function stopMS200PCapture() {
    if (ms200pPollingTimer) {
        clearInterval(ms200pPollingTimer);
        ms200pPollingTimer = null;
    }

    var btn = document.getElementById('btnMS200PCapture');
    btn.disabled = true;
    btn.textContent = '停止中...';

    fetch('/api/ms200p/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: ms200pSessionId })
    })
        .then(function (r) { return r.json(); })
        .then(function () {
            ms200pCapturing = false;
            btn.disabled = false;
            btn.textContent = '捕获';
            btn.classList.remove('capturing');
            showMS200PStatus('info', '数据捕获已停止，累积点云已保留');
            setStatus('MS200P捕获已停止');
        })
        .catch(function () {
            ms200pCapturing = false;
            btn.disabled = false;
            btn.textContent = '捕获';
            btn.classList.remove('capturing');
        });
}

/**
 * 启动MS200P数据轮询（每300ms获取累积点云）
 */
function startMS200PPolling() {
    if (ms200pPollingTimer) clearInterval(ms200pPollingTimer);

    // 使用采集间隔控制前端轮询频率（转为毫秒）
    var interval = parseFloat(document.getElementById('ms200pInterval').value);
    var pollMs = Math.max(100, Math.round(interval * 1000));  // 最少 100ms

    ms200pPollingTimer = setInterval(function () {
        if (!ms200pCapturing || !ms200pSessionId) return;

        fetch('/api/ms200p/data', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: ms200pSessionId })
        })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.point_cloud && data.point_cloud.num_points > 0) {
                    renderLidarPoints(data.point_cloud);
                    updateMS200PStatusBar(data.status);
                }
            })
            .catch(function () {});
    }, pollMs);
}

/**
 * 显示MS200P状态信息
 */
function showMS200PStatus(type, message) {
    var el = document.getElementById('ms200pStatus');
    el.style.display = 'block';
    el.className = 'info-box';
    if (type === 'error') el.classList.add('error-box');
    else if (type === 'success') el.classList.add('success-box');
    el.textContent = message;
}

/**
 * 更新MS200P状态栏信息
 * 显示电机旋转进度、角度、耗时、点云统计等
 */
function updateMS200PStatusBar(status) {
    if (!status) return;
    var el = document.getElementById('ms200pStatus');
    if (el.style.display === 'none') return;

    // 计算进度百分比
    var progressPercent = 0;
    if (status.total_time > 0) {
        progressPercent = Math.min(100, (status.elapsed / status.total_time) * 100);
    }

    var html = '';

    // 进度条：可视化电机旋转进度
    html += '<div style="margin-bottom:6px;background:#21262d;border-radius:3px;height:6px;overflow:hidden;">';
    html += '<div style="width:' + progressPercent.toFixed(0) + '%;height:100%;background:#238636;transition:width 0.3s;"></div>';
    html += '</div>';

    // 关键状态行：电机角度 + 耗时
    var totalAngle = (status.total_time * 2.25).toFixed(0);
    html += '<span class="status-line">电机角度: ' + status.azimuth + '° / ' + totalAngle + '°</span>';
    html += '<span class="status-line">耗时: ' + status.elapsed + 's / ' + status.total_time + 's</span>';

    // 详细数据行
    html += '<div style="font-size:11px;color:#8b949e;margin-top:4px;">';
    html += '累积点数: ' + status.num_points.toLocaleString() + ' | ';
    html += '数据包: ' + status.packets_received.toLocaleString() + ' | ';
    html += 'FPS: ' + status.fps + ' | ';
    html += '转速: ' + status.rotation_speed + '°/s';
    html += '</div>';

    // 诊断统计行：显示数据包被丢弃的原因
    var hasDiag = status.diag_crc_fail > 0 || status.diag_n_mismatch > 0 || status.diag_angle_filtered > 0;
    if (hasDiag) {
        html += '<div style="font-size:10px;color:#f85149;margin-top:3px;border-top:1px solid #21262d;padding-top:3px;">';
        html += '诊断: 成功=' + (status.diag_parsed_ok || 0).toLocaleString();
        if (status.diag_crc_fail > 0) html += ' | CRC失败=' + status.diag_crc_fail.toLocaleString() + ' (' + (status.diag_crc_fail_rate || 0) + '%)';
        if (status.diag_n_mismatch > 0) html += ' | N不匹配=' + status.diag_n_mismatch.toLocaleString();
        if (status.diag_angle_filtered > 0) html += ' | 角度过滤=' + status.diag_angle_filtered.toLocaleString();
        html += '</div>';
    }

    el.innerHTML = html;
}

/**
 * 将MS200P累积点云导入为文件处理会话
 */
function useMS200PData() {
    if (!ms200pSessionId) return;

    showLoading('导入MS200P累积点云...');
    fetch('/api/ms200p/to_session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ms200p_session_id: ms200pSessionId })
    })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            hideLoading();
            if (data.error) { showError(data.error); return; }

            sessionId = data.session_id;
            currentPointCloud = data.point_cloud;

            var info = data.info;
            document.getElementById('fileInfo').style.display = 'block';
            document.getElementById('fileInfo').innerHTML =
                '来源: MS200P<br>' +
                '点数: ' + info.num_points.toLocaleString() + '<br>' +
                '中心: [' + info.center.map(function (v) { return v.toFixed(3); }).join(', ') + ']';

            renderPointCloud(currentPointCloud);
            fitCamera();

            document.getElementById('btnPreprocess').disabled = false;
            document.getElementById('btnSpatialModel').disabled = false;
            if (frameObj) { scene.remove(frameObj); frameObj = null; }
            document.getElementById('spatialModelInfo').style.display = 'none';

            document.querySelectorAll('.tab').forEach(function (t) { t.classList.remove('active'); });
            document.querySelector('.tab[data-view="pointcloud"]').classList.add('active');
            activeView = 'pointcloud';
            updateVisibility();

            setStatus('MS200P累积数据已导入: ' + info.num_points.toLocaleString() + ' 点');
        })
        .catch(function (err) { hideLoading(); showError('导入失败: ' + err.message); });
}

/**
 * 导出MS200P累积点云数据为文件
 */
function exportMS200PPointcloud() {
    if (!ms200pSessionId) return;

    var format = prompt('请输入导出格式 (ply / pcd / xyz):', 'ply');
    if (!format) return;
    format = format.trim().toLowerCase();
    if (['ply', 'pcd', 'xyz'].indexOf(format) === -1) {
        alert('不支持的格式，请选择: ply, pcd, xyz');
        return;
    }

    showLoading('导出MS200P点云...');
    fetch('/api/ms200p/export_pointcloud', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ms200p_session_id: ms200pSessionId, format: format })
    })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            hideLoading();
            if (data.error) { showError(data.error); return; }

            var link = document.createElement('a');
            link.href = data.download_url;
            link.download = data.filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            setStatus('MS200P点云已导出: ' + data.filename);
        })
        .catch(function (err) { hideLoading(); showError('导出失败: ' + err.message); });
}
