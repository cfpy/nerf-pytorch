# 导入操作系统库，用于文件路径处理
import os
# 导入 NumPy，用于数值运算
import numpy as np
# 导入 imageio，用于图像读取
import imageio


# 加载 DeepVoxels 数据集
# scene: 场景名称；basedir: 数据目录；testskip: 测试集采样步长

def load_dv_data(scene='cube', basedir='/data/deepvoxels', testskip=8):


    # 解析相机内参文件的辅助函数
    def parse_intrinsics(filepath, trgt_sidelength, invert_y=False):
        # 获取相机内参参数
        with open(filepath, 'r') as file:
            f, cx, cy = list(map(float, file.readline().split()))[:3]
            grid_barycenter = np.array(list(map(float, file.readline().split())))
            near_plane = float(file.readline())
            scale = float(file.readline())
            height, width = map(float, file.readline().split())

            try:
                world2cam_poses = int(file.readline())
            except ValueError:
                world2cam_poses = None

        # 若最后一行无值则视为 False
        if world2cam_poses is None:
            world2cam_poses = False

        # 转为布尔值便于判断
        world2cam_poses = bool(world2cam_poses)

        # 打印原始内参值
        print(cx,cy,f,height,width)

        # 将主点坐标按目标边长归一化
        cx = cx / width * trgt_sidelength
        cy = cy / height * trgt_sidelength
        # 根据目标尺寸调整焦距
        f = trgt_sidelength / height * f

        # x 方向焦距
        fx = f
        # 根据是否翻转 y 轴确定 fy 正负
        if invert_y:
            fy = -f
        else:
            fy = f

        # 构建完整的 4x4 内参矩阵
        full_intrinsic = np.array([[fx, 0., cx, 0.],
                                   [0., fy, cy, 0],
                                   [0., 0, 1, 0],
                                   [0, 0, 0, 1]])

        # 返回内参、网格质心、尺度、近裁剪面以及是否使用 world2cam
        return full_intrinsic, grid_barycenter, scale, near_plane, world2cam_poses


    # 加载单个 4x4 位姿矩阵的辅助函数
    def load_pose(filename):
        # 确认文件存在
        assert os.path.isfile(filename)
        # 读取所有数字
        nums = open(filename).read().split()
        # 将数字重塑成 4x4 矩阵并转为 float32
        return np.array([float(x) for x in nums]).reshape([4,4]).astype(np.float32)


    # 默认图像高度
    H = 512
    # 默认图像宽度
    W = 512
    # 训练集基础路径
    deepvoxels_base = '{}/train/{}/'.format(basedir, scene)

    # 解析内参文件并获取参数
    full_intrinsic, grid_barycenter, scale, near_plane, world2cam_poses = parse_intrinsics(os.path.join(deepvoxels_base, 'intrinsics.txt'), H)
    # 打印解析后的结果
    print(full_intrinsic, grid_barycenter, scale, near_plane, world2cam_poses)
    # 提取焦距
    focal = full_intrinsic[0,0]
    # 打印高度、宽度和焦距
    print(H, W, focal)


    # 将位姿文件夹解析为位姿数组
    def dir2poses(posedir):
        # 读取目录下所有 txt 文件的位姿并堆叠
        poses = np.stack([load_pose(os.path.join(posedir, f)) for f in sorted(os.listdir(posedir)) if f.endswith('txt')], 0)
        # 定义将坐标系从右手转为左手的转换矩阵
        transf = np.array([
            [1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,1.],
        ])
        # 将转换矩阵应用到所有位姿
        poses = poses @ transf
        # 仅保留前 3x4 的相机外参部分
        poses = poses[:,:3,:4].astype(np.float32)
        # 返回处理后的位姿
        return poses

    # 训练集位姿目录
    posedir = os.path.join(deepvoxels_base, 'pose')
    # 读取训练集位姿
    poses = dir2poses(posedir)
    # 读取测试集位姿并按步长采样
    testposes = dir2poses('{}/test/{}/pose'.format(basedir, scene))
    testposes = testposes[::testskip]
    # 读取验证集位姿并按步长采样
    valposes = dir2poses('{}/validation/{}/pose'.format(basedir, scene))
    valposes = valposes[::testskip]

    # 列出训练集 RGB 图像文件
    imgfiles = [f for f in sorted(os.listdir(os.path.join(deepvoxels_base, 'rgb'))) if f.endswith('png')]
    # 读取并归一化训练集图像
    imgs = np.stack([imageio.imread(os.path.join(deepvoxels_base, 'rgb', f))/255. for f in imgfiles], 0).astype(np.float32)


    # 测试集 RGB 目录
    testimgd = '{}/test/{}/rgb'.format(basedir, scene)
    # 列出测试集图片
    imgfiles = [f for f in sorted(os.listdir(testimgd)) if f.endswith('png')]
    # 读取并按步长采样测试集图片
    testimgs = np.stack([imageio.imread(os.path.join(testimgd, f))/255. for f in imgfiles[::testskip]], 0).astype(np.float32)

    # 验证集 RGB 目录
    valimgd = '{}/validation/{}/rgb'.format(basedir, scene)
    # 列出验证集图片
    imgfiles = [f for f in sorted(os.listdir(valimgd)) if f.endswith('png')]
    # 读取并按步长采样验证集图片
    valimgs = np.stack([imageio.imread(os.path.join(valimgd, f))/255. for f in imgfiles[::testskip]], 0).astype(np.float32)

    # 将训练、验证、测试图像组成列表
    all_imgs = [imgs, valimgs, testimgs]
    # 计算每部分的数量并构建累积计数
    counts = [0] + [x.shape[0] for x in all_imgs]
    counts = np.cumsum(counts)
    # 生成三组数据的索引区间
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    # 合并所有图像
    imgs = np.concatenate(all_imgs, 0)
    # 合并训练、验证、测试位姿
    poses = np.concatenate([poses, valposes, testposes], 0)

    # 将测试集位姿用于渲染
    render_poses = testposes

    # 打印位姿和图像的形状
    print(poses.shape, imgs.shape)

    # 返回图像、位姿、渲染位姿、相机参数以及数据切片
    return imgs, poses, render_poses, [H,W,focal], i_split


