# 导入操作系统接口库，用于路径拼接和文件管理
import os
# 导入 PyTorch 主库，用于张量计算
import torch
# 导入 NumPy，用于数值计算
import numpy as np
# 导入 imageio，用于读取图像数据
import imageio
# 导入 json，用于读取数据集的元数据文件
import json
# 导入 PyTorch 的函数式 API，后续可能用于插值等操作
import torch.nn.functional as F
# 导入 OpenCV，用于图像尺寸变换
import cv2


# 定义一个匿名函数，根据位移 t 生成沿 z 轴平移的 4x4 齐次变换矩阵
trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

# 定义绕 x 轴旋转 phi 角度的 4x4 旋转矩阵
rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

# 定义绕 y 轴旋转 theta 角度的 4x4 旋转矩阵
rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


# 根据球坐标参数生成一个位姿矩阵
# theta: 绕 y 轴角度，phi: 绕 x 轴角度，radius: 距离原点的半径
# 返回一个 c2w（camera-to-world）变换矩阵

def pose_spherical(theta, phi, radius):
    # 先做平移，将相机放在半径对应的距离
    c2w = trans_t(radius)
    # 绕 x 轴旋转 phi，将相机抬起或压低
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    # 再绕 y 轴旋转 theta，将相机绕目标旋转
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    # 将坐标轴从 OpenGL 风格转换到 NeRF 期望的坐标系
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    # 返回最终的位姿矩阵
    return c2w


# 加载 LINEMOD 数据集并返回图像、位姿等信息
# basedir: 数据集根目录；half_res: 是否缩小分辨率；testskip: 测试集采样间隔

def load_LINEMOD_data(basedir, half_res=False, testskip=1):
    # 定义数据划分
    splits = ['train', 'val', 'test']
    # 用于存放每个划分的元信息
    metas = {}
    # 读取每个划分的 transforms_*.json 元数据
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    # 存放全部图像
    all_imgs = []
    # 存放全部相机位姿
    all_poses = []
    # 记录每个分割之前的样本累积数量
    counts = [0]
    # 遍历 train/val/test 三个集合
    for s in splits:
        # 当前集合的元信息
        meta = metas[s]
        # 当前集合的图像列表
        imgs = []
        # 当前集合的位姿列表
        poses = []
        # 根据是否训练集决定采样步长
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip

        # 遍历按步长采样后的帧
        for idx_test, frame in enumerate(meta['frames'][::skip]):
            # 每个帧的文件路径
            fname = frame['file_path']
            # 如果是测试集，打印当前帧索引和路径
            if s == 'test':
                print(f"{idx_test}th test frame: {fname}")
            # 读取图像并追加
            imgs.append(imageio.imread(fname))
            # 追加对应的变换矩阵
            poses.append(np.array(frame['transform_matrix']))
        # 归一化图像到 0~1，并保留 RGBA 四个通道
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        # 将位姿转为 float32 数组
        poses = np.array(poses).astype(np.float32)
        # 更新计数列表，记录到当前集合末尾的累计数量
        counts.append(counts[-1] + imgs.shape[0])
        # 将当前集合的数据加入总列表
        all_imgs.append(imgs)
        all_poses.append(poses)

    # 生成三个集合在合并后数组中的索引切片
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    # 将所有集合的图像拼接成一个数组
    imgs = np.concatenate(all_imgs, 0)
    # 将所有集合的位姿拼接成一个数组
    poses = np.concatenate(all_poses, 0)

    # 从第一张图像获取高度和宽度
    H, W = imgs[0].shape[:2]
    # 从元数据中读取相机焦距
    focal = float(meta['frames'][0]['intrinsic_matrix'][0][0])
    # 读取相机内参矩阵
    K = meta['frames'][0]['intrinsic_matrix']
    # 打印焦距信息
    print(f"Focal: {focal}")

    # 生成一圈渲染位姿，用于新视角合成
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    # 如果需要半分辨率
    if half_res:
        # 高度和宽度减半
        H = H//2
        W = W//2
        # 焦距也减半
        focal = focal/2.

        # 准备一个空数组存放缩放后的图像
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
        # 对每张图像做插值缩放
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        # 用缩放后的结果替换原图
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    # 取训练集和测试集 near/far 的上下界，分别向下/向上取整
    near = np.floor(min(metas['train']['near'], metas['test']['near']))
    far = np.ceil(max(metas['train']['far'], metas['test']['far']))
    # 返回图像、位姿、渲染位姿、相机参数、内参矩阵、数据切片以及近远平面
    return imgs, poses, render_poses, [H, W, focal], K, i_split, near, far


