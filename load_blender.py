# 导入操作系统库，处理路径和文件
import os
# 导入 PyTorch，用于张量运算
import torch
# 导入 NumPy，用于数值计算
import numpy as np
# 导入 imageio，用于图像读取
import imageio
# 导入 json，用于解析元数据
import json
# 导入 PyTorch 的函数式 API
import torch.nn.functional as F
# 导入 OpenCV，用于图像缩放
import cv2


# 定义沿 z 轴平移 t 的 4x4 变换矩阵
trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

# 定义绕 x 轴旋转 phi 的 4x4 旋转矩阵
rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

# 定义绕 y 轴旋转 theta 的 4x4 旋转矩阵
rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


# 将球坐标参数转换成 c2w 位姿矩阵
# theta: 绕 y 轴角度；phi: 绕 x 轴角度；radius: 距离原点半径

def pose_spherical(theta, phi, radius):
    # 先平移到半径距离
    c2w = trans_t(radius)
    # 绕 x 轴旋转
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    # 绕 y 轴旋转
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    # 将坐标系转换为 NeRF 期望的形式
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    # 返回位姿
    return c2w


# 加载 Blender 数据集
# basedir: 数据集根目录；half_res: 是否半分辨率；testskip: 测试集采样间隔

def load_blender_data(basedir, half_res=False, testskip=1):
    # 数据集划分
    splits = ['train', 'val', 'test']
    # 存储元信息
    metas = {}
    # 读取各分割的 transforms_*.json 文件
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    # 存储全部图像
    all_imgs = []
    # 存储全部位姿
    all_poses = []
    # 记录每个分割的累积数量
    counts = [0]
    # 遍历数据划分
    for s in splits:
        # 当前分割的元信息
        meta = metas[s]
        # 当前分割的图像列表
        imgs = []
        # 当前分割的位姿列表
        poses = []
        # 训练集或不跳过时的步长
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip

        # 按步长读取帧
        for frame in meta['frames'][::skip]:
            # 拼接 PNG 路径
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            # 读取图像
            imgs.append(imageio.imread(fname))
            # 追加位姿矩阵
            poses.append(np.array(frame['transform_matrix']))
        # 归一化图像到 0~1，保留 RGBA
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        # 转为 float32 的位姿数组
        poses = np.array(poses).astype(np.float32)
        # 更新累积数量
        counts.append(counts[-1] + imgs.shape[0])
        # 收集本分割的数据
        all_imgs.append(imgs)
        all_poses.append(poses)

    # 生成三份数据的切片索引
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    # 合并所有图像
    imgs = np.concatenate(all_imgs, 0)
    # 合并所有位姿
    poses = np.concatenate(all_poses, 0)

    # 从首张图像获取尺寸
    H, W = imgs[0].shape[:2]
    # 从元数据获取水平视角
    camera_angle_x = float(meta['camera_angle_x'])
    # 根据视角计算焦距
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    # 预生成一组渲染位姿
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    # 如果需要半分辨率
    if half_res:
        # 高宽减半
        H = H//2
        W = W//2
        # 焦距也减半
        focal = focal/2.

        # 预分配半分辨率图像数组
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        # 逐张缩放
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        # 替换为缩放后的结果
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()


    # 返回图像、位姿、渲染位姿、相机尺寸与焦距以及索引切片
    return imgs, poses, render_poses, [H, W, focal], i_split


