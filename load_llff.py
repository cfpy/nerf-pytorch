# 导入 os 和 imageio，用于文件操作和图像读取
import os, imageio
# 导入 numpy 以及数学函数
import numpy as np


########## Slightly modified version of LLFF data loading code
##########  see https://github.com/Fyusion/LLFF for original

# 将原始图片按照缩放因子或指定分辨率缩小
# basedir: 数据目录；factors/resolutions: 缩放参数列表

def _minify(basedir, factors=[], resolutions=[]):
    # 标记是否需要生成缩小版图像
    needtoload = False
    # 检查按倍数缩放的目标目录是否存在
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    # 检查按分辨率缩放的目标目录是否存在
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    # 如果都存在则直接返回
    if not needtoload:
        return

    # 延迟导入复制和子进程工具
    from shutil import copy
    from subprocess import check_output

    # 原始图像目录
    imgdir = os.path.join(basedir, 'images')
    # 获取所有图像文件路径
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    # 过滤出常见扩展名的图片
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    # 记录原始目录
    imgdir_orig = imgdir

    # 记录当前工作目录
    wd = os.getcwd()

    # 针对每个缩放请求生成缩放后的图片
    for r in factors + resolutions:
        # 判断是缩放比例还是固定分辨率
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        # 目标目录
        imgdir = os.path.join(basedir, name)
        # 若已存在则跳过
        if os.path.exists(imgdir):
            continue

        # 打印缩小信息
        print('Minifying', r, basedir)

        # 创建目标目录
        os.makedirs(imgdir)
        # 将原图复制到目标目录
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        # 获取原始图片扩展名
        ext = imgs[0].split('.')[-1]
        # 构造 mogrify 缩放命令
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        # 打印命令
        print(args)
        # 进入目标目录执行缩放
        os.chdir(imgdir)
        check_output(args, shell=True)
        # 返回原工作目录
        os.chdir(wd)

        # 如果原始格式不是 png，删除旧格式副本
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        # 完成提示
        print('Done')


# 读取 LLFF 数据并可选地缩放
# factor: 缩放倍数；width/height: 指定宽高；load_imgs: 是否加载图像

def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):

    # 读取位姿和视锥界的 npy 文件
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    # 提取位姿部分并整理维度为 [3,5,N]
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    # 提取近远平面数据
    bds = poses_arr[:, -2:].transpose([1,0])

    # 获取任意一张原始图像以确定尺寸
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    # 读取图像形状
    sh = imageio.imread(img0).shape

    # 后缀初始化
    sfx = ''

    # 根据传入参数决定缩放方式
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    # 确定缩放后图像目录
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return

    # 列出缩放后所有图片
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    # 确保图片数量与位姿数量匹配
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return

    # 读取缩放后尺寸
    sh = imageio.imread(imgfiles[0]).shape
    # 将图像尺寸写入位姿的第 4 列
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor

    # 如果不加载图像，则直接返回位姿和边界
    if not load_imgs:
        return poses, bds

    # 自定义图像读取以处理 png 的 gamma
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    # 读取所有图像并归一化到 0-1
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    # 堆叠成形状 [H,W,3,N]
    imgs = np.stack(imgs, -1)

    # 打印加载信息并返回
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs




# 将向量归一化

def normalize(x):
    return x / np.linalg.norm(x)

# 根据 z 方向、up 向量和位置生成视角矩阵

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

# 将世界坐标点转换到相机坐标系

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

# 计算多个位姿的平均位姿

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


# 生成螺旋路径的渲染位姿

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


# 将位姿重心平移到坐标原点

def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


#####################


# 将位姿映射到球面并生成均匀分布的渲染路径

def spherify_poses(poses, bds):

    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)

    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])
    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))

    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []

    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)

    return poses_reset, new_poses, bds


# 主入口：加载 LLFF 数据并生成渲染路径

def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):


    # 加载原始数据（默认缩放 8 倍）
    poses, bds, imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())

    # 调整旋转矩阵顺序并将变长维度移到前面
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # 如果提供边界缩放因子则进行尺度调整
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc

    # 是否将场景中心对齐到原点
    if recenter:
        poses = recenter_poses(poses)

    # 如果需要球面路径则生成球面渲染位姿
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:

        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
  #             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)


    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)

    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test


