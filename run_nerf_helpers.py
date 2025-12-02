# 导入 torch，用于张量计算
import torch
# 导入 numpy，用于数值运算
import numpy as np
# 导入 torch.nn 模块以构建网络
import torch.nn as nn
# 导入函数式接口以使用 ReLU 等激活函数
import torch.nn.functional as F

# 将两张图像的均方误差作为损失计算
img2mse = lambda x, y : torch.mean((x - y) ** 2)
# 将 MSE 转换为 PSNR 指标
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
# 将浮点图像裁剪到 0-1 后转换为 8bit
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# 位置编码模块（论文 5.1 节）
class Embedder:
    def __init__(self, **kwargs):
        # 保存配置参数
        self.kwargs = kwargs
        # 根据参数生成编码函数列表
        self.create_embedding_fn()

    def create_embedding_fn(self):
        # 存放具体的编码函数
        embed_fns = []
        # 输入维度
        d = self.kwargs['input_dims']
        # 输出维度初始化
        out_dim = 0
        # 是否包含原始输入
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        # 最大频率指数
        max_freq = self.kwargs['max_freq_log2']
        # 频率数量
        N_freqs = self.kwargs['num_freqs']

        # 根据是否对数采样生成频带
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        # 为每个频率和周期函数生成编码函数
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        # 保存编码函数和输出维度
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        # 将所有编码结果在最后一个维度上拼接
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


# 获取位置编码器的工厂函数

def get_embedder(multires, i=0):
    # 当 i 为 -1 时返回恒等映射
    if i == -1:
        return nn.Identity(), 3

    # 配置编码参数
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    # 构造编码器对象
    embedder_obj = Embedder(**embed_kwargs)
    # 提供一个便捷的调用接口
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    # 返回编码函数和输出维度
    return embed, embedder_obj.out_dim


# NeRF 主网络模型
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """
        NeRF 模型的多层感知机定义
        """
        super(NeRF, self).__init__()
        # 总层数
        self.D = D
        # 每层宽度
        self.W = W
        # 位置编码后的维度
        self.input_ch = input_ch
        # 视角编码后的维度
        self.input_ch_views = input_ch_views
        # 最终输出通道
        self.skips = skips
        # 是否使用视角依赖分支
        self.use_viewdirs = use_viewdirs

        # 点的线性层序列，部分层拼接跳跃连接
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        # 根据是否使用视角分支定义输出层
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        # 将输入拆分为位置编码和视角编码部分
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        # 初始隐藏状态为位置向量
        h = input_pts
        # 遍历点的线性层
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            # 在指定层添加跳跃连接
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        # 若使用视角分支
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def load_weights_from_keras(self, weights):
        # 仅支持包含视角分支的模型
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # 依次加载点 MLP 的权重
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))

        # 加载特征和输出层权重
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # 加载视角分支权重
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # 加载 RGB 和透明度层权重
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # 加载 alpha 分支权重
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))


# 光线辅助函数

def get_rays(H, W, K, c2w):
    # 构建像素网格，注意 PyTorch meshgrid 的索引顺序
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    # 计算每个像素对应的方向向量
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # 将方向从相机坐标系旋转到世界坐标系
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # 相机中心在世界坐标系下的坐标，作为光线原点
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


# NumPy 版本的射线生成

def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # 方向旋转到世界坐标系
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # 光线原点展开到与方向相同的形状
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


# 将射线投影到 NDC 坐标系

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # 将光线原点平移到 near 平面
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # 进行投影变换
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d


# 分层采样函数（论文 5.2 节）

def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # 为避免数值问题给权重加上极小值
    weights = weights + 1e-5 # prevent nans
    # 计算概率密度函数
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    # 计算累积分布函数
    cdf = torch.cumsum(pdf, -1)
    # 在开头添加 0 以方便后续查找区间
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # 根据是否确定性采样生成均匀样本
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # 在单元测试中使用固定的随机数
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # 通过反转 CDF 将均匀样本映射到目标分布
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # 获取对应的 cdf 和 bin 界限
    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    # 避免分母为 0 的情况
    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    # 线性插值得到最终样本位置
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

