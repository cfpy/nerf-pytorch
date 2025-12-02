# 为了便于逐行理解，本文件在每一行代码前添加了一条中文注释，直接说明该行的具体内容。
# 这一行执行: import os, sys
import os, sys
# 这一行执行: import numpy as np
import numpy as np
# 这一行执行: import imageio
import imageio
# 这一行执行: import json
import json
# 这一行执行: import random
import random
# 这一行执行: import time
import time
# 这一行执行: import torch
import torch
# 这一行执行: import torch.nn as nn
import torch.nn as nn
# 这一行执行: import torch.nn.functional as F
import torch.nn.functional as F
# 这一行执行: from tqdm import tqdm, trange
from tqdm import tqdm, trange

# 这一行执行: import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# 这一行执行: from run_nerf_helpers import *
from run_nerf_helpers import *

# 这一行执行: from load_llff import load_llff_data
from load_llff import load_llff_data
# 这一行执行: from load_deepvoxels import load_dv_data
from load_deepvoxels import load_dv_data
# 这一行执行: from load_blender import load_blender_data
from load_blender import load_blender_data
# 这一行执行: from load_LINEMOD import load_LINEMOD_data
from load_LINEMOD import load_LINEMOD_data


# 这一行执行: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 这一行执行: np.random.seed(0)
np.random.seed(0)
# 这一行执行: DEBUG = False
DEBUG = False


# 这一行执行: def batchify(fn, chunk):
def batchify(fn, chunk):
# 这一行执行:     """Constructs a version of 'fn' that applies to smaller batches.
    """Constructs a version of 'fn' that applies to smaller batches.
# 这一行执行:     """
    """
# 这一行执行:     if chunk is None:
    if chunk is None:
# 这一行执行:         return fn
        return fn
# 这一行执行:     def ret(inputs):
    def ret(inputs):
# 这一行执行:         return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
# 这一行执行:     return ret
    return ret


# 这一行执行: def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
# 这一行执行:     """Prepares inputs and applies network 'fn'.
    """Prepares inputs and applies network 'fn'.
# 这一行执行:     """
    """
# 这一行执行:     inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
# 这一行执行:     embedded = embed_fn(inputs_flat)
    embedded = embed_fn(inputs_flat)

# 这一行执行:     if viewdirs is not None:
    if viewdirs is not None:
# 这一行执行:         input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs = viewdirs[:,None].expand(inputs.shape)
# 这一行执行:         input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
# 这一行执行:         embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded_dirs = embeddirs_fn(input_dirs_flat)
# 这一行执行:         embedded = torch.cat([embedded, embedded_dirs], -1)
        embedded = torch.cat([embedded, embedded_dirs], -1)

# 这一行执行:     outputs_flat = batchify(fn, netchunk)(embedded)
    outputs_flat = batchify(fn, netchunk)(embedded)
# 这一行执行:     outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
# 这一行执行:     return outputs
    return outputs


# 这一行执行: def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
# 这一行执行:     """Render rays in smaller minibatches to avoid OOM.
    """Render rays in smaller minibatches to avoid OOM.
# 这一行执行:     """
    """
# 这一行执行:     all_ret = {}
    all_ret = {}
# 这一行执行:     for i in range(0, rays_flat.shape[0], chunk):
    for i in range(0, rays_flat.shape[0], chunk):
# 这一行执行:         ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
# 这一行执行:         for k in ret:
        for k in ret:
# 这一行执行:             if k not in all_ret:
            if k not in all_ret:
# 这一行执行:                 all_ret[k] = []
                all_ret[k] = []
# 这一行执行:             all_ret[k].append(ret[k])
            all_ret[k].append(ret[k])

# 这一行执行:     all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
# 这一行执行:     return all_ret
    return all_ret


# 这一行执行: def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
# 这一行执行:                   near=0., far=1.,
                  near=0., far=1.,
# 这一行执行:                   use_viewdirs=False, c2w_staticcam=None,
                  use_viewdirs=False, c2w_staticcam=None,
# 这一行执行:                   **kwargs):
                  **kwargs):
# 这一行执行:     """Render rays
    """Render rays
# 这一行执行:     Args:
    Args:
# 这一行执行:       H: int. Height of image in pixels.
      H: int. Height of image in pixels.
# 这一行执行:       W: int. Width of image in pixels.
      W: int. Width of image in pixels.
# 这一行执行:       focal: float. Focal length of pinhole camera.
      focal: float. Focal length of pinhole camera.
# 这一行执行:       chunk: int. Maximum number of rays to process simultaneously. Used to
      chunk: int. Maximum number of rays to process simultaneously. Used to
# 这一行执行:         control maximum memory usage. Does not affect final results.
        control maximum memory usage. Does not affect final results.
# 这一行执行:       rays: array of shape [2, batch_size, 3]. Ray origin and direction for
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
# 这一行执行:         each example in batch.
        each example in batch.
# 这一行执行:       c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
# 这一行执行:       ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
# 这一行执行:       near: float or array of shape [batch_size]. Nearest distance for a ray.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
# 这一行执行:       far: float or array of shape [batch_size]. Farthest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
# 这一行执行:       use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
# 这一行执行:       c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
# 这一行执行:        camera while using other c2w argument for viewing directions.
       camera while using other c2w argument for viewing directions.
# 这一行执行:     Returns:
    Returns:
# 这一行执行:       rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
# 这一行执行:       disp_map: [batch_size]. Disparity map. Inverse of depth.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
# 这一行执行:       acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
# 这一行执行:       extras: dict with everything returned by render_rays().
      extras: dict with everything returned by render_rays().
# 这一行执行:     """
    """
# 这一行执行:     if c2w is not None:
    if c2w is not None:
# 这一行执行:         # special case to render full image
        # special case to render full image
# 这一行执行:         rays_o, rays_d = get_rays(H, W, K, c2w)
        rays_o, rays_d = get_rays(H, W, K, c2w)
# 这一行执行:     else:
    else:
# 这一行执行:         # use provided ray batch
        # use provided ray batch
# 这一行执行:         rays_o, rays_d = rays
        rays_o, rays_d = rays

# 这一行执行:     if use_viewdirs:
    if use_viewdirs:
# 这一行执行:         # provide ray directions as input
        # provide ray directions as input
# 这一行执行:         viewdirs = rays_d
        viewdirs = rays_d
# 这一行执行:         if c2w_staticcam is not None:
        if c2w_staticcam is not None:
# 这一行执行:             # special case to visualize effect of viewdirs
            # special case to visualize effect of viewdirs
# 这一行执行:             rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
# 这一行执行:         viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
# 这一行执行:         viewdirs = torch.reshape(viewdirs, [-1,3]).float()
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

# 这一行执行:     sh = rays_d.shape # [..., 3]
    sh = rays_d.shape # [..., 3]
# 这一行执行:     if ndc:
    if ndc:
# 这一行执行:         # for forward facing scenes
        # for forward facing scenes
# 这一行执行:         rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

# 这一行执行:     # Create ray batch
    # Create ray batch
# 这一行执行:     rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_o = torch.reshape(rays_o, [-1,3]).float()
# 这一行执行:     rays_d = torch.reshape(rays_d, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

# 这一行执行:     near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
# 这一行执行:     rays = torch.cat([rays_o, rays_d, near, far], -1)
    rays = torch.cat([rays_o, rays_d, near, far], -1)
# 这一行执行:     if use_viewdirs:
    if use_viewdirs:
# 这一行执行:         rays = torch.cat([rays, viewdirs], -1)
        rays = torch.cat([rays, viewdirs], -1)

# 这一行执行:     # Render and reshape
    # Render and reshape
# 这一行执行:     all_ret = batchify_rays(rays, chunk, **kwargs)
    all_ret = batchify_rays(rays, chunk, **kwargs)
# 这一行执行:     for k in all_ret:
    for k in all_ret:
# 这一行执行:         k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
# 这一行执行:         all_ret[k] = torch.reshape(all_ret[k], k_sh)
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

# 这一行执行:     k_extract = ['rgb_map', 'disp_map', 'acc_map']
    k_extract = ['rgb_map', 'disp_map', 'acc_map']
# 这一行执行:     ret_list = [all_ret[k] for k in k_extract]
    ret_list = [all_ret[k] for k in k_extract]
# 这一行执行:     ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
# 这一行执行:     return ret_list + [ret_dict]
    return ret_list + [ret_dict]


# 这一行执行: def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

# 这一行执行:     H, W, focal = hwf
    H, W, focal = hwf

# 这一行执行:     if render_factor!=0:
    if render_factor!=0:
# 这一行执行:         # Render downsampled for speed
        # Render downsampled for speed
# 这一行执行:         H = H//render_factor
        H = H//render_factor
# 这一行执行:         W = W//render_factor
        W = W//render_factor
# 这一行执行:         focal = focal/render_factor
        focal = focal/render_factor

# 这一行执行:     rgbs = []
    rgbs = []
# 这一行执行:     disps = []
    disps = []

# 这一行执行:     t = time.time()
    t = time.time()
# 这一行执行:     for i, c2w in enumerate(tqdm(render_poses)):
    for i, c2w in enumerate(tqdm(render_poses)):
# 这一行执行:         print(i, time.time() - t)
        print(i, time.time() - t)
# 这一行执行:         t = time.time()
        t = time.time()
# 这一行执行:         rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
# 这一行执行:         rgbs.append(rgb.cpu().numpy())
        rgbs.append(rgb.cpu().numpy())
# 这一行执行:         disps.append(disp.cpu().numpy())
        disps.append(disp.cpu().numpy())
# 这一行执行:         if i==0:
        if i==0:
# 这一行执行:             print(rgb.shape, disp.shape)
            print(rgb.shape, disp.shape)

# 这一行执行:         """
        """
# 这一行执行:         if gt_imgs is not None and render_factor==0:
        if gt_imgs is not None and render_factor==0:
# 这一行执行:             p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
# 这一行执行:             print(p)
            print(p)
# 这一行执行:         """
        """

# 这一行执行:         if savedir is not None:
        if savedir is not None:
# 这一行执行:             rgb8 = to8b(rgbs[-1])
            rgb8 = to8b(rgbs[-1])
# 这一行执行:             filename = os.path.join(savedir, '{:03d}.png'.format(i))
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
# 这一行执行:             imageio.imwrite(filename, rgb8)
            imageio.imwrite(filename, rgb8)


# 这一行执行:     rgbs = np.stack(rgbs, 0)
    rgbs = np.stack(rgbs, 0)
# 这一行执行:     disps = np.stack(disps, 0)
    disps = np.stack(disps, 0)

# 这一行执行:     return rgbs, disps
    return rgbs, disps


# 这一行执行: def create_nerf(args):
def create_nerf(args):
# 这一行执行:     """Instantiate NeRF's MLP model.
    """Instantiate NeRF's MLP model.
# 这一行执行:     """
    """
# 这一行执行:     embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

# 这一行执行:     input_ch_views = 0
    input_ch_views = 0
# 这一行执行:     embeddirs_fn = None
    embeddirs_fn = None
# 这一行执行:     if args.use_viewdirs:
    if args.use_viewdirs:
# 这一行执行:         embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
# 这一行执行:     output_ch = 5 if args.N_importance > 0 else 4
    output_ch = 5 if args.N_importance > 0 else 4
# 这一行执行:     skips = [4]
    skips = [4]
# 这一行执行:     model = NeRF(D=args.netdepth, W=args.netwidth,
    model = NeRF(D=args.netdepth, W=args.netwidth,
# 这一行执行:                  input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
# 这一行执行:                  input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
# 这一行执行:     grad_vars = list(model.parameters())
    grad_vars = list(model.parameters())

# 这一行执行:     model_fine = None
    model_fine = None
# 这一行执行:     if args.N_importance > 0:
    if args.N_importance > 0:
# 这一行执行:         model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
# 这一行执行:                           input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
# 这一行执行:                           input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
# 这一行执行:         grad_vars += list(model_fine.parameters())
        grad_vars += list(model_fine.parameters())

# 这一行执行:     network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
# 这一行执行:                                                                 embed_fn=embed_fn,
                                                                embed_fn=embed_fn,
# 这一行执行:                                                                 embeddirs_fn=embeddirs_fn,
                                                                embeddirs_fn=embeddirs_fn,
# 这一行执行:                                                                 netchunk=args.netchunk)
                                                                netchunk=args.netchunk)

# 这一行执行:     # Create optimizer
    # Create optimizer
# 这一行执行:     optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

# 这一行执行:     start = 0
    start = 0
# 这一行执行:     basedir = args.basedir
    basedir = args.basedir
# 这一行执行:     expname = args.expname
    expname = args.expname

# 这一行执行:     ##########################
    ##########################

# 这一行执行:     # Load checkpoints
    # Load checkpoints
# 这一行执行:     if args.ft_path is not None and args.ft_path!='None':
    if args.ft_path is not None and args.ft_path!='None':
# 这一行执行:         ckpts = [args.ft_path]
        ckpts = [args.ft_path]
# 这一行执行:     else:
    else:
# 这一行执行:         ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

# 这一行执行:     print('Found ckpts', ckpts)
    print('Found ckpts', ckpts)
# 这一行执行:     if len(ckpts) > 0 and not args.no_reload:
    if len(ckpts) > 0 and not args.no_reload:
# 这一行执行:         ckpt_path = ckpts[-1]
        ckpt_path = ckpts[-1]
# 这一行执行:         print('Reloading from', ckpt_path)
        print('Reloading from', ckpt_path)
# 这一行执行:         ckpt = torch.load(ckpt_path)
        ckpt = torch.load(ckpt_path)

# 这一行执行:         start = ckpt['global_step']
        start = ckpt['global_step']
# 这一行执行:         optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

# 这一行执行:         # Load model
        # Load model
# 这一行执行:         model.load_state_dict(ckpt['network_fn_state_dict'])
        model.load_state_dict(ckpt['network_fn_state_dict'])
# 这一行执行:         if model_fine is not None:
        if model_fine is not None:
# 这一行执行:             model_fine.load_state_dict(ckpt['network_fine_state_dict'])
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

# 这一行执行:     ##########################
    ##########################

# 这一行执行:     render_kwargs_train = {
    render_kwargs_train = {
# 这一行执行:         'network_query_fn' : network_query_fn,
        'network_query_fn' : network_query_fn,
# 这一行执行:         'perturb' : args.perturb,
        'perturb' : args.perturb,
# 这一行执行:         'N_importance' : args.N_importance,
        'N_importance' : args.N_importance,
# 这一行执行:         'network_fine' : model_fine,
        'network_fine' : model_fine,
# 这一行执行:         'N_samples' : args.N_samples,
        'N_samples' : args.N_samples,
# 这一行执行:         'network_fn' : model,
        'network_fn' : model,
# 这一行执行:         'use_viewdirs' : args.use_viewdirs,
        'use_viewdirs' : args.use_viewdirs,
# 这一行执行:         'white_bkgd' : args.white_bkgd,
        'white_bkgd' : args.white_bkgd,
# 这一行执行:         'raw_noise_std' : args.raw_noise_std,
        'raw_noise_std' : args.raw_noise_std,
# 这一行执行:     }
    }

# 这一行执行:     # NDC only good for LLFF-style forward facing data
    # NDC only good for LLFF-style forward facing data
# 这一行执行:     if args.dataset_type != 'llff' or args.no_ndc:
    if args.dataset_type != 'llff' or args.no_ndc:
# 这一行执行:         print('Not ndc!')
        print('Not ndc!')
# 这一行执行:         render_kwargs_train['ndc'] = False
        render_kwargs_train['ndc'] = False
# 这一行执行:         render_kwargs_train['lindisp'] = args.lindisp
        render_kwargs_train['lindisp'] = args.lindisp

# 这一行执行:     render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
# 这一行执行:     render_kwargs_test['perturb'] = False
    render_kwargs_test['perturb'] = False
# 这一行执行:     render_kwargs_test['raw_noise_std'] = 0.
    render_kwargs_test['raw_noise_std'] = 0.

# 这一行执行:     return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer
    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


# 这一行执行: def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
# 这一行执行:     """Transforms model's predictions to semantically meaningful values.
    """Transforms model's predictions to semantically meaningful values.
# 这一行执行:     Args:
    Args:
# 这一行执行:         raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
# 这一行执行:         z_vals: [num_rays, num_samples along ray]. Integration time.
        z_vals: [num_rays, num_samples along ray]. Integration time.
# 这一行执行:         rays_d: [num_rays, 3]. Direction of each ray.
        rays_d: [num_rays, 3]. Direction of each ray.
# 这一行执行:     Returns:
    Returns:
# 这一行执行:         rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
# 这一行执行:         disp_map: [num_rays]. Disparity map. Inverse of depth map.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
# 这一行执行:         acc_map: [num_rays]. Sum of weights along each ray.
        acc_map: [num_rays]. Sum of weights along each ray.
# 这一行执行:         weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
# 这一行执行:         depth_map: [num_rays]. Estimated distance to object.
        depth_map: [num_rays]. Estimated distance to object.
# 这一行执行:     """
    """
# 这一行执行:     raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

# 这一行执行:     dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = z_vals[...,1:] - z_vals[...,:-1]
# 这一行执行:     dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

# 这一行执行:     dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

# 这一行执行:     rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
# 这一行执行:     noise = 0.
    noise = 0.
# 这一行执行:     if raw_noise_std > 0.:
    if raw_noise_std > 0.:
# 这一行执行:         noise = torch.randn(raw[...,3].shape) * raw_noise_std
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

# 这一行执行:         # Overwrite randomly sampled data if pytest
        # Overwrite randomly sampled data if pytest
# 这一行执行:         if pytest:
        if pytest:
# 这一行执行:             np.random.seed(0)
            np.random.seed(0)
# 这一行执行:             noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
# 这一行执行:             noise = torch.Tensor(noise)
            noise = torch.Tensor(noise)

# 这一行执行:     alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
# 这一行执行:     # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
# 这一行执行:     weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
# 这一行执行:     rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

# 这一行执行:     depth_map = torch.sum(weights * z_vals, -1)
    depth_map = torch.sum(weights * z_vals, -1)
# 这一行执行:     disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
# 这一行执行:     acc_map = torch.sum(weights, -1)
    acc_map = torch.sum(weights, -1)

# 这一行执行:     if white_bkgd:
    if white_bkgd:
# 这一行执行:         rgb_map = rgb_map + (1.-acc_map[...,None])
        rgb_map = rgb_map + (1.-acc_map[...,None])

# 这一行执行:     return rgb_map, disp_map, acc_map, weights, depth_map
    return rgb_map, disp_map, acc_map, weights, depth_map


# 这一行执行: def render_rays(ray_batch,
def render_rays(ray_batch,
# 这一行执行:                 network_fn,
                network_fn,
# 这一行执行:                 network_query_fn,
                network_query_fn,
# 这一行执行:                 N_samples,
                N_samples,
# 这一行执行:                 retraw=False,
                retraw=False,
# 这一行执行:                 lindisp=False,
                lindisp=False,
# 这一行执行:                 perturb=0.,
                perturb=0.,
# 这一行执行:                 N_importance=0,
                N_importance=0,
# 这一行执行:                 network_fine=None,
                network_fine=None,
# 这一行执行:                 white_bkgd=False,
                white_bkgd=False,
# 这一行执行:                 raw_noise_std=0.,
                raw_noise_std=0.,
# 这一行执行:                 verbose=False,
                verbose=False,
# 这一行执行:                 pytest=False):
                pytest=False):
# 这一行执行:     """Volumetric rendering.
    """Volumetric rendering.
# 这一行执行:     Args:
    Args:
# 这一行执行:       ray_batch: array of shape [batch_size, ...]. All information necessary
      ray_batch: array of shape [batch_size, ...]. All information necessary
# 这一行执行:         for sampling along a ray, including: ray origin, ray direction, min
        for sampling along a ray, including: ray origin, ray direction, min
# 这一行执行:         dist, max dist, and unit-magnitude viewing direction.
        dist, max dist, and unit-magnitude viewing direction.
# 这一行执行:       network_fn: function. Model for predicting RGB and density at each point
      network_fn: function. Model for predicting RGB and density at each point
# 这一行执行:         in space.
        in space.
# 这一行执行:       network_query_fn: function used for passing queries to network_fn.
      network_query_fn: function used for passing queries to network_fn.
# 这一行执行:       N_samples: int. Number of different times to sample along each ray.
      N_samples: int. Number of different times to sample along each ray.
# 这一行执行:       retraw: bool. If True, include model's raw, unprocessed predictions.
      retraw: bool. If True, include model's raw, unprocessed predictions.
# 这一行执行:       lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
# 这一行执行:       perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
# 这一行执行:         random points in time.
        random points in time.
# 这一行执行:       N_importance: int. Number of additional times to sample along each ray.
      N_importance: int. Number of additional times to sample along each ray.
# 这一行执行:         These samples are only passed to network_fine.
        These samples are only passed to network_fine.
# 这一行执行:       network_fine: "fine" network with same spec as network_fn.
      network_fine: "fine" network with same spec as network_fn.
# 这一行执行:       white_bkgd: bool. If True, assume a white background.
      white_bkgd: bool. If True, assume a white background.
# 这一行执行:       raw_noise_std: ...
      raw_noise_std: ...
# 这一行执行:       verbose: bool. If True, print more debugging info.
      verbose: bool. If True, print more debugging info.
# 这一行执行:     Returns:
    Returns:
# 这一行执行:       rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
# 这一行执行:       disp_map: [num_rays]. Disparity map. 1 / depth.
      disp_map: [num_rays]. Disparity map. 1 / depth.
# 这一行执行:       acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
# 这一行执行:       raw: [num_rays, num_samples, 4]. Raw predictions from model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
# 这一行执行:       rgb0: See rgb_map. Output for coarse model.
      rgb0: See rgb_map. Output for coarse model.
# 这一行执行:       disp0: See disp_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
# 这一行执行:       acc0: See acc_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
# 这一行执行:       z_std: [num_rays]. Standard deviation of distances along ray for each
      z_std: [num_rays]. Standard deviation of distances along ray for each
# 这一行执行:         sample.
        sample.
# 这一行执行:     """
    """
# 这一行执行:     N_rays = ray_batch.shape[0]
    N_rays = ray_batch.shape[0]
# 这一行执行:     rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
# 这一行执行:     viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
# 这一行执行:     bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
# 这一行执行:     near, far = bounds[...,0], bounds[...,1] # [-1,1]
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

# 这一行执行:     t_vals = torch.linspace(0., 1., steps=N_samples)
    t_vals = torch.linspace(0., 1., steps=N_samples)
# 这一行执行:     if not lindisp:
    if not lindisp:
# 这一行执行:         z_vals = near * (1.-t_vals) + far * (t_vals)
        z_vals = near * (1.-t_vals) + far * (t_vals)
# 这一行执行:     else:
    else:
# 这一行执行:         z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

# 这一行执行:     z_vals = z_vals.expand([N_rays, N_samples])
    z_vals = z_vals.expand([N_rays, N_samples])

# 这一行执行:     if perturb > 0.:
    if perturb > 0.:
# 这一行执行:         # get intervals between samples
        # get intervals between samples
# 这一行执行:         mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
# 这一行执行:         upper = torch.cat([mids, z_vals[...,-1:]], -1)
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
# 这一行执行:         lower = torch.cat([z_vals[...,:1], mids], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
# 这一行执行:         # stratified samples in those intervals
        # stratified samples in those intervals
# 这一行执行:         t_rand = torch.rand(z_vals.shape)
        t_rand = torch.rand(z_vals.shape)

# 这一行执行:         # Pytest, overwrite u with numpy's fixed random numbers
        # Pytest, overwrite u with numpy's fixed random numbers
# 这一行执行:         if pytest:
        if pytest:
# 这一行执行:             np.random.seed(0)
            np.random.seed(0)
# 这一行执行:             t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = np.random.rand(*list(z_vals.shape))
# 这一行执行:             t_rand = torch.Tensor(t_rand)
            t_rand = torch.Tensor(t_rand)

# 这一行执行:         z_vals = lower + (upper - lower) * t_rand
        z_vals = lower + (upper - lower) * t_rand

# 这一行执行:     pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


# 这一行执行: #     raw = run_network(pts)
#     raw = run_network(pts)
# 这一行执行:     raw = network_query_fn(pts, viewdirs, network_fn)
    raw = network_query_fn(pts, viewdirs, network_fn)
# 这一行执行:     rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

# 这一行执行:     if N_importance > 0:
    if N_importance > 0:

# 这一行执行:         rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

# 这一行执行:         z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
# 这一行执行:         z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
# 这一行执行:         z_samples = z_samples.detach()
        z_samples = z_samples.detach()

# 这一行执行:         z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
# 这一行执行:         pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

# 这一行执行:         run_fn = network_fn if network_fine is None else network_fine
        run_fn = network_fn if network_fine is None else network_fine
# 这一行执行: #         raw = run_network(pts, fn=run_fn)
#         raw = run_network(pts, fn=run_fn)
# 这一行执行:         raw = network_query_fn(pts, viewdirs, run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

# 这一行执行:         rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

# 这一行执行:     ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
# 这一行执行:     if retraw:
    if retraw:
# 这一行执行:         ret['raw'] = raw
        ret['raw'] = raw
# 这一行执行:     if N_importance > 0:
    if N_importance > 0:
# 这一行执行:         ret['rgb0'] = rgb_map_0
        ret['rgb0'] = rgb_map_0
# 这一行执行:         ret['disp0'] = disp_map_0
        ret['disp0'] = disp_map_0
# 这一行执行:         ret['acc0'] = acc_map_0
        ret['acc0'] = acc_map_0
# 这一行执行:         ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

# 这一行执行:     for k in ret:
    for k in ret:
# 这一行执行:         if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
# 这一行执行:             print(f"! [Numerical Error] {k} contains nan or inf.")
            print(f"! [Numerical Error] {k} contains nan or inf.")

# 这一行执行:     return ret
    return ret


# 这一行执行: def config_parser():
def config_parser():

# 这一行执行:     import configargparse
    import configargparse
# 这一行执行:     parser = configargparse.ArgumentParser()
    parser = configargparse.ArgumentParser()
# 这一行执行:     parser.add_argument('--config', is_config_file=True, 
    parser.add_argument('--config', is_config_file=True, 
# 这一行执行:                         help='config file path')
                        help='config file path')
# 这一行执行:     parser.add_argument("--expname", type=str, 
    parser.add_argument("--expname", type=str, 
# 这一行执行:                         help='experiment name')
                        help='experiment name')
# 这一行执行:     parser.add_argument("--basedir", type=str, default='./logs/', 
    parser.add_argument("--basedir", type=str, default='./logs/', 
# 这一行执行:                         help='where to store ckpts and logs')
                        help='where to store ckpts and logs')
# 这一行执行:     parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
# 这一行执行:                         help='input data directory')
                        help='input data directory')

# 这一行执行:     # training options
    # training options
# 这一行执行:     parser.add_argument("--netdepth", type=int, default=8, 
    parser.add_argument("--netdepth", type=int, default=8, 
# 这一行执行:                         help='layers in network')
                        help='layers in network')
# 这一行执行:     parser.add_argument("--netwidth", type=int, default=256, 
    parser.add_argument("--netwidth", type=int, default=256, 
# 这一行执行:                         help='channels per layer')
                        help='channels per layer')
# 这一行执行:     parser.add_argument("--netdepth_fine", type=int, default=8, 
    parser.add_argument("--netdepth_fine", type=int, default=8, 
# 这一行执行:                         help='layers in fine network')
                        help='layers in fine network')
# 这一行执行:     parser.add_argument("--netwidth_fine", type=int, default=256, 
    parser.add_argument("--netwidth_fine", type=int, default=256, 
# 这一行执行:                         help='channels per layer in fine network')
                        help='channels per layer in fine network')
# 这一行执行:     parser.add_argument("--N_rand", type=int, default=32*32*4, 
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
# 这一行执行:                         help='batch size (number of random rays per gradient step)')
                        help='batch size (number of random rays per gradient step)')
# 这一行执行:     parser.add_argument("--lrate", type=float, default=5e-4, 
    parser.add_argument("--lrate", type=float, default=5e-4, 
# 这一行执行:                         help='learning rate')
                        help='learning rate')
# 这一行执行:     parser.add_argument("--lrate_decay", type=int, default=250, 
    parser.add_argument("--lrate_decay", type=int, default=250, 
# 这一行执行:                         help='exponential learning rate decay (in 1000 steps)')
                        help='exponential learning rate decay (in 1000 steps)')
# 这一行执行:     parser.add_argument("--chunk", type=int, default=1024*32, 
    parser.add_argument("--chunk", type=int, default=1024*32, 
# 这一行执行:                         help='number of rays processed in parallel, decrease if running out of memory')
                        help='number of rays processed in parallel, decrease if running out of memory')
# 这一行执行:     parser.add_argument("--netchunk", type=int, default=1024*64, 
    parser.add_argument("--netchunk", type=int, default=1024*64, 
# 这一行执行:                         help='number of pts sent through network in parallel, decrease if running out of memory')
                        help='number of pts sent through network in parallel, decrease if running out of memory')
# 这一行执行:     parser.add_argument("--no_batching", action='store_true', 
    parser.add_argument("--no_batching", action='store_true', 
# 这一行执行:                         help='only take random rays from 1 image at a time')
                        help='only take random rays from 1 image at a time')
# 这一行执行:     parser.add_argument("--no_reload", action='store_true', 
    parser.add_argument("--no_reload", action='store_true', 
# 这一行执行:                         help='do not reload weights from saved ckpt')
                        help='do not reload weights from saved ckpt')
# 这一行执行:     parser.add_argument("--ft_path", type=str, default=None, 
    parser.add_argument("--ft_path", type=str, default=None, 
# 这一行执行:                         help='specific weights npy file to reload for coarse network')
                        help='specific weights npy file to reload for coarse network')

# 这一行执行:     # rendering options
    # rendering options
# 这一行执行:     parser.add_argument("--N_samples", type=int, default=64, 
    parser.add_argument("--N_samples", type=int, default=64, 
# 这一行执行:                         help='number of coarse samples per ray')
                        help='number of coarse samples per ray')
# 这一行执行:     parser.add_argument("--N_importance", type=int, default=0,
    parser.add_argument("--N_importance", type=int, default=0,
# 这一行执行:                         help='number of additional fine samples per ray')
                        help='number of additional fine samples per ray')
# 这一行执行:     parser.add_argument("--perturb", type=float, default=1.,
    parser.add_argument("--perturb", type=float, default=1.,
# 这一行执行:                         help='set to 0. for no jitter, 1. for jitter')
                        help='set to 0. for no jitter, 1. for jitter')
# 这一行执行:     parser.add_argument("--use_viewdirs", action='store_true', 
    parser.add_argument("--use_viewdirs", action='store_true', 
# 这一行执行:                         help='use full 5D input instead of 3D')
                        help='use full 5D input instead of 3D')
# 这一行执行:     parser.add_argument("--i_embed", type=int, default=0, 
    parser.add_argument("--i_embed", type=int, default=0, 
# 这一行执行:                         help='set 0 for default positional encoding, -1 for none')
                        help='set 0 for default positional encoding, -1 for none')
# 这一行执行:     parser.add_argument("--multires", type=int, default=10, 
    parser.add_argument("--multires", type=int, default=10, 
# 这一行执行:                         help='log2 of max freq for positional encoding (3D location)')
                        help='log2 of max freq for positional encoding (3D location)')
# 这一行执行:     parser.add_argument("--multires_views", type=int, default=4, 
    parser.add_argument("--multires_views", type=int, default=4, 
# 这一行执行:                         help='log2 of max freq for positional encoding (2D direction)')
                        help='log2 of max freq for positional encoding (2D direction)')
# 这一行执行:     parser.add_argument("--raw_noise_std", type=float, default=0., 
    parser.add_argument("--raw_noise_std", type=float, default=0., 
# 这一行执行:                         help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

# 这一行执行:     parser.add_argument("--render_only", action='store_true', 
    parser.add_argument("--render_only", action='store_true', 
# 这一行执行:                         help='do not optimize, reload weights and render out render_poses path')
                        help='do not optimize, reload weights and render out render_poses path')
# 这一行执行:     parser.add_argument("--render_test", action='store_true', 
    parser.add_argument("--render_test", action='store_true', 
# 这一行执行:                         help='render the test set instead of render_poses path')
                        help='render the test set instead of render_poses path')
# 这一行执行:     parser.add_argument("--render_factor", type=int, default=0, 
    parser.add_argument("--render_factor", type=int, default=0, 
# 这一行执行:                         help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

# 这一行执行:     # training options
    # training options
# 这一行执行:     parser.add_argument("--precrop_iters", type=int, default=0,
    parser.add_argument("--precrop_iters", type=int, default=0,
# 这一行执行:                         help='number of steps to train on central crops')
                        help='number of steps to train on central crops')
# 这一行执行:     parser.add_argument("--precrop_frac", type=float,
    parser.add_argument("--precrop_frac", type=float,
# 这一行执行:                         default=.5, help='fraction of img taken for central crops') 
                        default=.5, help='fraction of img taken for central crops') 

# 这一行执行:     # dataset options
    # dataset options
# 这一行执行:     parser.add_argument("--dataset_type", type=str, default='llff', 
    parser.add_argument("--dataset_type", type=str, default='llff', 
# 这一行执行:                         help='options: llff / blender / deepvoxels')
                        help='options: llff / blender / deepvoxels')
# 这一行执行:     parser.add_argument("--testskip", type=int, default=8, 
    parser.add_argument("--testskip", type=int, default=8, 
# 这一行执行:                         help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

# 这一行执行:     ## deepvoxels flags
    ## deepvoxels flags
# 这一行执行:     parser.add_argument("--shape", type=str, default='greek', 
    parser.add_argument("--shape", type=str, default='greek', 
# 这一行执行:                         help='options : armchair / cube / greek / vase')
                        help='options : armchair / cube / greek / vase')

# 这一行执行:     ## blender flags
    ## blender flags
# 这一行执行:     parser.add_argument("--white_bkgd", action='store_true', 
    parser.add_argument("--white_bkgd", action='store_true', 
# 这一行执行:                         help='set to render synthetic data on a white bkgd (always use for dvoxels)')
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
# 这一行执行:     parser.add_argument("--half_res", action='store_true', 
    parser.add_argument("--half_res", action='store_true', 
# 这一行执行:                         help='load blender synthetic data at 400x400 instead of 800x800')
                        help='load blender synthetic data at 400x400 instead of 800x800')

# 这一行执行:     ## llff flags
    ## llff flags
# 这一行执行:     parser.add_argument("--factor", type=int, default=8, 
    parser.add_argument("--factor", type=int, default=8, 
# 这一行执行:                         help='downsample factor for LLFF images')
                        help='downsample factor for LLFF images')
# 这一行执行:     parser.add_argument("--no_ndc", action='store_true', 
    parser.add_argument("--no_ndc", action='store_true', 
# 这一行执行:                         help='do not use normalized device coordinates (set for non-forward facing scenes)')
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
# 这一行执行:     parser.add_argument("--lindisp", action='store_true', 
    parser.add_argument("--lindisp", action='store_true', 
# 这一行执行:                         help='sampling linearly in disparity rather than depth')
                        help='sampling linearly in disparity rather than depth')
# 这一行执行:     parser.add_argument("--spherify", action='store_true', 
    parser.add_argument("--spherify", action='store_true', 
# 这一行执行:                         help='set for spherical 360 scenes')
                        help='set for spherical 360 scenes')
# 这一行执行:     parser.add_argument("--llffhold", type=int, default=8, 
    parser.add_argument("--llffhold", type=int, default=8, 
# 这一行执行:                         help='will take every 1/N images as LLFF test set, paper uses 8')
                        help='will take every 1/N images as LLFF test set, paper uses 8')

# 这一行执行:     # logging/saving options
    # logging/saving options
# 这一行执行:     parser.add_argument("--i_print",   type=int, default=100, 
    parser.add_argument("--i_print",   type=int, default=100, 
# 这一行执行:                         help='frequency of console printout and metric loggin')
                        help='frequency of console printout and metric loggin')
# 这一行执行:     parser.add_argument("--i_img",     type=int, default=500, 
    parser.add_argument("--i_img",     type=int, default=500, 
# 这一行执行:                         help='frequency of tensorboard image logging')
                        help='frequency of tensorboard image logging')
# 这一行执行:     parser.add_argument("--i_weights", type=int, default=10000, 
    parser.add_argument("--i_weights", type=int, default=10000, 
# 这一行执行:                         help='frequency of weight ckpt saving')
                        help='frequency of weight ckpt saving')
# 这一行执行:     parser.add_argument("--i_testset", type=int, default=50000, 
    parser.add_argument("--i_testset", type=int, default=50000, 
# 这一行执行:                         help='frequency of testset saving')
                        help='frequency of testset saving')
# 这一行执行:     parser.add_argument("--i_video",   type=int, default=50000, 
    parser.add_argument("--i_video",   type=int, default=50000, 
# 这一行执行:                         help='frequency of render_poses video saving')
                        help='frequency of render_poses video saving')

# 这一行执行:     return parser
    return parser


# 这一行执行: def train():
def train():

# 这一行执行:     parser = config_parser()
    parser = config_parser()
# 这一行执行:     args = parser.parse_args()
    args = parser.parse_args()

# 这一行执行:     # Load data
    # Load data
# 这一行执行:     K = None
    K = None
# 这一行执行:     if args.dataset_type == 'llff':
    if args.dataset_type == 'llff':
# 这一行执行:         images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
# 这一行执行:                                                                   recenter=True, bd_factor=.75,
                                                                  recenter=True, bd_factor=.75,
# 这一行执行:                                                                   spherify=args.spherify)
                                                                  spherify=args.spherify)
# 这一行执行:         hwf = poses[0,:3,-1]
        hwf = poses[0,:3,-1]
# 这一行执行:         poses = poses[:,:3,:4]
        poses = poses[:,:3,:4]
# 这一行执行:         print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
# 这一行执行:         if not isinstance(i_test, list):
        if not isinstance(i_test, list):
# 这一行执行:             i_test = [i_test]
            i_test = [i_test]

# 这一行执行:         if args.llffhold > 0:
        if args.llffhold > 0:
# 这一行执行:             print('Auto LLFF holdout,', args.llffhold)
            print('Auto LLFF holdout,', args.llffhold)
# 这一行执行:             i_test = np.arange(images.shape[0])[::args.llffhold]
            i_test = np.arange(images.shape[0])[::args.llffhold]

# 这一行执行:         i_val = i_test
        i_val = i_test
# 这一行执行:         i_train = np.array([i for i in np.arange(int(images.shape[0])) if
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
# 这一行执行:                         (i not in i_test and i not in i_val)])
                        (i not in i_test and i not in i_val)])

# 这一行执行:         print('DEFINING BOUNDS')
        print('DEFINING BOUNDS')
# 这一行执行:         if args.no_ndc:
        if args.no_ndc:
# 这一行执行:             near = np.ndarray.min(bds) * .9
            near = np.ndarray.min(bds) * .9
# 这一行执行:             far = np.ndarray.max(bds) * 1.
            far = np.ndarray.max(bds) * 1.

# 这一行执行:         else:
        else:
# 这一行执行:             near = 0.
            near = 0.
# 这一行执行:             far = 1.
            far = 1.
# 这一行执行:         print('NEAR FAR', near, far)
        print('NEAR FAR', near, far)

# 这一行执行:     elif args.dataset_type == 'blender':
    elif args.dataset_type == 'blender':
# 这一行执行:         images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
# 这一行执行:         print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
# 这一行执行:         i_train, i_val, i_test = i_split
        i_train, i_val, i_test = i_split

# 这一行执行:         near = 2.
        near = 2.
# 这一行执行:         far = 6.
        far = 6.

# 这一行执行:         if args.white_bkgd:
        if args.white_bkgd:
# 这一行执行:             images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
# 这一行执行:         else:
        else:
# 这一行执行:             images = images[...,:3]
            images = images[...,:3]

# 这一行执行:     elif args.dataset_type == 'LINEMOD':
    elif args.dataset_type == 'LINEMOD':
# 这一行执行:         images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
# 这一行执行:         print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
# 这一行执行:         print(f'[CHECK HERE] near: {near}, far: {far}.')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
# 这一行执行:         i_train, i_val, i_test = i_split
        i_train, i_val, i_test = i_split

# 这一行执行:         if args.white_bkgd:
        if args.white_bkgd:
# 这一行执行:             images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
# 这一行执行:         else:
        else:
# 这一行执行:             images = images[...,:3]
            images = images[...,:3]

# 这一行执行:     elif args.dataset_type == 'deepvoxels':
    elif args.dataset_type == 'deepvoxels':

# 这一行执行:         images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
# 这一行执行:                                                                  basedir=args.datadir,
                                                                 basedir=args.datadir,
# 这一行执行:                                                                  testskip=args.testskip)
                                                                 testskip=args.testskip)

# 这一行执行:         print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
# 这一行执行:         i_train, i_val, i_test = i_split
        i_train, i_val, i_test = i_split

# 这一行执行:         hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
# 这一行执行:         near = hemi_R-1.
        near = hemi_R-1.
# 这一行执行:         far = hemi_R+1.
        far = hemi_R+1.

# 这一行执行:     else:
    else:
# 这一行执行:         print('Unknown dataset type', args.dataset_type, 'exiting')
        print('Unknown dataset type', args.dataset_type, 'exiting')
# 这一行执行:         return
        return

# 这一行执行:     # Cast intrinsics to right types
    # Cast intrinsics to right types
# 这一行执行:     H, W, focal = hwf
    H, W, focal = hwf
# 这一行执行:     H, W = int(H), int(W)
    H, W = int(H), int(W)
# 这一行执行:     hwf = [H, W, focal]
    hwf = [H, W, focal]

# 这一行执行:     if K is None:
    if K is None:
# 这一行执行:         K = np.array([
        K = np.array([
# 这一行执行:             [focal, 0, 0.5*W],
            [focal, 0, 0.5*W],
# 这一行执行:             [0, focal, 0.5*H],
            [0, focal, 0.5*H],
# 这一行执行:             [0, 0, 1]
            [0, 0, 1]
# 这一行执行:         ])
        ])

# 这一行执行:     if args.render_test:
    if args.render_test:
# 这一行执行:         render_poses = np.array(poses[i_test])
        render_poses = np.array(poses[i_test])

# 这一行执行:     # Create log dir and copy the config file
    # Create log dir and copy the config file
# 这一行执行:     basedir = args.basedir
    basedir = args.basedir
# 这一行执行:     expname = args.expname
    expname = args.expname
# 这一行执行:     os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
# 这一行执行:     f = os.path.join(basedir, expname, 'args.txt')
    f = os.path.join(basedir, expname, 'args.txt')
# 这一行执行:     with open(f, 'w') as file:
    with open(f, 'w') as file:
# 这一行执行:         for arg in sorted(vars(args)):
        for arg in sorted(vars(args)):
# 这一行执行:             attr = getattr(args, arg)
            attr = getattr(args, arg)
# 这一行执行:             file.write('{} = {}\n'.format(arg, attr))
            file.write('{} = {}\n'.format(arg, attr))
# 这一行执行:     if args.config is not None:
    if args.config is not None:
# 这一行执行:         f = os.path.join(basedir, expname, 'config.txt')
        f = os.path.join(basedir, expname, 'config.txt')
# 这一行执行:         with open(f, 'w') as file:
        with open(f, 'w') as file:
# 这一行执行:             file.write(open(args.config, 'r').read())
            file.write(open(args.config, 'r').read())

# 这一行执行:     # Create nerf model
    # Create nerf model
# 这一行执行:     render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
# 这一行执行:     global_step = start
    global_step = start

# 这一行执行:     bds_dict = {
    bds_dict = {
# 这一行执行:         'near' : near,
        'near' : near,
# 这一行执行:         'far' : far,
        'far' : far,
# 这一行执行:     }
    }
# 这一行执行:     render_kwargs_train.update(bds_dict)
    render_kwargs_train.update(bds_dict)
# 这一行执行:     render_kwargs_test.update(bds_dict)
    render_kwargs_test.update(bds_dict)

# 这一行执行:     # Move testing data to GPU
    # Move testing data to GPU
# 这一行执行:     render_poses = torch.Tensor(render_poses).to(device)
    render_poses = torch.Tensor(render_poses).to(device)

# 这一行执行:     # Short circuit if only rendering out from trained model
    # Short circuit if only rendering out from trained model
# 这一行执行:     if args.render_only:
    if args.render_only:
# 这一行执行:         print('RENDER ONLY')
        print('RENDER ONLY')
# 这一行执行:         with torch.no_grad():
        with torch.no_grad():
# 这一行执行:             if args.render_test:
            if args.render_test:
# 这一行执行:                 # render_test switches to test poses
                # render_test switches to test poses
# 这一行执行:                 images = images[i_test]
                images = images[i_test]
# 这一行执行:             else:
            else:
# 这一行执行:                 # Default is smoother render_poses path
                # Default is smoother render_poses path
# 这一行执行:                 images = None
                images = None

# 这一行执行:             testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
# 这一行执行:             os.makedirs(testsavedir, exist_ok=True)
            os.makedirs(testsavedir, exist_ok=True)
# 这一行执行:             print('test poses shape', render_poses.shape)
            print('test poses shape', render_poses.shape)

# 这一行执行:             rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
# 这一行执行:             print('Done rendering', testsavedir)
            print('Done rendering', testsavedir)
# 这一行执行:             imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

# 这一行执行:             return
            return

# 这一行执行:     # Prepare raybatch tensor if batching random rays
    # Prepare raybatch tensor if batching random rays
# 这一行执行:     N_rand = args.N_rand
    N_rand = args.N_rand
# 这一行执行:     use_batching = not args.no_batching
    use_batching = not args.no_batching
# 这一行执行:     if use_batching:
    if use_batching:
# 这一行执行:         # For random ray batching
        # For random ray batching
# 这一行执行:         print('get rays')
        print('get rays')
# 这一行执行:         rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
# 这一行执行:         print('done, concats')
        print('done, concats')
# 这一行执行:         rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
# 这一行执行:         rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
# 这一行执行:         rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
# 这一行执行:         rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
# 这一行执行:         rays_rgb = rays_rgb.astype(np.float32)
        rays_rgb = rays_rgb.astype(np.float32)
# 这一行执行:         print('shuffle rays')
        print('shuffle rays')
# 这一行执行:         np.random.shuffle(rays_rgb)
        np.random.shuffle(rays_rgb)

# 这一行执行:         print('done')
        print('done')
# 这一行执行:         i_batch = 0
        i_batch = 0

# 这一行执行:     # Move training data to GPU
    # Move training data to GPU
# 这一行执行:     if use_batching:
    if use_batching:
# 这一行执行:         images = torch.Tensor(images).to(device)
        images = torch.Tensor(images).to(device)
# 这一行执行:     poses = torch.Tensor(poses).to(device)
    poses = torch.Tensor(poses).to(device)
# 这一行执行:     if use_batching:
    if use_batching:
# 这一行执行:         rays_rgb = torch.Tensor(rays_rgb).to(device)
        rays_rgb = torch.Tensor(rays_rgb).to(device)


# 这一行执行:     N_iters = 200000 + 1
    N_iters = 200000 + 1
# 这一行执行:     print('Begin')
    print('Begin')
# 这一行执行:     print('TRAIN views are', i_train)
    print('TRAIN views are', i_train)
# 这一行执行:     print('TEST views are', i_test)
    print('TEST views are', i_test)
# 这一行执行:     print('VAL views are', i_val)
    print('VAL views are', i_val)

# 这一行执行:     # Summary writers
    # Summary writers
# 这一行执行:     # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

# 这一行执行:     start = start + 1
    start = start + 1
# 这一行执行:     for i in trange(start, N_iters):
    for i in trange(start, N_iters):
# 这一行执行:         time0 = time.time()
        time0 = time.time()

# 这一行执行:         # Sample random ray batch
        # Sample random ray batch
# 这一行执行:         if use_batching:
        if use_batching:
# 这一行执行:             # Random over all images
            # Random over all images
# 这一行执行:             batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
# 这一行执行:             batch = torch.transpose(batch, 0, 1)
            batch = torch.transpose(batch, 0, 1)
# 这一行执行:             batch_rays, target_s = batch[:2], batch[2]
            batch_rays, target_s = batch[:2], batch[2]

# 这一行执行:             i_batch += N_rand
            i_batch += N_rand
# 这一行执行:             if i_batch >= rays_rgb.shape[0]:
            if i_batch >= rays_rgb.shape[0]:
# 这一行执行:                 print("Shuffle data after an epoch!")
                print("Shuffle data after an epoch!")
# 这一行执行:                 rand_idx = torch.randperm(rays_rgb.shape[0])
                rand_idx = torch.randperm(rays_rgb.shape[0])
# 这一行执行:                 rays_rgb = rays_rgb[rand_idx]
                rays_rgb = rays_rgb[rand_idx]
# 这一行执行:                 i_batch = 0
                i_batch = 0

# 这一行执行:         else:
        else:
# 这一行执行:             # Random from one image
            # Random from one image
# 这一行执行:             img_i = np.random.choice(i_train)
            img_i = np.random.choice(i_train)
# 这一行执行:             target = images[img_i]
            target = images[img_i]
# 这一行执行:             target = torch.Tensor(target).to(device)
            target = torch.Tensor(target).to(device)
# 这一行执行:             pose = poses[img_i, :3,:4]
            pose = poses[img_i, :3,:4]

# 这一行执行:             if N_rand is not None:
            if N_rand is not None:
# 这一行执行:                 rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

# 这一行执行:                 if i < args.precrop_iters:
                if i < args.precrop_iters:
# 这一行执行:                     dH = int(H//2 * args.precrop_frac)
                    dH = int(H//2 * args.precrop_frac)
# 这一行执行:                     dW = int(W//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
# 这一行执行:                     coords = torch.stack(
                    coords = torch.stack(
# 这一行执行:                         torch.meshgrid(
                        torch.meshgrid(
# 这一行执行:                             torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
# 这一行执行:                             torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
# 这一行执行:                         ), -1)
                        ), -1)
# 这一行执行:                     if i == start:
                    if i == start:
# 这一行执行:                         print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
# 这一行执行:                 else:
                else:
# 这一行执行:                     coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

# 这一行执行:                 coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
# 这一行执行:                 select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
# 这一行执行:                 select_coords = coords[select_inds].long()  # (N_rand, 2)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
# 这一行执行:                 rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
# 这一行执行:                 rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
# 这一行执行:                 batch_rays = torch.stack([rays_o, rays_d], 0)
                batch_rays = torch.stack([rays_o, rays_d], 0)
# 这一行执行:                 target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

# 这一行执行:         #####  Core optimization loop  #####
        #####  Core optimization loop  #####
# 这一行执行:         rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
# 这一行执行:                                                 verbose=i < 10, retraw=True,
                                                verbose=i < 10, retraw=True,
# 这一行执行:                                                 **render_kwargs_train)
                                                **render_kwargs_train)

# 这一行执行:         optimizer.zero_grad()
        optimizer.zero_grad()
# 这一行执行:         img_loss = img2mse(rgb, target_s)
        img_loss = img2mse(rgb, target_s)
# 这一行执行:         trans = extras['raw'][...,-1]
        trans = extras['raw'][...,-1]
# 这一行执行:         loss = img_loss
        loss = img_loss
# 这一行执行:         psnr = mse2psnr(img_loss)
        psnr = mse2psnr(img_loss)

# 这一行执行:         if 'rgb0' in extras:
        if 'rgb0' in extras:
# 这一行执行:             img_loss0 = img2mse(extras['rgb0'], target_s)
            img_loss0 = img2mse(extras['rgb0'], target_s)
# 这一行执行:             loss = loss + img_loss0
            loss = loss + img_loss0
# 这一行执行:             psnr0 = mse2psnr(img_loss0)
            psnr0 = mse2psnr(img_loss0)

# 这一行执行:         loss.backward()
        loss.backward()
# 这一行执行:         optimizer.step()
        optimizer.step()

# 这一行执行:         # NOTE: IMPORTANT!
        # NOTE: IMPORTANT!
# 这一行执行:         ###   update learning rate   ###
        ###   update learning rate   ###
# 这一行执行:         decay_rate = 0.1
        decay_rate = 0.1
# 这一行执行:         decay_steps = args.lrate_decay * 1000
        decay_steps = args.lrate_decay * 1000
# 这一行执行:         new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
# 这一行执行:         for param_group in optimizer.param_groups:
        for param_group in optimizer.param_groups:
# 这一行执行:             param_group['lr'] = new_lrate
            param_group['lr'] = new_lrate
# 这一行执行:         ################################
        ################################

# 这一行执行:         dt = time.time()-time0
        dt = time.time()-time0
# 这一行执行:         # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
# 这一行执行:         #####           end            #####
        #####           end            #####

# 这一行执行:         # Rest is logging
        # Rest is logging
# 这一行执行:         if i%args.i_weights==0:
        if i%args.i_weights==0:
# 这一行执行:             path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
# 这一行执行:             torch.save({
            torch.save({
# 这一行执行:                 'global_step': global_step,
                'global_step': global_step,
# 这一行执行:                 'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
# 这一行执行:                 'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
# 这一行执行:                 'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
# 这一行执行:             }, path)
            }, path)
# 这一行执行:             print('Saved checkpoints at', path)
            print('Saved checkpoints at', path)

# 这一行执行:         if i%args.i_video==0 and i > 0:
        if i%args.i_video==0 and i > 0:
# 这一行执行:             # Turn on testing mode
            # Turn on testing mode
# 这一行执行:             with torch.no_grad():
            with torch.no_grad():
# 这一行执行:                 rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
# 这一行执行:             print('Done, saving', rgbs.shape, disps.shape)
            print('Done, saving', rgbs.shape, disps.shape)
# 这一行执行:             moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
# 这一行执行:             imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
# 这一行执行:             imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

# 这一行执行:             # if args.use_viewdirs:
            # if args.use_viewdirs:
# 这一行执行:             #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
# 这一行执行:             #     with torch.no_grad():
            #     with torch.no_grad():
# 这一行执行:             #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
# 这一行执行:             #     render_kwargs_test['c2w_staticcam'] = None
            #     render_kwargs_test['c2w_staticcam'] = None
# 这一行执行:             #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

# 这一行执行:         if i%args.i_testset==0 and i > 0:
        if i%args.i_testset==0 and i > 0:
# 这一行执行:             testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
# 这一行执行:             os.makedirs(testsavedir, exist_ok=True)
            os.makedirs(testsavedir, exist_ok=True)
# 这一行执行:             print('test poses shape', poses[i_test].shape)
            print('test poses shape', poses[i_test].shape)
# 这一行执行:             with torch.no_grad():
            with torch.no_grad():
# 这一行执行:                 render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
# 这一行执行:             print('Saved test set')
            print('Saved test set')



# 这一行执行:         if i%args.i_print==0:
        if i%args.i_print==0:
# 这一行执行:             tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
# 这一行执行:         """
        """
# 这一行执行:             print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
# 这一行执行:             print('iter time {:.05f}'.format(dt))
            print('iter time {:.05f}'.format(dt))

# 这一行执行:             with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
# 这一行执行:                 tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('loss', loss)
# 这一行执行:                 tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.scalar('psnr', psnr)
# 这一行执行:                 tf.contrib.summary.histogram('tran', trans)
                tf.contrib.summary.histogram('tran', trans)
# 这一行执行:                 if args.N_importance > 0:
                if args.N_importance > 0:
# 这一行执行:                     tf.contrib.summary.scalar('psnr0', psnr0)
                    tf.contrib.summary.scalar('psnr0', psnr0)


# 这一行执行:             if i%args.i_img==0:
            if i%args.i_img==0:

# 这一行执行:                 # Log a rendered validation view to Tensorboard
                # Log a rendered validation view to Tensorboard
# 这一行执行:                 img_i=np.random.choice(i_val)
                img_i=np.random.choice(i_val)
# 这一行执行:                 target = images[img_i]
                target = images[img_i]
# 这一行执行:                 pose = poses[img_i, :3,:4]
                pose = poses[img_i, :3,:4]
# 这一行执行:                 with torch.no_grad():
                with torch.no_grad():
# 这一行执行:                     rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
# 这一行执行:                                                         **render_kwargs_test)
                                                        **render_kwargs_test)

# 这一行执行:                 psnr = mse2psnr(img2mse(rgb, target))
                psnr = mse2psnr(img2mse(rgb, target))

# 这一行执行:                 with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

# 这一行执行:                     tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
# 这一行执行:                     tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
# 这一行执行:                     tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

# 这一行执行:                     tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.scalar('psnr_holdout', psnr)
# 这一行执行:                     tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


# 这一行执行:                 if args.N_importance > 0:
                if args.N_importance > 0:

# 这一行执行:                     with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
# 这一行执行:                         tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
# 这一行执行:                         tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
# 这一行执行:                         tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
# 这一行执行:         """
        """

# 这一行执行:         global_step += 1
        global_step += 1


# 这一行执行: if __name__=='__main__':
if __name__=='__main__':
# 这一行执行:     torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# 这一行执行:     train()
    train()
