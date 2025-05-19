import copy
import torch

from utils.config_utils import GlobalState, get_instance_from_config, config
from utils.matrix_utils import create_camera_plane, quaternion_multiply, quaternion_to_matrix, quaternion_to_rotation, quaternion_translation_inverse

import os
from errno import EEXIST
from plyfile import PlyData, PlyElement
import numpy as np

from gsplat.rendering import rasterization, rasterization_2dgs

if GlobalState["dim_mode"].lower() == '2d':
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
elif GlobalState["dim_mode"].lower() == '3d':
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def render(cameras, GS_params):
    render_config = config["render"]
    render_params = render_config["params"]
        
    # batch rendering
    if GlobalState["dim_mode"].lower() == '3d':
        rets = render_gsplat_3d(cameras, GS_params, render_params)
        
    else:
        if render_config["implementation"] == "gsplat":
            rets = render_gsplat_2d(cameras, GS_params, render_params)
            
        elif render_config["implementation"] == "official":
            rets = render_official_2d(cameras, GS_params, render_params)
        
    return rets


def render_gsplat_3d(cameras, GS_params, render_params):
    viewmats = cameras.w2c.float()
    Ks = cameras.K.float()
    color_list = []
    alpha_list = []
    for b in range(GS_params["xyz"].shape[0]):
        colors, alphas, meta = rasterization(
            GS_params["xyz"][b].float(), 
            GS_params["rotation"][b].float(), 
            GS_params["scale"][b].float(), 
            GS_params["opacity"][b, ..., 0].float(), 
            GS_params["features"][b].float(), 
            viewmats=viewmats[b:b+1],
            Ks=Ks[b:b+1],
            width=cameras.width, 
            height=cameras.height,
            sh_degree=GS_params["sh_degree"],
            render_mode="RGB+ED",
            **render_params
            )
        color_list.append(colors)
        alpha_list.append(alphas)
    colors = torch.cat(color_list, dim=0).permute(0, 3, 1, 2)
    alphas = torch.cat(alpha_list, dim=0).permute(0, 3, 1, 2)

    rets =  {
        "render": colors[:, :-1],
        "surf_depth": colors[:, -1:],
        "rend_alpha": alphas,
        }
    return rets
    
def render_gsplat_2d(cameras, GS_params, render_params):
    viewmats = cameras.w2c.float()
    Ks = cameras.K.float()
    color_list = []
    alpha_list = []
    for b in range(GS_params["xyz"].shape[0]):
        colors, alphas, normals, surf_normals, distort, median_depth, meta = rasterization_2dgs(
            GS_params["xyz"][b].float(),
            GS_params["rotation"][b].float(),
            torch.cat([GS_params["scale"][b], torch.zeros_like(GS_params["scale"][b][..., :1])], dim=-1).float(),
            GS_params["opacity"][b, ..., 0].float(),
            GS_params["features"][b].float(),
            viewmats=viewmats[b:b+1],
            Ks=Ks[b:b+1],
            width=cameras.width,
            height=cameras.height,
            sh_degree=GS_params["sh_degree"],
            render_mode="RGB+ED",
            **render_params
        )
        color_list.append(colors)
        alpha_list.append(alphas)
    colors = torch.cat(color_list, dim=0).permute(0, 3, 1, 2)
    alphas = torch.cat(alpha_list, dim=0).permute(0, 3, 1, 2)

    rets =  {
        "render": colors[:, :-1],
        "surf_depth": colors[:, -1:],
        "rend_alpha": alphas,
        }
    return rets

def render_official_2d(cameras, GS_params, render_params):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(GS_params["xyz"][0], dtype=torch.float32, requires_grad=True) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    bg_color = torch.zeros(3, device="cuda")
    
    # Prepare camera params
    camera_params = {}
    camera_params["tanhalffovx"] = cameras.tanhalffovx.float()
    camera_params["tanhalffovy"] = cameras.tanhalffovy.float()
    camera_params["width"] = cameras.width
    camera_params["height"] = cameras.height
    camera_params["world_view_transform"] = cameras.world_view_transform.contiguous().float()
    camera_params["full_proj_transform"] = cameras.full_proj_transform.contiguous().float()
    camera_params["camera_center"] = cameras.t.contiguous().float()

    rendered_image_list = []
    radii_list = []
    allmap_list = []
    for b in range(GS_params["xyz"].shape[0]):
        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera_params["height"]),
            image_width=int(camera_params["width"]),
            tanfovx=camera_params["tanhalffovx"][b].item(),
            tanfovy=camera_params["tanhalffovy"][b].item(),
            bg=bg_color,
            scale_modifier=render_params.get("scale_modifier", 1.0),
            viewmatrix=camera_params["world_view_transform"][b],
            projmatrix=camera_params["full_proj_transform"][b],
            sh_degree=GS_params["sh_degree"],
            campos=camera_params["camera_center"][b],
            prefiltered=render_params.get("prefiltered", False),
            debug=render_params.get("debug", False),
        )
        
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        means3D = GS_params["xyz"][b].contiguous().float()
        means2D = screenspace_points
        opacity = GS_params["opacity"][b].contiguous().float()
        
        scales = None
        rotations = None
        cov3D_precomp = None
        if True: # diff of Rt and K
            # currently don't support normal consistency loss if use precomputed covariance
            splat2world = build_covariance_from_scaling_rotation(GS_params["xyz"][b], GS_params["scale"][b], render_params.get("scale_modifier", 1.0), GS_params["rotation"][b])
            W, H = cameras.width, cameras.height
            near, far = cameras.znear, cameras.zfar
            ndc2pix = torch.tensor([
                [W / 2, 0., 0., (W-1) / 2],
                [0., H / 2, 0., (H-1) / 2],
                [0., 0., far-near, near],
                [0., 0., 0., 1.]], device=GS_params["xyz"].device).T
            world2pix =  camera_params["full_proj_transform"][b] @ ndc2pix
            cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9).contiguous().float() # column major
        else:
            scales = GS_params["scale"][b].contiguous().float()
            rotations = GS_params["rotation"][b].contiguous().float()
        
        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        shs = GS_params["features"][b].contiguous().float()

        rendered_image, radii, allmap = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp
        )
        rendered_image_list.append(rendered_image)
        radii_list.append(radii)
        allmap_list.append(allmap)

    rendered_image = torch.stack(rendered_image_list, dim=0)
    radii = torch.stack(radii_list, dim=0)
    allmap = torch.stack(allmap_list, dim=0)
    
    rets =  {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
    }
    
    # additional regularizations
    render_alpha = allmap[:, 1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[:, 2:5]
    render_normal = (render_normal.permute(0, 2, 3, 1) @ (cameras.world_view_transform[:, None, :3, :3].transpose(-1, -2))).permute(0, 3, 1, 2)
    
    # get median depth map
    render_depth_median = allmap[:, 5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[:, 0:1]
    render_depth_expected = (render_depth_expected / render_alpha.clamp(1e-5))
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[:, 6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1 - render_params.get("depth_ratio", 0.)) + render_params.get("depth_ratio", 0.) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal, surf_point = depth_to_normal(cameras, surf_depth)
    surf_normal = surf_normal.permute(0, 3, 1, 2)
    surf_point = surf_point.permute(0, 3, 1, 2)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * render_alpha.detach()

    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
            'surf_point': surf_point,
    })
    
    return rets

class CameraOptimizer:
    def __init__(self, **config):
        self.config = config
        self.losses = []
        for loss_name, loss_config in config["losses"].items():
            loss_weight = loss_config["weight"]
            if loss_weight > 0.:
                loss_function = get_instance_from_config(loss_config)
                self.losses.append((loss_name, loss_function, loss_weight))
                
    def __call__(self, target, cameras, GS_params):
        cameras.t = torch.nn.Parameter(cameras.t)
        cameras.quaternion = torch.nn.Parameter(cameras.quaternion)
        self.optimizer = get_instance_from_config(self.config["optimizer"], [cameras.t, cameras.quaternion])
        
        inputs = {"video_tensor": target.unsqueeze(1),
                  "rets_dict": {}}
        with torch.enable_grad():
            for i in range(self.config["n_iter"]):
                rendered = render(cameras, GS_params)
                inputs["rets_dict"][("all", 0)] = rendered
                loss = 0.
                for loss_name, loss_function, loss_weight in self.losses:
                    loss += loss_weight * loss_function(inputs)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
        return cameras

def gs_cat(gs_list: list, dim=1):
    if len(gs_list) == 1:
        return gs_list[0]

    sh_degree = gs_list[0]["sh_degree"]
    assert all(sh_degree == gs["sh_degree"] for gs in gs_list), "sh_degree in gs_cat is not same."
    GS_params = {}
    GS_params["sh_degree"] = sh_degree
    GS_params["scale"] = torch.cat([gs["scale"] for gs in gs_list], dim=dim)
    GS_params["opacity"] = torch.cat([gs["opacity"] for gs in gs_list], dim=dim)
    GS_params["features"] = torch.cat([gs["features"] for gs in gs_list], dim=dim)
    GS_params["xyz"] = torch.cat([gs["xyz"] for gs in gs_list], dim=dim)
    GS_params["rotation"] = torch.cat([gs["rotation"] for gs in gs_list], dim=dim)
    return GS_params

def gs_trans(GS_params, q, t):
    # GS_params: B, N, 3/4
    # q: B, 4, t: B, 3
    R = quaternion_to_matrix(q)
    GS_params["xyz"] = GS_params["xyz"] @ R.transpose(-1, -2) + t.unsqueeze(1)
    GS_params["rotation"] = quaternion_multiply(q.unsqueeze(1), GS_params["rotation"])
    # TODO feature transformation
    return GS_params

def build_scaling_rotation(s, r):
    R = quaternion_to_rotation(r)
    L = torch.zeros_like(R)

    L[..., 0, 0] = s[..., 0]
    L[..., 1, 1] = s[..., 1]
    L[..., 2, 2] = s[..., 2]

    L = R @ L
    return L

def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
    RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0, 2, 1)
    trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device=center.device)
    trans[:,:3,:3] = RS
    trans[:, 3,:3] = center
    trans[:, 3, 3] = 1
    return trans

def depths_to_points(view, depthmap):
    # c2w = view.c2w
    W, H = view.width, view.height
    # fx = view.fx
    # fy = view.fy
    # intrins = torch.tensor(
    #     [[fx, 0., W/2.],
    #     [0., fy, H/2.],
    #     [0., 0., 1.0]]
    # ).float().cuda()
    # grid_x, grid_y = torch.meshgrid(torch.arange(W, device="cuda"), torch.arange(H, device="cuda"), indexing='xy')
    # points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3).float()
    # rays_d = (points @ intrins.inverse().T).unsqueeze(0) @ c2w[:, :3,:3].permute(0, 2, 1)
    points = create_camera_plane(view) # B, 3, H, W
    rays_d = view.R @ points.reshape(points.shape[0], 3, -1) # B, 3, H*W
    rays_d = rays_d.permute(0, 2, 1) # B, H*W, 3
    rays_o = view.t.unsqueeze(1)
    points = depthmap.reshape(depthmap.shape[0], H*W, 1) * rays_d + rays_o
    return points.reshape(depthmap.shape[0], H, W, 3)

def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth)
    output = torch.zeros_like(points)
    # dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    # dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    dx = points[..., 2:, 1:-1, :] - points[..., :-2, 1:-1, :]
    dy = points[..., 1:-1, 2:, :] - points[..., 1:-1, :-2, :]
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[..., 1:-1, 1:-1, :] = normal_map
    return output, points

def map_to_GS(param_map):
    """
    param_map: B, N, F, H, W
    output: B, N*H*W, F
    """
    B, N, F, H, W = param_map.shape
    return param_map.permute(0, 1, 3, 4, 2).reshape(B, N*H*W, F)

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5


def construct_list_of_attributes(rest_sh_dim):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(3 * 1):
        l.append('f_dc_{}'.format(i))
    for i in range(3 * rest_sh_dim):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(2):
        l.append('scale_{}'.format(i))
    for i in range(4):
        l.append('rot_{}'.format(i))
    return l

def save_ply(gs_params, ply_path):
    folder_path = os.path.dirname(ply_path)

    try:
        os.makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and os.path.isdir(folder_path):
            pass
        else:
            raise

    assert gs_params["xyz"].ndim == 3, "B, N, 3"
    B = gs_params["xyz"].shape[0]
    xyz = gs_params["xyz"].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = gs_params["features"][:, :, 0, :].detach().cpu().numpy()
    rest_sh_dim = gs_params["features"].shape[-2] - 1
    f_rest = gs_params["features"][:, :, 1:, :].detach().transpose(-2, -1).flatten(start_dim=-2).contiguous().cpu().numpy()
    opacities = gs_params["opacity"].detach().cpu().numpy()
    scale = gs_params["scale"].detach().cpu().numpy()
    rotation = gs_params["rotation"].detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(rest_sh_dim)]

    for b in range(B):
        elements = np.empty(xyz[b].shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz[b], normals[b], f_dc[b], f_rest[b], opacities[b], scale[b], rotation[b]), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(ply_path[:-4] + "_" + str(b) + ".ply")