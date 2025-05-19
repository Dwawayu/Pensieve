import numpy as np
import torch
import lpips
import torch.nn.functional as F
from pytorch3d.transforms import so3_relative_angle

def create_camera_plane(camera, pix_residual=None):
    return create_param_plane(camera.width, camera.height, camera.K_inv, camera.device, pix_residual)

def create_param_plane(width, height, K_inv, device, pix_residual=None):
    width_list = torch.arange(width, device=device) + 0.5
    height_list = torch.arange(height, device=device) + 0.5
    meshgrid = torch.meshgrid(width_list, height_list, indexing='xy')
    pix_coords = torch.stack(meshgrid, axis=0) # 2, H, W
    pix_coords = pix_coords.unsqueeze(0) # 1, 2, H, W
    if pix_residual is not None:
        pix_coords = pix_coords[(None,) * (pix_residual.ndim - pix_coords.ndim)]
        pix_coords = pix_coords + pix_residual # B, (N), 2, H, W
    ones = torch.ones(*pix_coords.shape[:-3], 1, height * width, device=device) # B, (N), 1, H, W

    pix_coords = pix_coords.reshape(*pix_coords.shape[:-2], -1)
    pix_coords = torch.cat([pix_coords, ones], dim=-2) # B, (N), 3, H*W

    K_inv = K_inv
    cam_points = K_inv[(slice(None),) * (K_inv.ndim - 2) + ((None,)*(pix_coords.ndim - K_inv.ndim))] @ pix_coords
    cam_points = cam_points.reshape(*cam_points.shape[:-2], 3, height, width)
    return cam_points

def quaternion_to_rotation(q):
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-5)

    R = torch.zeros(list(q.shape[:-1]) + [3, 3], device=q.device)

    r = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    R[..., 0, 0] = 1 - 2 * (y*y + z*z)
    R[..., 0, 1] = 2 * (x*y - r*z)
    R[..., 0, 2] = 2 * (x*z + r*y)
    R[..., 1, 0] = 2 * (x*y + r*z)
    R[..., 1, 1] = 1 - 2 * (x*x + z*z)
    R[..., 1, 2] = 2 * (y*z - r*x)
    R[..., 2, 0] = 2 * (x*z - r*y)
    R[..., 2, 1] = 2 * (y*z + r*x)
    R[..., 2, 2] = 1 - 2 * (x*x + y*y)
    return R

# TODO: There are two Q2R functions in this file. We should test if they are equivalent and remove one of them.
def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / ((quaternions * quaternions).sum(-1) + 1e-5)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

# TODO: There is a for loop in this function. Can we vectorize it?
def matrix_to_quaternion(M: torch.Tensor) -> torch.Tensor:
    """
    Matrix-to-quaternion conversion method. Equation taken from 
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    Args:
        M: rotation matrices, (... x 3 x 3)
    Returns:
        q: quaternion of shape (... x 4)
    """
    prefix_shape = M.shape[:-2]
    Ms = M.reshape(-1, 3, 3)

    trs = 1 + Ms[:, 0, 0] + Ms[:, 1, 1] + Ms[:, 2, 2]

    Qs = []

    for i in range(Ms.shape[0]):
        M = Ms[i]
        tr = trs[i]
        if tr > 0:
            r = torch.sqrt(tr) / 2.0
            x = ( M[ 2, 1] - M[ 1, 2] ) / ( 4 * r )
            y = ( M[ 0, 2] - M[ 2, 0] ) / ( 4 * r )
            z = ( M[ 1, 0] - M[ 0, 1] ) / ( 4 * r )
        elif ( M[ 0, 0] > M[ 1, 1]) and (M[ 0, 0] > M[ 2, 2]):
            S = torch.sqrt(1.0 + M[ 0, 0] - M[ 1, 1] - M[ 2, 2]) * 2 # S=4*qx 
            r = (M[ 2, 1] - M[ 1, 2]) / S
            x = 0.25 * S
            y = (M[ 0, 1] + M[ 1, 0]) / S 
            z = (M[ 0, 2] + M[ 2, 0]) / S 
        elif M[ 1, 1] > M[ 2, 2]: 
            S = torch.sqrt(1.0 + M[ 1, 1] - M[ 0, 0] - M[ 2, 2]) * 2 # S=4*qy
            r = (M[ 0, 2] - M[ 2, 0]) / S
            x = (M[ 0, 1] + M[ 1, 0]) / S
            y = 0.25 * S
            z = (M[ 1, 2] + M[ 2, 1]) / S
        else:
            S = torch.sqrt(1.0 + M[ 2, 2] - M[ 0, 0] -  M[ 1, 1]) * 2 # S=4*qz
            r = (M[ 1, 0] - M[ 0, 1]) / S
            x = (M[ 0, 2] + M[ 2, 0]) / S
            y = (M[ 1, 2] + M[ 2, 1]) / S
            z = 0.25 * S
        Q = torch.stack([r, x, y, z], dim=-1)
        Qs += [Q]

    return torch.stack(Qs, dim=0).reshape(*prefix_shape, 4)


def quaternion_multiply(q1, q2):
    r_0 = q1[..., 0] * q2[..., 0] - q1[..., 1] * q2[..., 1] - q1[..., 2] * q2[..., 2] - q1[..., 3] * q2[..., 3]
    r_1 = q1[..., 0] * q2[..., 1] + q1[..., 1] * q2[..., 0] + q1[..., 2] * q2[..., 3] - q1[..., 3] * q2[..., 2]
    r_2 = q1[..., 0] * q2[..., 2] - q1[..., 1] * q2[..., 3] + q1[..., 2] * q2[..., 0] + q1[..., 3] * q2[..., 1]
    r_3 = q1[..., 0] * q2[..., 3] + q1[..., 1] * q2[..., 2] - q1[..., 2] * q2[..., 1] + q1[..., 3] * q2[..., 0]
    return torch.stack([r_0, r_1, r_2, r_3], dim=-1)

def quaternion_translation_multiply(q1, t1, q2, t2):
    q = quaternion_multiply(q1, q2)
    R1 = quaternion_to_rotation(q1)
    t = (R1 @ t2.unsqueeze(-1)).squeeze(-1) + t1
    
    return q, t

def quaternion_inverse(q):
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)

def quaternion_translation_inverse(q, t):
    R = quaternion_to_rotation(q)
    q_inv = quaternion_inverse(q)
    t_inv = -(t.unsqueeze(-2) @ R).squeeze(-2)
    return q_inv, t_inv

def quaternion_t_to_matrix(q, t):
    R = quaternion_to_matrix(q)
    Rt = torch.cat([R, t.unsqueeze(-1)], dim=-1)
    bottom_row = torch.zeros_like(Rt[..., 0:1, :])
    bottom_row[..., -1] = 1.
    Rt = torch.cat([Rt, bottom_row], dim=-2)
    return Rt

def apply_bilgrid(bilgrid, image):
    # bilgrid: B, 12, D, H, W
    B, _, H, W = image.shape
    rgb2gray_weight = torch.tensor([0.2989, 0.5870, 0.1140], device=image.device)
    gray = (image * rgb2gray_weight[None, :, None, None]).sum(1, True) # B, 1, H, W
    
    width_list = torch.arange(W, device=image.device) / (W - 1.)
    height_list = torch.arange(H, device=image.device) / (H - 1.)
    meshgrid = torch.meshgrid(width_list, height_list, indexing='xy')
    pix_coords = torch.stack(meshgrid, axis=0) # 2, H, W
    pix_coords = pix_coords.unsqueeze(0).expand(B, -1, -1, -1) # B, 2, H, W
    
    grid_xyz = torch.cat([pix_coords, gray], dim=1) # B, 3, H, W
    grid_xyz = grid_xyz * 2. - 1.
    grid_xyz = grid_xyz.permute(0, 2, 3, 1).unsqueeze(1) # B, 1, H, W, 3

    affine_mats = F.grid_sample(bilgrid, grid_xyz, mode='bilinear', align_corners=True, padding_mode='border').squeeze(2)  # (B, 12, H, W)
    affine_mats = affine_mats.permute(0, 2, 3, 1).reshape(B, H, W, 3, 4)
    image = affine_mats[..., :3] @ image.permute(0, 2, 3, 1)[..., None] + affine_mats[..., 3:4]
    image = image.squeeze(-1).permute(0, 3, 1, 2)
    return image

def camera_list_to_dict(camera_list):
    camera_dict = {}
    camera_dict["quaternion"] = torch.stack([c.quaternion for c in camera_list], dim=1).float()
    camera_dict["t"] = torch.stack([c.t for c in camera_list], dim=1).float()
    camera_dict["fx"] = torch.stack([c.fx for c in camera_list], dim=1).float()
    camera_dict["fy"] = torch.stack([c.fy for c in camera_list], dim=1).float()
    camera_dict["width"] = camera_list[0]["width"]
    camera_dict["height"] = camera_list[0]["height"]
    if camera_list[0]._cx is not None:
        camera_dict["_cx"] = torch.stack([c._cx for c in camera_list], dim=1).float()
        camera_dict["_cy"] = torch.stack([c._cy for c in camera_list], dim=1).float()
    else:
        camera_dict["_cx"] = None
        camera_dict["_cy"] = None
    return camera_dict

def camera_dict_to_list(camera_dict):
    from utils.camera import BatchCameras
    camera_list = []
    for i in range(camera_dict["quaternion"].shape[1]):
        camera = BatchCameras()
        camera.device = camera_dict["quaternion"].device
        if isinstance(camera_dict["width"], int):
            camera.width = camera_dict["width"]
            camera.height = camera_dict["height"]
        else:
            camera.width = camera_dict["width"][0].item()
            assert (camera_dict["width"][0] == camera_dict["width"]).all()
            camera.height = camera_dict["height"][0].item()
            assert (camera_dict["height"][0] == camera_dict["height"]).all()
        camera.quaternion = camera_dict["quaternion"][:, i]
        camera.t = camera_dict["t"][:, i]
        camera.fx = camera_dict["fx"][:, i]
        camera.fy = camera_dict["fy"][:, i]
        if camera_dict["_cx"] is not None:
            camera.cx = camera_dict["_cx"][:, i]
            camera.cy = camera_dict["_cy"][:, i]
        else:
            camera.cx = None
            camera.cy = None
            
        if camera_dict.get("bilgrid") is not None:
            camera.bilgrid = camera_dict["bilgrid"][:, i]
        else:
            camera.bilgrid = None

        camera_list.append(camera)
    return camera_list

def regularize_camera_list(camera_list):
    q_0 = camera_list[0]["quaternion"]
    t_0 = camera_list[0]["t"]
    q_0_inv, t_0_inv = quaternion_translation_inverse(q_0, t_0)
    for i in range(0, len(camera_list)):
        q, t = quaternion_translation_multiply(q_0_inv, t_0_inv, camera_list[i]["quaternion"], camera_list[i]["t"])
        camera_list[i]["quaternion"] = q
        camera_list[i]["t"] = t
    return camera_list

def regularize_camera_dict(camera_dict):
    q_0 = camera_dict["quaternion"][:, 0:1]
    t_0 = camera_dict["t"][:, 0:1]
    q_0_inv, t_0_inv = quaternion_translation_inverse(q_0, t_0)
    camera_dict["quaternion"], camera_dict["t"] = quaternion_translation_multiply(q_0_inv, t_0_inv, camera_dict["quaternion"], camera_dict["t"])
    return camera_dict

def umeyama(src, dst, estimate_scale=True):
    '''
    src, dst: ..., N, D
    '''
    src = src.float()
    dst = dst.float()
    *batch_dims, num_points, dim = src.shape

    src_mean = src.mean(-2, True) # ..., 1, D
    dst_mean = dst.mean(-2, True)

    src_demean = src - src_mean # ..., N, D
    dst_demean = dst - dst_mean

    covariance_matrix = torch.matmul(dst_demean.transpose(-1, -2), src_demean) / num_points # ..., D, D

    U, S, Vh = torch.linalg.svd(covariance_matrix, full_matrices=False)

    d = torch.ones((*batch_dims, dim), dtype=src.dtype, device=src.device) # ..., D
    det = torch.linalg.det(torch.matmul(U, Vh))
    d[..., -1] = torch.where(det < 0, -1.0, 1.0)

    D = torch.diag_embed(d)  # ..., D, D

    R = torch.matmul(torch.matmul(U, D), Vh)

    if estimate_scale:
        var_src = (src_demean ** 2).sum(dim=(-2, -1)) / num_points
        scale = (S * d).sum(dim=-1) / var_src
    else:
        scale = torch.ones((*batch_dims,), dtype=src.dtype, device=src.device)

    t = dst_mean.squeeze(-2) - scale.unsqueeze(-1) * torch.matmul(R, src_mean.squeeze(-2).unsqueeze(-1)).squeeze(-1)

    T = torch.eye(dim + 1, dtype=src.dtype, device=src.device).expand(*batch_dims, dim + 1, dim + 1).clone()
    T[..., :dim, :dim] = scale.unsqueeze(-1).unsqueeze(-1) * R
    T[..., :dim, -1] = t

    return T

def align_camera_dict(pred_dict, gt_dict, mode="all"):
    if mode == "first":
        rel_q = quaternion_multiply(gt_dict["quaternion"][:, 0], quaternion_inverse(pred_dict["quaternion"][:, 0])) # B, 4
    elif mode == "all":
        rel_q = quaternion_multiply(gt_dict["quaternion"], quaternion_inverse(pred_dict["quaternion"])) # B, N, 4
        rel_q = quatWAvgMarkley(rel_q) # B, 4

    pred_t = pred_dict["t"] # B, N, 3
    pred_t_trans = pred_t @ quaternion_to_matrix(rel_q).transpose(-1, -2) # B, N, 3
    
    if mode == "first":
        scale = (gt_dict["t"] - gt_dict["t"][:, 0:1]).norm(dim=-1).mean(-1, True) / (pred_t_trans - pred_t_trans[:, 0:1]).norm(dim=-1).mean(-1, True).clamp(1e-5) # B, 1
    elif mode == "all":
        scale = (gt_dict["t"] - gt_dict["t"].mean(1, True)).norm(dim=-1).mean(-1, True) / (pred_t_trans - pred_t_trans.mean(1, True)).norm(dim=-1).mean(-1, True).clamp(1e-5) # B, 1
    
    pred_t_trans = scale[:, None] * pred_t_trans
    
    if mode == "first":
        rel_t = gt_dict["t"][:, 0] - pred_t_trans[:, 0] # B, 3
    elif mode == "all":
        rel_t = gt_dict["t"].mean(1) - pred_t_trans.mean(1) # B, 3
        
    return rel_q, scale, rel_t
    
        
def quatWAvgMarkley(Q, weights=None):
    '''
    Averaging Quaternions.

    Arguments:
        Q(ndarray):    ..., N, 4
        weights(list): ..., N
    '''
    if torch.allclose(Q, Q.mean(dim=-2, keepdim=True), atol=1e-4):
        return Q.mean(dim=-2)
    
    if weights is None:
        weights = torch.ones(Q.shape[:-1], device=Q.device, dtype=Q.dtype)
    weights = weights / weights.sum(-1, True)
    
    Q = torch.linalg.eigh(torch.einsum('...ij,...ik,...i->...jk', Q, Q, weights).float())[1][..., :, -1]
    Q = Q / (Q.norm(dim=-1, keepdim=True) + 1e-5)
    return Q

def closed_form_inverse(se3):
    """
    Computes the inverse of each 4x4 SE3 matrix in the batch.

    Args:
    - se3 (Tensor): Nx4x4 tensor of SE3 matrices.

    Returns:
    - Tensor: Nx4x4 tensor of inverted SE3 matrices.
    """
    R = se3[..., :3, :3]
    T = se3[..., 3:, :3]

    # Compute the transpose of the rotation
    R_transposed = R.transpose(-1, -2)

    # Compute the left part of the inverse transformation
    left_bottom = -T @ R_transposed
    left_combined = torch.cat((R_transposed, left_bottom), dim=-2)

    # Keep the right-most column as it is
    right_col = se3[..., 3:].detach().clone()
    inverted_matrix = torch.cat((left_combined, right_col), dim=-1)

    return inverted_matrix


def rotation_angle(rot_gt, rot_pred):
    # rot_gt, rot_pred (B, 3, 3)
    *batch, T1, T2 = rot_gt.shape
    rel_angle_cos = so3_relative_angle(rot_gt.reshape(-1, T1, T2), rot_pred.reshape(-1, T1, T2), eps=1e-4).reshape(*batch)
    rel_rangle_deg = rel_angle_cos * 180 / np.pi
    return rel_rangle_deg

def translation_angle(tvec_gt, tvec_pred):
    # tvec_gt, tvec_pred (B, 3,)
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi
    return rel_tangle_deg

def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e5):
    """Normalize the translation vectors and compute the angle between them."""
    t_norm = torch.norm(t, dim=-1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=-1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=-1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t

def linear_insert_camera(camera_1, camera_2, num_frames):
    from utils.camera import BatchCameras
    assert camera_1.width == camera_2.width
    assert camera_1.height == camera_2.height
    
    camera_list = [camera_1]
    for i in range(1, num_frames):
        alpha = i / num_frames
        camera = BatchCameras()
        
        camera.width = camera_1.width
        camera.height = camera_1.height
        camera.fx = camera_1.fx * (1 - alpha) + camera_2.fx * alpha
        camera.fy = camera_1.fy * (1 - alpha) + camera_2.fy * alpha
        if camera_1._cx is not None and camera_2._cx is not None:
            camera._cx = camera_1._cx * (1 - alpha) + camera_2._cx * alpha
            camera._cy = camera_1._cy * (1 - alpha) + camera_2._cy * alpha
        camera.t = camera_1.t * (1 - alpha) + camera_2.t * alpha
        q_weight = torch.tensor([[1 - alpha, alpha]], device=camera_1.quaternion.device)
        camera.quaternion = quatWAvgMarkley(torch.stack([camera_1.quaternion, camera_2.quaternion], dim=-2), q_weight)
        if camera_1.bilgrid is not None and camera_2.bilgrid is not None:
            camera.bilgrid = camera_1.bilgrid * (1 - alpha) + camera_2.bilgrid * alpha
        
        camera.device = camera.quaternion.device
        camera_list.append(camera)
    camera_list.append(camera_2)
    return camera_list
    
def axis_angle_to_quaternion(axis, angle):
    '''
    axis: ..., 3
    angle: ..., 1
    '''
    axis = axis / axis.norm(dim=-1, keepdim=True)
    half_angle = angle / 2.0
    sin_half_angle = torch.sin(half_angle)
    cos_half_angle = torch.cos(half_angle)
    return torch.cat([cos_half_angle, axis * sin_half_angle], dim=-1)
    
def camera_translate_cycle(camera, center, axis, num_frames):
    '''
    center and axis should be in the camera coordinate system
    '''
    c2w_R = quaternion_to_matrix(camera.quaternion)
    c2w_t = camera.t
    if isinstance(center, list):
        center = torch.tensor(center, device=camera.quaternion.device).float().unsqueeze(0)
    center_w = (c2w_R @ center.unsqueeze(-1)).squeeze(-1) + c2w_t
    if isinstance(axis, list):
        axis = torch.tensor(axis, device=camera.quaternion.device).float().unsqueeze(0)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    axis_w = (c2w_R @ axis.unsqueeze(-1)).squeeze(-1)
    
    from utils.camera import BatchCameras
    camera_list = [camera]
    for i in range(1, num_frames):
        alpha = 2 * torch.pi * i / num_frames * torch.ones_like(axis_w[..., 0:1])
        camera = BatchCameras()
        camera.width = camera_list[0].width
        camera.height = camera_list[0].height
        camera.fx = camera_list[0].fx
        camera.fy = camera_list[0].fy
        if camera_list[0]._cx is not None:
            camera._cx = camera_list[0]._cx
            camera._cy = camera_list[0]._cy
        camera.quaternion = camera_list[0].quaternion
        
        rotation_quaternion = axis_angle_to_quaternion(axis_w, alpha)
        t = camera_list[0].t - center_w
        t = (quaternion_to_matrix(rotation_quaternion) @ t.unsqueeze(-1)).squeeze(-1) + center_w
        camera.t = t
        
        camera.device = camera.quaternion.device
        camera_list.append(camera)
    camera_list.append(camera_list[0])
    return camera_list

def generate_camera_trajectory_demo(camera_list, t_N=10, cycle_N=30):
    middle_idx = len(camera_list) // 2
    results = []
    for i in range(len(camera_list)-1):
        if i == middle_idx:
            norm = torch.norm(camera_list[i].t - camera_list[i+1].t, dim=-1, keepdim=True) / 2
            axis = torch.zeros_like(camera_list[i].t)
            axis[..., -1] = 1
            center = torch.zeros_like(camera_list[i].t)
            center[..., 0] = -1
            results += camera_translate_cycle(camera_list[i], center*norm, axis, cycle_N)
            center = torch.zeros_like(camera_list[i].t)
            center[..., 0] = 1
            results += camera_translate_cycle(camera_list[i], center*norm, axis, cycle_N)
            center = torch.zeros_like(camera_list[i].t)
            center[..., 1] = -1
            results += camera_translate_cycle(camera_list[i], center*norm, axis, cycle_N)
            center = torch.zeros_like(camera_list[i].t)
            center[..., 1] = 1
            results += camera_translate_cycle(camera_list[i], center*norm, axis, cycle_N)
        results += linear_insert_camera(camera_list[i], camera_list[i+1], t_N)
    return results