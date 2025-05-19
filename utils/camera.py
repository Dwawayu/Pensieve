import torch
from utils.matrix_utils import create_camera_plane, quaternion_inverse, quaternion_multiply, quaternion_to_matrix, quaternion_to_rotation

class Camera:
    def __init__(self):
        self.device = None
        
        # int
        self.width = None
        self.height = None
        
        # [] or [B]
        self._cx = None
        self._cy = None
        self.fx = None
        self.fy = None
        
        self._quaternion = None # [4] or [B, 4]
        self.t = None # [3] or [B, 3]
        
        self.zfar = 100.0
        self.znear = 0.01
        
        self.bilgrid = None
    
    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except:
            raise KeyError(f"{key} does not exist")
        
    @property
    def quaternion(self):
        self._quaternion = self._quaternion / (torch.norm(self._quaternion, dim=-1, keepdim=True)+1e-5)
        return self._quaternion
    @quaternion.setter
    def quaternion(self, value):
        self._quaternion = value

    @property
    def cx(self):
        if self._cx is None:
            return self.width / 2 * torch.ones_like(self.fx)
        return self._cx
    @cx.setter
    def cx(self, value):
        self._cx = value

    @property
    def cy(self):
        if self._cy is None:
            return self.height / 2 * torch.ones_like(self.fy)
        return self._cy
    @cy.setter
    def cy(self, value):
        self._cy = value
        
    # we recompute the following value to maintain the compute graph

    @property
    def K(self):
        if isinstance(self, BatchCameras):
            K = torch.zeros([self.fx.shape[0], 3, 3], device=self.device)
        else:
            K = torch.zeros([3, 3], device=self.device)
        K[..., 0, 0] = self.fx
        K[..., 1, 1] = self.fy
        K[..., 0, 2] = self.cx
        K[..., 1, 2] = self.cy
        K[..., 2, 2] = 1.
        self._K = K
        return K
        
    @property
    def K_inv(self):
        return torch.inverse(self.K)
        
    @property
    def fovx(self):
        return 2 * torch.atan(self.width / (2 * self.fx))
    
    @property
    def fovy(self):
        return 2 * torch.atan(self.height / (2 * self.fy))
    
    @property
    def tanhalffovx(self):
        return self.width / (2 * self.fx)
    
    @property
    def tanhalffovy(self):
        return self.height / (2 * self.fy)
    
    def resize_(self, width, height):
        ratio_x = width / self.width
        ratio_y = height / self.height
        self.width = width
        self.height = height
        if self._cx is not None:
            self._cx *= ratio_x
        if self._cy is not None:
            self._cy *= ratio_y
        if self.fx is not None:
            self.fx *= ratio_x
        if self.fy is not None:
            self.fy *= ratio_y
        return self
    
    def resize(self, width, height):
        resized_camera = self.__class__()
        resized_camera.width = width
        resized_camera.height = height
        resized_camera._quaternion = self._quaternion
        resized_camera.t = self.t
        resized_camera.bilgrid = self.bilgrid
        resized_camera.device = self.device
        if self._cx is not None:
            resized_camera._cx = self._cx * width / self.width
        if self._cy is not None:
            resized_camera._cy = self._cy * height / self.height
        if self.fx is not None:
            resized_camera.fx = self.fx * width / self.width
        if self.fy is not None:
            resized_camera.fy = self.fy * height / self.height
        
        return resized_camera
        
    @property
    def R(self):
        return quaternion_to_matrix(self.quaternion)
    
    @property
    def R_inv(self):
        return self.R.inverse()

    @property
    def Rt(self):
        Rt = torch.cat([self.R, self.t.unsqueeze(-1)], -1) # [3, 4] or [B, 3, 4]
        bottom_row = torch.zeros_like(Rt[..., 0:1, :]) # [1, 4] or [B, 1, 4]
        bottom_row[..., -1] = 1.
        Rt = torch.cat([Rt, bottom_row], dim=-2) # [4, 4] or [B, 4, 4]
        return Rt
    @property
    def c2w(self):
        return self.Rt
    @property
    def w2c(self):
        return self.c2w.inverse()

    @property
    def world_view_transform(self):
        # defined as GS
        return self.w2c.transpose(-1, -2)
        
    @property
    def full_proj_transform(self):
        # defined as GS
        return self.world_view_transform @ self.P.transpose(-1, -2)
        
    @property
    def P(self):
        # defined as GS
        tanHalfFovX = self.tanhalffovx
        tanHalfFovY = self.tanhalffovy
        znear = self.znear
        zfar = self.zfar

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        if isinstance(self, BatchCameras):
            P = torch.zeros([self.fx.shape[0], 4, 4], device=self.device)
        else:
            P = torch.zeros([4, 4], device=self.device)

        z_sign = 1.0

        P[..., 0, 0] = 2.0 * znear / (right - left)
        P[..., 1, 1] = 2.0 * znear / (top - bottom)
        P[..., 0, 2] = (right + left) / (right - left)
        P[..., 1, 2] = (top + bottom) / (top - bottom)
        P[..., 3, 2] = z_sign
        P[..., 2, 2] = z_sign * zfar / (zfar - znear)
        P[..., 2, 3] = -(zfar * znear) / (zfar - znear)
        return P
    
    @property
    def plucker_ray(self):
        # [B, 6, H, W]
        cam_points = create_camera_plane(self) # B, 3, H, W
        cam_points = cam_points / cam_points.norm(dim=1, keepdim=True) # B, 3, H, W
        B, _, H, W = cam_points.shape
        d = self.R @ cam_points.flatten(-2, -1) # B, 3, H*W
        p = d + self.t.unsqueeze(-1) # B, 3, H*W
        o = self.t.unsqueeze(-1) # B, 3, 1
        m = torch.cross(o, p, dim=1) # B, 3, H*W
        embedding = torch.cat([d, m], dim=1) # B, 6, H*W
        return embedding.reshape(B, 6, H, W)

class BatchCameras(Camera):
    def __init__(self):
        super(BatchCameras, self).__init__()

def camera_cat(cameras):
    if len(cameras) == 0:
        return None
    camera = BatchCameras()
    camera.device = cameras[0].device
    camera.width = cameras[0].width
    camera.height = cameras[0].height
    camera.zfar = cameras[0].zfar
    camera.znear = cameras[0].znear
    for key in ["_cx", "_cy", "fx", "fy", "_quaternion", "t", "bilgrid"]:
        value = []
        for c in cameras:
            if getattr(c, key) is not None:
                value.append(getattr(c, key))
        if len(value) < len(cameras):
            setattr(camera, key, None)
        setattr(camera, key, torch.cat(value, dim=0))
    return camera

def average_intrinsics(cameras_list):
    fx = 0.
    fy = 0.
    cx = 0.
    cy = 0.
    for camera in cameras_list:
        fx = fx + camera.fx
        fy = fy + camera.fy
        cx = cx + camera.cx
        cy = cy + camera.cy
    fx = fx / len(cameras_list)
    fy = fy / len(cameras_list)
    cx = cx / len(cameras_list)
    cy = cy / len(cameras_list)
    for camera in cameras_list:
        camera.fx = fx
        camera.fy = fy
        camera.cx = cx
        camera.cy = cy
    return cameras_list


def norm_extrinsics(cameras_list, idx=0):
    t = cameras_list[idx].t
    q = cameras_list[idx].quaternion
    q_inv = quaternion_inverse(q)
    R_inv = quaternion_to_matrix(q_inv)
    
    for camera in cameras_list:
        camera.t = (R_inv @ (camera.t - t).unsqueeze(-1)).squeeze(-1)
        camera.quaternion = quaternion_multiply(q_inv, camera.quaternion)

    return cameras_list