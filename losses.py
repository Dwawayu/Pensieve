import numpy as np
import torch
import torch.nn.functional as F
from math import exp

import torchvision
from evaluations import LPIPS

from utils.matrix_utils import align_camera_dict, camera_list_to_dict, create_camera_plane, quaternion_multiply, quaternion_to_matrix

from pytorch3d.loss import chamfer_distance

from utils.config_utils import GlobalState

class FrameWeightedLoss:
    '''
    FrameWeightedLoss is a base class for loss functions that are weighted by the frame index.
    All render loss of ret in rets_dict.
    '''
    def loss_func(self, x, y):
        raise NotImplementedError("loss_func should be implemented in the subclass")
    
    
    def __call__(self, inputs):
        total_loss = 0.
        total_weight = 0.
        for key, ret in inputs["rets_dict"].items():
            if len(key) == 1 or not isinstance(key[1], int):
                # in this case, ret["render"] rendered the whole video or a subset of the video
                # and the loss_weight is a list of length ret["render"].shape[1]
                # key[1] is a list of index
                if len(key) == 1:
                    loss = self.loss_func(ret["render"], inputs['video_tensor'])
                else:
                    loss = self.loss_func(ret["render"], inputs['video_tensor'][:, key[1]])
                loss = loss.mean(dim=[0, 2, 3, 4]) # L
                if "loss_weight" in ret:
                    loss_weight = ret["loss_weight"]
                    loss_weight = torch.tensor(loss_weight, device=ret["render"].device)
                    loss = loss * loss_weight # L
                    total_weight += loss_weight.sum()
                else:
                    total_weight += loss.shape[0]
                total_loss += loss.sum()
                    
            else:
                # in this case, ret["render"] rendered a frame of the video
                # key[1] is a int index
                # loss_weight is a scalar
                loss = self.loss_func(ret["render"], inputs['video_tensor'][:, key[1]]) # B, 3, H, W
                loss = loss.mean() # scalar
                if "loss_weight" in ret:
                    loss = loss * ret["loss_weight"]
                    total_weight += ret["loss_weight"]
                else:
                    total_weight += 1.
                total_loss += loss
            
        return total_loss / total_weight

class ImageL1Loss(FrameWeightedLoss):
    def loss_func(self, x, y):
        return (x - y).abs()
            
class ImageL2Loss(FrameWeightedLoss):
    def loss_func(self, x, y):
        return (x - y).pow(2)
    
class ImageSSIMLoss:
    def __init__(self, window_size=11, size_average=True):
        self.window_size = window_size
        self.size_average = size_average
        self.window = self.create_window()
        
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()
        
    def create_window(self):
        _1D_window = self.gaussian(self.window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0) # 1, 1, window_size, window_size
        # window = torch.autograd.Variable(_2D_window.expand(channel, 1, self.window_size, self.window_size).contiguous())
        return _2D_window
        
    def _ssim(self, img1, img2, window, channel):
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def __call__(self, inputs):
        raise NotImplementedError("ImageSSIMLoss should be used with FrameWeightedLoss")
        channel = inputs['video_tensor'].size(-3)
        window = self.window.repeat(channel, 1, 1, 1)

        if inputs['video_tensor'].is_cuda:
            window = window.cuda(inputs['video_tensor'].get_device())
        window = window.type_as(inputs['video_tensor'])
        
        ssim_loss = 0.
        for key, ret in inputs["rets_dict"].items():
            ssim_loss += (1. - self._ssim(ret["render"], inputs['video_tensor'][:, key[1]], window, channel)).mean()
        return ssim_loss / len(inputs["rets_dict"])

class DepthProjectionLoss:
    def __init__(self, fwd_flow_weight=0., use_predict_depth=False, max_step=None, proj_window=[-3, -2, -1, 1, 2, 3]):
        self.fwd_flow_weight = fwd_flow_weight
        self.use_predict_depth = use_predict_depth
        self.proj_window = proj_window
        self.max_step = max_step

    def back_projection(self, depth ,camera):
        camera_planes = create_camera_plane(camera)
        xyz = camera_planes * depth # B, 3, H, W
        B, _, H, W = xyz.shape
        xyz = camera.R @ xyz.reshape(B, 3, H*W) + camera.t[:, :, None]
        return xyz.reshape(B, 3, H, W)

    def projection(self, xyz, camera):
        B, _, H, W = xyz.shape
        xyz = xyz.reshape(B, 3, H*W)
        xyz = xyz - camera.t[:, :, None]
        xyz = camera.K @ camera.R_inv @ xyz
        xyz = xyz.reshape(B, 3, H, W)
        z = xyz[:, 2:, ...]
        z_vaild = z > 1e-5
        z[~z_vaild] = 1e-5
        xy = xyz[:, :2] / z
        vaild_mask = (xy[:, 0:1] >= 0.) * (xy[:, 0:1] <= camera.width - 1) * \
                     (xy[:, 1:2] >= 0.) * (xy[:, 1:2] <= camera.height - 1) * \
                     z_vaild

        return xy, vaild_mask

    def projection_sample(self, xyz, camera, reference_image):
        uv, vaild_mask = self.projection(xyz, camera)
        uv[:, 0, ...] /= camera.width - 1.
        uv[:, 1, ...] /= camera.height - 1.
        xy = uv.permute(0, 2, 3, 1) # B, H, W, 2
        xy = (xy - 0.5) * 2
        self_image = F.grid_sample(reference_image, xy, padding_mode="border", align_corners=True)
        return self_image, vaild_mask, uv

    def __call__(self, inputs):
        if self.max_step is not None and GlobalState["global_step"] >= self.max_step:
            return 0.
        depth_sample_loss = 0.
        sample_loss_count = 0
        fwd_flow_loss = 0.
        fwd_loss_count = 0
        for key, ret in inputs["rets_dict"].items():
            if "surf_depth" not in ret:
                continue
            pos_mask = ret["surf_depth"] > 0
            if not pos_mask.any():
                continue
            xyz = self.back_projection(ret["surf_depth"], inputs["cameras_list"][key[1]])
            
            for r in self.proj_window:
                if r != 0 and key[1] + r < inputs["video_tensor"].shape[1] and key[1] + r >= 0:
                    self_image, vaild_mask, uv = self.projection_sample(xyz, inputs["cameras_list"][key[1] + r], inputs["video_tensor"][:, key[1] + r])
                    # self_image = self_image * vaild_mask + (1. - vaild_mask) * inputs["video_tensor"][:, key[1]]
                    vaild_mask = vaild_mask * pos_mask
                    depth_sample_loss += ((self_image - inputs["video_tensor"][:, key[1]]).abs() * vaild_mask).mean()
                    sample_loss_count += 1

                    if r == 1 and self.fwd_flow_weight > 0.:
                        H, W = uv.shape[-2], uv.shape[-1]
                        width_list = torch.arange(W, device=uv.device) / (W - 1)
                        height_list = torch.arange(H, device=uv.device) / (H - 1)
                        meshgrid = torch.meshgrid(width_list, height_list, indexing='xy')
                        pix_coords = torch.stack(meshgrid, axis=0) # 2, H, W
                        pred_flow = uv - pix_coords.unsqueeze(0) # B, 2, H, W
                        fwd_flow_loss += (pred_flow - inputs["flow_tensor"][:, key[1]]).abs().mean()
                        fwd_loss_count += 1

        if self.use_predict_depth:
            assert inputs["predict_depth"].shape[2] == 1
            for l in range(inputs["video_tensor"].shape[1]):
                pos_mask = inputs["predict_depth"][:, l, 0] > 0
                if not pos_mask.any():
                    continue
                xyz = self.back_projection(inputs["predict_depth"][:, l, 0], inputs["cameras_list"][l])
                for r in self.proj_window:
                    if r != 0 and l + r < inputs["video_tensor"].shape[1] and l + r >= 0:
                        self_image, vaild_mask, uv = self.projection_sample(xyz, inputs["cameras_list"][l + r], inputs["video_tensor"][:, l + r])
                        # self_image = self_image * vaild_mask + (1. - vaild_mask) * inputs["video_tensor"][:, key[1]]
                        vaild_mask = vaild_mask * pos_mask
                        depth_sample_loss += ((self_image - inputs["video_tensor"][:, l]).abs() * vaild_mask).mean()
                        sample_loss_count += 1

                        if r == 1 and self.fwd_flow_weight > 0.:
                            H, W = uv.shape[-2], uv.shape[-1]
                            width_list = torch.arange(W, device=uv.device) / (W - 1)
                            height_list = torch.arange(H, device=uv.device) / (H - 1)
                            meshgrid = torch.meshgrid(width_list, height_list, indexing='xy')
                            pix_coords = torch.stack(meshgrid, axis=0) # 2, H, W
                            pred_flow = uv - pix_coords.unsqueeze(0) # B, 2, H, W
                            fwd_flow_loss += (pred_flow - inputs["flow_tensor"][:, l]).abs().mean()
                            fwd_loss_count += 1

        total_loss = depth_sample_loss / sample_loss_count
        if fwd_loss_count > 0:
            total_loss += self.fwd_flow_weight * fwd_flow_loss / fwd_loss_count
        if self.max_step is not None:
            total_loss = total_loss * (self.max_step - GlobalState["global_step"]) / self.max_step
        return total_loss

class DepthSmoothLoss:
    def __init__(self, gamma=1, inv=True, normalize=True, use_predict_depth=False):
        self.gamma = gamma
        self.inv = inv
        self.normalize = normalize
        self.use_predict_depth = use_predict_depth

    def get_smooth_loss(self, depth, img):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        depth: ..., 1, H, W
        img: ..., 3, H, W
        """
        if self.inv:
            mask = depth < 1e-5
            depth = 1. / depth.clamp_min(1e-5)
            depth[mask] = 0.
        if self.normalize:
            depth = depth / depth.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        grad_depth_x = torch.abs(depth[..., :, :-1] - depth[..., :, 1:])
        grad_depth_y = torch.abs(depth[..., :-1, :] - depth[..., 1:, :])

        grad_img_x = torch.mean(torch.abs(img[..., :, :-1] - img[..., :, 1:]), -3, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[..., :-1, :] - img[..., 1:, :]), -3, keepdim=True)

        grad_depth_x *= torch.exp(-self.gamma*grad_img_x)
        grad_depth_y *= torch.exp(-self.gamma*grad_img_y)

        return grad_depth_x.mean() + grad_depth_y.mean()
    
    def __call__(self, inputs):
        depth_smooth_loss = 0.
        loss_count = 0
        for key, ret in inputs["rets_dict"].items():
            if "surf_depth" not in ret:
                continue
            depth = ret["surf_depth"]
            depth_smooth_loss += self.get_smooth_loss(depth, inputs['video_tensor'][:, key[1]])
            loss_count += 1
            
        if self.use_predict_depth:
            assert inputs["predict_depth"].shape[2] == 1 # B, L, 1, 1, H, W
            depth_smooth_loss += self.get_smooth_loss(inputs["predict_depth"][:, :, 0], inputs['video_tensor'])
            loss_count += inputs["predict_depth"].shape[1]
        return depth_smooth_loss / loss_count


class CameraInverseLoss:
    def __init__(self, q_weight, t_weight):
        self.q_weight = q_weight
        self.t_weight = t_weight
    
    def __call__(self, inputs):
        q_loss = 0.

        for quaternion_loss in inputs["inverse_quaternion_loss"]:
            q_loss += quaternion_loss.mean()
        q_loss = q_loss / len(inputs["inverse_quaternion_loss"])

        t_loss = 0.
        for translation_loss in inputs["inverse_translation_loss"]:
            t_loss += translation_loss.mean()
        t_loss = t_loss / len(inputs["inverse_translation_loss"])
        
        return q_loss * self.q_weight + t_loss * self.t_weight


class PushScaleLoss:
    def __init__(self):
        pass
    def __call__(self, inputs):
        pass

class PushAlphaL1Loss:
    def __call__(self, inputs):
        alpha_loss = 0.
        for key, ret in inputs["rets_dict"].items():
            alpha_loss += torch.relu(1. - ret["rend_alpha"]).mean()
        return alpha_loss / len(inputs["rets_dict"])


class PushAlphaLogLoss:
    def __init__(self, weight=2.):
        self.weight = weight

    def __call__(self, inputs):
        alpha_loss = 0.
        loss_count = 0.
        for key, ret in inputs["rets_dict"].items():
            if "rend_alpha" in ret:
                alpha_loss -= self.weight * ret["rend_alpha"].log().mean()
                loss_count += 1.
        return alpha_loss / loss_count

class RegOpacityLoss:
    def __call__(self, inputs):
        opacity_loss = 0.
        for gs in inputs["gs_list"]:
            opacity_loss += gs["opacity"].abs().mean()
        return opacity_loss / len(inputs["gs_list"])

class CameraSupervisedLoss:
    def __call__(self, inputs):
        camera_loss = 0.
        
        predict_cameras_dict = camera_list_to_dict(inputs["cameras_list"])
        g2p_q, g2p_scale, g2p_t = align_camera_dict(inputs["camera_dict"], predict_cameras_dict, "all")

        gt_t = (quaternion_to_matrix(g2p_q) @ inputs["camera_dict"]["t"].unsqueeze(-1)).squeeze(-1) * g2p_scale + g2p_t
        gt_quaternion = quaternion_multiply(g2p_q, inputs["camera_dict"]["quaternion"])
        gt_rfx = inputs["camera_dict"]["fx"] / inputs["camera_dict"]["width"]
        gt_rfy = inputs["camera_dict"]["fy"] / inputs["camera_dict"]["height"]
        
        predicted_t = predict_cameras_dict["t"]
        predicted_quaternion = predict_cameras_dict["quaternion"]
        predicted_rfx = predict_cameras_dict["fx"] / predict_cameras_dict["width"]
        predicted_rfy = predict_cameras_dict["fy"] / predict_cameras_dict["height"]
        
        quaternion_loss = - (predicted_quaternion * gt_quaternion).sum(-1).abs()
        translation_loss = (predicted_t - gt_t).norm(dim=-1).clamp(1e-6) / (gt_t.norm(dim=-1) + predicted_t.norm(dim=-1)).clamp(1e-6)
        focal_loss = (predicted_rfx - gt_rfx).abs() + (predicted_rfy - gt_rfy).abs()
        
        camera_loss += quaternion_loss.mean() + translation_loss.mean() + focal_loss.mean()

        return camera_loss

class CameraConsistencyLoss:
    def __call__(self, inputs):
        camera_loss = 0.
        for (idx_0, idx_1) in inputs["paired_idx"]:
            camera_0 = inputs["cameras_list"][idx_0]
            camera_1 = inputs["cameras_list"][idx_1]
            
            quaternion_0 = camera_0.quaternion
            t_0 = camera_0.t
            rfx_0 = camera_0.fx / camera_0.width
            rfy_0 = camera_0.fy / camera_0.height
            
            quaternion_1 = camera_1.quaternion
            t_1 = camera_1.t
            rfx_1 = camera_1.fx / camera_1.width
            rfy_1 = camera_1.fy / camera_1.height
            
            quaternion_loss = - (quaternion_0 * quaternion_1).sum(-1).abs()
            translation_loss = (t_0 - t_1).norm(dim=-1).clamp(1e-6) / (t_0.norm(dim=-1) + t_1.norm(dim=-1)).clamp(1e-6)
            focal_loss = (rfx_0 - rfx_1).abs() + (rfy_0 - rfy_1).abs()
            
            camera_loss += quaternion_loss.mean() + translation_loss.mean() + focal_loss.mean()
            
        return camera_loss / len(inputs["paired_idx"])


class DepthDistortionLoss:
    def __call__(self, inputs):
        dist_loss = 0.
        for key, ret in inputs["rets_dict"].items():
            dist_loss += ret["rend_dist"].mean()
        return dist_loss / len(inputs["rets_dict"])

class NormalConsistencyLoss:
    def __call__(self, inputs):
        normal_loss = 0.
        for key, ret in inputs["rets_dict"].items():
            normal_loss += (1. - (ret["rend_normal"] * ret["surf_normal"]).sum(dim=1)).mean()
        return normal_loss / len(inputs["rets_dict"])


class DepthSupervisedLoss:
    def __init__(self, inv=False, normalize=True):
        self.inv = inv
        self.normalize = normalize

    def __call__(self, inputs):
        depth_loss = 0.
        for key, ret in inputs["rets_dict"].items():
            pred = ret["surf_depth"]
            gt = inputs['depth_tensor'][:, key[1]]
            if self.inv:
                pred = 1. / pred.clamp_min(1e-5)
                gt = 1. / gt.clamp_min(1e-5)
            if self.normalize:
                pred = pred / pred.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
                gt = gt / gt.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
            depth_loss += (pred - gt).abs().mean()
        return depth_loss / len(inputs["rets_dict"])


class LpipsLoss(FrameWeightedLoss):
    def loss_func(self, x, y):
        loss = LPIPS.compute_lpips(x, y, 'vgg') # B, (N)
        return loss[..., None, None, None]
    

class PerceptualLoss:
    def __init__(self, pc_net="vgg19"):
        if pc_net == "vgg19":
            class vgg19(torch.nn.Module):
                def __init__(self, requires_grad=False):
                    super(vgg19, self).__init__()
                    vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features

                    # This has Vgg config E:
                    # partial convolution paper uses up to pool3
                    # [64,'r', 64,r, 'M', 128,'r', 128,r, 'M', 256,'r', 256,r, 256,r, 256,r, 'M', 512,'r', 512,r, 512,r, 512,r]
                    self.slice1 = torch.nn.Sequential()
                    self.slice2 = torch.nn.Sequential()
                    self.slice3 = torch.nn.Sequential()
                    self.slice4 = torch.nn.Sequential()
                    n_new = 0
                    for x in range(5):  # pool1,
                        self.slice1.add_module(str(n_new), vgg_pretrained_features[x])
                        n_new += 1
                    for x in range(5, 10):  # pool2
                        self.slice2.add_module(str(n_new), vgg_pretrained_features[x])
                        n_new += 1
                    for x in range(10, 19):  # pool3
                        self.slice3.add_module(str(n_new), vgg_pretrained_features[x])
                        n_new += 1
                    for x in range(19, 28):  # pool4
                        self.slice4.add_module(str(n_new), vgg_pretrained_features[x])
                        n_new += 1
                    for param in self.parameters():
                        param.requires_grad = requires_grad
                    # norm as torch
                    self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                    # norm as FalNet
                    # self.normalize = torchvision.transforms.Normalize(mean=[0.411, 0.432, 0.45],
                    #                         std=[1, 1, 1])

                def forward(self, x, full=False):
                    x = self.normalize(x)
                    h_relu1_2 = self.slice1(x)
                    h_relu2_2 = self.slice2(h_relu1_2)
                    h_relu3_4 = self.slice3(h_relu2_2)
                    if full:
                        h_relu4_4 = self.slice4(h_relu3_4)
                        return h_relu1_2, h_relu2_2, h_relu3_4, h_relu4_4
                    else:
                        return h_relu1_2, h_relu2_2, h_relu3_4
            self.pc_net = vgg19().cuda()
        else:
            raise NotImplementedError("pc_net should be 'vgg19'")
        
    def __call__(self, inputs):
        perceptual_loss = 0.
        total_weight = 0.
        for key, ret in inputs["rets_dict"].items():
            if len(key) >= 3:
                weight = key[2]
            else:
                weight = 1.
            pred = ret["render"]
            gt = inputs['video_tensor'][:, key[1]]
            pred_features = self.pc_net(pred)
            gt_features = self.pc_net(gt)
            pc_error = 0.
            for i in range(len(pred_features)):
                pc_error += (pred_features[i] - gt_features[i]).square().mean()
            perceptual_loss += pc_error * weight
            total_weight += weight
        return perceptual_loss / total_weight


class DepthMultiModalLoss:
    def __init__(self, dist="lap"):
        if dist == "gaussian":
            self.distribution = self.gaussian
        elif dist == "lap":
            self.distribution = self.laplacian
        else:
            raise ValueError("dist should be 'gaussian' or 'lap'")

    def gaussian(self, error, sigma):
        return torch.exp(-0.5*error** 2/ sigma** 2)/sigma/(2*np.pi)**0.5
    def laplacian(self, error, b):
        return 0.5 * torch.exp(-(torch.abs(error)/b))/b

    def multimodal_loss(self, error, sigma, pi):
        return - torch.log(torch.sum(pi * self.distribution(error, sigma), dim=2) + 1e-5)

    def __call__(self, inputs):
        error = (inputs["predict_depth"] - inputs["depth_tensor"].unsqueeze(2)) / (inputs["depth_tensor"].unsqueeze(2).detach().amax(dim=(-2, -1), keepdim=True)+1e-5) # B, L, N, 1, H, W
        loss = self.multimodal_loss(error, inputs["lap_sigma"], inputs["lap_pi"])
        return loss.mean()
    
    
class BilGridTVLoss:
    def __call__(self, inputs):
        total_loss = 0.
        loss_count = 0
        for camera in inputs["cameras_list"]:
            if camera.bilgrid is not None:
                bilgrid = camera.bilgrid
                loss = (bilgrid[:, :, :, :, :-1] - bilgrid[:, :, :, :, 1:]).square().mean() + \
                       (bilgrid[:, :, :, :-1, :] - bilgrid[:, :, :, 1:, :]).square().mean() + \
                       (bilgrid[:, :, :-1, :, :] - bilgrid[:, :, 1:, :, :]).square().mean()

                total_loss += loss
                loss_count += 1
        return total_loss / loss_count


class ChamferDistanceLoss:
    def __init__(self, ignore_quantile=None):
        self.ignore_quantile = ignore_quantile
        
    def __call__(self, inputs):
        total_loss = 0.
        loss_count = 0
        for l in range(len(inputs["gs_list"]) - 1):
            x = torch.cat([inputs["gs_list"][l]["xyz"] / inputs["gs_list"][l]["xyz"].norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True), inputs["gs_list"][l]["features"].reshape(*inputs["gs_list"][l]["features"].shape[:2], -1)], dim=-1)
            y = torch.cat([inputs["gs_list"][l+1]["xyz"] / inputs["gs_list"][l+1]["xyz"].norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True), inputs["gs_list"][l+1]["features"].reshape(*inputs["gs_list"][l+1]["features"].shape[:2], -1)], dim=-1)
            if self.ignore_quantile is not None:
                cd_error = chamfer_distance(x, y, batch_reduction=None, point_reduction=None)[0]
                cd_error_0 = cd_error[0] # B, N
                cd_error_1 = cd_error[1] # B, N
                threshold_0 = torch.quantile(cd_error_0, self.ignore_quantile, dim=-1, keepdim=True)
                mask_0 = cd_error_0 < threshold_0
                threshold_1 = torch.quantile(cd_error_1, self.ignore_quantile, dim=-1, keepdim=True)
                mask_1 = cd_error_1 < threshold_1
                total_loss += (cd_error_0[mask_0].mean() + cd_error_1[mask_1].mean()) / 2
            else:
                total_loss += chamfer_distance(x, y)[0]
            loss_count += 1
        return total_loss / loss_count
    
    
class PixelProjAlignLoss:
    def __init__(self, near=0.2):
        self.near = near
        
    def __call__(self, inputs):
        total_loss = 0.
        H, W = inputs["video_tensor"].shape[-2:]
        N = inputs["gs_list"][0]["xyz"].shape[1]
        radio = (N / (H * W)) ** 0.5
        H = int(H * radio)
        W = int(W * radio)
        meshgrid = torch.meshgrid(torch.arange(W, device=inputs["video_tensor"].device) / W, torch.arange(H, device=inputs["video_tensor"].device) / H, indexing='xy')
        pix_coords = torch.stack(meshgrid, axis=0) # 2, H, W
        pix_coords = pix_coords.permute(1, 2, 0).reshape(-1, 2).unsqueeze(0) # 1, H*W, 2
        if "gs_idx" in inputs:
            gs_cameras_list = [inputs["cameras_list"][idx] for idx in inputs["gs_idx"]]
        else:
            gs_cameras_list = inputs["cameras_list"]
        assert len(inputs["gs_list"]) == len(gs_cameras_list)
        for i in range(len(inputs["gs_list"])):
            uvd = (inputs["gs_list"][i]["xyz"] - gs_cameras_list[i].t.detach().unsqueeze(-2)) @ gs_cameras_list[i].R.detach() @ gs_cameras_list[i].K.detach().transpose(-2, -1) # B, N, 3
            visiable_mask = uvd[:, :, 2] > self.near
            uv = uvd[:, :, :2] / uvd[:, :, 2:]
            # uv = uv.clone()
            uv[..., 0] /= W - 1
            uv[..., 1] /= H - 1
            uv_loss = (uv - pix_coords).square().sum(-1)[visiable_mask].mean()
            # uv_loss = (uv - pix_coords).abs().sum(-1)[visiable_mask].mean()
            # uv_loss = (uv - 0.25).abs().sum(-1)[visiable_mask].mean()
            if not visiable_mask.all():
                depth_loss = (-uvd[:, :, 2][~visiable_mask]).mean()
            else:
                depth_loss = 0.
            total_loss += uv_loss + depth_loss
        return total_loss / len(inputs["gs_list"])
    

class PixelDirAlignLoss:
    def __call__(self, inputs):
        total_loss = 0.
        
        H, W = inputs["video_tensor"].shape[-2:]
        N = inputs["gs_list"][0]["xyz"].shape[1]
        radio = (N / (H * W)) ** 0.5
        H = int(H * radio)
        W = int(W * radio)
        meshgrid = torch.meshgrid(torch.arange(W, device=inputs["video_tensor"].device), torch.arange(H, device=inputs["video_tensor"].device), indexing='xy')
        pix_coords = torch.stack(meshgrid, axis=0) + 0.5 # 2, H, W
        pix_coords = pix_coords.permute(1, 2, 0).reshape(1, -1, 2) # 1, H*W, 2
        pix_coords = torch.cat([pix_coords, torch.ones_like(pix_coords[..., :1])], dim=-1) # 1, H*W, 3
        
        if "gs_idx" in inputs:
            gs_cameras_list = [inputs["cameras_list"][idx] for idx in inputs["gs_idx"]]
        else:
            gs_cameras_list = inputs["cameras_list"]
        assert len(inputs["gs_list"]) == len(gs_cameras_list)
        
        for i in range(len(inputs["gs_list"])):
            camera = gs_cameras_list[i]
            pix_dir = pix_coords @ camera.K_inv.detach().transpose(-2, -1) @ camera.R_inv.detach() # B, H*W, 3
            
            gs_dir = inputs["gs_list"][i]["xyz"] - camera.t.detach().unsqueeze(-2) # B, N, 3
            
            dir_loss = 1. - (pix_dir * gs_dir).sum(-1) / (pix_dir.norm(dim=-1) * gs_dir.norm(dim=-1))
            
            total_loss += dir_loss.mean()
            
        return total_loss / len(inputs["gs_list"])
    

class DirectLoss:
    def __init__(self, key_weight_dict):
        self.key_weight_dict = key_weight_dict
        
    def __call__(self, inputs):
        total_loss = 0.
        for key, weight in self.key_weight_dict.items():
            total_loss += weight * inputs[key]
            
        return total_loss