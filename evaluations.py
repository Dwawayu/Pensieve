import copy
from einops import reduce
import imageio
from jaxtyping import Float
import lpips
import numpy as np
from skimage.metrics import structural_similarity
from PIL import Image, ImageDraw, ImageFont
import os

import torch
from torch import Tensor
import torch.distributed as dist
from torchvision.transforms import v2
import torch.nn.functional as F

from utils.GS_utils import render, gs_cat
from utils.camera import average_intrinsics, norm_extrinsics
from utils.config_utils import get_instance_from_config
from utils.matrix_utils import align_camera_dict, camera_dict_to_list, camera_list_to_dict, closed_form_inverse, generate_camera_trajectory_demo, matrix_to_quaternion, quatWAvgMarkley, quaternion_multiply, quaternion_t_to_matrix, quaternion_to_matrix, quaternion_translation_inverse, quaternion_translation_multiply, rotation_angle, translation_angle, umeyama

class AlignPoseEvaluation:
    def __init__(self, trainer, **config):
        super().__init__()
        self.trainer = trainer
        self.config = config

        self.transforms = None
        if "transforms" in self.config:
            transforms_list = []
            for transform_name, transform_config in self.config["transforms"].items():
                transforms_list.append(get_instance_from_config(transform_config))
            self.transforms = v2.Compose(transforms_list)
            
        self.camera_optimizer = None
        if "camera_optimizer" in self.config:
            self.camera_optimizer = get_instance_from_config(self.config["camera_optimizer"])
            
        self.tgt_pose = self.config.get("tgt_pose", "align")

        self.reset_metrics()

    def reset_metrics(self):
        self.metrics = {"image_count": 0, "pose_count": 0,
                        "psnr": 0., "lpips": 0., "ssim": 0.,
                        "Racc_5":0, "Racc_15":0, "Racc_30":0,
                        "Tacc_5":0, "Tacc_15":0, "Tacc_30":0}

    def mean_metrics(self, metrics):
        return {
            "psnr": metrics["psnr"] / metrics["image_count"],
            "lpips": metrics["lpips"] / metrics["image_count"],
            "ssim": metrics["ssim"] / metrics["image_count"],
            "Racc_5": metrics["Racc_5"] / metrics["pose_count"],
            "Racc_15": metrics["Racc_15"] / metrics["pose_count"],
            "Racc_30": metrics["Racc_30"] / metrics["pose_count"],
            "Tacc_5": metrics["Tacc_5"] / metrics["pose_count"],
            "Tacc_15": metrics["Tacc_15"] / metrics["pose_count"],
            "Tacc_30": metrics["Tacc_30"] / metrics["pose_count"]
        }

    def get_metrics(self):
        return self.mean_metrics(self.metrics)
        
    def get_metrics_dist(self):
        dist_metrics = {}
        for key, value in self.metrics.items():
            dist_metrics[key] = torch.tensor(value, device="cuda")
            dist.all_reduce(dist_metrics[key], op=dist.ReduceOp.SUM)
        return self.mean_metrics(dist_metrics)

    def visualize(self, inputs):
        # folder = os.path.join("./visualization", inputs["key"][0])
        folder = "./visualization"
        os.makedirs(folder, exist_ok=True)
        from utils.general_utils import tensor2image, visualize_cameras
        video = inputs["video_tensor"]
        # inputs["cameras_list"] += inputs["gt_cameras_list"]
        cameras = visualize_cameras(inputs, width=video.shape[-1], height=video.shape[-2])
        video = tensor2image(video)
        # video = np.concatenate([video, video], axis=-4)
        video = np.concatenate([video, cameras], axis=-2)
        for b in range(video.shape[0]):
            video_writer = imageio.get_writer("{}/{}_{}.mp4".format(folder, dist.get_rank(), self.metrics["image_count"]+b), fps=1)
            for i in range(video.shape[1]):
                frame = video[b, i]
                video_writer.append_data(frame)
            video_writer.close()
            Image.fromarray(cameras[b, -1]).save("{}/{}_{}_cameras.png".format(folder, dist.get_rank(), self.metrics["image_count"]+b))
        
        for key, value in inputs["rets_dict"].items():
            predict = value["render"]
            gt = inputs["target_images"][:, key[1]]
            predict = tensor2image(predict)
            gt = tensor2image(gt)
            for b in range(predict.shape[0]):
                Image.fromarray(predict[b]).save("{}/{}_{}_{}_predict.png".format(folder, dist.get_rank(), self.metrics["image_count"]+b, key[1]))
                Image.fromarray(gt[b]).save("{}/{}_{}_{}_gt.png".format(folder, dist.get_rank(), self.metrics["image_count"]+b, key[1]))
                depth = tensor2image(value["surf_depth"][b], normalize=True, colorize=True)
                Image.fromarray(depth).save("{}/{}_{}_{}_depth.png".format(folder, dist.get_rank(), self.metrics["image_count"]+b, key[1]))

    def evaluate_metrics(self, inputs):
        
        for key, value in inputs["rets_dict"].items():

            predict = value["render"]
            gt = inputs["target_images"][:, key[1]]
            if self.transforms is not None:
                predict = self.transforms(predict)
                gt = self.transforms(gt)
        
            self.metrics["image_count"] += predict.shape[0]
            self.metrics["psnr"] += compute_psnr(gt, predict).sum().item()
            self.metrics["lpips"] += LPIPS.compute_lpips(gt, predict).sum().item()
            self.metrics["ssim"] += compute_ssim(gt, predict).sum().item()
        
            
        predict_cameras_dict = camera_list_to_dict(inputs["cameras_list"])
        rel_rangle_deg, rel_tangle_deg = camera_to_rel_deg(quaternion_t_to_matrix(predict_cameras_dict["quaternion"], predict_cameras_dict["t"]),
                                                           quaternion_t_to_matrix(inputs["camera_dict"]["quaternion"], inputs["camera_dict"]["t"]))
        rel_rangle_deg = rel_rangle_deg.reshape(-1)
        rel_tangle_deg = rel_tangle_deg.reshape(-1)
        assert rel_rangle_deg.shape == rel_tangle_deg.shape
        self.metrics["pose_count"] += rel_rangle_deg.shape[0]
        self.metrics["Racc_5"] += (rel_rangle_deg < 5).sum().item()
        self.metrics["Racc_15"] += (rel_rangle_deg < 15).sum().item()
        self.metrics["Racc_30"] += (rel_rangle_deg < 30).sum().item()
        self.metrics["Tacc_5"] += (rel_tangle_deg < 5).sum().item()
        self.metrics["Tacc_15"] += (rel_tangle_deg < 15).sum().item()
        self.metrics["Tacc_30"] += (rel_tangle_deg < 30).sum().item()
        
    def inference(self, inputs):
        inputs = self.trainer.init_results_list(inputs)
        inputs = self.trainer.inference(inputs)
        return inputs

    @torch.no_grad()
    def __call__(self, inputs):
        inputs = self.inference(inputs)

        
        predict_cameras_dict = camera_list_to_dict(inputs["cameras_list"])
        g2p_q, g2p_scale, g2p_t = align_camera_dict(inputs["camera_dict"], predict_cameras_dict, "all")
       
        gs_to_render = gs_cat(inputs["gs_list"])
        gt_cameras_list = camera_dict_to_list(inputs["target_cameras"])
        
        
        for i, camera in enumerate(gt_cameras_list):
            
            
            camera.t = (quaternion_to_matrix(g2p_q) @ camera.t.unsqueeze(-1)).squeeze(-1) * g2p_scale + g2p_t
            camera.quaternion = quaternion_multiply(g2p_q, camera.quaternion)
            camera.fx = inputs["cameras_list"][0].fx
            camera.fy = inputs["cameras_list"][0].fy
            if self.camera_optimizer is not None:
                camera = self.camera_optimizer(inputs["target_images"][:, i], camera, gs_to_render)
            
            inputs["rets_dict"][("all", i)] = render(camera, gs_to_render)
                

        # self.visualize(inputs)
        self.evaluate_metrics(inputs)
        
        
class RefineEvaluation(AlignPoseEvaluation):
    
    @torch.no_grad()
    def inference(self, inputs):
        backbone = self.trainer.backbones["shared_backbone"].eval()
        camera_decoder = self.trainer.decoders["camera_decoder"].eval()
        gs_decoder = self.trainer.decoders["gs_decoder"].eval()
        
        inputs = self.trainer.init_results_list(inputs)
        
        if self.tgt_pose == "align":
            gs_idx = list(range(inputs["video_tensor"].shape[1]))
        elif self.tgt_pose == "predict":
            inputs["video_tensor"] = torch.cat([inputs["video_tensor"][:, 0:1], inputs["target_images"], inputs["video_tensor"][:, -1:]], dim=1)
            gs_idx = [0, inputs["video_tensor"].shape[1]-1]
        inputs = backbone(inputs, gs_idx)

        for l in range(inputs["camera_features"].shape[1]):
            inputs["now_idx"] = l
            inputs = camera_decoder(inputs)
        if self.trainer.config["single_intrinsic"]:
            inputs["cameras_list"] = average_intrinsics(inputs["cameras_list"])
        if self.trainer.config["norm_extrinsic"]:
            inputs["cameras_list"] = norm_extrinsics(inputs["cameras_list"], idx=gs_idx[0])
            
        if self.tgt_pose == "predict":
            inputs["video_tensor"] = torch.stack([inputs["video_tensor"][:, 0], inputs["video_tensor"][:, -1]], dim=1)
            inputs["gt_cameras_list"] = inputs["cameras_list"][1:-1]
            inputs["cameras_list"] = [inputs["cameras_list"][0], inputs["cameras_list"][-1]]
            
        for l in range(inputs["gs_features"].shape[1]):
            inputs["now_idx"] = l
            inputs = gs_decoder(inputs)
            
        # debug
        if self.tgt_pose == "align":
            lvsm_decoder = self.trainer.decoders["lvsm_decoder"].eval()
            src_plucker = []
            for camera in inputs["cameras_list"]:
                plucker_embedding = camera.plucker_ray
                src_plucker.append(plucker_embedding)
            src_plucker = torch.stack(src_plucker, dim=1)
            gs_features_plucker = torch.cat([inputs["gs_features"], src_plucker], dim=2) # B, G, F+6, H, W
            zero_gs = torch.zeros_like(inputs["gs_features"][:, 0]) # B, F, H, W
            camera = inputs["cameras_list"][0]
            Q = torch.stack([inputs["cameras_list"][0].quaternion, inputs["cameras_list"][1].quaternion], dim=1)
            camera._quaternion = quatWAvgMarkley(Q)
            camera.t = (inputs["cameras_list"][0].t + inputs["cameras_list"][1].t) / 2.
            plucker_embedding = camera.plucker_ray
            plucker_embedding = torch.cat([zero_gs, plucker_embedding], dim=1)
            render_image, others = lvsm_decoder.module.infer_lvsm(gs_features_plucker, plucker_embedding)
            
            inputs["video_tensor"] = torch.stack([inputs["video_tensor"][:, 0], render_image, inputs["video_tensor"][:, 1]], dim=1)
            inputs = self.trainer.init_results_list(inputs)
            gs_idx = list(range(inputs["video_tensor"].shape[1]))
            inputs = backbone(inputs, gs_idx)
            
            for l in range(inputs["camera_features"].shape[1]):
                inputs["now_idx"] = l
                inputs = camera_decoder(inputs)
            if self.trainer.config["single_intrinsic"]:
                inputs["cameras_list"] = average_intrinsics(inputs["cameras_list"])
            if self.trainer.config["norm_extrinsic"]:
                inputs["cameras_list"] = norm_extrinsics(inputs["cameras_list"], idx=gs_idx[0])
                
            for l in range(inputs["gs_features"].shape[1]):
                inputs["now_idx"] = l
                inputs = gs_decoder(inputs)
            
            inputs["cameras_list"] = [inputs["cameras_list"][0], inputs["cameras_list"][-1]]
            inputs["gs_features"] = torch.stack([inputs["gs_features"][:, 0], inputs["gs_features"][:, -1]], dim=1)
            inputs["video_tensor"] = torch.stack([inputs["video_tensor"][:, 0], inputs["video_tensor"][:, -1]], dim=1)
        
        return inputs

    @torch.no_grad()
    def __call__(self, inputs):
        inputs = self.inference(inputs)
        
        lvsm_decoder = self.trainer.decoders["lvsm_decoder"].eval()
        src_plucker = []
        for camera in inputs["cameras_list"]:
            plucker_embedding = camera.plucker_ray
            src_plucker.append(plucker_embedding)
        src_plucker = torch.stack(src_plucker, dim=1)
        gs_features_plucker = torch.cat([inputs["gs_features"], src_plucker], dim=2) # B, G, F+6, H, W
        zero_gs = torch.zeros_like(inputs["gs_features"][:, 0]) # B, F, H, W
        
        if self.tgt_pose == "align":
            predict_cameras_dict = camera_list_to_dict(inputs["cameras_list"])
            g2p_q, g2p_scale, g2p_t = align_camera_dict(inputs["camera_dict"], predict_cameras_dict, "all")
            gt_cameras_list = camera_dict_to_list(inputs["target_cameras"])
        elif self.tgt_pose == "predict":
            gt_cameras_list = inputs["gt_cameras_list"]
            
        gs_to_render = gs_cat(inputs["gs_list"])
        for i, camera in enumerate(gt_cameras_list):
            
            if self.tgt_pose == "align":
                camera.t = (quaternion_to_matrix(g2p_q) @ camera.t.unsqueeze(-1)).squeeze(-1) * g2p_scale + g2p_t
                camera.quaternion = quaternion_multiply(g2p_q, camera.quaternion)
                camera.fx = inputs["cameras_list"][0].fx
                camera.fy = inputs["cameras_list"][0].fy
            if self.camera_optimizer is not None:
                camera = self.camera_optimizer(inputs["target_images"][:, i], camera, gs_to_render)
        

            plucker_embedding = camera.plucker_ray
            plucker_embedding = torch.cat([zero_gs, plucker_embedding], dim=1)
            render_image, others = lvsm_decoder.module.infer_lvsm(gs_features_plucker, plucker_embedding)
            
            
            lvsm_result = render_image
            inputs["rets_dict"][("all", i)] = {"render": render_image,
                                               "surf_depth": torch.zeros_like(render_image)[:, 0:1]}
            
            
        # self.visualize(inputs)
        self.evaluate_metrics(inputs)
        if dist.get_rank() == 0:
            print(self.get_metrics())
            

@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()

class LPIPS:
    loss_fn = None

    @classmethod
    def create_loss_fn(cls, model, device):
        cls.loss_fn = lpips.LPIPS(net=model).to(device)

    @classmethod
    def compute_lpips(cls, x, y, model="vgg"):
        if cls.loss_fn is None:
            cls.create_loss_fn(model, x.device)
        N = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        y = y.reshape(-1, *y.shape[-3:])
        loss = cls.loss_fn.forward(x, y, normalize=True)
        loss = loss.reshape(*N)
        return loss

@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ground_truth = ground_truth.reshape(-1, *ground_truth.shape[-3:])
    predicted = predicted.reshape(-1, *predicted.shape[-3:])
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)


@torch.no_grad()
def camera_to_rel_deg(pred_c2w, gt_c2w):
    """
    Calculate relative rotation and translation angles between predicted and ground truth cameras.

    Args:
    - pred_cameras: Predicted camera.
    - gt_cameras: Ground truth camera.s
    - accelerator: The device for moving tensors to GPU or others.
    - batch_size: Number of data samples in one batch.

    Returns:
    - rel_rotation_angle_deg, rel_translation_angle_deg: Relative rotation and translation angles in degrees.
    """
    
    B, N, _, _ = pred_c2w.shape
    
    pair_idx_i1, pair_idx_i2 = torch.combinations(torch.arange(N, device=pred_c2w.device), 2, with_replacement=False).unbind(-1) # NN
    relative_pose_gt = gt_c2w[:, pair_idx_i1].inverse() @ gt_c2w[:, pair_idx_i2] # B, NN, 4, 4
    relative_pose_pred = pred_c2w[:, pair_idx_i1].inverse() @ pred_c2w[:, pair_idx_i2] # B, NN, 4, 4
    
    rel_rangle_deg = rotation_angle(relative_pose_gt[..., :3, :3], relative_pose_pred[..., :3, :3])
    rel_tangle_deg = translation_angle(relative_pose_gt[..., :3, 3], relative_pose_pred[..., :3, 3])

    return rel_rangle_deg, rel_tangle_deg