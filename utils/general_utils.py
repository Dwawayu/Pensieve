
import math
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import os
import random
import shutil
import warnings

import cv2
import imageio

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.optim.lr_scheduler import _LRScheduler

from evaluations import compute_psnr, LPIPS

def save_code(srcfile, log_path, dir_level=0):
    # Save a file or directory to the log path, appending _0, _1, etc. if a file already exists
    if not os.path.exists(srcfile):
        print(f"{srcfile} does not exist!")
    else:
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        
        def get_unique_path(path):
            base, extension = os.path.splitext(path)
            counter = 0
            unique_path = path
            while os.path.exists(unique_path):
                unique_path = f"{base}_{counter}{extension}"
                counter += 1
            return unique_path
        
        def copy_dir(srcdir, destdir, dir_level):
            if os.path.isdir(srcdir) and dir_level >= 0:
                os.makedirs(destdir, exist_ok=True)
                for item in os.listdir(srcdir):
                    copy_dir(os.path.join(srcdir, item), os.path.join(destdir, item), dir_level-1)
            elif os.path.isfile(srcdir):
                shutil.copy(srcdir, destdir)
        
        if os.path.isfile(srcfile):
            dest_file = get_unique_path(os.path.join(log_path, os.path.basename(srcfile)))
            shutil.copy(srcfile, dest_file)
            print(f"Copied file {srcfile} -> {dest_file}")
        elif os.path.isdir(srcfile):
            dest_dir = get_unique_path(os.path.join(log_path, os.path.basename(srcfile)))
            if dir_level < 0:
                shutil.copytree(srcfile, dest_dir)
            else:
                copy_dir(srcfile, dest_dir, dir_level)
            print(f"Copied directory {srcfile} -> {dest_dir}")

def batch_opration(batch_inputs, operation):
    """
    Perform an operation on data in batches.
    """
    outputs = []
    for batch in batch_inputs:
        outputs.append(operation(batch))
    return outputs


def batch_operation_parallel(batch_inputs, operation):
    """
    Perform an operation on data in batches using multiple processes.
    """
    with Pool(processes=len(batch_inputs)) as pool:
        outputs = pool.map(operation, batch_inputs)
    return outputs

def batch_operation_threaded(batch_inputs, operation):
    """
    Perform an operation on data in batches using multiple threads.
    """
    with ThreadPoolExecutor(max_workers=len(batch_inputs)) as executor:
        outputs = list(executor.map(operation, batch_inputs))
    return outputs


def unwrap_ddp_model(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model

def collate_fn(batch):
    batch = [data for data in batch if data is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)
        
class cached_property:
    def __init__(self, func):
        self.func = func
        self.key = func.__name__

    def __get__(self, instance, cls=None):
        if instance is None:
            return self
        if self.key not in instance._cache:
            instance._cache[self.key] = self.func(instance)
        return instance._cache[self.key]

def listdir_nohidden(path):
    no_hidden_list = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            no_hidden_list.append(f)
    return no_hidden_list

def sample_sublist(list, min_N, max_N, min_step, max_step, step_mode="constant"):
    '''
    min_N will be satisfied.
    when max_N < 0, we will sample as many as possible, start = 0, step is random.
    ''' 
    max_step = min(math.floor((len(list)-1) / (min_N-1)), max_step)
    min_step = min(min_step, max_step)
    step_size = random.randint(min_step, max_step)
    if max_N < 0:
        start_index = 0
    else:
        if step_mode == "constant":
            start_index = random.randint(0, len(list) - (min_N-1) * step_size - 1)
        elif step_mode == "random":
            start_index = random.randint(0, len(list) - (min_N-1) * max_step - 1)

    if step_mode == "constant":
        if max_N < 0:
            sub_slice = slice(start_index, None, step_size)
            return list[sub_slice], sub_slice
        else:
            sub_slice = slice(start_index, start_index + random.randint(min_N, max_N) * step_size, step_size)
            return list[sub_slice], sub_slice
        
    
    elif step_mode == "random":
        sub_list = [list[start_index]]
        sub_slice = [start_index]
        if max_N < 0:
            N = len(list)
        else:
            N = random.randint(min_N, max_N)
        for i in range(1, N):
            start_index += step_size
            if start_index >= len(list):
                break
            sub_list.append(list[start_index])
            sub_slice.append(start_index)
            step_size = random.randint(min_step, max_step)
            
        return sub_list, sub_slice

def tensor2image(tensor, normalize=False, colorize=False):
    '''
    Convert a tensor to an image.
    tensor: (..., C, H, W), [0, 1]
    output: (..., H, W, C), [0, 255]
    '''
    ndim = tensor.dim()
    if normalize:
        tensor = tensor - tensor.amin(dim=-1, keepdim=True).amin(dim=-2, keepdim=True).amin(dim=-3, keepdim=True)
        tensor = tensor / (tensor.amax(dim=-1, keepdim=True).amax(dim=-2, keepdim=True).amax(dim=-3, keepdim=True) + 1e-6)
    tensor = tensor * 255
    tensor = tensor.clamp(0, 255)
    permute_dims = list(range(ndim - 3)) + [ndim - 2, ndim - 1, ndim - 3]
    tensor = tensor.permute(*permute_dims).byte().cpu().numpy()
    if colorize:
        if ndim > 3:
            new_tensor = np.zeros(tensor.shape[:-1]+(3,), dtype=np.uint8)
            for idx in np.ndindex(tensor.shape[:-3]):
                new_tensor[idx] = cv2.cvtColor(cv2.applyColorMap(tensor[idx], cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB) # cv2.COLORMAP_TURBO
            tensor = new_tensor
        else:
            tensor = cv2.cvtColor(cv2.applyColorMap(tensor, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    return tensor


def outputs2video(outputs, video_path, multi_results=1):
    def extract_rend_from_outputs(outputs, rend_key):
        rend = []
        example_shape = None
        for key, ret in outputs["rets_dict"].items():
            if rend_key not in ret:
                if len(key) > 1 and isinstance(key[1], int):
                    r = ret["render"].unsqueeze(1).shape[:-3]
                else:
                    r = ret["render"].shape[:-3]
            else:
                r = ret[rend_key]
                if len(key) > 1 and isinstance(key[1], int):
                    r = r.unsqueeze(1)
                example_shape = r.shape[-3:]
            rend.append(r)
        if example_shape is None:
            example_shape = list(outputs["video_tensor"].shape[-3:])
            example_shape[0] = 1
        rend = [r if isinstance(r, torch.Tensor) else torch.zeros(*r, *example_shape, device=outputs["video_tensor"].device) for r in rend]
        rend = torch.cat(rend, dim=1)
        return rend
    
    # inputs
    input_video = tensor2image(outputs["video_tensor"])
    
    # rendered
    render_now = extract_rend_from_outputs(outputs, "render")
    # rend_normal_now = extract_rend_from_outputs(outputs, "rend_normal")
    # surf_normal_now = extract_rend_from_outputs(outputs, "surf_normal")
    depth_now = extract_rend_from_outputs(outputs, "surf_depth")
    
    render_now = tensor2image(render_now)
    # rend_normal_now = (rend_normal_now + 1.) / 2.
    # rend_normal_now = tensor2image(rend_normal_now)
    # surf_normal_now = (surf_normal_now + 1.) / 2.
    # surf_normal_now = tensor2image(surf_normal_now)
    depth_now = tensor2image(depth_now, normalize=True, colorize=True)

    # vis cameras
    cameras = visualize_cameras(outputs, width=input_video.shape[-2], height=input_video.shape[-3])

    # create video
    video = np.concatenate([input_video, cameras], axis=-2)
    video = np.tile(video, (1, multi_results, 1, 1, 1))
    # video = np.concatenate([video, np.concatenate([surf_normal_now, rend_normal_now], axis=-2)], axis=-3)
    video = np.concatenate([video, np.concatenate([render_now, depth_now], axis=-2)], axis=-3)

    print("Saving videos to " + video_path)
    for b in range(video.shape[0]):
        # video_writer = cv2.VideoWriter(video_path.format(b), cv2.VideoWriter_fourcc(*'mp4v'), 1, (video.shape[-2], video.shape[-3]))
        video_writer = imageio.get_writer(video_path.format(b), fps=1)
        for i in range(video.shape[1]):
            frame = video[b, i]
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # video_writer.write(frame)
            video_writer.append_data(frame)
        video_writer.close()

def get_camera_mesh(pose, depth=1):
    vertices = (
        torch.tensor(
            [[-0.5, -0.5, 1.], [0.5, -0.5, 1.], [0.5, 0.5, 1.], [-0.5, 0.5, 1.], [0, 0, 0]]
        )
        * depth
    )
    faces = torch.tensor(
        [[0, 1, 2], [0, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
    )
    vertices = vertices @ pose[:, :3, :3].transpose(-1, -2)
    vertices += pose[:, None, :3, 3]
    wireframe = vertices[:, [0, 1, 2, 3, 0, 4, 1, 2, 4, 3]]
    return vertices, faces, wireframe

def merge_wireframes(wireframe):
    wireframe_merged = [[], [], []]
    for w in wireframe:
        wireframe_merged[0] += [-float(n) for n in w[:, 0]]
        wireframe_merged[1] += [float(n) for n in w[:, 1]]
        wireframe_merged[2] += [float(n) for n in w[:, 2]] # change the sign of axis
    return wireframe_merged

def draw_poses(poses, width, height, no_background=True, depth=None, ms=None, ps=None, mean=None):
    inches_width = 16
    inches_height = (height / width) * inches_width
    dpi = width / inches_width

    colours = ["C1"] * poses.shape[0] + ["C2"]
    fig = plt.figure(figsize=(inches_width, inches_height), dpi=dpi)
    ax = fig.add_subplot(projection='3d')
    
    if no_background:
        ax.set_facecolor((0, 0, 0, 0))
        fig.patch.set_facecolor((0, 0, 0, 0))
        ax.grid(False)
        # ax.set_axis_off()
        
    

    centered_poses = poses.clone()
    if mean is None:
        mean = torch.mean(centered_poses[:, :3, 3], dim=0, keepdim=True)
    centered_poses[:, :3, 3] -= mean
    
    if depth is None:
        depth = centered_poses[:, :3, 3]
        if depth.shape[0] > 1:
            depth = (depth[1:] - depth[:-1]).norm(dim=-1).min().item()
        else:
            depth = 0.

    vertices, faces, wireframe = get_camera_mesh(
        centered_poses, max(depth, 0.1)
    )
    center = vertices[:, -1]
    if ps is None:
        ps = max(torch.max(center).item(), 0.1)
    if ms is None:
        ms = min(torch.min(center).item(), -0.1)
    ax.set_xlim3d(ms, ps)
    ax.set_ylim3d(ms, ps)
    ax.set_zlim3d(ms, ps)
    wireframe_merged = merge_wireframes(wireframe)
    for c in range(center.shape[0]):
        ax.plot(
            wireframe_merged[2][c * 10 : (c + 1) * 10],
            wireframe_merged[0][c * 10 : (c + 1) * 10],
            wireframe_merged[1][c * 10 : (c + 1) * 10], # change axis
            color=colours[c],
        )

    plt.tight_layout()
    fig.canvas.draw()
    # img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img

def visualize_cameras(outputs, width, height):
    Rts = []
    for camera in outputs["cameras_list"]:
        Rts.append(camera.Rt) # list of B, 4, 4
    Rts = torch.stack(Rts, dim=1) # B, N, 4, 4
    Rts = Rts.detach().cpu()
    
    # debug
    means = Rts[:, :, :3, 3].mean(dim=1, keepdim=True) # B, 1, 3
    depth = Rts[:, :, :3, 3]
    depth = (depth[:, 1:] - depth[:, :-1]).norm(dim=-1).amin(dim=-1) # B
    center = Rts[:, :, :3, 3] - means # B, N, 3
    ps = center.amax(dim=(-1, -2)).clamp_min(0.1)
    ms = center.amin(dim=(-1, -2)).clamp_max(-0.1)
    
    videos = []
    for b in range(Rts.shape[0]):
        frames = []
        for n in range(Rts.shape[1]):
            frames.append(draw_poses(Rts[b, :n+1, ...], width, height, depth=depth[b].item(), ms=ms[b].item(), ps=ps[b].item(), mean=means[b]))
        frames = np.stack(frames) # N, H, W, 3
        videos.append(frames)
    return np.stack(videos) # B, N, H, W, 3

def evaluate_render(outputs):
    color_gt = outputs["video_tensor"] # B, L, C, H, W
    B, L, C, H, W = color_gt.shape
    depth_tensor = outputs.get("depth_tensor", None) # B, L, 1, H, W

    render_evaluation = {}

    for key, ret in outputs["rets_dict"].items():
        color_pred = ret["render"].detach()
        r = key[1] - key[0]
        if f"color/{r}/psnr" not in render_evaluation:
            render_evaluation[f"color/{r}/psnr"] = []
            render_evaluation[f"color/{r}/lpips"] = []
            if depth_tensor is not None:
                render_evaluation[f"depth/{r}/a1"] = []
                render_evaluation[f"depth/{r}/abs_rel"] = []
        render_evaluation[f"color/{r}/psnr"].append(compute_psnr(color_gt[:, key[1]], color_pred).mean().item())
        render_evaluation[f"color/{r}/lpips"].append(LPIPS.compute_lpips(color_gt[:, key[1]], color_pred, 'vgg').mean().item())

        if depth_tensor is not None:
            depth_pred = ret["surf_depth"].detach()
            depth_gt = depth_tensor[:, key[1]]
            depth_pred = torch.nn.functional.interpolate(depth_pred, size=depth_gt.shape[-2:], mode='bilinear', align_corners=False)
            mask = depth_gt > 0.
            if mask.sum() > 0.:
                depth_gt = depth_gt[mask]
                depth_pred = depth_pred[mask]
                thresh = torch.max((depth_gt / depth_pred), (depth_pred / depth_gt))
                a1 = (thresh < 1.25).float().mean()
                abs_rel = torch.mean(torch.abs(depth_gt - depth_pred) / depth_gt)
                render_evaluation[f"depth/{r}/a1"].append(a1.item())
                render_evaluation[f"depth/{r}/abs_rel"].append(abs_rel.item())

    for key, value in render_evaluation.items():
        render_evaluation[key] = sum(value) / (len(value)+1e-6)

    return render_evaluation


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super(WarmUpLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < 0:
            return [0.0 for _ in self.base_lrs]
        elif self.last_epoch < self.total_iters:
            return [base_lr * (self.last_epoch + 1) / self.total_iters for base_lr in self.base_lrs]
        else:
            return [base_lr for base_lr in self.base_lrs]


class WarmupCosineAnnealing(_LRScheduler):
    def __init__(
        self,
        optimizer,
        T_warmup: int,
        T_cosine: int,
        eta_min=0,
        last_epoch=-1
    ):
        self.T_warmup = T_warmup
        self.T_cosine = T_cosine
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < 0:
            return [0.0 for _ in self.base_lrs]
        
        elif self.last_epoch < self.T_warmup:
            return [base_lr * self.last_epoch / self.T_warmup for base_lr in self.base_lrs]
        
        elif self.last_epoch < self.T_cosine:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * (self.last_epoch - self.T_warmup) / (self.T_cosine - self.T_warmup)))
                / 2
                for base_lr in self.base_lrs
            ]

        else:
            return [self.eta_min for base_lr in self.base_lrs]
       

class FlipLR(_LRScheduler):
    def __init__(self, optimizer, T_flip, multiple_1, multiple_2, last_epoch=-1):
        self.T_flip = T_flip
        self.multiple_1 = multiple_1
        self.multiple_2 = multiple_2
        super(FlipLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < 0:
            return [0.0 for _ in self.base_lrs]
        elif (self.last_epoch // self.T_flip) % 2 == 0:
            return [base_lr * self.multiple_1 for base_lr in self.base_lrs]
        else:
            return [base_lr * self.multiple_2 for base_lr in self.base_lrs]