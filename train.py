import copy
import glob
import os

from utils.config_utils import config, get_instance_from_config, load_config, save_config, GlobalState
from utils.GS_utils import render, gs_cat
from utils.general_utils import evaluate_render, save_code, unwrap_ddp_model, tensor2image, outputs2video, collate_fn
from utils.matrix_utils import apply_bilgrid

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, set_seed, GradientAccumulationPlugin
from safetensors.torch import save_file, load_file, load_model, save_model

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

from alive_progress import alive_bar, alive_it

class Trainer:
    def __init__(self, config):
        '''
        Prepare the following components:
            self.config
            self.exp_folder
            self.accelerator
            self.train_loader
            self.backbones, self.shared_by
            self.decoders
            self.optimizers
            self.schedulers
        '''
        
        self.config = config
        self.exp_folder = os.path.join(config["log_folder"], config["exp_name"])

        if config['seed']:
            set_seed(config["seed"])

        accelerator_project_config = ProjectConfiguration(project_dir=self.exp_folder)
        gradient_accumulation_config = GradientAccumulationPlugin(num_steps=config['training']['gradient_accumulation_steps'], sync_with_dataloader=False)
        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            # kwargs_handlers=[ddp_kwargs]
            # gradient_accumulation_steps = config['training']['gradient_accumulation_steps'],
            mixed_precision = config['training']['mixed_precision'],
            log_with = "tensorboard",
            project_config=accelerator_project_config,
            gradient_accumulation_plugin=gradient_accumulation_config,
        )
        self.device = self.accelerator.device

        if self.accelerator.is_main_process and not config.get("only_evaluation", False):
            if os.path.exists(self.exp_folder) and os.path.exists(os.path.join(self.exp_folder, "config.yaml")):
                exist_config = load_config(os.path.join(self.exp_folder, "config.yaml"))
                config["training"]["start_epoch"] = exist_config["training"]["start_epoch"]
                # import pdb; pdb.set_trace()
                # assert config == exist_config, "log_folder exists with different config."
            else:
                os.makedirs(self.exp_folder, exist_ok=True)
                save_config(config, os.path.join(self.exp_folder, "config.yaml"))
                
            save_code_path = os.path.join(self.exp_folder, "code")
            save_code("./train.py", save_code_path)
            save_code("./losses.py", save_code_path)
            save_code("./evaluations.py", save_code_path)
            save_code("./inferences.py", save_code_path)
            save_code("./backbones", save_code_path)
            save_code("./camera_decoders", save_code_path)
            save_code("./gs_decoders", save_code_path)
            save_code("./datasets", save_code_path)
            save_code("./utils", save_code_path)

        self.accelerator.init_trackers("tensorboard")
        
        if config["dataset"]["params"].get("data_cache", False):
            # Disable multiprocessing to avoid duplicate data_cache
            config["training"]["num_workers"] = 0

        train_dataset = get_instance_from_config(config["dataset"])
        
        if "sampler" in config["dataset"]:
            sampler = get_instance_from_config(config["dataset"]["sampler"], dataset=train_dataset)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        self.train_loader = DataLoader(train_dataset, 
                                        batch_size=config["training"]["batch_size"], 
                                        num_workers=config["training"]["num_workers"], 
                                        collate_fn=collate_fn, 
                                        shuffle=shuffle,
                                        sampler=sampler,
                                        persistent_workers=False)

        self.backbones = {}
        self.decoders = {}
        self.optimizers = {}
        self.schedulers = {}
        # Process shared backbone first
        if "shared_backbone" in config["models"] and "shared_by" in config["models"]["shared_backbone"]:
            self.shared_by = config["models"]["shared_backbone"]["shared_by"]
            self._regist_model("shared_backbone", config["models"]["shared_backbone"])
        else:
            self.shared_by = []
        
        # Process other models
        for model_name, model_config in config["models"].items():
            if model_name == "shared_backbone":
                continue
            
            if model_name not in self.shared_by:
                self._regist_model(model_name+"_backbone", model_config["backbone"])
                self._regist_model(model_name+"_decoder", model_config["decoder"], self.backbones[model_name+"_backbone"].ch_feature)
            else:
                self._regist_model(model_name+"_decoder", model_config["decoder"], self.backbones["shared_backbone"].ch_feature)

        # log_folder overrides load_folder
        if len(glob.glob(os.path.join(self.exp_folder, f"ckpts/*_latest.safetensors"))) > 0 and config.get("load_from", "latest") == "latest":
            # Load the models
            ckpt_folder = os.path.join(self.exp_folder, "ckpts")
            self.load(ckpt_folder, extend_name="_latest", models_to_load=config.get("models_to_load", []))
        elif config["load_folder"]:
            self.load(config["load_folder"], extend_name="_latest", models_to_load=config.get("models_to_load", []))
        
        # for model in self.backbones.values():
        #     model.train()
        # for model in self.decoders.values():
        #     model.train()
        #TODO: weight_dtype
        # weight_dtype = torch.float32
        # if self.accelerator.mixed_precision == "fp16":
        #     weight_dtype = torch.float16
        # elif self.accelerator.mixed_precision == "bf16":
        #     weight_dtype = torch.bfloat16
        # TODO: Models and Optimizers have different order
        # if self.config['training']["gradient_checkpointing"]:
        #     for model in self.backbones.values():
        #         model.gradient_checkpointing_enable()
        #     for model in self.decoders.values():
        #         model.gradient_checkpointing_enable()
        prepared = self.accelerator.prepare(*list(self.backbones.values()), *list(self.decoders.values()), *list(self.optimizers.values()), *list(self.schedulers.values()), self.train_loader)
        self.backbones = dict(zip(self.backbones.keys(), prepared[:len(self.backbones)]))
        self.decoders = dict(zip(self.decoders.keys(), prepared[len(self.backbones):len(self.backbones)+len(self.decoders)]))
        self.optimizers = dict(zip(self.optimizers.keys(), prepared[len(self.backbones)+len(self.decoders):len(self.backbones)+len(self.decoders)+len(self.optimizers)]))
        self.schedulers = dict(zip(self.schedulers.keys(), prepared[len(self.backbones)+len(self.decoders)+len(self.optimizers):len(self.backbones)+len(self.decoders)+len(self.optimizers)+len(self.schedulers)]))
        
        if "sampler" not in config["dataset"]:
            # Only prepare train_loader if not using sampler, otherwise the sampler will be replaced
            self.train_loader = prepared[-1]
        
        # prepare loss class
        self.losses = []
        for loss_name, loss_config in config["losses"].items():
            loss_weight = loss_config["weight"]
            if loss_weight > 0.:
                loss_function = get_instance_from_config(loss_config)
                self.losses.append((loss_name, loss_function, loss_weight))
                
        if "inference" in config:
            self.inference_method = get_instance_from_config(config["inference"], self)
        else:
            self.inference_method = None
            
        
    def _regist_model(self, model_name, config, *args, **kwargs):
        
        model = get_instance_from_config(config, *args, **kwargs)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        if model_name.endswith("backbone"):
            self.backbones[model_name] = model
        elif model_name.endswith("decoder"):
            self.decoders[model_name] = model
        else:
            raise ValueError("model_name should end with 'backbone' or 'decoder'")
        
        model_parameters = model.parameters()
        if len(list(model_parameters)) > 0.:
            self.optimizers[model_name] = get_instance_from_config(config["optimizer"], model.parameters())
            self.schedulers[model_name] = get_instance_from_config(config["scheduler"], self.optimizers[model_name])
        
    def train(self):
        for epoch in range(self.config["training"]["start_epoch"], self.config["training"]["num_epochs"]):
            self.epoch = epoch
            for model in self.backbones.values():
                model.train()
            for model in self.decoders.values():
                model.train()
                
            train_iterator = self.train_loader
            self.train_dataset_len = len(train_iterator)
            self.steps = self.epoch * self.train_dataset_len
            GlobalState["global_step"] = self.steps
            if self.accelerator.is_main_process:
                train_iterator = alive_it(train_iterator, title=f"epoch {self.epoch}:")
            train_iterator = enumerate(train_iterator)
            
            valid_inputs = []
            iter_to_sync = self.config['training']['gradient_accumulation_steps']
            for iteration, inputs in train_iterator:
                if inputs is None:
                    print("skip iteration.")
                    continue
                elif len(valid_inputs) < config['training']['gradient_accumulation_steps']:
                    # Store the first valid inputs
                    valid_inputs.append(copy.deepcopy(inputs))
                    
                self.iteration = iteration

                with self.accelerator.accumulate(*list(self.backbones.values()), *list(self.decoders.values())):
                    outputs, total_loss = self.process_batch(inputs)
                    self.accelerator.backward(total_loss)

                    for model_name, optimizer in self.optimizers.items():
                        optimizer.step()
                        optimizer.zero_grad()
                    for model_name, scheduler in self.schedulers.items():
                        scheduler.step()

                if self.accelerator.is_main_process:
                    with torch.no_grad():
                        if self.steps % self.config["training"]["visualization_steps"] == 0:
                            self.visualize(outputs)
                        # if self.steps % self.config["training"]["eval_steps"] == 0:
                        #     self.evaluate(outputs)
                        if self.steps % self.config["training"]["log_steps"] == 0:
                            self.log(outputs)
                            
                iter_to_sync -= 1
                if self.accelerator.sync_gradients:
                    all_flags = [False] * self.accelerator.state.num_processes
                    dist.all_gather_object(all_flags, iteration == self.train_dataset_len - 1)
                    if any(all_flags):
                        break
                    iter_to_sync = self.config['training']['gradient_accumulation_steps']

            for _ in range(iter_to_sync):
                inputs = valid_inputs.pop(0)
                with self.accelerator.accumulate(*list(self.backbones.values()), *list(self.decoders.values())):
                    outputs, total_loss = self.process_batch(inputs)
                    self.accelerator.backward(total_loss)
                    if self.accelerator.sync_gradients:
                        for model in self.backbones.values():
                            # for param in model.parameters():
                            #     if param.grad is not None:
                            #         torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
                            self.accelerator.clip_grad_norm_(model.parameters(), self.config['training']['max_grad_norm'])
                        for model in self.decoders.values():
                            # for param in model.parameters():
                            #     if param.grad is not None:
                            #         torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
                            self.accelerator.clip_grad_norm_(model.parameters(), self.config['training']['max_grad_norm'])
                    for model_name, optimizer in self.optimizers.items():
                        optimizer.step()
                        optimizer.zero_grad()
                    for model_name, scheduler in self.schedulers.items():
                        scheduler.step()
            if iter_to_sync > 0:
                all_flags = [False] * self.accelerator.state.num_processes
                dist.all_gather_object(all_flags, True)

            if self.accelerator.is_main_process and "evaluations" in self.config:
                self.evaluate()

            if self.accelerator.is_main_process and self.epoch % self.config["training"]["save_ckpt_epochs"] == 0:
                self.save("_latest")

    @torch.no_grad()
    def evaluate(self):
        for model in self.backbones.values():
            model.eval()
        for model in self.decoders.values():
            model.eval()
        for name, data_method in self.config["evaluations"].items():
            evaluation_dataset = get_instance_from_config(data_method["evaluation_dataset"])
            evaluation_method = get_instance_from_config(data_method["evaluation_method"], self)
            evaluation_loader = DataLoader(evaluation_dataset,
                                    batch_size=self.config["training"]["batch_size"],
                                    num_workers=self.config["training"]["num_workers"],
                                    collate_fn=collate_fn,
                                    shuffle=False)
            evaluation_loader = self.accelerator.prepare(evaluation_loader)
            print(f"Begin evaluation {name}")
            if self.accelerator.is_main_process:
                evaluation_loader = alive_it(evaluation_loader)
            for inputs in evaluation_loader:
                if not inputs:
                    print("skip iteration.")
                    continue
                evaluation_method(inputs)
            
            if dist.is_initialized():
                metrics = evaluation_method.get_metrics_dist()
                if self.accelerator.is_main_process:
                    print(f"{name} results:")
                    print(metrics)
            else:
                print(f"{name} results:")
                print(evaluation_method.get_metrics())

                
    def process_batch(self, inputs):
        # with self.accelerator.accumulate(model):
        video_tensor = inputs['video_tensor']

        B, L, C, H, W = video_tensor.shape
        self.steps = self.epoch * self.train_dataset_len + self.iteration
        
        inputs = self.init_results_list(inputs)
        
        GlobalState["global_step"] = self.steps
        
        if self.inference_method:
            inputs = self.inference_method(inputs)

        else:
            inputs["render_window"] = self.config["training"]["render_window"]
            
            # predict
            inputs = self.inference(inputs)
            
            # breakpoint()
            # self.backbones["shared_backbone"].model.unpatchify.linear_proj.weight

            # rendering
            render_list = self.config["training"]["render_list"] if "render_list" in self.config["training"] else range(L)
            for l in render_list:
                gs_to_render = []
                gs_idx = []
                for gs_cat_window in self.config["training"]["gs_cat_window"]:
                    if l + gs_cat_window < L and l + gs_cat_window >= 0:
                        gs_to_render.append(inputs["gs_list"][l+gs_cat_window])
                        gs_idx.append(l+gs_cat_window)
                gs_to_render = gs_cat(gs_to_render)

                gs_idx = tuple(gs_idx)
                for r in self.config["training"]["render_window"]:
                    if l + r < L and l + r >= 0:
                        inputs["rets_dict"][(gs_idx, l+r)] = render(inputs["cameras_list"][l+r], gs_to_render)
                        if inputs["cameras_list"][l+r].bilgrid is not None:
                            inputs["rets_dict"][(gs_idx, l+r)]["render"] = apply_bilgrid(inputs["cameras_list"][l+r].bilgrid, inputs["rets_dict"][(gs_idx, l+r)]["render"])
                        if self.config["alpha_bg"] == "GT":
                            inputs["rets_dict"][(gs_idx, l+r)]["render"] = inputs["rets_dict"][(gs_idx, l+r)]["rend_alpha"].detach() * inputs["rets_dict"][(gs_idx, l+r)]["render"] + (1. - inputs["rets_dict"][(gs_idx, l+r)]["rend_alpha"].detach()) * inputs["video_tensor"][:, l+r]
                        elif self.config["alpha_bg"] == "noise":
                            inputs["rets_dict"][(gs_idx, l+r)]["render"] = inputs["rets_dict"][(gs_idx, l+r)]["rend_alpha"] * inputs["rets_dict"][(gs_idx, l+r)]["render"] + (1. - inputs["rets_dict"][(gs_idx, l+r)]["rend_alpha"]) * torch.rand_like(inputs["rets_dict"][(gs_idx, l+r)]["render"])
        
        total_loss = 0.
        inputs["losses"] = {}
        for loss_name, loss_function, loss_weight in self.losses:
            loss = loss_function(inputs)
            inputs["losses"][f"loss/{loss_name}"] = loss
            total_loss += loss_weight * loss
        inputs["losses"]["loss/total_loss"] = total_loss
        
        return inputs, total_loss

    def init_results_list(self, inputs):
        B, L, C, H, W = inputs['video_tensor'].shape
        # first_camera = BatchCameras()
        # first_camera.t = torch.zeros([B, 3], device=self.device)
        # first_camera._quaternion = torch.zeros([B, 4], device=self.device)
        # first_camera._quaternion[..., 0] = 1.
        # first_camera.width = W
        # first_camera.height = H
        # first_camera.device = self.device
        # inputs["cameras_list"] = [first_camera]
        inputs["cameras_list"] = []
        inputs["gs_list"] = []
        inputs["rets_dict"] = {}
        return inputs

    def inference(self, inputs):
        
        # the backbone outputs should be # B, L, F, H, W
        for backbone_name, backbone_model in self.backbones.items():
            # if backbone_name.startswith("camera") and l == L-1:
            #     continue
            features, inputs = torch.utils.checkpoint.checkpoint(backbone_model, inputs, use_reentrant=False)
            # features, inputs = backbone_model(inputs)
            if backbone_name == "shared_backbone":
                for model_name in self.shared_by:
                    inputs[model_name+"_features"] = features
            else:
                inputs[backbone_name.replace("_backbone", "_features")] = features
        
        L = inputs['video_tensor'].shape[1]
        # prepare inputs
        for decoder_name, decoder_model in self.decoders.items():
            # if decoder_name.startswith("camera") and l == L-1:
            #     continue
            for l in range(L):
                inputs["now_idx"] = l
                # inputs = decoder_model(inputs)
                inputs = torch.utils.checkpoint.checkpoint(decoder_model, inputs, use_reentrant=False)
                
            if decoder_name.startswith("camera") and self.config["single_intrinsic"]:
                inputs["cameras_list"] = self.average_intrinsics(inputs["cameras_list"])
        
        return inputs
    
    def average_intrinsics(self, cameras_list):
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
            

    def visualize(self, outputs):
        visualization_path = os.path.join(self.exp_folder, "visualization", f"{self.epoch}")
        os.makedirs(visualization_path, exist_ok=True)
        video_path = os.path.join(visualization_path, f"{self.iteration:06d}"+"_{:03d}_video.mp4")
        outputs2video(outputs, video_path, self.config["training"]["vis_multi_results"])

    # def evaluate(self, outputs):
    #     outputs["losses"].update(evaluate_render(outputs))

    def log(self, outputs):
        self.accelerator.log(outputs["losses"], step=self.steps)
        
    def load(self, ckpt_folder, extend_name, models_to_load=[]):
        print(f"Loading ckpts from {ckpt_folder}.")

        for name, model in self.backbones.items():
            if not models_to_load or name in models_to_load:
                # model.load_state_dict(load_file(os.path.join(ckpt_folder, f"{name}{extend_name}.safetensors")))
                load_model(model, os.path.join(ckpt_folder, f"{name}{extend_name}.safetensors"), strict=False, device=str(self.device))
            else:
                print(f"Skip loading {name}")
        for name, model in self.decoders.items():
            if not models_to_load or name in models_to_load:
                # model.load_state_dict(load_file(os.path.join(ckpt_folder, f"{name}{extend_name}.safetensors")))
                load_model(model, os.path.join(ckpt_folder, f"{name}{extend_name}.safetensors"), strict=False, device=str(self.device))
            else:
                print(f"Skip loading {name}")

        if not config.get("only_evaluation", False) and config["load_optimizer"]:
            for name, optimizer in self.optimizers.items():
                if not models_to_load or name in models_to_load:
                    optimizer_load_path = os.path.join(ckpt_folder, f"{name}_optimizer{extend_name}.pth")
                    scheduler_load_path = os.path.join(ckpt_folder, f"{name}_scheduler{extend_name}.pth")
                    try:
                        optimizer_dict = torch.load(optimizer_load_path, map_location=self.device, weights_only=False)
                        optimizer.load_state_dict(optimizer_dict)
                        scheduler_dict = torch.load(scheduler_load_path, map_location=self.device, weights_only=False)
                        self.schedulers[name].load_state_dict(scheduler_dict)
                    except:
                        print(f"Cannot load optimizer weights for {name} so optimizer is randomly initialized")

    def save(self, extend_name):
        self.config["training"]["start_epoch"] = self.epoch + 1
        save_config(self.config, os.path.join(self.exp_folder, "config.yaml"))
        ckpt_folder = os.path.join(self.exp_folder, "ckpts")
        print(f"Saving ckpts to {ckpt_folder}.")
        os.makedirs(ckpt_folder, exist_ok=True)

        for name, model in self.backbones.items():
            # save_file(unwrap_ddp_model(model).state_dict(), os.path.join(ckpt_folder, f"{name}{extend_name}.safetensors"))
            save_model(unwrap_ddp_model(model), os.path.join(ckpt_folder, f"{name}{extend_name}.safetensors"))
        for name, model in self.decoders.items():
            # save_file(unwrap_ddp_model(model).state_dict(), os.path.join(ckpt_folder, f"{name}{extend_name}.safetensors"))
            save_model(unwrap_ddp_model(model), os.path.join(ckpt_folder, f"{name}{extend_name}.safetensors"))

        for name, optimizer in self.optimizers.items():
            torch.save(optimizer.state_dict(), os.path.join(ckpt_folder, f"{name}_optimizer{extend_name}.pth"))
            
        for name, scheduler in self.schedulers.items():
            torch.save(scheduler.state_dict(), os.path.join(ckpt_folder, f"{name}_scheduler{extend_name}.pth"))
    
if __name__ == "__main__":
    # Get the configuration
    trainer = Trainer(config)
    # debug
    # torch.autograd.set_detect_anomaly(True)
    if not config.get("only_evaluation", False):
        trainer.train()
    else:
        trainer.evaluate()