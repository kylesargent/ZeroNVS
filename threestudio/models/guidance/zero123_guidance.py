import importlib
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from tqdm import tqdm

import threestudio
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.typing import *
import copy


import sys

sys.path.insert(
    0, "/home/jupyter/enter_the_photo_image2nerf/enter_the_photo_diffusion/zero123"
)

from ldm.data import common


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def walk(dictionary):
    for key, value in list(dictionary.items()):
        if isinstance(value, dict):
            yield from walk(value)
        else:
            yield dictionary, key, value


def _get_relative_transformations(batch, scales):
    # this is the correct representation to use because it's invariant to
    # shift but not scale

    # scales = batch['metadata_scene_radius'][:, None]
    # assert centers.ndim == 2
    # assert scales.ndim == 2

    target_cam2world = batch["target_cam2world"].detach().clone()
    cond_cam2world = batch["cond_cam2world"].detach().clone()
    # print(target_cam2world[:, :3, -1].shape, centers.shape)

    # target_cam2world[:, :3, -1] /= scales
    # cond_cam2world[:, :3, -1] /= scales

    batch_size = target_cam2world.shape[0]

    relative_target_transformation = torch.linalg.inv(cond_cam2world) @ target_cam2world
    relative_target_transformation[:, :3, -1] /= scales

    assert relative_target_transformation.shape == (
        batch_size,
        4,
        4,
    ), relative_target_transformation.shape
    return relative_target_transformation


def v_get_angle_between_rotations(pose1, pose2):
    assert pose1.shape == pose2.shape
    assert pose1.shape[1:] == (4, 4)

    r1 = pose1[:, :3, :3]
    r2 = pose2[:, :3, :3]

    r_diff = torch.bmm(r1.permute((0, 2, 1)), r2)

    r_diff_trace = r_diff.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    cos_theta = (r_diff_trace - 1) / 2
    theta = torch.arccos(torch.clip(cos_theta, -1, 1))
    return theta


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")

    # for dictionary, key, value in walk(dict(config)):
    #     print(key)
    #     if key == 'target' and value.startswith('ldm'):
    #         old_target = value
    #         new_target = 'extern.ldm_zero123' + dictionary[key][3:]
    #         print(old_target, new_target)
    #         dictionary[key] = new_target

    return get_obj_from_str(config["target"])(**config.get("params", dict()))


# load model
def load_model_from_config(config, ckpt, device, vram_O=True, verbose=False):
    pl_sd = torch.load(ckpt, map_location="cpu")

    if "global_step" in pl_sd and verbose:
        print(f'[INFO] Global Step: {pl_sd["global_step"]}')

    sd = pl_sd["state_dict"]

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("[INFO] missing keys: \n", m)
    if len(u) > 0 and verbose:
        print("[INFO] unexpected keys: \n", u)

    # manually load ema and delete it to save GPU memory
    if model.use_ema:
        if verbose:
            print("[INFO] loading EMA...")
        model.model_ema.copy_to(model.model)
        del model.model_ema

    if vram_O:
        # we don't need decoder
        del model.first_stage_model.decoder

    torch.cuda.empty_cache()

    model.eval().to(device)

    return model


@threestudio.register("zero123-guidance")
class Zero123Guidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "load/zero123/105000.ckpt"
        pretrained_config: str = "load/zero123/sd-objaverse-finetune-c_concat-256.yaml"
        vram_O: bool = True

        cond_image_path: str = "load/images/hamburger_rgba.png"
        cond_elevation_deg: float = 0.0
        cond_azimuth_deg: float = 0.0
        cond_camera_distance: float = 1.2

        guidance_scale: float = 5.0
        guidance_scale_aux: float = 5.0
        precomputed_scale: Optional[float] = None
        cond_fov_deg: Optional[float] = None

        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = False

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4

        p_use_aux_cameras: float = 0.0

        use_anisotropic_schedule: bool = False
        anisotropic_offset: int = 0

        depth_threshold_for_anchor_guidance: float = 0.

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Zero123 ...")

        self.config = OmegaConf.load(self.cfg.pretrained_config)
        self.config.model.params.conditioning_config.params.depth_model_name = None

        # TODO: seems it cannot load into fp16...
        self.weights_dtype = torch.float32
        self.model = load_model_from_config(
            self.config,
            self.cfg.pretrained_model_name_or_path,
            device=self.device,
            vram_O=self.cfg.vram_O,
        )

        for p in self.model.parameters():
            p.requires_grad_(False)

        # timesteps: use diffuser for convenience... hope it's alright.
        self.num_train_timesteps = self.config.model.params.timesteps

        self.scheduler = DDIMScheduler(
            self.num_train_timesteps,
            self.config.model.params.linear_start,
            self.config.model.params.linear_end,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        self.prepare_embeddings(self.cfg.cond_image_path)

        self.p_use_aux_cameras_val = 0.0

        # Only set if aux cameras are passed
        self.all_conditioning = None
        self.aux_rgbs = None

        self.global_step = 0  # keep track of it

        threestudio.info(f"Loaded Zero123!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def prepare_embeddings(self, image_path: str) -> None:
        # load cond image for zero123
        assert os.path.exists(image_path)
        rgba = cv2.cvtColor(
            cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
        )
        rgba = (
            cv2.resize(rgba, (256, 256), interpolation=cv2.INTER_AREA).astype(
                np.float32
            )
            / 255.0
        )
        rgb = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])
        self.rgb_256: Float[Tensor, "1 3 H W"] = (
            torch.from_numpy(rgb)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .contiguous()
            .to(self.device)
        )
        self.c_crossattn, self.c_concat = self.get_img_embeds(self.rgb_256)

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_img_embeds(
        self,
        img: Float[Tensor, "B 3 256 256"],
    ) -> Tuple[Float[Tensor, "B 1 768"], Float[Tensor, "B 4 32 32"]]:
        img = img * 2.0 - 1.0
        c_crossattn = self.model.get_learned_conditioning(img.to(self.weights_dtype))
        c_concat = self.model.encode_first_stage(img.to(self.weights_dtype)).mode()
        return c_crossattn, c_concat

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(imgs.to(self.weights_dtype))
        )
        return latents.to(input_dtype)  # [B, 4, 32, 32] Latent space image

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        image = self.model.decode_first_stage(latents)
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_cond_from_known_camera(self, batch, c_crossattn, c_concat):
        T = common.compute_T(
            self.config.model.params.conditioning_config,
            None,
            batch,
            precomputed_scale=self.cfg.precomputed_scale,
        )

        # print(T.shape)
        T = T[:, None, :].to(self.device)
        ##

        # need to handle the case where c_crossattn are batched or unbatched
        b_repeats = len(T) // len(c_crossattn)
        assert len(c_crossattn) == len(c_concat)

        cond = {}
        clip_emb = self.model.cc_projection(
            torch.cat(
                [
                    (c_crossattn).repeat(b_repeats, 1, 1),
                    T,
                ],
                dim=-1,
            )
        )

        cond["c_crossattn"] = [
            torch.cat([torch.zeros_like(clip_emb).to(self.device), clip_emb], dim=0)
        ]
        cond["c_concat"] = [
            torch.cat(
                [
                    torch.zeros_like(c_concat)
                    .repeat(b_repeats, 1, 1, 1)
                    .to(self.device),
                    (c_concat).repeat(b_repeats, 1, 1, 1),
                ],
                dim=0,
            )
        ]
        return cond

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_cond(
        self,
        camera,
        c_crossattn=None,
        c_concat=None,
        **kwargs,
    ) -> dict:
        # print("Using new camera API")
        elevation = camera["elevation"]
        azimuth = camera["azimuth"]
        camera_distances = camera["camera_distances"]

        # print(camera)

        assert "canonical_c2w" in camera
        assert "c2w" in camera

        condition_mode = "7dof"

        target_cam2world = camera["c2w"].to(self.device)
        batch_size = target_cam2world.shape[0]

        cond_cam2world = camera["canonical_c2w"].to(self.device)
        assert target_cam2world.shape == (batch_size, 4, 4)
        assert cond_cam2world.shape == (1, 4, 4)

        cond_cam2world = cond_cam2world.broadcast_to(target_cam2world.shape)

        batch = {
            "target_cam2world": target_cam2world,
            "cond_cam2world": cond_cam2world,
            "fov_deg": camera["fov_deg"],
        }

        if camera["aux_c2ws"]:
            # compute conditionings if not computed already
            if self.all_conditioning is None:
                assert self.aux_rgbs is None
                aux_rgbs = []
                all_conditioning = [(self.c_crossattn, self.c_concat)]
                for aux_c2w in camera["aux_c2ws"]:
                    aux_batch = {
                        "target_cam2world": aux_c2w,
                        "cond_cam2world": cond_cam2world[:1],
                        "fov_deg": camera["fov_deg"][:1],
                    }
                    aux_cond = self.get_cond_from_known_camera(
                        aux_batch, self.c_crossattn, self.c_concat
                    )

                    aux_rgb = self.gen_from_cond(cond=aux_cond, ddim_steps=500)
                    aux_rgbs.append(aux_rgb)

                    aux_rgb = (
                        torch.from_numpy(aux_rgb)
                        .permute(0, 3, 1, 2)
                        .contiguous()
                        .to(self.device)
                    )

                    aux_c_crossattn, aux_c_concat = self.get_img_embeds(aux_rgb)
                    all_conditioning.append((aux_c_crossattn, aux_c_concat))

                self.all_conditioning = all_conditioning
                self.aux_rgbs = aux_rgbs

                # import pdb
                # pdb.set_trace()

            all_c2ws = torch.concatenate([camera["canonical_c2w"], *camera["aux_c2ws"]])
            assert len(all_c2ws) == len(self.all_conditioning)

            use_auxes = np.random.binomial(
                n=1, p=self.p_use_aux_cameras_val, size=batch_size
            )
            nearest_idxs = [
                torch.argmin(
                    v_get_angle_between_rotations(
                        target_cam2world_i[None].broadcast_to(all_c2ws.shape),
                        all_c2ws,
                    )
                )
                for target_cam2world_i in target_cam2world
            ]
            # print(torch.stack(nearest_idxs))
            # print(use_auxes)
            # import pdb
            # pdb.set_trace()

            batches = []
            for i, (use_aux, nearest_idx) in enumerate(zip(use_auxes, nearest_idxs)):
                if use_aux:
                    batch = {
                        "target_cam2world": target_cam2world[i],
                        "cond_cam2world": all_c2ws[nearest_idx],
                        "fov_deg": camera["fov_deg"][0],
                        "c_crossattn": self.all_conditioning[nearest_idx][0][0],
                        "c_concat": self.all_conditioning[nearest_idx][1][0],
                    }
                else:
                    batch = {
                        "target_cam2world": target_cam2world[i],
                        "cond_cam2world": cond_cam2world[0],
                        "fov_deg": camera["fov_deg"][0],
                        "c_crossattn": self.c_crossattn[0],
                        "c_concat": self.c_concat[0],
                    }
                batch['c_crossattn_nearest'] = self.all_conditioning[nearest_idx][0][0]

                batches.append(batch)
            batch = torch.utils.data.default_collate(batches)
            c_crossattn_nearest = batch['c_crossattn_nearest']
            cond = self.get_cond_from_known_camera(
                batch, c_crossattn=batch["c_crossattn"], c_concat=batch["c_concat"]
            )

        else:
            nearest_idxs = None
            c_crossattn_nearest = None
            cond = self.get_cond_from_known_camera(
                batch, c_crossattn=self.c_crossattn, c_concat=self.c_concat
            )
        # cond[''] = c_crossattn_nearest
        return cond, {'c_crossattn_nearest': c_crossattn_nearest, 'nearest_idxs': nearest_idxs}

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        camera,
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
        if rgb_as_latents:
            latents = (
                F.interpolate(rgb_BCHW, (32, 32), mode="bilinear", align_corners=False)
                * 2
                - 1
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (256, 256), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)

        cond, aux = self.get_cond(camera)

        if self.cfg.use_anisotropic_schedule:
            azimuth_deg = camera["azimuth"] % 360
            degrees_from_canonical = torch.min(azimuth_deg, 360.0 - azimuth_deg)
            anisotropic_frac = degrees_from_canonical / 180.0

            # at 180 degrees from the canonical view, the offset is maximized
            anisotropic_steps = torch.clamp(
                self.global_step - self.cfg.anisotropic_offset * anisotropic_frac,
                min=0,
                max=None,
            ).to(torch.int32)

            ts = []
            for anisotropic_step in anisotropic_steps.tolist():
                min_step_percent = C(self.cfg.min_step_percent, 0, anisotropic_step)
                max_step_percent = C(self.cfg.max_step_percent, 0, anisotropic_step)

                min_step = int(self.num_train_timesteps * min_step_percent)
                max_step = int(self.num_train_timesteps * max_step_percent)
                t = torch.randint(
                    min_step,
                    max_step + 1,
                    [1],
                    dtype=torch.long,
                    device=self.device,
                )
                ts.append(t)
            t = torch.cat(ts)

            # print(azimuth_deg)
            # print(anisotropic_steps)
            # print()
            # import pdb
            # pdb.set_trace()

        else:
            min_step = self.min_step
            max_step = self.max_step

            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            t = torch.randint(
                min_step,
                max_step + 1,
                [batch_size],
                dtype=torch.long,
                device=self.device,
            )

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            x_in = torch.cat([latents_noisy] * 2)
            t_in = torch.cat([t] * 2)
            noise_pred = self.model.apply_model(x_in, t_in, cond)

        # perform guidance
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

        # import pdb
        # pdb.set_trace()

        if aux['nearest_idxs'] is not None:
            is_main_camera = (torch.stack(aux['nearest_idxs']) == 0)[:, None, None, None]
            is_aux_camera = ~is_main_camera 

            v_guidance_scale = self.cfg.guidance_scale * is_main_camera + self.cfg.guidance_scale_aux * is_aux_camera
        else:
            v_guidance_scale = self.cfg.guidance_scale
            is_main_camera = True
        # print(self.cfg.guidance_scale, self.cfg.guidance_scale_aux)

        noise_pred = noise_pred_uncond + v_guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        w = (1 - self.alphas[t]).reshape(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        if aux['c_crossattn_nearest'] is not None:
            rgb_for_clip = rgb.permute((0, 3, 1,2))  # NHWC -> NCHW
            rgb_for_clip = rgb_for_clip *2 - 1
            rgb_for_clip = F.interpolate(rgb_for_clip, (256, 256), mode='bilinear')

            clip_emb = self.model.get_learned_conditioning(rgb_for_clip.to(self.weights_dtype))

            loss_clip = F.mse_loss(aux['c_crossattn_nearest'], clip_emb, reduction='sum') / batch_size
            # import pdb
            # pdb.set_trace()
        else:
            loss_clip = None

        guidance_out = {
            "loss_sds": loss_sds,
            "loss_clip": loss_clip,
            "grad_norm": grad.norm(),
            "min_step": min_step,
            "max_step": max_step,
            "is_main_camera": is_main_camera,
        }



        # import pdb
        # pdb.set_trace()

        if guidance_eval:
            guidance_eval_utils = {
                "cond": cond,
                "t_orig": t,
                "latents_noisy": latents_noisy,
                "noise_pred": noise_pred,
            }
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
            texts = []
            for n, e, a, c in zip(
                guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances
            ):
                texts.append(
                    f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                )
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})

        return guidance_out

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(self, cond, t_orig, latents_noisy, noise_pred):
        # use only 50 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = (
            min(self.cfg.max_items_eval, latents_noisy.shape[0])
            if self.cfg.max_items_eval > 0
            else latents_noisy.shape[0]
        )  # batch size
        large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[
            :bs
        ].unsqueeze(
            -1
        )  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = self.decode_latents(latents_noisy[:bs]).permute(0, 2, 3, 1)

        # get prev latent
        latents_1step = []
        pred_1orig = []
        for b in range(bs):
            step_output = self.scheduler.step(
                noise_pred[b : b + 1], t[b], latents_noisy[b : b + 1], eta=1
            )
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = self.decode_latents(latents_1step).permute(0, 2, 3, 1)
        imgs_1orig = self.decode_latents(pred_1orig).permute(0, 2, 3, 1)

        latents_final = []
        for b, i in enumerate(idxs):
            latents = latents_1step[b : b + 1]
            c = {
                "c_crossattn": [cond["c_crossattn"][0][[b, b + len(idxs)], ...]],
                "c_concat": [cond["c_concat"][0][[b, b + len(idxs)], ...]],
            }
            for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False):
                # pred noise
                x_in = torch.cat([latents] * 2)
                t_in = torch.cat([t.reshape(1)] * 2).to(self.device)
                noise_pred = self.model.apply_model(x_in, t_in, c)
                # perform guidance
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
                # get prev latent
                latents = self.scheduler.step(noise_pred, t, latents, eta=1)[
                    "prev_sample"
                ]
            latents_final.append(latents)

        latents_final = torch.cat(latents_final)
        imgs_final = self.decode_latents(latents_final).permute(0, 2, 3, 1)

        return {
            "bs": bs,
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
            "imgs_final": imgs_final,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.global_step = global_step

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )

        self.p_use_aux_cameras_val = C(self.cfg.p_use_aux_cameras, epoch, global_step)

    # verification - requires `vram_O = False` in load_model_from_config
    @torch.no_grad()
    def generate(
        self,
        image,  # image tensor [1, 3, H, W] in [0, 1]
        elevation=0,
        azimuth=0,
        camera_distances=0,  # new view params
        c_crossattn=None,
        c_concat=None,
        scale=3,
        ddim_steps=50,
        post_process=True,
        ddim_eta=1,
    ):
        raise NotImplementedError("Need to migrate to new camera API.")

    # verification - requires `vram_O = False` in load_model_from_config
    @torch.no_grad()
    def gen_from_cond(
        self,
        cond,
        scale=3,
        ddim_steps=50,
        post_process=True,
        ddim_eta=1,
    ):
        # produce latents loop
        B = cond["c_crossattn"][0].shape[0] // 2
        latents = torch.randn((B, 4, 32, 32), device=self.device)
        self.scheduler.set_timesteps(ddim_steps)

        for t in self.scheduler.timesteps:
            x_in = torch.cat([latents] * 2)
            t_in = torch.cat([t.reshape(1).repeat(B)] * 2).to(self.device)

            noise_pred = self.model.apply_model(x_in, t_in, cond)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + scale * (
                noise_pred_cond - noise_pred_uncond
            )

            latents = self.scheduler.step(noise_pred, t, latents, eta=ddim_eta)[
                "prev_sample"
            ]

        imgs = self.decode_latents(latents)
        imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1) if post_process else imgs

        return imgs
