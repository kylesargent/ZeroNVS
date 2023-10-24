import os
import random
import shutil
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchmetrics import PearsonCorrCoef

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from threestudio.models.guidance import zero123_guidance
from torch_efficient_distloss import (
    eff_distloss,
    eff_distloss_native,
    flatten_eff_distloss,
)
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import lpips
import json
import numpy as np
import pickle
import viz


def _quantize(t):
    # t is in range [0, 1]
    t = torch.clip(t, 0, 1)
    t = t * 255.0
    t = t.to(torch.uint8)
    t = t.to(torch.float32)
    t = t / 255.0
    return t


def move_batch_to_device(batch, device):
    batch = {
        k: v.to(device)[None] if isinstance(v, torch.Tensor) else v
        for (k, v) in batch.items()
    }
    return batch


class MaskedBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask_ref):
        ctx.mask_ref = mask_ref
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_output *= ctx.mask_ref["mask"]
        return grad_output, None


@threestudio.register("zero123-system")
class Zero123(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        freq: dict = field(default_factory=dict)
        refinement: bool = False
        ambient_ratio_min: float = 0.5

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # no prompt processor
        self.guidance: zero123_guidance.Zero123Guidance = threestudio.find(
            self.cfg.guidance_type
        )(self.cfg.guidance)

        # visualize all training images
        all_images = (
            self.trainer.datamodule.train_dataloader().dataset.get_all_images_fullres()
        )
        self.save_image_grid(
            "all_training_images.png",
            [
                {"type": "rgb", "img": image, "kwargs": {"data_format": "HWC"}}
                for image in all_images
            ],
            name="on_fit_start",
            step=self.true_global_step,
        )

        self.pearson = PearsonCorrCoef().to(self.device)

    def training_substep(self, batch, batch_idx, guidance: str):
        """
        Args:
            guidance: one of "ref" (reference image supervision), "zero123"
        """
        if guidance == "ref":
            # bg_color = torch.rand_like(batch['rays_o'])
            ambient_ratio = 1.0
            shading = "diffuse"
            batch["shading"] = shading
        elif guidance == "zero123":
            batch = batch["random_camera"]
            ambient_ratio = (
                self.cfg.ambient_ratio_min
                + (1 - self.cfg.ambient_ratio_min) * random.random()
            )

        batch["bg_color"] = None
        batch["ambient_ratio"] = ambient_ratio

        out = self(batch)
        loss_prefix = f"loss_{guidance}_"

        loss_terms = {}

        def set_loss(name, value):
            loss_terms[f"{loss_prefix}{name}"] = value

        guidance_eval = (
            guidance == "zero123"
            and self.cfg.freq.guidance_eval > 0
            and self.true_global_step % self.cfg.freq.guidance_eval == 0
        )

        if guidance == "ref":
            gt_mask = batch["mask"]
            gt_rgb = batch["rgb"]

            # color loss
            gt_rgb = gt_rgb * gt_mask.float() + out["comp_rgb_bg"] * (
                1 - gt_mask.float()
            )
            set_loss("rgb", F.mse_loss(gt_rgb, out["comp_rgb"]))

            # mask loss
            set_loss("mask", F.mse_loss(gt_mask.float(), out["opacity"]))

            # depth loss
            if self.C(self.cfg.loss.lambda_depth) > 0:
                valid_gt_depth = batch["ref_depth"][gt_mask.squeeze(-1)].unsqueeze(1)
                valid_pred_depth = out["depth"][gt_mask].unsqueeze(1)
                # with torch.no_grad():
                #     A = torch.cat(
                #         [valid_gt_depth, torch.ones_like(valid_gt_depth)], dim=-1
                #     )  # [B, 2]
                #     X = torch.linalg.lstsq(A, valid_pred_depth).solution  # [2, 1]
                #     valid_gt_depth = A @ X  # [B, 1]
                set_loss("depth", F.mse_loss(valid_gt_depth, valid_pred_depth))

            # relative depth loss
            if self.C(self.cfg.loss.lambda_depth_rel) > 0:
                valid_gt_depth = batch["ref_depth"][gt_mask.squeeze(-1)]  # [B,]
                valid_pred_depth = out["depth"][gt_mask]  # [B,]
                set_loss(
                    "depth_rel", 1 - self.pearson(valid_pred_depth, valid_gt_depth)
                )

            # normal loss
            if self.C(self.cfg.loss.lambda_normal) > 0:
                valid_gt_normal = (
                    1 - 2 * batch["ref_normal"][gt_mask.squeeze(-1)]
                )  # [B, 3]
                valid_pred_normal = (
                    2 * out["comp_normal"][gt_mask.squeeze(-1)] - 1
                )  # [B, 3]
                set_loss(
                    "normal",
                    1 - F.cosine_similarity(valid_pred_normal, valid_gt_normal).mean(),
                )
        elif guidance == "zero123":
            # zero123
            if self.cfg.guidance.depth_threshold_for_anchor_guidance > 0:
                mask_ref = {}
                sds_rgb = MaskedBackward().apply(out["comp_rgb"], mask_ref)

                guidance_out = self.guidance(
                    sds_rgb,
                    camera=batch,
                    rgb_as_latents=False,
                    guidance_eval=guidance_eval,
                )

                mask = guidance_out["is_main_camera"] | (
                    out["depth"] > self.cfg.guidance.depth_threshold_for_anchor_guidance
                )
                mask_ref["mask"] = mask
            else:
                guidance_out = self.guidance(
                    out["comp_rgb"],
                    camera=batch,
                    rgb_as_latents=False,
                    guidance_eval=guidance_eval,
                )

            # claforte: TODO: rename the loss_terms keys
            set_loss("sds", guidance_out["loss_sds"])

            if self.C(self.cfg.loss.lambda_clip) > 0:
                assert guidance_out["loss_clip"] is not None
                set_loss("clip", guidance_out["loss_clip"])

        if self.C(self.cfg.loss.lambda_normal_smooth) > 0:
            if "comp_normal" not in out:
                raise ValueError(
                    "comp_normal is required for 2D normal smooth loss, no comp_normal is found in the output."
                )
            normal = out["comp_normal"]
            set_loss(
                "normal_smooth",
                (normal[:, 1:, :, :] - normal[:, :-1, :, :]).square().mean()
                + (normal[:, :, 1:, :] - normal[:, :, :-1, :]).square().mean(),
            )

        # import pdb
        # pdb.set_trace()

        # assert self.C(self.cfg.loss.lambda_distortion) > 0
        if self.C(self.cfg.loss.lambda_distortion) > 0:
            distortion_loss = flatten_eff_distloss(
                out["weights"][..., 0],
                out["t_points"][..., 0],
                out["t_intervals"][..., 0],
                out["ray_indices"],
            )
            # print(distortion_loss)
            set_loss("distortion", distortion_loss)

        if self.C(self.cfg.loss.lambda_3d_normal_smooth) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for normal smooth loss, no normal is found in the output."
                )
            if "normal_perturb" not in out:
                raise ValueError(
                    "normal_perturb is required for normal smooth loss, no normal_perturb is found in the output."
                )
            normals = out["normal"]
            normals_perturb = out["normal_perturb"]
            set_loss("3d_normal_smooth", (normals - normals_perturb).abs().mean())

        if not self.cfg.refinement:
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                set_loss(
                    "orient",
                    (
                        out["weights"].detach()
                        * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                    ).sum()
                    / (out["opacity"] > 0).sum(),
                )

            if guidance != "ref" and self.C(self.cfg.loss.lambda_sparsity) > 0:
                set_loss("sparsity", (out["opacity"] ** 2 + 0.01).sqrt().mean())

            if self.C(self.cfg.loss.lambda_opaque) > 0:
                opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
                set_loss(
                    "opaque", binary_cross_entropy(opacity_clamped, opacity_clamped)
                )
        else:
            if self.C(self.cfg.loss.lambda_normal_consistency) > 0:
                set_loss("normal_consistency", out["mesh"].normal_consistency())
            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                set_loss("laplacian_smoothness", out["mesh"].laplacian())

        loss = 0.0
        for name, value in loss_terms.items():
            self.log(f"train/{name}", value)
            if name.startswith(loss_prefix):
                loss_weighted = value * self.C(
                    self.cfg.loss[name.replace(loss_prefix, "lambda_")]
                )
                self.log(f"train/{name}_w", loss_weighted)
                loss += loss_weighted

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        self.log(f"train/loss_{guidance}", loss)

        if guidance_eval:
            self.guidance_evaluation_save(
                out["comp_rgb"].detach()[: guidance_out["eval"]["bs"]],
                guidance_out["eval"],
            )

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        # import pdb
        # pdb.set_trace()

        # 1 + 1

        if self.cfg.freq.get("ref_or_zero123", "accumulate") == "accumulate":
            do_ref = True
            do_zero123 = True
        elif self.cfg.freq.get("ref_or_zero123", "accumulate") == "alternate":
            do_ref = (
                self.true_global_step < self.cfg.freq.ref_only_steps
                or self.true_global_step % self.cfg.freq.n_ref == 0
            )
            do_zero123 = not do_ref

        total_loss = 0.0
        if do_zero123:
            out = self.training_substep(batch, batch_idx, guidance="zero123")
            total_loss += out["loss"]

        if do_ref:
            out = self.training_substep(batch, batch_idx, guidance="ref")
            total_loss += out["loss"]

        self.log("train/loss", total_loss, prog_bar=True)

        # sch = self.lr_schedulers()
        # sch.step()

        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-val/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["depth"][0],
                        "kwargs": {},
                    }
                ]
                if "depth" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            # claforte: TODO: don't hardcode the frame numbers to record... read them from cfg instead.
            name=f"validation_step_batchidx_{batch_idx}"
            if batch_idx in [0, 7, 15, 23, 29]
            else None,
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        filestem = f"it{self.true_global_step}-val"
        self.save_img_sequence(
            filestem,
            filestem,
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="validation_epoch_end",
            step=self.true_global_step,
        )
        shutil.rmtree(
            os.path.join(self.get_save_dir(), f"it{self.true_global_step}-val")
        )

    def _maybe_set_gt_to_pred_scale(self):
        if self.trainer.datamodule.cfg.view_synthesis is None:
            return

        self.eval()
        self.cuda()
        assert not self.training

        if not self.trainer.datamodule.cfg.view_synthesis.rescale:
            return

        scene = self.trainer.datamodule.scene
        if self.trainer.datamodule.cfg.view_synthesis.rescale_mode == "manual":
            assert (
                self.trainer.datamodule.cfg.view_synthesis.manual_gt_to_pred_scale
                is not None
            )
            self.trainer.datamodule.gt_to_pred_scale = (
                self.trainer.datamodule.cfg.view_synthesis.manual_gt_to_pred_scale
            )
        elif self.trainer.datamodule.cfg.view_synthesis.rescale_mode == "icp":
            point_clouds = []
            for batch in self.trainer.datamodule.unscaled_test_dataset:
                batch = move_batch_to_device(batch, self.device)
                out = self(batch)

                assert out["comp_rgb"].shape[0] == out["depth"].shape[0] == 1

                image = out["comp_rgb"][0].cpu().numpy()
                depth = out["depth"][0].cpu().numpy()

                extrinsics = batch["c2w"][0].cpu().numpy()
                intrinsics = scene["intrinsics"]

                point_cloud, _ = viz.get_homogeneous_point_cloud(
                    image, depth, intrinsics, extrinsics
                )
                point_cloud = point_cloud[..., :3]
                point_clouds.append(point_cloud)

            predicted_point_cloud = np.concatenate(point_clouds, axis=0)
            gt_point_cloud = scene["point_cloud"]
            homogeneous_gt_point_cloud = np.concatenate(
                [gt_point_cloud, np.ones_like(gt_point_cloud[..., :1])], axis=1
            )
            gt_point_cloud = np.matmul(
                self.trainer.datamodule.dtu_to_threestudio_transform[0],
                homogeneous_gt_point_cloud.T,
            ).T

            with open("test_outputs/predicted_point_cloud.npy", "wb") as fp:
                np.save(fp, predicted_point_cloud)
            with open("test_outputs/gt_point_cloud.npy", "wb") as fp:
                np.save(fp, gt_point_cloud)
            raise

        elif self.trainer.datamodule.cfg.view_synthesis.rescale_mode in [
            "lstq_depth",
            "lstq_disparity",
        ]:
            batch = move_batch_to_device(
                self.trainer.datamodule.canonical_batch, self.device
            )

            with torch.no_grad():
                result = self(batch)

                pred_depth = result["depth"][:, None, ..., 0]
                pred_depth = torch.nn.functional.interpolate(
                    pred_depth, scale_factor=1, mode="nearest"
                )
                pred_depth = pred_depth[0, 0].cpu().numpy()

            gt_depth = self.trainer.datamodule.scene["test_input_depth_map"]
            valid_mask = gt_depth != 0
            pred_depth[~valid_mask] == 0

            if self.trainer.datamodule.cfg.view_synthesis.rescale_mode == "lstq_depth":
                # compute lstq scalar alignment of pred depth to GT depth
                pred_to_gt_scale = (pred_depth * gt_depth).sum() / (
                    gt_depth * gt_depth
                ).sum()
                gt_to_pred_scale = 1.0 / pred_to_gt_scale

                print("gt_to_pred_scale: ", gt_to_pred_scale)
                self.trainer.datamodule.gt_to_pred_scale = gt_to_pred_scale
            elif (
                self.trainer.datamodule.cfg.view_synthesis.rescale_mode
                == "lstq_disparity"
            ):
                # compute lstq scalar alignment of pred depth to GT depth
                gt_disparity = np.zeros_like(gt_depth)
                gt_disparity[valid_mask] = 1.0 / np.clip(
                    gt_depth[valid_mask], 1e-3, None
                )
                pred_disparity = np.zeros_like(pred_depth)
                pred_disparity[valid_mask] = 1.0 / np.clip(
                    pred_depth[valid_mask], 1e-3, None
                )

                pred_to_gt_disparity_scale = (pred_disparity * gt_disparity).sum() / (
                    gt_disparity * gt_disparity
                ).sum()
                gt_to_pred_scale = pred_to_gt_disparity_scale

                print("gt_to_pred_scale: ", gt_to_pred_scale)
                self.trainer.datamodule.gt_to_pred_scale = gt_to_pred_scale

                # print("dumping locals...")

                # # import pdb
                # # pdb.set_trace()
                # serializable_locals = {
                #     k:v
                #     for (k,v) in locals().items()
                #     if not callable(v)
                # }

                # with open("test_outputs/locals.pkl", "wb") as fp:
                #     pickle.dump(serializable_locals, fp)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def on_fit_end(self):
        super().on_fit_end()
        self._maybe_set_gt_to_pred_scale()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        out = self(batch)

        if "gt_rgb" in batch:
            gt_view = batch["gt_rgb"] / 2 + 0.5
            pred_view = out["comp_rgb"]

            view_synthesis_cfg = self.trainer.datamodule.cfg.view_synthesis

            if view_synthesis_cfg.quantize:
                gt_view = _quantize(gt_view)
                pred_view = _quantize(pred_view)

            # import pdb
            # pdb.set_trace()

            lpips = (
                self.lpips_fn(
                    pred_view.permute((0, 3, 1, 2)).to(torch.float32).to(self.device)
                    * 2
                    - 1,
                    gt_view.permute((0, 3, 1, 2)).to(torch.float32).to(self.device) * 2
                    - 1,
                )
                .cpu()
                .numpy()
            )
            lpips = lpips.item()

            psnr = peak_signal_noise_ratio(
                gt_view.cpu().numpy(), pred_view.cpu().numpy(), data_range=1.0
            )

            # assert pred_view.min() >= 0.
            # assert pred_view.max() <= 1.

            ssim = structural_similarity(
                gt_view[0].cpu().numpy(),
                pred_view[0].cpu().numpy(),
                data_range=1.0,
                channel_axis=2,
            )

            # import pdb
            # pdb.set_trace()

            metrics = dict(ssim=ssim, psnr=psnr, lpips=lpips)

            if (
                batch["index"] != batch["test_input_idx"]
                and batch["index"].item()
                not in self.trainer.datamodule.cfg.view_synthesis.excluded_views
            ):
                self.test_step_outputs.append(metrics)

        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["gt_rgb"][0] / 2 + 0.5,
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "gt_rgb" in batch
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["depth"][0],
                        "kwargs": {},
                    }
                ]
                if "depth" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self, *args, **kwargs):
        if self.test_step_outputs:
            reduced_metrics = {
                key: np.mean([output[key] for output in self.test_step_outputs]).item()
                for key in self.test_step_outputs[0]
            }
            print(reduced_metrics)

            with open(os.path.join(self.get_save_dir(), "metrics.json"), "w") as fp:
                json.dump(reduced_metrics, fp)

        if hasattr(self, "guidance") and self.guidance.aux_rgbs is not None:
            for i, image in enumerate(self.guidance.aux_rgbs):
                image = Image.fromarray(
                    np.clip(image[0] * 255, 0, 255).astype(np.uint8)
                )
                image.save(os.path.join(self.get_save_dir(), f"aux_pred_{i}.png"))

        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
        shutil.rmtree(
            os.path.join(self.get_save_dir(), f"it{self.true_global_step}-test")
        )
