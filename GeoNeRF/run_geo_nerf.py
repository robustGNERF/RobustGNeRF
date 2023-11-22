# GeoNeRF is a generalizable NeRF model that renders novel views
# without requiring per-scene optimization. This software is the 
# implementation of the paper "GeoNeRF: Generalizing NeRF with 
# Geometry Priors" by Mohammad Mahdi Johari, Yann Lepoittevin,
# and Francois Fleuret.

# Copyright (c) 2022 ams International AG

# This file is part of GeoNeRF.
# GeoNeRF is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.

# GeoNeRF is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GeoNeRF. If not, see <http://www.gnu.org/licenses/>.

# This file incorporates work covered by the following copyright and  
# permission notice:

    # MIT License

    # Copyright (c) 2021 apchenstu

    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:

    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.

    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer, loggers
from pytorch_lightning.loggers import WandbLogger

import os
import time
import numpy as np
import imageio
import lpips
from skimage.metrics import structural_similarity as ssim

from model.geo_reasoner import CasMVSNet
from model.self_attn_renderer import Renderer
from utils.rendering import render_rays
from utils.utils import (
    load_ckpt,
    init_log,
    get_rays_pts,
    SL1Loss,
    self_supervision_loss,
    img2mse,
    mse2psnr,
    acc_threshold,
    abs_error,
    visualize_depth,
)
from utils.options import config_parser
from data.get_datasets import (
    get_training_dataset,
    get_finetuning_dataset,
    get_validation_dataset,
)
from data.llff import de_linearize
from tqdm import tqdm
import torch.nn.functional as F
from utils.io_utils import colorize, img_HWC2CHW
lpips_fn = lpips.LPIPS(net="vgg")
ALPHA = 0.99997

class GeoNeRF(LightningModule):
    def __init__(self, hparams):
        super(GeoNeRF, self).__init__()
        self.hparams.update(vars(hparams))
        self.wr_cntr = 0
        self.iter_ = 0

        self.depth_loss = SL1Loss()
        self.learning_rate = hparams.lrate

        # Create geometry_reasoner and renderer models
        self.geo_reasoner = CasMVSNet(use_depth=hparams.use_depth, dip_feat=hparams.dip_feat).cuda()
        self.renderer = Renderer(
            nb_samples_per_ray=hparams.nb_coarse + hparams.nb_fine
        ).cuda()

        self.eval_metric = [0.01, 0.05, 0.1]
        self.automatic_optimization = False
        self.save_hyperparameters()

    def unpreprocess(self, data, shape=(1, 1, 3, 1, 1)):
        if self.hparams.burst_training:
            return data
        # to unnormalize image for visualization
        device = data.device
        mean = (
            torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225])
            .view(*shape)
            .to(device)
        )
        std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)

        return (data - mean) / std

    def prepare_data(self):
        if self.hparams.scene == "None":  ## Generalizable
            self.train_dataset, self.train_sampler = get_training_dataset(self.hparams)
            self.val_dataset = get_validation_dataset(self.hparams)
        else:  ## Fine-tune
            self.train_dataset, self.train_sampler = get_finetuning_dataset(
                self.hparams
            )
            self.val_dataset = get_validation_dataset(self.hparams)

    def configure_optimizers(self):
        eps = 1e-5

        opt = torch.optim.Adam(
            list(self.geo_reasoner.parameters()) + list(self.renderer.parameters()),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
        )
        sch = CosineAnnealingLR(opt, T_max=self.hparams.num_steps, eta_min=eps)

        return [opt], [sch]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            shuffle=True if self.train_sampler is None else False,
            num_workers=8,
            batch_size=1,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=1,
            batch_size=1,
            pin_memory=True,
        )

    def training_step(self, batch, batch_nb):
        loss = 0
        self.iter_ += 1
        nb_views = self.hparams.nb_views + (1 if self.geo_reasoner.dip_feat else 0)
        H, W = batch["images"].shape[-2:]
        H, W = int(H), int(W)

        ## Inferring Geometry Reasoner
        out = self.geo_reasoner(
            imgs=batch["images"][:, :nb_views] if not self.hparams.clean_src_imgs else batch["clean_images"][:, :nb_views],
            affine_mats=batch["affine_mats"][:, :nb_views],
            affine_mats_inv=batch["affine_mats_inv"][:, :nb_views],
            near_far=batch["near_fars"][:, :nb_views],
            closest_idxs=batch["closest_idxs"][:, :nb_views],
            gt_depths=batch["depths_aug"][:, :nb_views],
            c2ws=batch["c2ws"][:, :nb_views],
            w2cs=batch["w2cs"][:, :nb_views],
            intrinsics=batch["intrinsics"][:, :nb_views],
            training=True
        )
        if self.geo_reasoner.dip_feat:
            nb_views -= 1
            feats_vol, feats_fpn, depth_map, depth_values, reconst_imgs, patch_depth, stage_depths, input_rgbds = out
        else:
            reconst_imgs, patch_depth, stage_depths, input_rgbds = None, None, None, None
            feats_vol, feats_fpn, depth_map, depth_values = out
        ## Normalizing depth maps in NDC coordinate
        depth_map_norm = {}
        for l in range(3):
            depth_map_norm[f"level_{l}"] = (
                depth_map[f"level_{l}"].detach() - depth_values[f"level_{l}"][:, :, 0]
            ) / (
                depth_values[f"level_{l}"][:, :, -1]
                - depth_values[f"level_{l}"][:, :, 0]
            )

        unpre_imgs = self.unpreprocess(batch["images"] if not self.hparams.clean_src_imgs else batch["clean_images"])  
        if 'clean_images' in batch.keys():
            unpre_clean_imgs = self.unpreprocess(batch["clean_images"])
        else:
            unpre_clean_imgs = None

        if self.geo_reasoner.dip_feat:
            source_rgbs   = self.unpreprocess(reconst_imgs[None]) 
        else:
            source_rgbs   = unpre_imgs[:, :nb_views]
        target_rgb  = unpre_imgs[0, -1] if unpre_clean_imgs == None else unpre_clean_imgs[0, -1]

        (
            pts_depth,
            rays_pts,
            rays_pts_ndc,
            rays_dir,
            rays_gt_rgb,
            rays_gt_depth,
            rays_pixs,
        ) = get_rays_pts(
            H,
            W,
            batch["c2ws"],
            batch["w2cs"],
            batch["intrinsics"],
            batch["near_fars"],
            depth_values,
            self.hparams.nb_coarse,
            self.hparams.nb_fine,
            nb_views=nb_views,
            train=True,
            train_batch_size=self.hparams.batch_size,
            target_img=target_rgb,
            target_depth=batch["depths_h"][0, -1],
            target_alpha=batch['target_alpha']
            
        )

        ## Rendering
        rendered_rgb, rendered_depth = render_rays(
            c2ws=batch["c2ws"][0, :nb_views],
            rays_pts=rays_pts,
            rays_pts_ndc=rays_pts_ndc,
            pts_depth=pts_depth,
            rays_dir=rays_dir,
            feats_vol=feats_vol,
            feats_fpn=feats_fpn[:, :nb_views],
            imgs=source_rgbs,
            depth_map_norm=depth_map_norm,
            renderer_net=self.renderer,
        )


        mask = rays_gt_depth > 0
        depth_available = mask.sum() > 0

        if reconst_imgs != None:
            weight = max(ALPHA ** self.iter_, 0.01)
            reconst_loss = F.l1_loss(reconst_imgs, batch["clean_images"][0, :nb_views]) * 0.05 * weight
            loss += reconst_loss
            self.log("train/reconst_loss", reconst_loss.item(), prog_bar=False)
            self.log("train/weight", weight, prog_bar=False)

            with torch.no_grad():
                reconst_psnr= mse2psnr(img2mse(self.unpreprocess(reconst_imgs[None]), self.unpreprocess(batch["clean_images"][0, :nb_views])))
                noisy_psnr= mse2psnr(img2mse(self.unpreprocess(batch["images"][0, :nb_views]), self.unpreprocess(batch["clean_images"][0, :nb_views])))
                self.log("train/reconst_psnr", reconst_psnr.item(), prog_bar=False)
                self.log("train/noisy_psnr", noisy_psnr.item(), prog_bar=False)

        if patch_depth != None:
            depth_loss = 0 

            coords = torch.zeros_like(rays_pixs.permute(1,0))

            coords[:, 0] = rays_pixs[1]
            coords[:, 0] = (coords[:, 0] / W ) * 2 - 1  
            coords[:, 1] = rays_pixs[0]
            coords[:, 1] = (coords[:, 1] / H ) * 2 - 1  
            for k in stage_depths.keys():
                for depth in stage_depths[k]:
                    sel_ref_depth = F.grid_sample(depth[-1:], coords[None, None].to(depth.device)).squeeze()
                    depth_loss += F.smooth_l1_loss(sel_ref_depth, rendered_depth.detach()) * 0.01
            loss += depth_loss
            self.log("train/depth_loss", depth_loss.item(), prog_bar=False)

        ## Reconstruction loss
        # l1_loss = F.l1_loss(rendered_rgb, rays_gt_rgb)
        # loss = loss + l1_loss
        mse_loss = img2mse(rendered_rgb, rays_gt_rgb)
        loss = loss + mse_loss

        with torch.no_grad():
            self.log("train/loss", loss.item(), prog_bar=True)
            psnr = mse2psnr(mse_loss.detach())
            self.log("train/PSNR", psnr.item(), prog_bar=False)
            self.log("train/img_mse_loss", mse_loss.item(), prog_bar=False)

        # Manual Optimization
        self.manual_backward(loss)

        opt = self.optimizers()
        sch = self.lr_schedulers()

        # Warming up the learning rate
        if self.trainer.global_step < self.hparams.warmup_steps:
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams.warmup_steps
            )
            for pg in opt.param_groups:
                pg["lr"] = lr_scale * self.learning_rate

        self.log("train/lr", opt.param_groups[0]["lr"], prog_bar=False)

        if self.geo_reasoner.dip_feat:
            torch.nn.utils.clip_grad_norm_(self.geo_reasoner.patchmatch.parameters(), 0.1)
            torch.nn.utils.clip_grad_norm_(self.geo_reasoner.restormer.parameters(), 0.1)
            
        opt.step()
        opt.zero_grad()
        sch.step()

        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        ## This makes Batchnorm to behave like InstanceNorm
        self.geo_reasoner.train()

        log_keys = [
            "val_psnr",
            "val_ssim",
            "val_lpips",
            "val_depth_loss_r",
            "val_abs_err",
            "mask_sum",
        ] + [f"val_acc_{i}mm" for i in self.eval_metric]
        log = {}
        log = init_log(log, log_keys)

        H, W = batch["images"].shape[-2:]
        H, W = int(H), int(W)

        nb_views = self.hparams.nb_views

        if 'eval_gain' in batch.keys():
            log['eval_gain'] = int(batch['eval_gain'].item())

        start = time.time()
        with torch.no_grad():
            ## Inferring Geometry Reasoner
            out = self.geo_reasoner(
                imgs=batch["images"][:, :nb_views] if not self.hparams.clean_src_imgs else batch["clean_images"][:, :nb_views],
                affine_mats=batch["affine_mats"][:, :nb_views],
                affine_mats_inv=batch["affine_mats_inv"][:, :nb_views],
                near_far=batch["near_fars"][:, :nb_views],
                closest_idxs=batch["closest_idxs"][:, :nb_views],
                gt_depths=batch["depths_aug"][:, :nb_views],
                c2ws=batch["c2ws"][:, :nb_views],
                w2cs=batch["w2cs"][:, :nb_views],
                intrinsics=batch["intrinsics"][:, :nb_views],   
            )

            if self.geo_reasoner.dip_feat:
                feats_vol, feats_fpn, depth_map, depth_values, reconst_imgs, patch_depth, stage_depths, input_rgbds = out
            else:
                reconst_imgs, patch_depth, stage_depths, input_rgbds = None, None, None, None
                feats_vol, feats_fpn, depth_map, depth_values = out


            ## Normalizing depth maps in NDC coordinate
            depth_map_norm = {}
            for l in range(3):
                depth_map_norm[f"level_{l}"] = (
                    depth_map[f"level_{l}"] - depth_values[f"level_{l}"][:, :, 0]
                ) / (
                    depth_values[f"level_{l}"][:, :, -1]
                    - depth_values[f"level_{l}"][:, :, 0]
                )

            # unpre_imgs = self.unpreprocess(batch["images"])
            unpre_imgs = self.unpreprocess(batch["images"]  if not self.hparams.clean_src_imgs else batch["clean_images"]) 
            if 'clean_images' in batch.keys():
                unpre_clean_imgs = self.unpreprocess(batch["clean_images"])
            else:
                unpre_clean_imgs = None

            if self.geo_reasoner.dip_feat:
                source_rgbs   = self.unpreprocess(reconst_imgs[None]) 
            else:
                source_rgbs   = unpre_imgs[:, :nb_views]

            rendered_rgb, rendered_depth = [], []
            for chunk_idx in tqdm(range(
                H * W // self.hparams.chunk + int(H * W % self.hparams.chunk > 0)
            )):
                pts_depth, rays_pts, rays_pts_ndc, rays_dir, _, _, _ = get_rays_pts(
                    H,
                    W,
                    batch["c2ws"],
                    batch["w2cs"],
                    batch["intrinsics"],
                    batch["near_fars"],
                    depth_values,
                    self.hparams.nb_coarse,
                    self.hparams.nb_fine,
                    nb_views=nb_views,
                    chunk=self.hparams.chunk,
                    chunk_idx=chunk_idx,
                )

                ## Rendering
                rend_rgb, rend_depth = render_rays(
                    c2ws=batch["c2ws"][0, :nb_views],
                    rays_pts=rays_pts,
                    rays_pts_ndc=rays_pts_ndc,
                    pts_depth=pts_depth,
                    rays_dir=rays_dir,
                    feats_vol=feats_vol,
                    feats_fpn=feats_fpn[:, :nb_views],
                    imgs=source_rgbs,
                    depth_map_norm=depth_map_norm,
                    renderer_net=self.renderer,
                )
                rendered_rgb.append(rend_rgb)
                rendered_depth.append(rend_depth)
            rendered_rgb = torch.cat(rendered_rgb).reshape(H, W, 3).permute(2, 0, 1)
            rendered_depth = torch.cat(rendered_depth).reshape(H, W)
            # torch.cuda.synchronize()
            # print("Render time = ",round(time.time() - start,3))
            # exit()
            ## Check if there is any ground truth depth information for the dataset
            depth_available = batch["depths_h"].sum() > 0

            if not self.hparams.burst_training:
                gt_img = unpre_imgs[0, -1] if unpre_clean_imgs == None else unpre_clean_imgs[0, -1] 
            else:
                rendered_rgb = de_linearize(rendered_rgb, batch['white_level'])
                gt_img = de_linearize(batch["clean_images"][0, -1], batch['white_level']).clamp(0,1)

            rendered_rgb = torch.clamp(rendered_rgb, 0, 1)

            ## Evaluate only on pixels with meaningful ground truth depths
            if depth_available:
                mask = batch["depths_h"] > 0
                img_gt_masked = (gt_img * mask[0, -1][None]).cpu()
                rendered_rgb_masked = (rendered_rgb * mask[0, -1][None]).cpu()
            else:
                img_gt_masked = gt_img.cpu()
                rendered_rgb_masked = rendered_rgb.cpu()

            unpre_imgs = unpre_imgs.cpu()
            rendered_rgb, rendered_depth = rendered_rgb.cpu(), rendered_depth.cpu()
            img_err_abs = (rendered_rgb_masked - img_gt_masked).abs()

            depth_target = batch["depths_h"][0, -1].cpu()
            mask_target = depth_target > 0

            if depth_available:
                log["val_psnr"] = mse2psnr(torch.mean(img_err_abs[:, mask_target] ** 2))
            else:
                log["val_psnr"] = mse2psnr(torch.mean(img_err_abs**2))
            
            log["val_ssim"] = ssim( rendered_rgb_masked.permute(1, 2, 0).numpy(), img_gt_masked.permute(1, 2, 0).numpy(), data_range=1, multichannel=True, channel_axis=-1)
            log["val_lpips"] = lpips_fn(rendered_rgb_masked[None] * 2 - 1, img_gt_masked[None] * 2 - 1).item()  # Normalize to [-1,1]

            depth_minmax = [
                0.9 * batch["near_fars"].min().detach().cpu().numpy(),
                1.1 * batch["near_fars"].max().detach().cpu().numpy(),
            ]
            rendered_depth_vis, _ = visualize_depth(rendered_depth, depth_minmax)

            if depth_available:
                log["val_abs_err"] = abs_error(
                    rendered_depth, depth_target, mask_target
                ).sum()
                log[f"val_acc_{self.eval_metric[0]}mm"] = acc_threshold(
                    rendered_depth, depth_target, mask_target, self.eval_metric[0]
                ).sum()
                log[f"val_acc_{self.eval_metric[1]}mm"] = acc_threshold(
                    rendered_depth, depth_target, mask_target, self.eval_metric[1]
                ).sum()
                log[f"val_acc_{self.eval_metric[2]}mm"] = acc_threshold(
                    rendered_depth, depth_target, mask_target, self.eval_metric[2]
                ).sum()
                log["mask_sum"] = mask_target.float().sum()

            img_vis = (
                torch.cat(
                    (
                        gt_img[None].cpu(),
                        torch.stack([rendered_rgb, img_err_abs * 5]),
                        rendered_depth_vis[None],
                    ),
                    dim=0,
                )
                .clip(0, 1)
                .permute(2, 0, 3, 1)
                .reshape(H, -1, 3)
                .numpy()
            )

            vis_interval = 1
            os.makedirs(
                f"{self.hparams.logdir}/{self.hparams.dataset_name}/{self.hparams.expname}/rendered_results/",
                exist_ok=True,
            )
            os.makedirs(
                f"{self.hparams.logdir}/{self.hparams.dataset_name}/{self.hparams.expname}/evaluation/",
                exist_ok=True,
            )

            if batch_nb % vis_interval == 0:
                imageio.imwrite(
                    f"{self.hparams.logdir}/{self.hparams.dataset_name}/{self.hparams.expname}/rendered_results/{self.wr_cntr:03d}.png",
                    (
                        rendered_rgb.detach().permute(1, 2, 0).clip(0.0, 1.0).cpu().numpy()
                        * 255
                    ).astype("uint8"),
                )

                imageio.imwrite(
                    f"{self.hparams.logdir}/{self.hparams.dataset_name}/{self.hparams.expname}/evaluation/{self.global_step:08d}_{self.wr_cntr:02d}.png",
                    (img_vis * 255).astype("uint8"),
                )


            log['blur_level'] = batch['blur_level'][0]
            if reconst_imgs != None :
                with torch.no_grad():
                    unproc_noisy    = self.unpreprocess(batch["images"][0, :nb_views]).squeeze()
                    unproc_clean    = self.unpreprocess(batch["clean_images"][0, :nb_views]).squeeze()
                    unproc_reconst  = self.unpreprocess(reconst_imgs).squeeze()
                    reconst_psnr= mse2psnr(img2mse(unproc_reconst, unproc_clean)).cpu()
                    noisy_psnr= mse2psnr(img2mse(unproc_noisy, unproc_clean)).cpu()
                    if self.hparams.burst_training:
                        unproc_noisy = de_linearize(unproc_noisy, batch['white_level']).clamp(0,1)
                        unproc_clean = de_linearize(unproc_clean, batch['white_level']).clamp(0,1)
                        unproc_reconst = de_linearize(unproc_reconst, batch['white_level']).clamp(0,1)

                log['reconst_psnr'] = reconst_psnr
                log['noisy_psnr']   = noisy_psnr

                vis_imgs = torch.stack([unproc_noisy, unproc_reconst, unproc_clean], dim=0)
                unpreprocess_reconst = vis_imgs.permute(0,3,1,4,2).clamp(0,1)
                unpreprocess_reconst = unpreprocess_reconst.reshape(H * 3, -1, 3)
                if batch_nb % vis_interval == 0:
                    imageio.imwrite(
                        f"{self.hparams.logdir}/{self.hparams.dataset_name}/{self.hparams.expname}/evaluation/reconst_{self.global_step:08d}_{self.wr_cntr:02d}.png",
                        (unpreprocess_reconst.cpu().numpy() * 255).astype("uint8"),
                    )

            if patch_depth != None:
                patch_depth_colorized = img_HWC2CHW(colorize(patch_depth.squeeze().permute(1,0,2).reshape(H,-1)))
                if batch_nb % vis_interval == 0:
                    imageio.imwrite(
                        f"{self.hparams.logdir}/{self.hparams.dataset_name}/{self.hparams.expname}/evaluation/patch_depth_{self.global_step:08d}_{self.wr_cntr:02d}.png",
                        (patch_depth_colorized.cpu().permute(1,2,0).numpy() * 255).astype("uint8"),
                    )

            if input_rgbds != None:
                ref_warped_imgs = input_rgbds[:,:,:3].permute(2,0,3,1,4).reshape(3, H * input_rgbds.shape[0], -1)
                mask = ref_warped_imgs == 0
                ref_warped_imgs = self.unpreprocess(ref_warped_imgs).squeeze()
                if self.hparams.burst_training:
                    ref_warped_imgs = de_linearize(ref_warped_imgs, batch['white_level'])
                ref_warped_imgs[mask] = 0
                ref_warped_imgs = ref_warped_imgs.permute(1,2,0).clamp(0,1)
                if batch_nb % vis_interval == 0:
                    imageio.imwrite(
                        f"{self.hparams.logdir}/{self.hparams.dataset_name}/{self.hparams.expname}/evaluation/warped_{self.global_step:08d}_{self.wr_cntr:02d}.png",
                        (ref_warped_imgs.cpu().numpy() * 255).astype("uint8"),
                    )


            print(f"Image {self.wr_cntr:02d} rendered.")
            self.wr_cntr += 1

        return log

    def validation_epoch_end(self, outputs):
        psnr = torch.stack([x["val_psnr"] for x in outputs]).numpy()
        ssim = np.stack([x["val_ssim"] for x in outputs])
        lpips = np.stack([x["val_lpips"] for x in outputs])

        mean_psnr = psnr.mean()
        mean_ssim = ssim.mean()
        mean_lpips = lpips.mean()


        if 'blur_level' in outputs[0].keys():
            blur_levels = [x['blur_level'] for x in outputs]
            result_per_level = {}
            for level, p, s, l in zip(blur_levels, psnr, ssim, lpips):
                if f"{level}_psnr" not in result_per_level.keys():
                    result_per_level[f"{level}_psnr"]  = [p]
                    result_per_level[f"{level}_ssim"]  = [s]
                    result_per_level[f"{level}_lpips"] = [l]
                else:
                    result_per_level[f"{level}_psnr"]  += [p]
                    result_per_level[f"{level}_ssim"]  += [s]
                    result_per_level[f"{level}_lpips"] += [l]
            
            for k in result_per_level.keys():
                self.log(f"val/{k}", np.mean(result_per_level[k]), prog_bar=False)
                print(k, np.mean(result_per_level[k]))

        if 'reconst_psnr' in outputs[0].keys():
            reconst_psnr    = torch.stack([x["reconst_psnr"] for x in outputs]).numpy().mean()
            noisy_psnr      = np.stack([x["noisy_psnr"] for x in outputs]).mean()

            self.log("val/reconst_psnr", reconst_psnr.item(), prog_bar=False)
            self.log("val/noisy_psnr", noisy_psnr.item(), prog_bar=False)


        mask_sum = torch.stack([x["mask_sum"] for x in outputs]).sum()
        mean_d_loss_r = torch.stack([x["val_depth_loss_r"] for x in outputs]).mean()
        mean_abs_err = torch.stack([x["val_abs_err"] for x in outputs]).sum() / mask_sum
        mean_acc_1mm = (
            torch.stack([x[f"val_acc_{self.eval_metric[0]}mm"] for x in outputs]).sum()
            / mask_sum
        )
        mean_acc_2mm = (
            torch.stack([x[f"val_acc_{self.eval_metric[1]}mm"] for x in outputs]).sum()
            / mask_sum
        )
        mean_acc_4mm = (
            torch.stack([x[f"val_acc_{self.eval_metric[2]}mm"] for x in outputs]).sum()
            / mask_sum
        )

        self.log("val/PSNR", mean_psnr, prog_bar=False)
        self.log("val/SSIM", mean_ssim, prog_bar=False)
        self.log("val/LPIPS", mean_lpips, prog_bar=False)
        if mask_sum > 0:
            self.log("val/d_loss_r", mean_d_loss_r, prog_bar=False)
            self.log("val/abs_err", mean_abs_err, prog_bar=False)
            self.log(f"val/acc_{self.eval_metric[0]}mm", mean_acc_1mm, prog_bar=False)
            self.log(f"val/acc_{self.eval_metric[1]}mm", mean_acc_2mm, prog_bar=False)
            self.log(f"val/acc_{self.eval_metric[2]}mm", mean_acc_4mm, prog_bar=False)

        with open(
            f"{self.hparams.logdir}/{self.hparams.dataset_name}/{self.hparams.expname}/{self.hparams.expname}_metrics.txt",
            "w",
        ) as metric_file:
            metric_file.write(f"PSNR: {mean_psnr}\n")
            metric_file.write(f"SSIM: {mean_ssim}\n")
            metric_file.write(f"LPIPS: {mean_lpips}")

        if self.hparams.eval_mode:
            exit()
        return


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    args = config_parser()
    geonerf = GeoNeRF(args)

    ## Checking to logdir to see if there is any checkpoint file to continue with
    ckpt_path = f"{args.logdir}/{args.dataset_name}/{args.expname}/ckpts"
    if os.path.isdir(ckpt_path) and len(os.listdir(ckpt_path)) > 0:
        ckpt_files = os.listdir(ckpt_path)
        ckpt_files.sort()
        ckpt_file = os.path.join(ckpt_path, ckpt_files[-1])
        print("######  Loading", ckpt_file, "######")
    else:
        ckpt_file = None

    ## Setting a callback to automatically save checkpoints
    checkpoint_callback = ModelCheckpoint(
        f"{args.logdir}/{args.dataset_name}/{args.expname}/ckpts",
        filename="ckpt_step-{step:06d}",
        auto_insert_metric_name=False,
        save_top_k=-1,
    )

    ## Setting up a logger
    if args.logger == "wandb":
        logger = WandbLogger(
            name=args.expname,
            project="GeoNeRF",
            save_dir=f"{args.logdir}",
            resume="allow",
            id=args.expname,
        )
    elif args.logger == "tensorboard":
        logger = loggers.TestTubeLogger(
            save_dir=f"{args.logdir}/{args.dataset_name}/{args.expname}",
            name=args.expname + "_logs",
            debug=False,
            create_git_tag=False,
        )
    else:
        logger = None

    args.use_amp = False if args.eval else True
    trainer = Trainer(
        max_steps=args.num_steps,
        callbacks=checkpoint_callback,
        checkpoint_callback=True,
        resume_from_checkpoint=ckpt_file,
        logger=logger,
        progress_bar_refresh_rate=1,
        gpus=1,
        num_sanity_val_steps=1 if not args.eval_mode else len(get_validation_dataset(args)),
        # val_check_interval= 2.0, #2000 if args.scene == "None" else 1.0,
        check_val_every_n_epoch=1000 if args.scene != 'None' else 1,
        benchmark=True,
        precision=16 if args.use_amp else 32,
        amp_level="O1",
    )

    if not args.eval:  ## Train
        # ckpt_file = "pretrained_weights/pretrained.ckpt"
        if args.load_ckpt != None:
            load_ckpt(geonerf.geo_reasoner, args.load_ckpt, "geo_reasoner")
            load_ckpt(geonerf.renderer, args.load_ckpt, "renderer")
            print("Loaded model from", args.load_ckpt)
        trainer.fit(geonerf)
    else:  ## Eval
        geonerf = GeoNeRF(args)

        if ckpt_file is None:
            if args.use_depth:
                ckpt_file = "pretrained_weights/pretrained_w_depth.ckpt"
            else:
                ckpt_file = "pretrained_weights/pretrained.ckpt"
        load_ckpt(geonerf.geo_reasoner, ckpt_file, "geo_reasoner")
        load_ckpt(geonerf.renderer, ckpt_file, "renderer")

        trainer.validate(geonerf)
