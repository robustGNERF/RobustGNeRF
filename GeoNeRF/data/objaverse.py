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

from torch.utils.data import Dataset
from torchvision import transforms as T

import os
import glob
import numpy as np
from PIL import Image
import torch
from utils.utils import get_nearest_pose_ids
from data.objaverse_test_scenes import objaverse_test_scenes
import json
import imageio
import random 

toTensor = T.ToTensor()

def normalize(v):
    return v / np.linalg.norm(v)


def average_poses(poses):
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)

    # convert to homogeneous coordinate for faster computation
    # by simply adding 0, 0, 0, 1 as the last row
    pose_avg_homo[:3] = pose_avg
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)

    # (N_images, 4, 4) homogeneous coordinate
    poses_homo = np.concatenate([poses, last_row], 1)

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, np.linalg.inv(pose_avg_homo) @ blender2opencv



# From DeepRep
# https://github.com/goutamgmb/deep-rep/blob/master/data/postprocessing_functions.py#L73

t = 0.0031308
gamma = 2.4
a = 1. / (1. / (t ** (1 / gamma) * (1. - (1 / gamma))) - 1.)  # 0.055
# a = 0.055
k0 = (1 + a) * (1 / gamma) * t ** ((1 / gamma) - 1.)  # 12.92
# k0 = 12.92
inv_t = t * k0

transform = T.ToTensor()

def de_linearize(rgb, wl=1.):
    """
    Process the RGB values in the inverse process of the approximate linearization, in a differential format
    @param rgb:
    @param wl:
    @return:
    """
    completed = False
    if isinstance(wl, torch.Tensor):
        if wl.ndim > 1 and rgb.ndim == 4:
            assert wl.shape[0] == rgb.shape[0]
            completed = True
            srgb = []
            for signal, level in zip(rgb, wl):
                signal = signal / level
                srgb_ = torch.where(signal > t, (1 + a) * torch.clamp(signal, min=t) ** (1 / gamma) - a, k0 * signal)

                k1 = (1 + a) * (1 / gamma)
                srgb_ = torch.where(signal > 1, k1 * signal - k1 + 1, srgb_)
                srgb.append(srgb_)

            srgb = torch.stack(srgb)                
    if not completed:
        rgb = rgb / wl
        srgb = torch.where(rgb > t, (1 + a) * torch.clamp(rgb, min=t) ** (1 / gamma) - a, k0 * rgb)

        k1 = (1 + a) * (1 / gamma)
        srgb = torch.where(rgb > 1, k1 * rgb - k1 + 1, srgb)
    return srgb


def de_linearize_np(rgb, wl=1.):
    rgb = rgb / wl
    srgb = np.where(rgb > t, (1 + a) * np.clip(rgb, a_min=t, a_max=np.inf) ** (1 / gamma) - a, k0 * rgb)

    # From deep-rep/data/postprocessing_functions.py  DenoisingPostProcess
    k1 = (1 + a) * (1 / gamma)
    srgb = np.where(rgb > 1, k1 * rgb - k1 + 1, srgb)
    return srgb


def re_linearize(rgb, wl=1.):
    """
    Approximate re-linearization of RGB values by revert gamma correction and apply white level
    Revert gamma correction
    @param rgb:
    @param wl:
    @return:
    """
    # return rgb
    return wl * (rgb ** 2.2)
    # degamma = torch.where(rgb > inv_t, ((torch.clamp(rgb, min=inv_t) + a) / (1 + a)) ** gamma, rgb / k0)

def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses

def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)

    if poses.shape[-1] == 5:
        hwf = poses[0, :3, -1:]
        c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    else:
        hwf = None
        c2w = viewmatrix(vec2, up, center)

    return c2w



class Objaverse_Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        split,
        nb_views,
        downSample=1.0,
        max_len=-1,
        scene="None",
        imgs_folder_name="images",
        burst_training = False,
        eval_mode=False,
    ):
        self.root_dir = root_dir
        self.split = split
        self.nb_views = nb_views
        self.scene = scene
        self.imgs_folder_name = imgs_folder_name
        self.eval_mode = eval_mode
        self.downsample = downSample
        self.max_len = max_len
        self.val_subsample = 2
        self.img_wh = (600, 400) #(int(576 * self.downsample), int(384 * self.downsample))
        # self.img_wh = (int(960 * self.downsample), int(720 * self.downsample))

        self.blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        self.n_nearby =  4
        self.burst_training = burst_training
        self.std_range = [-3.0, -1.5, -2.0, -1.25] #[-3.0, -0.5, -2.0, -0.5]
        self.eval_gains = [4,2,1]        

        if self.split == 'train':
            assert len(self.std_range) == 4
            self.get_noise_params = self.get_noise_params_train
        else:
            # load gain data from KPN paper https://bmild.github.io/kpn/index.html
            noise_data = np.load('/home/chan/data/synthetic_5d_j2_16_noiselevels6_wide_438x202x320x8.npz')

            sig_read_list = np.unique(noise_data['sig_read'])[2:]
            sig_shot_list = np.unique(noise_data['sig_shot'])[2:]

            self.log_sig_read = np.log10(sig_read_list)
            self.log_sig_shot = np.log10(sig_shot_list)

            self.d_read = np.diff(self.log_sig_read)[0]
            self.d_shot = np.diff(self.log_sig_shot)[0]

        self.define_transforms()
        self.build_metas()


    def get_noise_params_train(self):
        sigma_read_lim = self.std_range[:2]
        sigma_shot_lim = self.std_range[2:]

        sigma_read_log = np.random.default_rng().uniform(low=sigma_read_lim[0], high=sigma_read_lim[1], size=1).item()
        sigma_shot_log = np.random.default_rng().uniform(low=sigma_shot_lim[0], high=sigma_shot_lim[1], size=1).item()

        sigma_read = 10 ** sigma_read_log
        sigma_shot = 10 ** sigma_shot_log

        return sigma_read, sigma_shot

    def get_noise_params_level(self, gain_level):

        gain_read = gain_shot = np.log2(gain_level)
        sig_read = 10 ** (self.log_sig_read[0] + self.d_read * gain_read)
        sig_shot = 10 ** (self.log_sig_shot[0] + self.d_shot * gain_shot)

        return sig_read, sig_shot

    def add_noise_level(self, rgb, gain_level):
        sig_read, sig_shot = self.get_noise_params_level(gain_level)
        std = self.get_std(rgb, sig_read, sig_shot)
        noise = std * torch.randn_like(rgb)

        noise_rgb = rgb + noise
        sigma_estimate = self.get_std(noise_rgb.clamp(0, 1), sig_read, sig_shot)
        return noise_rgb, sigma_estimate

    @classmethod
    def get_std(cls, rgb, sig_read, sig_shot):
        return (sig_read ** 2 + sig_shot ** 2 * rgb) ** 0.5

    def add_noise(self, rgb):
        sig_read, sig_shot = self.get_noise_params()
        std = self.get_std(rgb, sig_read, sig_shot)
        noise = std * torch.randn_like(rgb)
        noise_rgb = rgb + noise
        sigma_estimate = self.get_std(noise_rgb.clamp(0, 1), sig_read, sig_shot)
        return noise_rgb, sigma_estimate

    def define_transforms(self):
        transforms = [T.ToTensor()]
        if not self.burst_training:
            transforms += [T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.transform = T.Compose(transforms)

    def build_metas(self):
        if self.scene != "None":
            self.scans = [
                os.path.basename(scan_dir)
                for scan_dir in sorted(
                    glob.glob(os.path.join(self.root_dir, self.scene))
                )
            ]
        else:
            self.scans = [
                "/".join(scan_dir.split("/")[-3:])
                for scan_dir in sorted(glob.glob(os.path.join(self.root_dir, "*", "*", "*")))
                if 'blur' in scan_dir and 'blur_mix' not in scan_dir and 'test' not in scan_dir #'exr' not in scan_dir and 'png' not in scan_dir
            ]
        print(f"################## {self.split} dataset meta Info loading")
        test_scans = ["/".join(dir_.split("/")[-3:]) for dir_ in objaverse_test_scenes]
        if self.split != 'train':
            # self.root_dir = self.root_dir.replace('output_blur_level_0922', 'output_blur_level_0922_denoised')
            self.scans = test_scans
        else:
            self.scans = [scan for scan in self.scans if scan not in test_scans]
            # self.scans = ['fern', 'trex', 'leaves', 'horns']

        self.meta = []
        self.image_paths = {}
        self.clean_image_paths = {}
        self.near_far = {}
        self.id_list = {}
        self.closest_idxs = {}
        self.c2ws = {}
        self.w2cs = {}
        self.intrinsics = {}
        self.affine_mats = {}
        self.affine_mats_inv = {}
        filtered_scans = []
        for scan in self.scans:
            
            pose_file = os.path.join(self.root_dir, scan, 'transforms.json')
            if not os.path.exists(pose_file):
                print("Pop", pose_file)
                # self.scans.pop(self.scans.index(scan))
                continue
            else:
                filtered_scans.append(scan)
            # print(pose_file)
            # Load the JSON data
            
            with open(pose_file, 'r') as f:
                data = json.load(f)
            
            frames = data['frames']
            blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            # Extract RGB files
            rgb_files = [os.path.join(self.root_dir, scan, frame['file_path'].split(os.sep)[-1]) for frame in frames]

            clean_rgb_files = []
            for rgb_file in rgb_files:
                rgb_file_clean = str(rgb_file).split('/')
                rgb_file_clean[-1] = 'clean_' + rgb_file_clean[-1]
                rgb_file_clean = '/'.join(rgb_file_clean)
                clean_rgb_files.append(rgb_file_clean)

            self.image_paths[scan] = rgb_files
            self.clean_image_paths[scan] = clean_rgb_files

            sh = imageio.imread(rgb_files[0]).shape
            h, w = sh[:2]

            poses = []
            far = 0
            near = 100
            for frame in frames:
                matrix = np.array(frame['transform_matrix'])
                matrix = matrix @ blender2opencv
                matrix[:3, -1] = matrix[:3, -1]
                poses.append(matrix)

            near = data['near'] * 0.8
            far = data['far'] * 1.2
            # print("After :", near, far)

            poses = np.array(poses)
            poses = recenter_poses(poses)

            bds = [np.array([near, far]) for _ in frames] 
            bds  = np.array(bds)
            scale = 1. / (bds.min() * 0.75)

            bds *= scale
            poses[:, :3, 3] *= scale
            poses = poses[:, :3].astype(np.float32)

            camera_angle_x = data['camera_angle_x']
            f = 0.5 * w / np.tan(0.5 * camera_angle_x)  # original focal length

            intrinsic = np.array([[f, 0, w / 2.],
                                [0, f, h / 2.],
                                [0, 0, 1]]).astype(np.float32)


            ####
            self.near_far[scan] = bds

            num_viewpoint = len(self.image_paths[scan])
            val_ids = [idx for idx in range(0, num_viewpoint, 8)][:: self.val_subsample]
            if not self.eval_mode:
                val_ids = val_ids[:1]            
            w, h = self.img_wh

            self.id_list[scan] = []
            self.closest_idxs[scan] = []
            self.c2ws[scan] = []
            self.w2cs[scan] = []
            self.intrinsics[scan] = []
            self.affine_mats[scan] = []
            self.affine_mats_inv[scan] = []
            for idx in range(num_viewpoint):
                if (
                    (self.split == "val" and idx in val_ids)
                    or (
                        self.split == "train"
                        and self.scene != "None"
                        and idx not in val_ids
                    )
                    or (self.split == "train" and self.scene == "None")
                ):
                    self.meta.append({"scan": scan, "target_idx": idx})

                view_ids = get_nearest_pose_ids(
                    poses[idx, :, :],
                    ref_poses=poses[..., :],
                    num_select=self.nb_views + 1,
                    angular_dist_method="dist",
                )

                self.id_list[scan].append(view_ids)

                closest_idxs = []
                source_views = view_ids[1:]
                for iter_, vid in enumerate(list(source_views) + list(view_ids[:1])):
                    idxs =  get_nearest_pose_ids(
                            poses[vid, :, :],
                            ref_poses=poses[source_views],
                            num_select=self.n_nearby,
                            angular_dist_method="dist",
                    )
                    
                    # target 
                    if iter_ == len(view_ids) - 1:
                        idxs[1:] = idxs[:-1]
                        idxs[0] = -1
                        
                    closest_idxs.append(idxs)

                self.closest_idxs[scan].append(np.stack(closest_idxs, axis=0))

                c2w = np.eye(4).astype('float32')
                c2w[:3] = poses[idx]
                w2c = np.linalg.inv(c2w)
                self.c2ws[scan].append(c2w)
                self.w2cs[scan].append(w2c)

                self.intrinsics[scan].append(intrinsic)

        self.scans = filtered_scans
        print(f"{self.split} Dataset Size = ", len(self.meta), f" {len(self.scans)} Number of Scenes")

    def burst_transform(self, imgs, eval_gain=-1):
        if self.split =='train': 
            white_level = 10 ** -torch.rand(1) * 0.4 + 0.6
            relin_imgs = re_linearize(imgs, white_level)
            noisy_imgs, sigma_ests = self.add_noise(relin_imgs)        
        else:
            white_level = torch.Tensor([1])
            relin_imgs = re_linearize(imgs, white_level)
            noisy_imgs, sigma_ests = self.add_noise_level(relin_imgs, eval_gain)                        

        return relin_imgs, noisy_imgs, sigma_ests, white_level

    def __len__(self):
        dset_size = len(self.meta) if self.max_len <= 0 else self.max_len
        return min(10000, len(self.meta)) if self.split == 'train' else len(self.meta) * (len(self.eval_gains) if self.burst_training else 1)

    def __getitem__(self, idx):
        if self.split == "train":
            noisy_factor = 1.0 #float(np.random.choice([1.0, 0.75], 1)) # 0.5
            close_views = self.n_nearby #int(np.random.choice([3, 4, 5], 1))
            eval_gain = -1
            idx = random.choice(list(range(len(self.meta))))
        else:
            noisy_factor = 1.0
            close_views = 5
            eval_gain = self.eval_gains[idx // len(self.meta)]
            idx = idx % len(self.meta)

        scan = self.meta[idx]["scan"]
        target_idx = self.meta[idx]["target_idx"]

        view_ids = self.id_list[scan][target_idx]
        target_view = view_ids[0]
        src_views = view_ids[1:]
        view_ids = [vid for vid in src_views] + [target_view]

        closest_idxs = self.closest_idxs[scan][target_idx][:, :close_views]

        imgs, clean_imgs, depths, depths_h, depths_aug = [], [], [], [], []
        intrinsics, w2cs, c2ws, near_fars = [], [], [], []
        affine_mats, affine_mats_inv = [], []

        w, h = self.img_wh
        w, h = int(w * noisy_factor), int(h * noisy_factor)

        for vid in view_ids:
            img_filename = self.image_paths[scan][vid]
            level = img_filename.split("/")[-2]
            level = level if 'mix' not in level else 'blur_mix' 
            img = Image.open(img_filename).convert("RGB")

            clean_img_filename = self.clean_image_paths[scan][vid]
            clean_img = Image.open(clean_img_filename).convert("RGBA")
            alpha = toTensor(clean_img)[-1:]
            clean_img = clean_img.convert("RGB")

            if img.size != (w, h):
                img = img.resize((w, h), Image.BICUBIC)           
                clean_img = clean_img.resize((w, h), Image.BICUBIC)           
            img = self.transform(img)
            imgs.append(img)

            clean_img = self.transform(clean_img)
            clean_imgs.append(clean_img)

            intrinsic = self.intrinsics[scan][vid].copy()
            intrinsic[:2] = intrinsic[:2] * noisy_factor
            intrinsics.append(intrinsic)

            w2c = self.w2cs[scan][vid]
            w2cs.append(w2c)
            c2ws.append(self.c2ws[scan][vid])

            aff = []
            aff_inv = []
            for l in range(3):
                proj_mat_l = np.eye(4)
                intrinsic_temp = intrinsic.copy()
                intrinsic_temp[:2] = intrinsic_temp[:2] / (2**l)
                proj_mat_l[:3, :4] = intrinsic_temp @ w2c[:3, :4]
                aff.append(proj_mat_l.copy())
                aff_inv.append(np.linalg.inv(proj_mat_l))
            aff = np.stack(aff, axis=-1)
            aff_inv = np.stack(aff_inv, axis=-1)

            affine_mats.append(aff)
            affine_mats_inv.append(aff_inv)

            near_fars.append(self.near_far[scan][vid])

            depths_h.append(np.zeros([h, w]))
            depths.append(np.zeros([h // 4, w // 4]))
            depths_aug.append(np.zeros([h // 4, w // 4]))

        imgs = np.stack(imgs)
        clean_imgs = np.stack(clean_imgs)
        depths = np.stack(depths)
        depths_h = np.stack(depths_h)
        depths_aug = np.stack(depths_aug)
        affine_mats = np.stack(affine_mats)
        affine_mats_inv = np.stack(affine_mats_inv)
        intrinsics = np.stack(intrinsics)
        w2cs = np.stack(w2cs)
        c2ws = np.stack(c2ws)
        near_fars = np.stack(near_fars)

        sample = {}
        sample["images"] = imgs
        sample['clean_images'] = clean_imgs
        sample['target_alpha'] = alpha
        sample["depths"] = depths
        sample["depths_h"] = depths_h
        sample["depths_aug"] = depths_aug
        sample["w2cs"] = w2cs
        sample["c2ws"] = c2ws
        sample["near_fars"] = near_fars
        sample["affine_mats"] = affine_mats
        sample["affine_mats_inv"] = affine_mats_inv
        sample["intrinsics"] = intrinsics
        sample["closest_idxs"] = closest_idxs
        sample['blur_level'] = level

        if self.burst_training:
            relin_imgs, noisy_imgs, sigma_ests, white_level = self.burst_transform(imgs, eval_gain)            
            sample["images"] = noisy_imgs
            sample["clean_images"] = relin_imgs
            sample["sigma_ests"] = sigma_ests
            sample['white_level'] = white_level
            sample['eval_gain'] = eval_gain
            sample['blur_level'] = level + f"_gain{eval_gain}"

        return sample
