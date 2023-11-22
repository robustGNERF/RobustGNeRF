import sys
sys.path.append('../')
import imageio
from config import config_parser
from ibrnet.sample_ray import RaySamplerSingleImage, rng
from ibrnet.render_image import render_single_image
from ibrnet.render_ray import render_rays
from ibrnet.criterion import Criterion
from ibrnet.model import IBRNetModel
from utils import *
from ibrnet.projection import Projector
from ibrnet.data_loaders import dataset_dict
import tensorflow as tf
from lpips_tensorflow import lpips_tf
from torch.utils.data import DataLoader
import numpy as np

from train import calc_depth_var

import torch
import torch.nn as nn
from ibrnet.data_loaders.data_utils import get_nearest_pose_ids, get_depth_warp_img
from tqdm import tqdm

from geo_interp import interp3
from pc_grad import PCGrad



def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def calc_depth_smooth_loss(ret, patch_size, loss_type='l2'):
    depth = ret['depth']  # [N_rays,], i.e., [n_patches * patch_size * patch_size]
    
    depth = depth.reshape([-1, patch_size, patch_size])  # [n_patches, patch_size, patch_size]
    
    v00 = depth[:, :-1, :-1]
    v01 = depth[:, :-1, 1:]
    v10 = depth[:, 1:, :-1]

    if loss_type == 'l2':
        loss = ((v00 - v01) ** 2) + ((v00 - v10) ** 2)
    elif loss_type == 'l1':
        loss = torch.abs(v00 - v01) + torch.abs(v00 - v10)
    else:
        raise ValueError('Not supported loss type.')
                
    return loss.sum()


class SL1Loss(nn.Module):
    def __init__(self):
        super(SL1Loss, self).__init__()
        self.loss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, depth_pred, depth_gt, mask=None, useMask=True):
        if None == mask and useMask:
            mask = depth_gt > 0
        loss = self.loss(depth_pred[mask], depth_gt[mask])  # * 2 ** (1 - 2)
        return loss



def project_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src):  # (x, y) --> (xz, yz, z) -> (x', y', z') -> (x'/z' , y'/ z')
    ''''''
    ## depth_ref: [B, H, W]
    ## intrinsics_ref: [3, 3]
    ## extrinsics_ref: [4, 4]
    ''''''
    width, height = depth_ref.shape[2], depth_ref.shape[1]
    batchsize = depth_ref.shape[0]

    y_ref, x_ref = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth_ref.device),
                                   torch.arange(0, width, dtype=torch.float32, device=depth_ref.device)])
    y_ref, x_ref = y_ref.contiguous(), x_ref.contiguous()
    y_ref, x_ref = y_ref.view(height * width), x_ref.view(height * width)

    pts = torch.stack((x_ref, y_ref, torch.ones_like(x_ref))).unsqueeze(
        0) * (depth_ref.view(batchsize, -1).unsqueeze(1))   ## [B, 3, height * width]

    xyz_ref = torch.matmul(torch.inverse(intrinsics_ref), pts)   ## [B, 3, height * width]

    ### torch.cat((xyz_ref, torch.ones_like(x_ref.unsqueeze(0)).repeat(batchsize, 1, 1)), dim=1).shape: [B, 4, height * width]
    ### torch.matmul(extrinsics_src, torch.inverse(extrinsics_ref)).shape: [4, 4]
    xyz_src = torch.matmul(torch.matmul(torch.inverse(extrinsics_src), extrinsics_ref),
                           torch.cat((xyz_ref, torch.ones_like(x_ref.unsqueeze(0)).repeat(batchsize, 1, 1)), dim=1))[:, :3, :]  ## [B, 3, height * width]

    K_xyz_src = torch.matmul(intrinsics_src, xyz_src)  # B*3*20480, [B, 3, height * width]
    depth_src = K_xyz_src[:, 2:3, :]   ## [B, 1, height * width]
    xy_src = K_xyz_src[:, :2, :] / (K_xyz_src[:, 2:3, :] + 1e-9)   ## [B, 2, height * width]
    x_src = xy_src[:, 0, :].view([batchsize, height, width])   ## [B, height, width]
    y_src = xy_src[:, 1, :].view([batchsize, height, width])   ## [B, height, width]

    return x_src, y_src, depth_src


def forward_warp(selected_inds, rgb_ref, depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src, src2tar=True, derive_full_image=False, cpu_speedup=True):
    ''''''
    ## selected_inds: [Num_Sampled_Rays] from the target view
    ## rgb_ref: [H, W, C]
    ## depth_ref: [B, H, W] or [H, W]
    ## intrinsics_ref: [3, 3]
    ## extrinsics_ref: [4, 4]
    ## cpu_speedup: Put the indexing operations to CPU for further speed-up
    ''''''
    
    x_res, y_res, depth_src = project_with_depth(
        depth_ref, intrinsics_ref, extrinsics_ref, intrinsics_src, extrinsics_src)
    width, height = depth_ref.shape[2], depth_ref.shape[1]
    batchsize = depth_ref.shape[0]
    
    if cpu_speedup:
        new = torch.zeros([height, width, 3])  ## pseudo RGB for the image, [height, width, 3]
        # new = torch.zeros_like(rgb_ref)  ## deprecated, as rgb_ref may not have the same resolution as depth_ref
        
        depth_src = depth_src.reshape(height, width).cpu() ## [height, width], assume batch_size=1
        new_depth = torch.zeros_like(depth_src)  ## [height, width], depth for the whole source image
        x_res = x_res.cpu()
        y_res = y_res.cpu()
        yy_base, xx_base = torch.meshgrid([torch.arange(
            0, height, dtype=torch.long), torch.arange(0, width, dtype=torch.long)])
        
    else:
        new = torch.zeros([height, width, 3]).to(depth_ref.device)  ## pseudo RGB for the image, [height, width, 3]
        # new = torch.zeros_like(rgb_ref)  ## deprecated, as rgb_ref may not have the same resolution as depth_ref
        
        depth_src = depth_src.reshape(height, width) ## [height, width], assume batch_size=1
        new_depth = torch.zeros_like(depth_src)  ## [height, width], depth for the whole source image
        
        yy_base, xx_base = torch.meshgrid([torch.arange(
            0, height, dtype=torch.long, device=depth_ref.device), torch.arange(0, width, dtype=torch.long, device=depth_ref.device)])
    
    
    y_res = torch.clamp(y_res, 0, height - 1).to(torch.long)  ## [B, height, width]
    x_res = torch.clamp(x_res, 0, width - 1).to(torch.long)  ## [B, height, width]
    yy_base = yy_base.reshape(-1)
    xx_base = xx_base.reshape(-1)
    y_res = y_res.reshape(-1)  ## [height*width], assume batch_size=1
    x_res = x_res.reshape(-1)  ## [height*width], assume batch_size=1
        
    if not derive_full_image:
        if src2tar:
            inds_res = (y_res * width + x_res).cpu().numpy()
            
            for i, inds in enumerate(inds_res):
                if inds in selected_inds:
                    if new_depth[y_res[i], x_res[i]] == 0 or new_depth[y_res[i], x_res[i]] > depth_src[yy_base[i], xx_base[i]]:
                        new_depth[y_res[i], x_res[i]] = depth_src[yy_base[i], xx_base[i]]
                        new[y_res[i], x_res[i]] = rgb_ref[yy_base[i], xx_base[i]]

            depth_proj = new_depth.reshape(-1)[selected_inds]  ## [N_rand]
            rgb_proj = new.reshape(-1,3)[selected_inds]   ## [N_rand, 3]
            
            if cpu_speedup:
                new = new.to(depth_ref.device)
                new_depth = new_depth.to(depth_ref.device)
                rgb_proj = rgb_proj.to(depth_ref.device)
                depth_proj = depth_proj.to(depth_ref.device)
                    
            return new, new_depth, rgb_proj, depth_proj
        
        else:
            selected_inds_new = []
            for i in selected_inds:
                selected_inds_new.append((y_res[i]*width + x_res[i]).item())
                if new_depth[y_res[i], x_res[i]] == 0 or new_depth[y_res[i], x_res[i]] > depth_src[yy_base[i], xx_base[i]]:
                    new_depth[y_res[i], x_res[i]] = depth_src[yy_base[i], xx_base[i]]
                    new[y_res[i], x_res[i]] = rgb_ref[yy_base[i], xx_base[i]]
            
            depth_proj = new_depth.reshape(-1)[selected_inds_new]  ## [N_rand]
            rgb_proj = new.reshape(-1,3)[selected_inds_new]   ## [N_rand, 3]

            if cpu_speedup:
                new = new.to(depth_ref.device)
                new_depth = new_depth.to(depth_ref.device)
                rgb_proj = rgb_proj.to(depth_ref.device)
                depth_proj = depth_proj.to(depth_ref.device)
                
            return new, new_depth, rgb_proj, depth_proj, selected_inds_new
        
    else:
        # painter's algo
        for i in range(yy_base.shape[0]):
            if new_depth[y_res[i], x_res[i]] == 0 or new_depth[y_res[i], x_res[i]] > depth_src[yy_base[i], xx_base[i]]:
                new_depth[y_res[i], x_res[i]] = depth_src[yy_base[i], xx_base[i]]
                new[y_res[i], x_res[i]] = rgb_ref[yy_base[i], xx_base[i]]

        depth_proj = new_depth.reshape(-1)[selected_inds]  ## [N_rand]
        rgb_proj = new.reshape(-1,3)[selected_inds]   ## [N_rand, 3]

        if cpu_speedup:
            new = new.to(depth_ref.device)
            new_depth = new_depth.to(depth_ref.device)
            rgb_proj = rgb_proj.to(depth_ref.device)
            depth_proj = depth_proj.to(depth_ref.device)
                
        return new, new_depth, rgb_proj, depth_proj


def calc_rotation_matrix(rot_degree):
    # input: rot_degree - [3]
    # output: rotation matrix - [3, 3]
    
    dx, dy, dz = rot_degree
    
    rot_x = torch.zeros(3,3)
    rot_x[0,0]=torch.cos(dx)
    rot_x[0,1]=-torch.sin(dx)
    rot_x[1,0]=torch.sin(dx)
    rot_x[1,1]=torch.cos(dx)
    rot_x[2,2]=1
    
    rot_y = torch.zeros(3,3)
    rot_y[0,0]=torch.cos(dy)
    rot_y[0,2]=torch.sin(dy)
    rot_y[1,1]=1
    rot_y[2,0]=-torch.sin(dy)
    rot_y[2,2]=torch.cos(dy)

    rot_z = torch.zeros(3,3)
    rot_z[0,0]=1
    rot_z[1,1]=torch.cos(dz)
    rot_z[1,2]=-torch.sin(dz)
    rot_z[2,1]=torch.sin(dz)
    rot_z[2,2]=torch.cos(dz)

    rot_mat = rot_z.mm(rot_y.mm(rot_x))  # shape=[3, 3]
    
    return rot_mat


def transform_src_cameras(src_cameras, rot_param, trans_param, num_source_views):
    camera_pose = src_cameras[0, :, -16:].reshape(-1, 4, 4)  # [num_source_views, 4, 4]

    rot_mats = []
    for src_id in range(num_source_views):
        rot_mat = calc_rotation_matrix(rot_param[src_id])  # [3, 3]
        rot_mats.append(rot_mat.unsqueeze(0))
    rot_mats = torch.cat(rot_mats, dim=0).to(src_cameras)  # [num_source_views, 3, 3]

    rot_new = rot_mats.bmm(camera_pose[:, :3, :3])  # [num_source_views, 3, 3]       
    trans_new = camera_pose[:, :3, 3] + trans_param.to(src_cameras)  # [num_source_views, 3]
    rot_trans = torch.cat([rot_new, trans_new.unsqueeze(2)], dim=2)  # [num_source_views, 3, 4] 

    return rot_trans


def init_adv_perturb(args, src_ray_batch, epsilon, upper_limit, lower_limit):
    delta = torch.zeros_like(src_ray_batch['src_rgbs'])  # ray_batch['src_rgbs'].shape=[1, N_views, H, W, 3]
    delta.uniform_(-epsilon, epsilon)
    delta.data = clamp(delta, lower_limit - src_ray_batch['src_rgbs'], upper_limit - src_ray_batch['src_rgbs'])
    delta.requires_grad = True
    
    return delta

    
### Derive adv perturbations using PGD on training data
def optimize_adv_perturb(args, delta, model, projector, src_ray_batch, data, return_loss=True):
    load_gt_depth = False    
    ray_sampler_train = RaySamplerSingleImage(data, 'cuda:0', load_gt_depth=load_gt_depth)     
    train_ray_batch = ray_sampler_train.random_sample(args.N_rand, sample_mode=args.sample_mode, center_ratio=args.center_ratio)
    total_loss = {}    
    if args.nearby_imgs > 1:
        nearby_idxs = []
        org_src_rgbs = (src_ray_batch['src_rgbs'] + delta).to(device)
        src_poses = src_ray_batch['src_cameras'][:,:,-16:].reshape(-1, 4, 4).cpu().numpy()
        src_intrinsics = src_ray_batch['src_cameras'][0, :,2:18].reshape(-1, 4, 4)
        for pose in src_poses:
            ids = get_nearest_pose_ids(pose, src_poses, args.nearby_imgs, angular_dist_method='dist', sort_by_dist=True)
            nearby_idxs.append(ids)

        nearby_imgs = torch.stack([org_src_rgbs[0].permute(0,3,1,2)[ids] for ids in nearby_idxs])
        nearby_poses = torch.stack([torch.from_numpy(src_poses[ids]) for ids in nearby_idxs]).to(device)
        nearby_intrinsics = torch.stack([src_intrinsics[ids] for ids in nearby_idxs])[..., :3,:3]

        tar_pose = src_ray_batch['camera'][:,-16:].reshape(4, 4).to(device)
        tar_intrinsics = src_ray_batch['camera'][:,2:18].reshape(4, 4).to(device)
        tar_near_ids = get_nearest_pose_ids(tar_pose.cpu().numpy(), src_poses, args.nearby_imgs - 1, angular_dist_method='dist', sort_by_dist=True)

        tar_nearby_poses = torch.cat([tar_pose[None], torch.from_numpy(src_poses[tar_near_ids]).to(device)], dim=0)[None]
        tar_nearby_intrinsics = torch.cat([tar_intrinsics[None], src_intrinsics[tar_near_ids]], dim=0)[..., :3,:3][None]

        tar_noisy_rgb = ray_sampler_train.rgb.reshape(1,H,W,3).permute(0,3,1,2).to(device)
        tar_nearby_imgs = torch.cat([tar_noisy_rgb, org_src_rgbs[0].permute(0,3,1,2)[tar_near_ids]], dim=0)[None]

        nearby_imgs         = torch.cat([tar_nearby_imgs, nearby_imgs], dim=0)
        nearby_poses        = torch.cat([tar_nearby_poses, nearby_poses], dim=0)
        nearby_intrinsics   = torch.cat([tar_nearby_intrinsics, nearby_intrinsics], dim=0)

        extrinsics = torch.inverse(nearby_poses)
        input_imgs = [img for img in nearby_imgs.permute(1,0,2,3,4)]
        depth, _, stage_depths = model.patchmatch(input_imgs, nearby_intrinsics, extrinsics, src_ray_batch['depth_range'][0,0].repeat(args.num_source_views+1), src_ray_batch['depth_range'][0,1].repeat(args.num_source_views+1))            
        tar_depth, src_depths = depth[:1], depth[1:]
        
        warped_rgbds, coords = get_depth_warp_img(nearby_imgs[1:], nearby_poses[1:], src_intrinsics, depth[1:].detach(), nearby_idxs)
        ref_rgbd = torch.cat([nearby_imgs[1:][:,:1], src_depths.unsqueeze(1).detach()], dim=2)
        input_imgs = torch.cat([ref_rgbd, warped_rgbds], dim=1)
        reconst_img, feats = model.feature_net(input_imgs.reshape(args.num_source_views, -1, H, W))
        featmaps = [feats[:, :args.coarse_feat_dim], feats[:, args.coarse_feat_dim:]]
    else:
        featmaps = model.feature_net((src_ray_batch['src_rgbs'] + delta).squeeze(0).permute(0, 3, 1, 2))
    
    ret = render_rays(ray_batch=train_ray_batch,
                    model=model,
                    projector=projector,
                    featmaps=featmaps,
                    N_samples=args.N_samples,
                    inv_uniform=args.inv_uniform,
                    N_importance=args.N_importance,
                    det=args.det,
                    white_bkgd=args.white_bkgd,
                    args=args,
                    src_ray_batch=src_ray_batch)  ## set src_ray_batch to the test ray_batch

    loss_rgb, _ = criterion(ret['outputs_coarse'], train_ray_batch, scalars_to_log=None)
    if ret['outputs_fine'] is not None:
        fine_loss, _ = criterion(ret['outputs_fine'], train_ray_batch, scalars_to_log=None)
        loss_rgb += fine_loss  
    total_loss['rgb'] = loss_rgb
    loss = sum(total_loss.values())
    
    if return_loss:
        return loss, total_loss

    grad = torch.autograd.grad(loss, delta)[0].detach()
    
    return grad



    
    
if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    args.expname = args.expname + "_advAttack"
    args.distributed = False
    
    args.det = True  ## always use deterministic sampling for coarse and fine samples 

    # Create IBRNet model
    model = IBRNetModel(args, load_scheduler=False, load_opt=False)
    eval_dataset_name = args.eval_dataset
    extra_out_dir = '{}/{}'.format(eval_dataset_name, args.expname)
    print("saving results to eval/{}...".format(extra_out_dir))
    os.makedirs(extra_out_dir, exist_ok=True)

    device = 'cuda:0'
    projector = Projector(device=device)

    assert len(args.eval_scenes) == 1, "only accept single scene"
    scene_name = args.eval_scenes[0]
    out_scene_dir = os.path.join(extra_out_dir, '{}_{:06d}'.format(scene_name, model.start_step))
    os.makedirs(out_scene_dir, exist_ok=True)
    
    if not args.no_attack and not args.view_specific:  # optimize generalizable adv perturb across different views
        train_dataset = dataset_dict[args.eval_dataset](args, 'train', scenes=args.eval_scenes)
        train_loader = DataLoader(train_dataset, batch_size=1, worker_init_fn=lambda _: np.random.seed(), num_workers=args.workers, 
                                    pin_memory=True, shuffle=True)
    else:
        train_dataset = None
        train_loader = None

    test_dataset = dataset_dict[args.eval_dataset](args, 'test', scenes=args.eval_scenes)
    save_prefix = scene_name
    test_loader = DataLoader(test_dataset, batch_size=1)
    total_num = len(test_loader)
    
    results_dict = {scene_name: {}}
    sum_coarse_psnr = 0
    sum_fine_psnr = 0
    running_mean_coarse_psnr = 0
    running_mean_fine_psnr = 0
    sum_coarse_lpips = 0
    sum_fine_lpips = 0
    running_mean_coarse_lpips = 0
    running_mean_fine_lpips = 0
    sum_coarse_ssim = 0
    sum_fine_ssim = 0
    running_mean_coarse_ssim = 0
    running_mean_fine_ssim = 0

    if "2." in tf.__version__[:2]:
        tf.compat.v1.disable_eager_execution()
        tf = tf.compat.v1

    pred_ph = tf.placeholder(tf.float32)
    gt_ph = tf.placeholder(tf.float32)
    distance_t = lpips_tf.lpips(pred_ph, gt_ph, model='net-lin', net='vgg')
    ssim_tf = tf.image.ssim(pred_ph, gt_ph, max_val=1.)
    psnr_tf = tf.image.psnr(pred_ph, gt_ph, max_val=1.)
    
    criterion = Criterion()
    
    if args.gt_depth_path:
        load_gt_depth = True
    else:
        load_gt_depth = False
    
    if not args.view_specific and not args.no_attack:
        test_dataset_for_src = dataset_dict[args.eval_dataset](args, 'test', scenes=args.eval_scenes, use_glb_src=args.use_center_view)
        test_loader_for_src = DataLoader(test_dataset_for_src, batch_size=1)
        
        for i, data in enumerate(test_loader_for_src):   
            src_ray_sampler = RaySamplerSingleImage(data, device=device, load_gt_depth=load_gt_depth)
            src_ray_batch_glb = src_ray_sampler.get_all()  # global source views for all view directions
            break
        
    else:
        src_ray_batch_glb = None
        
    assert args.view_specific
    
    cnt = 1
    for i, data in tqdm(enumerate(test_loader), desc=f"############### Test Image {cnt} ##########"):
        cnt += 1 
        rgb_path = data['rgb_path'][0]
        file_id = os.path.basename(rgb_path).split('.')[0]
        src_rgbs = data['src_rgbs'][0].cpu().numpy()

        averaged_img = (np.mean(src_rgbs, axis=0) * 255.).astype(np.uint8)
        imageio.imwrite(os.path.join(out_scene_dir, '{}_average.png'.format(file_id)),
                        averaged_img)

        model.switch_to_eval()
        ray_sampler = RaySamplerSingleImage(data, device=device, load_gt_depth=load_gt_depth)
        ray_batch = ray_sampler.get_all()
        src_ray_batch = ray_batch
        H, W = ray_sampler.src_rgbs.shape[2:4]    
        ##########################
        ## Adversarial Attack ####
        ##########################
        epsilon = torch.tensor(args.epsilon / 255.).cuda()
        alpha = torch.tensor(args.adv_lr / 255.).cuda()
        upper_limit = 1
        lower_limit = 0
        
        delta = init_adv_perturb(args, src_ray_batch, epsilon, upper_limit, lower_limit)            
        params = [delta]                    
        opt = torch.optim.Adam(params, lr=args.adam_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step_size, gamma=args.lr_gamma)
            
        print("Start adversarial perturbation...")
        iter_ = 1
        for num_iter in tqdm(range(args.adv_iters), desc='Perturbation Learning'): 
            iter_ += 1
            loss, loss_dict = optimize_adv_perturb(args, delta, model, projector, src_ray_batch, data, return_loss=True)
            opt.zero_grad()                
            loss.backward()
            delta.grad.data *= -1                    
            opt.step()
            scheduler.step()
                    
            delta.data = clamp(delta.data, -epsilon, epsilon)
            delta.data = clamp(delta.data, lower_limit - src_ray_batch['src_rgbs'], upper_limit - src_ray_batch['src_rgbs'])  

        delta.requires_grad = False
        torch.cuda.empty_cache()

        if args.export_adv_source_img:
            adv_src_rgbs = src_ray_batch['src_rgbs'] + delta   # [1, N_views, H, W, 3]
            adv_src_rgbs = adv_src_rgbs[0]  # [N_views, H, W, 3]
            src_rgbs = src_ray_batch['src_rgbs'][0]  # [N_views, H, W, 3]
            
            for j in range(adv_src_rgbs.shape[0]):
                adv_src_img = adv_src_rgbs[j,:,:,:]
                adv_src_img = (255 * np.clip(adv_src_img.data.cpu().numpy(), a_min=0, a_max=1.)).astype(np.uint8)
                imageio.imwrite(os.path.join(out_scene_dir, 'adv_src_{}_{}.png'.format(i,j)), adv_src_img)
                
                src_img = src_rgbs[j,:,:,:]
                src_img = (255 * np.clip(src_img.data.cpu().numpy(), a_min=0, a_max=1.)).astype(np.uint8)
                imageio.imwrite(os.path.join(out_scene_dir, 'src_{}_{}.png'.format(i,j)), src_img)
            
            # sys.exit()
        
        ray_batch['src_rgbs']       = ray_batch['src_rgbs']+delta
        src_ray_batch['src_rgbs']   = src_ray_batch['src_rgbs']+delta
        with torch.no_grad():
            if args.nearby_imgs > 1:
                nearby_idxs = []
                org_src_rgbs = src_ray_batch['src_rgbs'].to(device)
                src_poses = src_ray_batch['src_cameras'][:,:,-16:].reshape(-1, 4, 4).cpu().numpy()
                src_intrinsics = src_ray_batch['src_cameras'][0, :,2:18].reshape(-1, 4, 4)
                for pose in src_poses:
                    ids = get_nearest_pose_ids(pose, src_poses, args.nearby_imgs, angular_dist_method='dist', sort_by_dist=True)
                    nearby_idxs.append(ids)

                nearby_imgs = torch.stack([org_src_rgbs[0].permute(0,3,1,2)[ids] for ids in nearby_idxs])
                nearby_poses = torch.stack([torch.from_numpy(src_poses[ids]) for ids in nearby_idxs]).to(device)
                nearby_intrinsics = torch.stack([src_intrinsics[ids] for ids in nearby_idxs])[..., :3,:3]

                tar_pose = src_ray_batch['camera'][:,-16:].reshape(4, 4).to(device)
                tar_intrinsics = src_ray_batch['camera'][:,2:18].reshape(4, 4).to(device)
                tar_near_ids = get_nearest_pose_ids(tar_pose.cpu().numpy(), src_poses, args.nearby_imgs - 1, angular_dist_method='dist', sort_by_dist=True)

                tar_nearby_poses = torch.cat([tar_pose[None], torch.from_numpy(src_poses[tar_near_ids]).to(device)], dim=0)[None]
                tar_nearby_intrinsics = torch.cat([tar_intrinsics[None], src_intrinsics[tar_near_ids]], dim=0)[..., :3,:3][None]

                tar_noisy_rgb = ray_sampler.rgb.reshape(1,H,W,3).permute(0,3,1,2).to(device)
                tar_nearby_imgs = torch.cat([tar_noisy_rgb, org_src_rgbs[0].permute(0,3,1,2)[tar_near_ids]], dim=0)[None]

                nearby_imgs         = torch.cat([tar_nearby_imgs, nearby_imgs], dim=0)
                nearby_poses        = torch.cat([tar_nearby_poses, nearby_poses], dim=0)
                nearby_intrinsics   = torch.cat([tar_nearby_intrinsics, nearby_intrinsics], dim=0)

                extrinsics = torch.inverse(nearby_poses)
                input_imgs = [img for img in nearby_imgs.permute(1,0,2,3,4)]
                depth, _, stage_depths = model.patchmatch(input_imgs, nearby_intrinsics, extrinsics, src_ray_batch['depth_range'][0,0].repeat(args.num_source_views+1), src_ray_batch['depth_range'][0,1].repeat(args.num_source_views+1))            
                tar_depth, src_depths = depth[:1], depth[1:]
                
                warped_rgbds, coords = get_depth_warp_img(nearby_imgs[1:], nearby_poses[1:], src_intrinsics, depth[1:].detach(), nearby_idxs)
                ref_rgbd = torch.cat([nearby_imgs[1:][:,:1], src_depths.unsqueeze(1).detach()], dim=2)
                input_imgs = torch.cat([ref_rgbd, warped_rgbds], dim=1)
                reconst_img, feats = model.feature_net(input_imgs.reshape(args.num_source_views, -1, H, W))
                featmaps = [feats[:, :args.coarse_feat_dim], feats[:, args.coarse_feat_dim:]]
                src_ray_batch['src_rgbs'] = reconst_img.permute(0,2,3,1)[None] #.detach()
            else:            
                featmaps = model.feature_net((src_ray_batch['src_rgbs']).squeeze(0).permute(0, 3, 1, 2))

            if args.use_clean_color or args.use_clean_density:
                featmaps_clean = model.feature_net((src_ray_batch['src_rgbs']).squeeze(0).permute(0, 3, 1, 2))
            else:
                featmaps_clean = None
                
            ret = render_single_image(ray_sampler=ray_sampler,
                                      ray_batch=ray_batch,
                                      model=model,
                                      projector=projector,
                                      chunk_size=args.chunk_size,
                                      det=True,
                                      N_samples=args.N_samples,
                                      inv_uniform=args.inv_uniform,
                                      N_importance=args.N_importance,
                                      white_bkgd=args.white_bkgd,
                                      featmaps=featmaps,
                                      args=args,
                                      featmaps_clean=featmaps_clean,
                                      src_ray_batch=src_ray_batch)
            
            gt_rgb = data['rgb'][0]
            coarse_pred_rgb = ret['outputs_coarse']['rgb'].detach().cpu()
            coarse_err_map = torch.sum((coarse_pred_rgb - gt_rgb) ** 2, dim=-1).numpy()
            coarse_err_map_colored = (colorize_np(coarse_err_map, range=(0., 1.)) * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(out_scene_dir, '{}_err_map_coarse.png'.format(file_id)),
                            coarse_err_map_colored)
            coarse_pred_rgb_np = np.clip(coarse_pred_rgb.numpy()[None, ...], a_min=0., a_max=1.)
            gt_rgb_np = gt_rgb.numpy()[None, ...]

            # different implementation of the ssim and psnr metrics can be different.
            # we use the tf implementation for evaluating ssim and psnr to match the setup of NeRF paper.

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session() as session:
                coarse_lpips = session.run(distance_t, feed_dict={pred_ph: coarse_pred_rgb_np, gt_ph: gt_rgb_np})[0]
                coarse_ssim = session.run(ssim_tf, feed_dict={pred_ph: coarse_pred_rgb_np, gt_ph: gt_rgb_np})[0]
                coarse_psnr = session.run(psnr_tf, feed_dict={pred_ph: coarse_pred_rgb_np, gt_ph: gt_rgb_np})[0]

            # saving outputs ...
            coarse_pred_rgb = (255 * np.clip(coarse_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
            imageio.imwrite(os.path.join(out_scene_dir, '{}_pred_coarse.png'.format(file_id)), coarse_pred_rgb)

            gt_rgb_np_uint8 = (255 * np.clip(gt_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
            imageio.imwrite(os.path.join(out_scene_dir, '{}_gt_rgb.png'.format(file_id)), gt_rgb_np_uint8)

            coarse_pred_depth = ret['outputs_coarse']['depth'].detach().cpu()
            imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_coarse.png'.format(file_id)),
                            (coarse_pred_depth.numpy().squeeze() * 1000.).astype(np.uint16))
            coarse_pred_depth_colored = colorize_np(coarse_pred_depth,
                                                    range=tuple(data['depth_range'].squeeze().cpu().numpy()))
            imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_vis_coarse.png'.format(file_id)),
                            (255 * coarse_pred_depth_colored).astype(np.uint8))
            coarse_acc_map = torch.sum(ret['outputs_coarse']['weights'], dim=-1).detach().cpu()
            coarse_acc_map_colored = (colorize_np(coarse_acc_map, range=(0., 1.)) * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(out_scene_dir, '{}_acc_map_coarse.png'.format(file_id)),
                            coarse_acc_map_colored)

            sum_coarse_psnr += coarse_psnr
            running_mean_coarse_psnr = sum_coarse_psnr / (i + 1)
            sum_coarse_lpips += coarse_lpips
            running_mean_coarse_lpips = sum_coarse_lpips / (i + 1)
            sum_coarse_ssim += coarse_ssim
            running_mean_coarse_ssim = sum_coarse_ssim / (i + 1)

            if ret['outputs_fine'] is not None:
                fine_pred_rgb = ret['outputs_fine']['rgb'].detach().cpu()
                fine_pred_rgb_np = np.clip(fine_pred_rgb.numpy()[None, ...], a_min=0., a_max=1.)

                with tf.Session() as session:
                    fine_lpips = session.run(distance_t, feed_dict={pred_ph: fine_pred_rgb_np, gt_ph: gt_rgb_np})[0]
                    fine_ssim = session.run(ssim_tf, feed_dict={pred_ph: fine_pred_rgb_np, gt_ph: gt_rgb_np})[0]
                    fine_psnr = session.run(psnr_tf, feed_dict={pred_ph: fine_pred_rgb_np, gt_ph: gt_rgb_np})[0]

                fine_err_map = torch.sum((fine_pred_rgb - gt_rgb) ** 2, dim=-1).numpy()
                fine_err_map_colored = (colorize_np(fine_err_map, range=(0., 1.)) * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(out_scene_dir, '{}_err_map_fine.png'.format(file_id)),
                                fine_err_map_colored)

                fine_pred_rgb = (255 * np.clip(fine_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
                imageio.imwrite(os.path.join(out_scene_dir, '{}_pred_fine.png'.format(file_id)), fine_pred_rgb)
                fine_pred_depth = ret['outputs_fine']['depth'].detach().cpu()
                imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_fine.png'.format(file_id)),
                                (fine_pred_depth.numpy().squeeze() * 1000.).astype(np.uint16))
                
                fine_pred_depth_colored = colorize_np(fine_pred_depth,
                                                      range=tuple(data['depth_range'].squeeze().cpu().numpy()))
                imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_vis_fine.png'.format(file_id)),
                                (255 * fine_pred_depth_colored).astype(np.uint8))

                if 'depth' in data:
                    fine_gt_depth_colored = colorize_np(data['depth'][0],
                                                        range=tuple(data['depth_range'].squeeze().cpu().numpy()))
                    imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_vis_GT.png'.format(file_id)),
                                    (255 * fine_gt_depth_colored).astype(np.uint8))
                
                fine_acc_map = torch.sum(ret['outputs_fine']['weights'], dim=-1).detach().cpu()
                fine_acc_map_colored = (colorize_np(fine_acc_map, range=(0., 1.)) * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(out_scene_dir, '{}_acc_map_fine.png'.format(file_id)),
                                fine_acc_map_colored)
            else:
                fine_ssim = fine_lpips = fine_psnr = 0.

            sum_fine_psnr += fine_psnr
            running_mean_fine_psnr = sum_fine_psnr / (i + 1)
            sum_fine_lpips += fine_lpips
            running_mean_fine_lpips = sum_fine_lpips / (i + 1)
            sum_fine_ssim += fine_ssim
            running_mean_fine_ssim = sum_fine_ssim / (i + 1)

            print("==================\n"
                  "{}, curr_id: {} \n"
                  "current coarse psnr: {:03f}, current fine psnr: {:03f} \n"
                  "running mean coarse psnr: {:03f}, running mean fine psnr: {:03f} \n"
                  "current coarse ssim: {:03f}, current fine ssim: {:03f} \n"
                  "running mean coarse ssim: {:03f}, running mean fine ssim: {:03f} \n" 
                  "current coarse lpips: {:03f}, current fine lpips: {:03f} \n"
                  "running mean coarse lpips: {:03f}, running mean fine lpips: {:03f} \n"
                  "===================\n"
                  .format(scene_name, file_id,
                          coarse_psnr, fine_psnr,
                          running_mean_coarse_psnr, running_mean_fine_psnr,
                          coarse_ssim, fine_ssim,
                          running_mean_coarse_ssim, running_mean_fine_ssim,
                          coarse_lpips, fine_lpips,
                          running_mean_coarse_lpips, running_mean_fine_lpips
                          ))

            results_dict[scene_name][file_id] = {'coarse_psnr': coarse_psnr,
                                                 'fine_psnr': fine_psnr,
                                                 'coarse_ssim': coarse_ssim,
                                                 'fine_ssim': fine_ssim,
                                                 'coarse_lpips': coarse_lpips,
                                                 'fine_lpips': fine_lpips,
                                                 }

    mean_coarse_psnr = sum_coarse_psnr / total_num
    mean_fine_psnr = sum_fine_psnr / total_num
    mean_coarse_lpips = sum_coarse_lpips / total_num
    mean_fine_lpips = sum_fine_lpips / total_num
    mean_coarse_ssim = sum_coarse_ssim / total_num
    mean_fine_ssim = sum_fine_ssim / total_num

    print('------{}-------\n'
          'final coarse psnr: {}, final fine psnr: {}\n'
          'fine coarse ssim: {}, final fine ssim: {} \n'
          'final coarse lpips: {}, fine fine lpips: {} \n'
          .format(scene_name, mean_coarse_psnr, mean_fine_psnr,
                  mean_coarse_ssim, mean_fine_ssim,
                  mean_coarse_lpips, mean_fine_lpips,
                  ))

    results_dict[scene_name]['coarse_mean_psnr'] = mean_coarse_psnr
    results_dict[scene_name]['fine_mean_psnr'] = mean_fine_psnr
    results_dict[scene_name]['coarse_mean_ssim'] = mean_coarse_ssim
    results_dict[scene_name]['fine_mean_ssim'] = mean_fine_ssim
    results_dict[scene_name]['coarse_mean_lpips'] = mean_coarse_lpips
    results_dict[scene_name]['fine_mean_lpips'] = mean_fine_lpips

    f = open("{}/psnr_{}_{}.txt".format(extra_out_dir, save_prefix, model.start_step), "w")
    f.write(str(results_dict))
    f.close()