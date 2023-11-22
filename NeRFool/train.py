import os
import time
import numpy as np
import shutil
import torch
import torch.utils.data.distributed

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from ibrnet.data_loaders import dataset_dict
from ibrnet.render_ray import render_rays
from ibrnet.render_image import render_single_image
from ibrnet.model import IBRNetModel
from ibrnet.sample_ray import RaySamplerSingleImage
from ibrnet.criterion import Criterion
from utils import img2mse, mse2psnr, img_HWC2CHW, colorize, cycle, img2psnr, save_current_code
import config
import torch.distributed as dist
from ibrnet.projection import Projector
from ibrnet.data_loaders.create_training_dataset import create_training_dataset
from ibrnet.data_loaders.data_utils import get_nearest_pose_ids, get_depth_warp_img
import torch.nn.functional as F

import cv2
cv2.setNumThreads(1)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def worker_init_fn(_):
    return np.random.seed()


def train(args):

    device = "cuda:{}".format(args.local_rank)
    out_folder = os.path.join(args.rootdir, 'MetaFool', 'out', args.expname)
    print('outputs will be saved to {}'.format(out_folder))
    os.makedirs(out_folder, exist_ok=True)

    # save the args and config files
    f = os.path.join(out_folder, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    if args.config is not None:
        f = os.path.join(out_folder, 'config.txt')
        if not os.path.isfile(f):
            shutil.copy(args.config, f)

    # create training dataset
    train_dataset, train_sampler = create_training_dataset(args)
    # currently only support batch_size=1 (i.e., one set of target and source views) for each GPU node
    # please use distributed parallel on multiple GPUs to train multiple target views per batch
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                               worker_init_fn=worker_init_fn,
                                               num_workers=1, #args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               shuffle=True if train_sampler is None else False)

    # create validation dataset
    val_dataset = dataset_dict[args.eval_dataset](args, 'validation',
                                                  scenes=args.eval_scenes)

    val_loader = DataLoader(val_dataset, batch_size=1)
    val_loader_iterator = iter(cycle(val_loader))

    # Create IBRNet model
    model = IBRNetModel(args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler)
    
    # create projector
    projector = Projector(device=device)

    # Create criterion
    criterion = Criterion()
    tb_dir = os.path.join(args.rootdir, 'MetaFool', 'logs/', args.expname)
    if args.local_rank == 0:
        writer = SummaryWriter(tb_dir)
        print('saving tensorboard files to {}'.format(tb_dir))
    scalars_to_log = {}

    global_step = model.start_step + 1
    epoch = 0
    while global_step < model.start_step + args.n_iters + 1:
        np.random.seed()
        for train_data in train_loader:
            time0 = time.time()
                        
            if args.distributed:
                train_sampler.set_epoch(epoch)

            # Start of core optimization loop

            # load training rays
            ray_sampler = RaySamplerSingleImage(train_data, device)
            N_rand = int(1.0 * args.N_rand * args.num_source_views / train_data['src_rgbs'][0].shape[0])
            
            ray_batch = ray_sampler.random_sample(N_rand,
                                                  sample_mode=args.sample_mode,
                                                  center_ratio=args.center_ratio,
                                                  )
            H, W = ray_sampler.src_rgbs.shape[2:4]
            ######### Without Adversarial Perturbation
            if args.use_adv_train:                
                epsilon = torch.tensor(args.epsilon / 255.).cuda()
                alpha = torch.tensor(args.adv_lr / 255.).cuda()
                upper_limit = 1
                lower_limit = 0

                delta = torch.zeros_like(ray_batch['src_rgbs'])  # ray_batch['src_rgbs'].shape=[1, N_views, H, W, 3]
                delta.uniform_(-epsilon, epsilon)
                delta.data = clamp(delta, lower_limit - ray_batch['src_rgbs'], upper_limit - ray_batch['src_rgbs'])
                delta.requires_grad = True
                
                for _ in range(args.adv_iters): 
                    if args.nearby_imgs > 1:
                        nearby_idxs = []
                        org_src_rgbs = (ray_batch['src_rgbs'] + delta).to(device)
                        src_poses = ray_batch['src_cameras'][:,:,-16:].reshape(-1, 4, 4).cpu().numpy()
                        src_intrinsics = ray_batch['src_cameras'][0, :,2:18].reshape(-1, 4, 4)
                        for pose in src_poses:
                            ids = get_nearest_pose_ids(pose, src_poses, args.nearby_imgs, angular_dist_method='dist', sort_by_dist=True)
                            nearby_idxs.append(ids)

                        nearby_imgs = torch.stack([org_src_rgbs[0].permute(0,3,1,2)[ids] for ids in nearby_idxs])
                        nearby_poses = torch.stack([torch.from_numpy(src_poses[ids]) for ids in nearby_idxs]).to(device)
                        nearby_intrinsics = torch.stack([src_intrinsics[ids] for ids in nearby_idxs])[..., :3,:3]

                        tar_pose = ray_batch['camera'][:,-16:].reshape(4, 4).to(device)
                        tar_intrinsics = ray_batch['camera'][:,2:18].reshape(4, 4).to(device)
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
                        depth, _, stage_depths = model.patchmatch(input_imgs, nearby_intrinsics, extrinsics, ray_batch['depth_range'][0,0].repeat(args.num_source_views+1), ray_batch['depth_range'][0,1].repeat(args.num_source_views+1))            
                        tar_depth, src_depths = depth[:1], depth[1:]
                        
                        warped_rgbds, coords = get_depth_warp_img(nearby_imgs[1:], nearby_poses[1:], src_intrinsics, depth[1:].detach(), nearby_idxs)
                        ref_rgbd = torch.cat([nearby_imgs[1:][:,:1], src_depths.unsqueeze(1).detach()], dim=2)
                        input_imgs = torch.cat([ref_rgbd, warped_rgbds], dim=1)
                        reconst_img, feats = model.feature_net(input_imgs.reshape(args.num_source_views, -1, H, W))
                        featmaps = [feats[:, :args.coarse_feat_dim], feats[:, args.coarse_feat_dim:]]
                    else:
                        featmaps = model.feature_net(((ray_batch['src_rgbs'] + delta)).squeeze(0).permute(0, 3, 1, 2))


                    ret = render_rays(ray_batch=ray_batch,
                                    model=model,
                                    projector=projector,
                                    featmaps=featmaps,
                                    N_samples=args.N_samples,
                                    inv_uniform=args.inv_uniform,
                                    N_importance=args.N_importance,
                                    det=args.det,
                                    white_bkgd=args.white_bkgd,
                                    args=args)
                    
                    loss, _ = criterion(ret['outputs_coarse'], ray_batch, scalars_to_log)
                    if ret['outputs_fine'] is not None:
                        fine_loss, _ = criterion(ret['outputs_fine'], ray_batch, scalars_to_log)
                        loss += fine_loss
                        
                    loss.backward()
                    grad = delta.grad.detach()
                    delta.data = delta.data + alpha * torch.sign(grad)
                    delta.grad.zero_()
            
                    delta.data = clamp(delta.data, -epsilon, epsilon)
                    delta.data = clamp(delta.data, lower_limit - ray_batch['src_rgbs'], upper_limit - ray_batch['src_rgbs'])  

                ##### Delta fixed after loop
                delta.requires_grad = False
                torch.cuda.empty_cache()
                ray_batch['src_rgbs'] = ray_batch['src_rgbs']+delta
                if args.nearby_imgs > 1:
                    nearby_idxs = []
                    org_src_rgbs = ray_batch['src_rgbs'].to(device)
                    src_poses = ray_batch['src_cameras'][:,:,-16:].reshape(-1, 4, 4).cpu().numpy()
                    src_intrinsics = ray_batch['src_cameras'][0, :,2:18].reshape(-1, 4, 4)
                    for pose in src_poses:
                        ids = get_nearest_pose_ids(pose, src_poses, args.nearby_imgs, angular_dist_method='dist', sort_by_dist=True)
                        nearby_idxs.append(ids)

                    nearby_imgs = torch.stack([org_src_rgbs[0].permute(0,3,1,2)[ids] for ids in nearby_idxs])
                    nearby_poses = torch.stack([torch.from_numpy(src_poses[ids]) for ids in nearby_idxs]).to(device)
                    nearby_intrinsics = torch.stack([src_intrinsics[ids] for ids in nearby_idxs])[..., :3,:3]

                    tar_pose = ray_batch['camera'][:,-16:].reshape(4, 4).to(device)
                    tar_intrinsics = ray_batch['camera'][:,2:18].reshape(4, 4).to(device)
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
                    depth, _, stage_depths = model.patchmatch(input_imgs, nearby_intrinsics, extrinsics, ray_batch['depth_range'][0,0].repeat(args.num_source_views+1), ray_batch['depth_range'][0,1].repeat(args.num_source_views+1))            
                    tar_depth, src_depths = depth[:1], depth[1:]
                    
                    warped_rgbds, coords = get_depth_warp_img(nearby_imgs[1:], nearby_poses[1:], src_intrinsics, depth[1:].detach(), nearby_idxs)
                    ref_rgbd = torch.cat([nearby_imgs[1:][:,:1], src_depths.unsqueeze(1).detach()], dim=2)
                    input_imgs = torch.cat([ref_rgbd, warped_rgbds], dim=1)
                    reconst_img, feats = model.feature_net(input_imgs.reshape(args.num_source_views, -1, H, W))
                    featmaps = [feats[:, :args.coarse_feat_dim], feats[:, args.coarse_feat_dim:]]
                    # ray_batch['src_rgbs'] = reconst_img.permute(0,2,3,1)[None] #.detach()
                else:            
                    featmaps = model.feature_net((ray_batch['src_rgbs']).squeeze(0).permute(0, 3, 1, 2))
                    reconst_img = None
                    tar_depth = None

            ######### Without Adversarial Perturbation
            else:
                if args.nearby_imgs > 1:
                    nearby_idxs = []
                    org_src_rgbs = ray_sampler.src_rgbs.to(device)
                    src_poses = ray_batch['src_cameras'][:,:,-16:].reshape(-1, 4, 4).cpu().numpy()
                    src_intrinsics = ray_batch['src_cameras'][0, :,2:18].reshape(-1, 4, 4)
                    for pose in src_poses:
                        ids = get_nearest_pose_ids(pose, src_poses, args.nearby_imgs, angular_dist_method='dist', sort_by_dist=True)
                        nearby_idxs.append(ids)

                    nearby_imgs = torch.stack([org_src_rgbs[0].permute(0,3,1,2)[ids] for ids in nearby_idxs])
                    nearby_poses = torch.stack([torch.from_numpy(src_poses[ids]) for ids in nearby_idxs]).to(device)
                    nearby_intrinsics = torch.stack([src_intrinsics[ids] for ids in nearby_idxs])[..., :3,:3]

                    tar_pose = ray_batch['camera'][:,-16:].reshape(4, 4).to(device)
                    tar_intrinsics = ray_batch['camera'][:,2:18].reshape(4, 4).to(device)
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
                    depth, _, stage_depths = model.patchmatch(input_imgs, nearby_intrinsics, extrinsics, ray_batch['depth_range'][0,0].repeat(args.num_source_views+1), ray_batch['depth_range'][0,1].repeat(args.num_source_views+1))            
                    tar_depth, src_depths = depth[:1], depth[1:]
                    
                    warped_rgbds, coords = get_depth_warp_img(nearby_imgs[1:], nearby_poses[1:], src_intrinsics, depth[1:].detach(), nearby_idxs)
                    ref_rgbd = torch.cat([nearby_imgs[1:][:,:1], src_depths.unsqueeze(1).detach()], dim=2)
                    input_imgs = torch.cat([ref_rgbd, warped_rgbds], dim=1)
                    reconst_img, feats = model.feature_net(input_imgs.reshape(args.num_source_views, -1, H, W))
                    featmaps = [feats[:, :args.coarse_feat_dim], feats[:, args.coarse_feat_dim:]]
                    # ray_batch['src_rgbs'] = reconst_img.permute(0,2,3,1)[None] #.detach()
                else:
                    featmaps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))
                    reconst_img = None
                    tar_depth = None

            ret = render_rays(ray_batch=ray_batch,
                              model=model,
                              projector=projector,
                              featmaps=featmaps,
                              N_samples=args.N_samples,
                              inv_uniform=args.inv_uniform,
                              N_importance=args.N_importance,
                              det=args.det,
                              white_bkgd=args.white_bkgd,
                              args=args,
                              geo_noise=args.geo_noise)

            # compute loss
            model.optimizer.zero_grad()
            loss, scalars_to_log = criterion(ret['outputs_coarse'], ray_batch, scalars_to_log)

            if ret['outputs_fine'] is not None:
                fine_loss, scalars_to_log = criterion(ret['outputs_fine'], ray_batch, scalars_to_log)
                loss += fine_loss
                
            if args.depth_var_loss > 0:
                depth_var_loss = args.depth_var_loss * calc_depth_var(ret['outputs_coarse'])
                
                if ret['outputs_fine'] is not None:
                    depth_var_loss += args.depth_var_loss * calc_depth_var(ret['outputs_fine']) 
                
                # print("loss: {%f}, depth_var_loss: {%f}" % (loss, depth_var_loss))
                
                loss += depth_var_loss

            if reconst_img != None:
                clean_src_imgs      = ray_sampler.src_rgbs.to(device)[0].permute(0,3,1,2)
                reconst_loss        = F.l1_loss(reconst_img, clean_src_imgs) # * 0.1
                loss += reconst_loss 
                scalars_to_log['train/reconst_loss'] = reconst_loss
                    

            if tar_depth != None:
                xy = ray_batch['xyz'][:,:2]
                pseudo_depth = (ret['outputs_coarse']['depth'] + ret['outputs_fine']['depth']) / 2
                depth_loss = 0

                coords = xy.clone().float()
                coords[:, 0] = (coords[:, 0] / W ) * 2 - 1  
                coords[:, 1] = (coords[:, 1] / H ) * 2 - 1  
                for k in stage_depths.keys():
                    for depth in stage_depths[k]:
                        sel_ref_depth = F.grid_sample(depth[:1], coords[None, None].cuda()).squeeze()
                        depth_loss += F.smooth_l1_loss(sel_ref_depth, pseudo_depth.detach()) * 0.01
                loss += depth_loss
                scalars_to_log['train/depth_loss'] = depth_loss

            loss.backward()
            scalars_to_log['loss'] = loss.item()
            model.optimizer.step()
            model.scheduler.step()

            scalars_to_log['lr'] = model.scheduler.get_last_lr()[0]
            # end of core optimization loop
            dt = time.time() - time0

            # Rest is logging
            if args.local_rank == 0:
                if global_step % args.i_print == 0 or global_step < 10:
                    # write mse and psnr stats
                    mse_error = img2mse(ret['outputs_coarse']['rgb'], ray_batch['rgb']).item()
                    scalars_to_log['train/coarse-loss'] = mse_error
                    scalars_to_log['train/coarse-psnr-training-batch'] = mse2psnr(mse_error)
                    if ret['outputs_fine'] is not None:
                        mse_error = img2mse(ret['outputs_fine']['rgb'], ray_batch['rgb']).item()
                        scalars_to_log['train/fine-loss'] = mse_error
                        scalars_to_log['train/fine-psnr-training-batch'] = mse2psnr(mse_error)

                    logstr = '{} Epoch: {}  step: {} '.format(args.expname, epoch, global_step)
                    for k in scalars_to_log.keys():
                        logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                        writer.add_scalar(k, scalars_to_log[k], global_step)
                    print(logstr)
                    print('each iter time {:.05f} seconds'.format(dt))

                if global_step % args.i_weights == 0:
                    print('Saving checkpoints at {} to {}...'.format(global_step, out_folder))
                    fpath = os.path.join(out_folder, 'model_{:06d}.pth'.format(global_step))
                    model.save_model(fpath)

                # print(global_step, len())
                if global_step % args.i_img == 0:
                    print('Logging a random validation view...')
                    val_data = next(val_loader_iterator)
                    tmp_ray_sampler = RaySamplerSingleImage(val_data, device, render_stride=args.render_stride)
                    H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
                    gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)
                    log_view_to_tb(writer, global_step, args, model, tmp_ray_sampler, projector, device,
                                gt_img, render_stride=args.render_stride, prefix='val/')
                    torch.cuda.empty_cache()

                    print('Logging current training view...')
                    tmp_ray_train_sampler = RaySamplerSingleImage(train_data, device,
                                                                render_stride=1)
                    H, W = tmp_ray_train_sampler.H, tmp_ray_train_sampler.W
                    gt_img = tmp_ray_train_sampler.rgb.reshape(H, W, 3)
                    log_view_to_tb(writer, global_step, args, model, tmp_ray_train_sampler, projector, device,
                                gt_img, render_stride=1, prefix='train/')
            global_step += 1
            if global_step > model.start_step + args.n_iters + 1:
                break
        epoch += 1


### Clean image logging
def log_view_to_tb(writer, global_step, args, model, ray_sampler, projector, device, gt_img,
                   render_stride=1, prefix=''):
    model.switch_to_eval()
    with torch.no_grad():
        ray_batch = ray_sampler.get_all()
        if model.feature_net is not None:
            if args.nearby_imgs > 1:
                nearby_idxs = []
                org_src_rgbs = ray_sampler.src_rgbs.to(device)
                H, W = org_src_rgbs.shape[-3:-1]
                src_poses = ray_batch['src_cameras'][:,:,-16:].reshape(-1, 4, 4).cpu().numpy()
                src_intrinsics = ray_batch['src_cameras'][0, :,2:18].reshape(-1, 4, 4)
                for pose in src_poses:
                    ids = get_nearest_pose_ids(pose, src_poses, args.nearby_imgs, angular_dist_method='dist', sort_by_dist=True)
                    nearby_idxs.append(ids)

                nearby_imgs = torch.stack([org_src_rgbs[0].permute(0,3,1,2)[ids] for ids in nearby_idxs])
                nearby_poses = torch.stack([torch.from_numpy(src_poses[ids]) for ids in nearby_idxs]).to(device)
                nearby_intrinsics = torch.stack([src_intrinsics[ids] for ids in nearby_idxs])[..., :3,:3]

                extrinsics = torch.inverse(nearby_poses)
                input_imgs = [img for img in nearby_imgs.permute(1,0,2,3,4)]
                depth, _, stage_depths = model.patchmatch(input_imgs, nearby_intrinsics, extrinsics, ray_batch['depth_range'][0,0].repeat(args.num_source_views), ray_batch['depth_range'][0,1].repeat(args.num_source_views))            
                
                warped_rgbds, coords = get_depth_warp_img(nearby_imgs, nearby_poses, src_intrinsics, depth.detach(), nearby_idxs)
                ref_rgbd = torch.cat([nearby_imgs[:,:1], depth.unsqueeze(1).detach()], dim=2)
                input_imgs = torch.cat([ref_rgbd, warped_rgbds], dim=1)
                reconst_img, feats = model.feature_net(input_imgs.reshape(args.num_source_views, -1, H, W))
                featmaps = [feats[:, :args.coarse_feat_dim], feats[:, args.coarse_feat_dim:]]
                # ray_batch['src_rgbs'] = reconst_img.permute(0,2,3,1)[None]            
            else:
                featmaps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))
        else:
            featmaps = [None, None]
        ret = render_single_image(ray_sampler=ray_sampler,
                                  ray_batch=ray_batch,
                                  model=model,
                                  projector=projector,
                                  chunk_size=args.chunk_size,
                                  N_samples=args.N_samples,
                                  inv_uniform=args.inv_uniform,
                                  det=True,
                                  N_importance=args.N_importance,
                                  white_bkgd=args.white_bkgd,
                                  render_stride=render_stride,
                                  featmaps=featmaps)

    average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))

    if args.render_stride != 1:
        gt_img = gt_img[::render_stride, ::render_stride]
        average_im = average_im[::render_stride, ::render_stride]

    rgb_gt = img_HWC2CHW(gt_img)
    average_im = img_HWC2CHW(average_im)

    rgb_pred = img_HWC2CHW(ret['outputs_coarse']['rgb'].detach().cpu())

    h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2], average_im.shape[-2])
    w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1], average_im.shape[-1])
    rgb_im = torch.zeros(3, h_max, 3*w_max)
    rgb_im[:, :average_im.shape[-2], :average_im.shape[-1]] = average_im
    rgb_im[:, :rgb_gt.shape[-2], w_max:w_max+rgb_gt.shape[-1]] = rgb_gt
    rgb_im[:, :rgb_pred.shape[-2], 2*w_max:2*w_max+rgb_pred.shape[-1]] = rgb_pred

    depth_im = ret['outputs_coarse']['depth'].detach().cpu()
    acc_map = torch.sum(ret['outputs_coarse']['weights'], dim=-1).detach().cpu()

    if ret['outputs_fine'] is None:
        depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))
        acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))
    else:
        rgb_fine = img_HWC2CHW(ret['outputs_fine']['rgb'].detach().cpu())
        rgb_fine_ = torch.zeros(3, h_max, w_max)
        rgb_fine_[:, :rgb_fine.shape[-2], :rgb_fine.shape[-1]] = rgb_fine
        rgb_im = torch.cat((rgb_im, rgb_fine_), dim=-1)
        depth_im = torch.cat((depth_im, ret['outputs_fine']['depth'].detach().cpu()), dim=-1)
        depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))
        acc_map = torch.cat((acc_map, torch.sum(ret['outputs_fine']['weights'], dim=-1).detach().cpu()), dim=-1)
        acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))

    if args.nearby_imgs > 1:
        postfix = ''
        vis_img = torch.cat([ray_sampler.src_rgbs, reconst_img.permute(0,2,3,1)[None].cpu(), (ray_sampler.src_rgbs - reconst_img.permute(0,2,3,1)[None].cpu())], dim=0)
        vis_img = vis_img[:,:2]
        vis_img = vis_img.permute(4,1,2,0,3).reshape(3, H * 2, -1)[:, ::render_stride, ::render_stride]
        vis_img = vis_img.cpu().clamp(0,1)
        writer.add_image(prefix + 'kernel_reconst'+ postfix, vis_img, global_step)

        pred_depth = depth.squeeze().permute(1,0,2).reshape(H,-1)[::render_stride, ::render_stride]
        pred_depth = img_HWC2CHW(colorize(pred_depth, cmap_name='jet', append_cbar=True))
        writer.add_image(prefix + 'patchmatch_depth'+ postfix, pred_depth, global_step)

        vis_imgs = input_imgs[:2, :, :3].permute(2,0,3,1,4).reshape(3, H * 2, -1)[:, ::render_stride, ::render_stride]
        writer.add_image(prefix + 'depth_warped_imgs'+ postfix, vis_imgs, global_step)

    # write the pred/gt rgb images and depths
    writer.add_image(prefix + 'rgb_gt-coarse-fine', rgb_im.clamp(0,1), global_step)
    writer.add_image(prefix + 'depth_gt-coarse-fine', depth_im, global_step)
    writer.add_image(prefix + 'acc-coarse-fine', acc_map, global_step)

    # write scalar
    pred_rgb = ret['outputs_fine']['rgb'] if ret['outputs_fine'] is not None else ret['outputs_coarse']['rgb']
    psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    writer.add_scalar(prefix + 'psnr_image', psnr_curr_img, global_step)

    model.switch_to_train()


def calc_depth_var(ret):
    depth = ret['depth']  # [N_rays,]
    weights = ret['weights']  # [N_rays, N_samples], T*alpha
    z_vals = ret['z_vals']  # [N_rays, N_samples]

    var = torch.sum(weights * (z_vals - depth.unsqueeze(dim=1)) ** 2, dim=1) / torch.sum(weights, dim=1)  # [N_rays,]
    var = torch.masked_select(var, ~torch.isnan(var)) # remove None value
        
    var_per_ray = torch.mean(var)
    
    return var_per_ray
    
    
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)



# def main_worker(local_rank, ngpus_per_node, args):
#     args.local_rank = local_rank
#     args.world_size = ngpus_per_node
#     torch.distributed.init_process_group(
#         backend='nccl', init_method=args.distributed_init_method, world_size=args.world_size, rank=args.local_rank
#     )

#     synchronize()
    
#     train(args)
    

if __name__ == '__main__':
    parser = config.config_parser()
    args = parser.parse_args()
    
    ngpus_per_node = torch.cuda.device_count()

    # if args.distributed:
    #     torch.cuda.set_device(args.local_rank)
    #     torch.distributed.init_process_group(backend="nccl", init_method="env://")
    #     synchronize()

    # train(args)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method=args.distributed_init_method, world_size=ngpus_per_node, rank=args.local_rank)
        synchronize()

    train(args)
    
    # if args.distributed:
    #     import torch.multiprocessing as mp
    #     mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


