import sys
sys.path.append('../')
import imageio
from config import config_parser
from ibrnet.sample_ray import RaySamplerSingleImage
from ibrnet.render_image import render_single_image
from ibrnet.model import IBRNetModel
from utils import *
from ibrnet.projection import Projector
from ibrnet.data_loaders import dataset_dict
import tensorflow as tf
from lpips_tensorflow import lpips_tf
from torch.utils.data import DataLoader
from tqdm import tqdm
from ibrnet.data_loaders.data_utils import get_nearest_pose_ids, get_depth_warp_img
import time
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    args.distributed = False

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

    cnt = 0
    total = 0 
    for i, data in tqdm(enumerate(test_loader), desc=f' ############# Test Image {cnt} ############'):
        cnt += 1
        rgb_path = data['rgb_path'][0]
        file_id = os.path.basename(rgb_path).split('.')[0]
        src_rgbs = data['src_rgbs'][0].cpu().numpy()

        averaged_img = (np.mean(src_rgbs, axis=0) * 255.).astype(np.uint8)
        imageio.imwrite(os.path.join(out_scene_dir, '{}_average.png'.format(file_id)),
                        averaged_img)

        model.switch_to_eval()
        with torch.no_grad():
            ray_sampler = RaySamplerSingleImage(data, device=device)
            ray_batch = ray_sampler.get_all()
            start = time.time()
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
                ray_batch['src_rgbs'] = reconst_img.permute(0,2,3,1)[None]            
            else:
                featmaps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))

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
                                      args=args)
            total += time.time() - start
            print(cnt,"Avg rendering time" ,round(total / cnt, 3))
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

