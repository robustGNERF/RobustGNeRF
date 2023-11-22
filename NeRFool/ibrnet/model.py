# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import os
from ibrnet.mlp_network import IBRNet
from ibrnet.feature_network import ResUNet
from ibrnet.patchmatch.net import PatchmatchNet
from ibrnet.restormer import Restormer
from ibrnet.transformer_network import GNT
from pathlib import Path
from torch import nn
from pathlib import Path

def get_latest_file(root: Path, suffix="*"):
    return max(root.glob(suffix), key=os.path.getmtime)

def de_parallel(model):
    return model.module if hasattr(model, 'module') else model

########################################################################################################################
# creation/saving/loading of nerf
########################################################################################################################


class IBRNetModel(nn.Module):
    def __init__(self, args, load_opt=True, load_scheduler=True):
        super().__init__()
        self.args = args
        device = torch.device('cuda:{}'.format(args.local_rank))
        # # create coarse IBRNet
        # self.net_coarse = IBRNet(args,
        #                          in_feat_ch=self.args.coarse_feat_dim,
        #                          n_samples=self.args.N_samples).to(device)
        self.net_coarse = GNT(
            args,
            in_feat_ch=self.args.coarse_feat_dim,
            posenc_dim=3 + 3 * 2 * 10,
            viewenc_dim=3 + 3 * 2 * 10,
            ret_alpha=args.N_importance > 0,
        ).to(device)
        
        if args.coarse_only:
            self.net_fine = None
        else:
            # # create coarse IBRNet
            # self.net_fine = IBRNet(args,
            #                        in_feat_ch=self.args.fine_feat_dim,
            #                        n_samples=self.args.N_samples+self.args.N_importance).to(device)
            self.net_fine = GNT(
                args,
                in_feat_ch=self.args.fine_feat_dim,
                posenc_dim=3 + 3 * 2 * 10,
                viewenc_dim=3 + 3 * 2 * 10,
                ret_alpha=True,
            ).to(device)

        if args.nearby_imgs > 1:
            self.patchmatch = PatchmatchNet(
                patchmatch_interval_scale=[0.005, 0.0125, 0.025],
                propagation_range=[4,3,2], #[3,2,1], #[7,5,3], [8,6,4],  #, #[6, 4, 2], # [4,3,2], 
                patchmatch_iteration=[1, 2, 2],
                patchmatch_num_sample=[8, 8, 16],
                propagate_neighbors=[0, 8, 16] ,#8, 16],
                evaluate_neighbors=[9, 9, 17] #[9, 9, 9],
            ).to(device)
            self.feature_net = Restormer(inp_channels=(3 + 1) * args.nearby_imgs, dim=16, num_blocks=[1,1,1,1], heads=[1,2,4,4], ffn_expansion_factor=1.5, dual_pixel_task=False, num_refinement_blocks=1, LayerNorm_type='BiasFree', pixelshuffle=False).to(device)
        else:
            # create feature extraction network
            self.feature_net = ResUNet(coarse_out_ch=self.args.coarse_feat_dim,
                                    fine_out_ch=self.args.fine_feat_dim,
                                    coarse_only=self.args.coarse_only).cuda()

        # optimizer and learning rate scheduler
        learnable_params = list(self.net_coarse.parameters())
        learnable_params += list(self.feature_net.parameters())
        if self.net_fine is not None:
            learnable_params += list(self.net_fine.parameters())

        if self.net_fine is not None:
            self.optimizer = torch.optim.Adam([
                {'params': self.net_coarse.parameters()},
                {'params': self.net_fine.parameters()},
                {'params': self.feature_net.parameters(), 'lr': args.lrate_feature}],
                lr=args.lrate_mlp)
        else:
            self.optimizer = torch.optim.Adam([
                {'params': self.net_coarse.parameters()},
                {'params': self.feature_net.parameters(), 'lr': args.lrate_feature}],
                lr=args.lrate_mlp)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=args.lrate_decay_steps,
                                                         gamma=args.lrate_decay_factor)

        out_folder = os.path.join(args.rootdir, 'MetaFool', 'out', args.expname)
        self.start_step = self.load_from_ckpt(Path(out_folder))

        if args.distributed:
            self.net_coarse = torch.nn.parallel.DistributedDataParallel(
                self.net_coarse,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )

            self.feature_net = torch.nn.parallel.DistributedDataParallel(
                self.feature_net,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )

            if self.net_fine is not None:
                self.net_fine = torch.nn.parallel.DistributedDataParallel(
                    self.net_fine,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank
                )

        elif args.use_dp:
            self.net_coarse = torch.nn.parallel.DataParallel(
                self.net_coarse,
            )

            self.feature_net = torch.nn.parallel.DataParallel(
                self.feature_net,
            )

            if self.net_fine is not None:
                self.net_fine = torch.nn.parallel.DataParallel(
                    self.net_fine,
                )


    def switch_to_eval(self):
        self.net_coarse.eval()
        self.feature_net.eval()
        if self.net_fine is not None:
            self.net_fine.eval()

    def switch_to_train(self):
        self.net_coarse.train()
        self.feature_net.train()
        if self.net_fine is not None:
            self.net_fine.train()

    def save_model(self, filename):
        to_save = {'optimizer': self.optimizer.state_dict(),
                   'scheduler': self.scheduler.state_dict(),
                   'model' : de_parallel(self).state_dict()}

        torch.save(to_save, filename)

    def load_model(self, filename):
        load_dict = torch.load(filename, map_location=torch.device(f'cuda:{self.args.local_rank}'))
        if 'model' not in load_dict:
            # for old version of ckpt
            load_dict = self.convert_state_to_model(load_dict)

        model_dict = load_dict['model'].copy()
        for key in load_dict['model']:
            if "spatial_views_attention" in key:
                print(f"[**] removing key {key}")
                del model_dict[key]

        load_dict['model'] = model_dict

        if not self.args.no_load_opt:
            self.optimizer.load_state_dict(load_dict['optimizer'])
        if not self.args.no_load_scheduler:
            self.scheduler.load_state_dict(load_dict['scheduler'])

        self.load_weights_to_net(self, load_dict['model'])

    @staticmethod
    def convert_state_to_model(load_dict):
        new_load_dict = {'optimizer': load_dict['optimizer'], 'scheduler': load_dict['scheduler'], 'model': {}}
        for net, weights in load_dict.items():
            if net not in ['optimizer', 'scheduler']:
                new_load_dict['model'].update({f"{net}.{k}": w for k, w in weights.items()})

        return new_load_dict

    @staticmethod
    def load_weights_to_net(net, pretrained_dict, allow_weights_mismatch=False):
        try:
            net.load_state_dict(pretrained_dict)
        except RuntimeError:
            if not allow_weights_mismatch:
                raise
            else:
                # from https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3
                new_model_dict = net.state_dict()

                # 1. filter of weights with shape mismatch
                for pre_k, pre_v in pretrained_dict.items():
                    if pre_k in new_model_dict:
                        new_v = new_model_dict[pre_k]
                        if new_v.shape == pre_v.shape:
                            new_model_dict[pre_k] = new_v
                        else:
                            pass
                            # if we want to load partial layers, it can be done with:
                            # new_model_dict[pre_k][torch.where(torch.ones_like(new_v))] = new_v.view(-1).clone()
                # 3. load the new state dict
                net.load_state_dict(new_model_dict)

    def load_from_ckpt(self, out_folder: Path):
        """
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        """

        # all existing ckpts
        ckpt = None

        if self.args.ckpt_path is not None: #and not self.args.resume_training:
            # if not Path(self.args.ckpt_path).exists():  # load the specified ckpt
            #     raise FileNotFoundError(f"requested ckpt_path does not exist: {self.args.ckpt_path}")
            ckpt = self.args.ckpt_path
        if ckpt is not None and not self.args.no_reload:
            step = int(Path(ckpt).stem[-6:])
            # print_link(ckpt, '[*] Reloading from', f"starting at step={step}")
            # step = 0
            print("Loading step", step, " from", ckpt)
            self.load_model(ckpt)
        else:
            if ckpt is None:
                print('[*] No ckpts found, training from scratch...')
                print(str(self.args.ckpt_path))
            if self.args.no_reload:
                print('[*] no_reload, training from scratch...')

            step = 0

        return step


    # def load_model(self, filename, load_opt=True, load_scheduler=True):
    #     if self.args.distributed:
    #         to_load = torch.load(filename, map_location='cuda:{}'.format(self.args.local_rank))
    #     else:
    #         to_load = torch.load(filename)

    #     if load_opt:
    #         self.optimizer.load_state_dict(to_load['optimizer'])
    #     if load_scheduler:
    #         self.scheduler.load_state_dict(to_load['scheduler'])

    #     missing_keys, unexpected_keys = self.net_coarse.load_state_dict(to_load['net_coarse'], strict=False)
    #     for missing_key in missing_keys:
    #         assert 'pos_encoding' in missing_key
    #     assert len(unexpected_keys) == 0

    #     self.feature_net.load_state_dict(to_load['feature_net'])

    #     if self.net_fine is not None and 'net_fine' in to_load.keys():
    #         missing_keys, unexpected_keys = self.net_fine.load_state_dict(to_load['net_fine'], strict=False)
            
    #         for missing_key in missing_keys:
    #             assert 'pos_encoding' in missing_key
    #         assert len(unexpected_keys) == 0

    # def load_from_ckpt(self, out_folder,
    #                    load_opt=True,
    #                    load_scheduler=True,
    #                    force_latest_ckpt=False):
    #     '''
    #     load model from existing checkpoints and return the current step
    #     :param out_folder: the directory that stores ckpts
    #     :return: the current starting step
    #     '''

    #     # all existing ckpts
    #     ckpts = []
    #     if os.path.exists(out_folder):
    #         ckpts = [os.path.join(out_folder, f)
    #                  for f in sorted(os.listdir(out_folder)) if f.endswith('.pth')]

    #     if self.args.ckpt_path is not None and not force_latest_ckpt:
    #         if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
    #             ckpts = [self.args.ckpt_path]

    #     if len(ckpts) > 0 and not self.args.no_reload:
    #         fpath = ckpts[-1]
    #         self.load_model(fpath, load_opt, load_scheduler)
    #         step = int(fpath[-10:-4])
    #         print('Reloading from {}, starting at step={}'.format(fpath, step))
    #     else:
    #         print('No ckpts found, training from scratch...')
    #         step = 0

    #     return step

