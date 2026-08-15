import math
import os
from copy import deepcopy

import einops
import imageio
import torch
import torch.distributed as dist
import wandb
from tqdm import tqdm

import utils
import models as model_registry
from models.embed import LabelEmbedder
from models.norm import RMSNorm
from trainers import register
from trainers.base_trainer import BaseTrainer
from trainers.larp_tokenizer_trainer import optimizer_dict


def save_video(video, path):
    imageio.mimwrite(path, video.numpy(), fps=25)


@register('vfm_ar_trainer')
class VFMARTrainer(BaseTrainer):
    def __init__(self, rank, cfg):
        super().__init__(rank, cfg)
        self.num_samples = cfg['ar']['num_samples']
        self.num_save_wandb = cfg['ar']['num_save_wandb']
        self.sample_batch_size = cfg['ar']['sample_batch_size']
        self.cfg_scale = cfg['ar']['cfg_scale']
        self.cfg_interval = cfg['ar']['cfg_interval']
        self.sampling_temperture = cfg['ar']['temperature']
        self.sampling_top_k = cfg['ar']['top_k']
        self.sampling_top_p = cfg['ar']['top_p']
        cfg['vae']['checkpoint'] = cfg['vae']['checkpoint'].strip("'").strip('"')
        assert os.path.exists(cfg['vae']['checkpoint']), \
            f"VAE checkpoint not found: {cfg['vae']['checkpoint']}"

        vae_ckpt = torch.load(cfg['vae']['checkpoint'], map_location='cpu', weights_only=False)
        vae_spec = vae_ckpt['model']
        vae_version = cfg['vae'].get('version', 'sd')
        if vae_version == 'sd':
            vae_sd = vae_spec['sd']
        elif vae_version.startswith('ema'):
            alpha = float(vae_version.split('_')[1])
            vae_sd = vae_spec['ema_sd'][alpha]
        else:
            raise ValueError(f"Unknown VAE version: {vae_version}")

        self.vae = model_registry.make(vae_spec, load_sd=False)
        self.vae.load_state_dict(vae_sd, strict=False)
        self.vae = self.vae.to(self.device)
        del vae_ckpt

        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        self.log(f'Loaded VAE ({vae_spec["name"]}) from {cfg["vae"]["checkpoint"]}')
        self.vae_force_fp32 = cfg.get('vae_force_fp32', False)

        self.seq_length = self.vae.bottleneck_token_num
        self.cfg['model']['args']['max_seq_len'] = self.seq_length
        self.cfg['model']['args']['vocab_size'] = self.vae.codebook_size

        self.num_codebooks = self.vae.quantize.num_codebooks
        self.cfg['model']['args']['num_codebooks'] = self.num_codebooks

        assert self.vae.codebook_size % self.num_codebooks == 0
        self.sub_vocab_size = self.vae.codebook_size // self.num_codebooks

        self.log(f'Using sequence length: {self.seq_length}')
        self.log(f'Using vocab size:      {self.vae.codebook_size} '
                 f'(num_codebooks={self.num_codebooks}, sub_vocab={self.sub_vocab_size})')

        self.periodic_save_epoch = cfg.get('periodic_save_epoch', 100)
        if cfg.get('save_epoch', cfg['max_epoch'] + 1) > cfg['max_epoch']:
            cfg['save_epoch'] = self.periodic_save_epoch
            self.log(f'Periodic checkpoint saving enabled every {self.periodic_save_epoch} epochs '
                     f'(epoch_XXXXXX.pth)')

    def update_model_spec(self, model_spec):
        """Resume 时仍与当前 VAE 的 seq_len / vocab / num_codebooks 对齐。"""
        model_spec = deepcopy(model_spec)
        args = model_spec.setdefault('args', {})
        args['max_seq_len'] = self.seq_length
        args['vocab_size'] = self.vae.codebook_size
        args['num_codebooks'] = self.num_codebooks
        return model_spec

    def save_checkpoint(self, filename, save_best=False, model_sd_only=False):
        if filename.startswith('epoch-') and filename.endswith('.pth'):
            suffix = filename[6:-4]
            if suffix.isdigit():
                filename = f'epoch_{int(suffix):06d}.pth'
        super().save_checkpoint(filename, save_best=save_best, model_sd_only=model_sd_only)

    @staticmethod
    def get_exp_name(base_exp_name, cfg, args):
        exp_name = f"{base_exp_name}/"
        if len(cfg.vae.checkpoint) < 8:
            exp_name += cfg.vae.checkpoint + "_"
        if float(cfg.optimizer.args.lr) != 0.0001:
            exp_name += f"lr{cfg.optimizer.args.lr}_"
        if 'weight_decay' in cfg.optimizer.args:
            if cfg.optimizer.args.weight_decay != 0.0:
                exp_name += f"wd{cfg.optimizer.args.weight_decay}_"
        exp_name += f'{cfg.model.name}_'
        exp_name += f'_{args.tag}'
        return exp_name

    def make_model(self, model_spec=None, load_sd=False):
        super().make_model(model_spec, load_sd)
        vae_size_str = utils.compute_num_params(self.vae)
        self.log(f'vae size: {vae_size_str}')
        if self.enable_wandb:
            wandb.run.summary['vae_size'] = vae_size_str

    def configure_optimizers(self, config, load_sd=False):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (
            torch.nn.LayerNorm,
            torch.nn.Embedding,
            LabelEmbedder,
            RMSNorm,
        )
        for mn, m in self.orig_model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        cases = ['pos_emb', 'abs_pe']
        for case in cases:
            if hasattr(self.orig_model, case) and isinstance(getattr(self.orig_model, case), torch.nn.Parameter):
                no_decay.add(case)

        param_dict = {pn: p for pn, p in self.orig_model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, \
            "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, \
            "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)

        config_no_decay = deepcopy({k: v for k, v in config.items() if k != 'sd'})
        if 'weight_decay' in config_no_decay['args']:
            del config_no_decay['args']['weight_decay']

        optimizer_groups = [
            {'params': [param_dict[pn] for pn in sorted(list(decay))],
             'weight_decay': config['args']['weight_decay']},
            {'params': [param_dict[pn] for pn in sorted(list(no_decay))],
             'weight_decay': 0.0},
        ]

        optimizer = optimizer_dict[config['name']](
            optimizer_groups, **config_no_decay['args']
        )
        if load_sd:
            optimizer.load_state_dict(config['sd'])
        self.optimizer = optimizer

    # =====================================================================
    # MCQ 工具函数
    # =====================================================================
    def _ensure_multi_codebook(self, z):
        if z.ndim == 2:
            assert self.num_codebooks == 1
            z = z.unsqueeze(-1)
        elif z.ndim == 3:
            assert z.shape[-1] == self.num_codebooks
        else:
            raise ValueError(f'Unexpected bottleneck_rep shape: {tuple(z.shape)}')
        return z.long()

    def forward_ar_model(self, z, c):
        input_tokens = z[:, :-1]
        with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            logits, loss = self.model_ddp(cond_idx=c, idx=input_tokens, targets=z)
        loss = loss.mean()
        return logits, loss

    def _iter_step(self, data, is_train):
        x = data.pop('gt').to(self.device, non_blocking=True)
        c = data.pop('label').to(self.device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=self.amp_dtype,
                            enabled=not self.vae_force_fp32 and self.use_amp):
            with torch.no_grad():
                z = self.vae.encode_tokens(x)['bottleneck_rep']  # [B, N, M]

        z = self._ensure_multi_codebook(z)

        logits, loss = self.forward_ar_model(z, c)

        if logits.ndim == 4:
            B, N, M, V = logits.shape
            logits_flat = logits.reshape(B, N * M, V)
            z_flat = z.reshape(B, N * M)
            topk_accuracies = utils.calculate_topk_accuracy(logits_flat, z_flat, topk=(1, 5))
        else:
            topk_accuracies = utils.calculate_topk_accuracy(logits, z.squeeze(-1), topk=(1, 5))

        if is_train:
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            for ema_decay, ema_model in self.ema_model_dict.items():
                self.update_ema(ema_model, decay=ema_decay)

        return_dict = {'loss': loss.item(), **topk_accuracies}
        return return_dict

    def train_step(self, data):
        return self._iter_step(data, is_train=True)

    def evaluate_step(self, data):
        with torch.no_grad():
            return self._iter_step(data, is_train=False)

    @torch.inference_mode()
    def visualize_epoch(self, c_distribution=None, logging=True,
                        use_ema=False, force_fp32=False):
        if use_ema:
            model = self.ema_model_dict[self.ema_decay_list[0]]
            self.log(f'Using EMA model {self.ema_decay_list[0]} for visualization')
        else:
            model = self.model
            self.log('Using current model for visualization')
        model = deepcopy(model)
        if not force_fp32:
            model = model.to(device=self.device, dtype=torch.bfloat16)
        model.eval()

        fp16_enabled = (not force_fp32) and self.use_amp
        self.log(f'Visualizing with fp16: {fp16_enabled}')

        out_dir = os.path.join(self.cfg['env']['save_dir'], 'visualize')
        epoch_tag = f'epoch_{self.epoch:06d}' if hasattr(self, 'epoch') else 'latest'
        vid_dir = os.path.join(out_dir, 'vid', epoch_tag)
        grid_dir = os.path.join(out_dir, 'grid')
        if self.is_master:
            os.makedirs(vid_dir, exist_ok=True)
            os.makedirs(grid_dir, exist_ok=True)

        sample_i3d_feats = None
        sample_inception_feats = None

        if dist.is_initialized():
            dist.barrier()
            world_size = dist.get_world_size()
        else:
            world_size = 1

        assert self.sample_batch_size % world_size == 0
        n = int(self.sample_batch_size // world_size)
        total_samples = int(math.ceil(self.num_samples / self.sample_batch_size) * self.sample_batch_size)

        if self.is_master:
            self.log(f"Total number of images that will be sampled: {total_samples}")

        assert total_samples % world_size == 0
        samples_needed_this_gpu = int(total_samples // world_size)
        assert samples_needed_this_gpu % n == 0
        iterations = int(samples_needed_this_gpu // n)

        if c_distribution is None:
            assert hasattr(self, 'train_loader'), "train_loader is not defined"
            c_distribution = self.train_loader.dataset.label_count
            if c_distribution is not None:
                c_distribution = torch.tensor(c_distribution, device=self.device, dtype=torch.float32)
                c_distribution /= c_distribution.sum()

        pbar = range(iterations)
        pbar = tqdm(pbar) if self.is_master else pbar

        vis_res = []

        for _ in pbar:
            if c_distribution is not None:
                c = torch.multinomial(c_distribution, n, replacement=True).to(device=self.device)
            else:
                c = -1 * torch.ones(n, device=self.device, dtype=torch.int64)

            sampled_seqs = model.sample(
                c=c,
                cfg_scale=self.cfg_scale, cfg_interval=self.cfg_interval,
                temperature=self.sampling_temperture,
                top_k=self.sampling_top_k,
                top_p=self.sampling_top_p,
            )

            if sampled_seqs.ndim == 3 and self.num_codebooks == 1:
                sampled_seqs_for_vae = sampled_seqs.squeeze(-1)
            else:
                sampled_seqs_for_vae = sampled_seqs

            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                sampled_batch = self.vae.decode_from_bottleneck(sampled_seqs_for_vae)

            sampled_batch = sampled_batch.clamp(0., 1.)
            sampled_batch = sampled_batch.to(torch.float32).contiguous()

            if sampled_batch.shape[2] >= 10:
                sample_i3d_feats = self.fvd_calculator.get_feature_stats_for_batch(
                    sampled_batch, sample_i3d_feats)
            if self.fid_calculator is not None and sampled_batch.shape[2] == 1:
                sample_inception_feats = self.fid_calculator.get_feature_stats_for_batch(
                    sampled_batch.squeeze(), sample_inception_feats)

            if len(vis_res) < self.num_save_wandb and self.is_master:
                vis_res.append(sampled_batch * 255.)

        del model

        if dist.is_initialized():
            dist.barrier()

        if sample_i3d_feats is not None:
            assert sample_i3d_feats.num_items == total_samples
        if sample_inception_feats is not None:
            assert sample_inception_feats.num_items == total_samples

        fvd, fid = None, None

        if self.is_master:
            vis_res = torch.cat(vis_res, dim=0)[:self.num_save_wandb]
            vis_res = einops.rearrange(vis_res, 'b c t h w -> b t c h w').cpu()
            vis_res = vis_res.type(torch.uint8)

            assert self.num_save_wandb % 8 == 0
            col = 8
            row = self.num_save_wandb // col
            if vis_res.shape[0] < row * col:
                col = 4
                row = vis_res.shape[0] // col

            vis_res = einops.rearrange(vis_res, '(b row col) t c h w -> b t c (row h) (col w)',
                                       row=row, col=col)

            grid_to_save = einops.rearrange(vis_res, 'b t c h w -> b t h w c')
            for gi in range(grid_to_save.shape[0]):
                grid_path_base = os.path.join(grid_dir, f'grid_{epoch_tag}_{gi:02d}')
                if grid_to_save.shape[1] >= 2:
                    save_video(grid_to_save[gi], grid_path_base + '.mp4')
                else:
                    imageio.imwrite(grid_path_base + '.png', grid_to_save[gi, 0].numpy())
            self.log(f'Saved local visualizations to {vid_dir} and {grid_dir}')

            if self.enable_wandb:
                if vis_res.shape[1] >= 2:
                    wandb.log({'samples': [wandb.Video(v, fps=4, format="mp4") for v in vis_res]},
                              step=self.epoch)
                else:
                    vis_res_img = vis_res.squeeze(1)
                    wandb.log({'samples': [wandb.Image(v) for v in vis_res_img]},
                              step=self.epoch)

            if sample_i3d_feats is not None and self.fvd_real_stats is not None:
                self.log('Calculating FVD with loaded real stats')
                fvd = self.fvd_calculator.calculate_fvd(sample_i3d_feats, self.fvd_real_stats)
                if isinstance(fvd, torch.Tensor):
                    fvd = fvd.item()
                if logging:
                    self.log_temp_scalar(f'test/fvd', fvd)
                    self.log_buffer.append(f' sample_fvd={fvd:.4f}')
                if hasattr(self, 'current_fvd'):
                    self.current_fvd = fvd

            if sample_inception_feats is not None and self.fid_real_stats is not None \
                    and self.fid_calculator is not None:
                self.log('Calculating FID with loaded real stats')
                fid = self.fid_calculator.calculate_fid(sample_inception_feats, self.fid_real_stats)
                if isinstance(fid, torch.Tensor):
                    fid = fid.item()
                if logging:
                    self.log_temp_scalar(f'test/fid', fid)
                    self.log_buffer.append(f' sample_fid={fid:.4f}')

        if dist.is_initialized():
            dist.barrier()

        return {'fvd': fvd, 'fid': fid}
