"""
Sampling script for VFM AR generation model.

Usage:
    python sample_vfm_ar.py \
        --ar_model /path/to/ar_checkpoint.pth \
        --vae /path/to/vae_checkpoint.pth \
        --output_dir ./output/samples \
        --num_samples 10000 \
        --sample_batch_size 16 \
        --cfg_scale 2.0 \
        --temperature 1.0 \
        --top_k 0 \
        --top_p 1.0
"""
import argparse
import gc
import math
import os
from concurrent.futures import ThreadPoolExecutor

import einops
import imageio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from compute_gfvd import (
    build_real_dataset_subset,
    compute_and_report_gfvd,
    compute_gfvd_score,
    save_gfvd_result,
)
import models as model_registry
from models.larp_ar import LARP_AR
from utils import FVDCalculator, frames_to_horizontal_strip


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description='Sample videos with VFM AR model')

    parser.add_argument('--ar_model', type=str, required=True,
                        help='AR model checkpoint path')
    parser.add_argument('--vae', type=str, required=True,
                        help='VFM autoencoder checkpoint path')
    parser.add_argument('--vae_version', type=str, default='sd',
                        help='VAE version: sd or ema_<decay>')

    parser.add_argument('--output_dir', type=str, default='./output/vfm_ar_samples',
                        help='Directory to save the samples')
    parser.add_argument('--num_samples', type=int, default=256,
                        help='Number of samples to generate')
    parser.add_argument('--sample_batch_size', type=int, default=16,
                        help='Batch size for sampling')
    parser.add_argument('--num_classes', type=int, default=101,
                        help='Number of classes for conditional generation')

    parser.add_argument('--cfg_scale', type=float, default=2.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--cfg_interval', type=int, default=-1,
                        help='CFG interval (-1 = always apply)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=0,
                        help='Top-k sampling (0 = disabled)')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='Top-p / nucleus sampling (1.0 = disabled)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        help='Data type for AR model (bfloat16 / float16 / float32)')

    parser.add_argument('--class_ids', type=str, default=None,
                        help='Comma-separated class ids to sample (default: uniform random)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--random_labels', action='store_true',
                        help='Use random class labels instead of dataset labels (non-LARP protocol)')

    parser.add_argument('--skip_gfvd', action='store_true',
                        help='Skip gFVD computation after sampling (default: compute gFVD)')
    parser.add_argument('--dataset_csv', type=str, default='ucf101_train.csv',
                        help='UCF101 csv for real video reference set')
    parser.add_argument('--dataset_split_seed', type=int, default=42,
                        help='Random seed for selecting real video subset')
    parser.add_argument('--frame_num', type=int, default=16,
                        help='Number of frames for real video FVD reference')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='Crop size for real video FVD reference')
    parser.add_argument('--gfvd_batch_size', type=int, default=32,
                        help='Batch size for extracting real video features')
    parser.add_argument('--gfvd_num_workers', type=int, default=0,
                        help='DataLoader workers during sampling (must be 0: decord+CUDA fork segfaults)')
    parser.add_argument('--gt_stats', type=str, default=None,
                        help='Path to save/load real video feature stats pkl')
    parser.add_argument('--force_recompute_gt', action='store_true',
                        help='Recompute real video stats even if gt_stats exists')

    args = parser.parse_args(input_args)
    return args


def save_frame_strip(frames_u8, path):
    imageio.imwrite(path, frames_to_horizontal_strip(frames_u8))


def load_vae_from_checkpoint(ckpt_path, version='sd'):
    """Load VAE by checkpoint-registered name (supports MCQ multi-codebook models)."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_spec = ckpt['model']
    del ckpt
    gc.collect()

    if version == 'sd':
        sd = model_spec['sd']
    elif version.startswith('ema'):
        alpha = float(version.split('_')[1])
        sd = model_spec['ema_sd'][alpha]
    else:
        raise ValueError(f'Unknown VAE version: {version}')

    vae = model_registry.make(model_spec, load_sd=False)
    vae.load_state_dict(sd, strict=False)
    return vae


def decode_batch(vae, sampled_seqs, num_codebooks):
    if sampled_seqs.ndim == 2:
        sampled_seqs_for_vae = sampled_seqs
    elif sampled_seqs.ndim == 3 and num_codebooks == 1:
        sampled_seqs_for_vae = sampled_seqs.squeeze(-1)
    elif sampled_seqs.ndim == 3:
        sampled_seqs_for_vae = sampled_seqs
    else:
        raise ValueError(
            f'Unexpected sampled_seqs shape {tuple(sampled_seqs.shape)}, '
            f'num_codebooks={num_codebooks}'
        )
    sampled_batch = vae.decode_from_bottleneck(
        sampled_seqs_for_vae.to(dtype=torch.long)
    )
    return sampled_batch.float().clamp(0., 1.).contiguous()


@torch.inference_mode()
def sample_videos(
    ar_model: LARP_AR,
    vae: torch.nn.Module,
    output_dir: str,
    num_samples: int = 256,
    sample_batch_size: int = 16,
    num_classes: int = 101,
    cfg_scale: float = 2.0,
    cfg_interval: int = -1,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    class_ids=None,
    seed: int = 42,
    use_dataset_labels: bool = True,
    compute_gfvd: bool = True,
    dataset_csv: str = 'ucf101_train.csv',
    dataset_split_seed: int = 42,
    frame_num: int = 16,
    crop_size: int = 256,
    gfvd_batch_size: int = 32,
    gfvd_num_workers: int = 0,
    gt_stats: str = None,
    force_recompute_gt: bool = False,
    save_videos: bool = True,
):
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    device = ar_model.device
    num_codebooks = getattr(ar_model, 'num_codebooks', 1)
    print(f'AR model: num_codebooks={num_codebooks}, '
          f'sub_vocab={ar_model.sub_vocab_size}, '
          f'vocab_size={ar_model.vocab_size}')
    print(f'Label mode: {"dataset (LARP-aligned)" if use_dataset_labels else "random uniform"}')

    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(output_dir, 'images')
    executor = None
    save_futures = []
    if save_videos:
        os.makedirs(image_dir, exist_ok=True)
        executor = ThreadPoolExecutor(max_workers=4)

    fvd_calculator = FVDCalculator(device=device)
    sample_feats = None
    gt_feats = None
    generated_count = 0

    if use_dataset_labels and class_ids is None:
        eval_subset = build_real_dataset_subset(
            dataset_csv=dataset_csv,
            num_samples=num_samples,
            dataset_split_seed=dataset_split_seed,
            frame_num=frame_num,
            crop_size=crop_size,
        )
        loader = DataLoader(
            eval_subset,
            batch_size=sample_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=gfvd_num_workers,
            pin_memory=True,
        )
        pbar = tqdm(loader, desc='Sampling (gen+gt stats together)')
        torch.manual_seed(seed)

        for batch_idx, data in enumerate(pbar):
            c = data['label'].to(device, non_blocking=True)
            orig_video = data['gt'].clamp(0., 1.).to(device, non_blocking=True)
            n = c.shape[0]

            sampled_seqs = ar_model.sample(
                c=c,
                cfg_scale=cfg_scale,
                cfg_interval=cfg_interval,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            if hasattr(ar_model, 'reset_caches'):
                ar_model.reset_caches()

            sampled_batch = decode_batch(vae, sampled_seqs, num_codebooks)

            if (batch_idx + 1) * sample_batch_size > num_samples:
                cut = num_samples - batch_idx * sample_batch_size
                sampled_batch = sampled_batch[:cut]
                orig_video = orig_video[:cut]
                c = c[:cut]
                n = cut

            if sampled_batch.shape[2] >= 10:
                sample_feats = fvd_calculator.get_feature_stats_for_batch(
                    sampled_batch, sample_feats)
                gt_feats = fvd_calculator.get_feature_stats_for_batch(
                    orig_video, gt_feats)

            if save_videos:
                sb_u8 = (sampled_batch * 255.).to(torch.uint8)
                sb_u8 = einops.rearrange(sb_u8, 'b c t h w -> b t h w c').cpu()
                cls_ids = c.cpu().tolist()
                for i in range(n):
                    idx = generated_count + i
                    cls_id = cls_ids[i]
                    fname = f'sample_{idx:05d}_cls{cls_id}'
                    save_futures.append(executor.submit(
                        save_frame_strip,
                        sb_u8[i],
                        os.path.join(image_dir, f'{fname}.png'),
                    ))

            generated_count += n
            pbar.set_postfix(generated=generated_count)
            del orig_video, sampled_batch, sampled_seqs
    else:
        total_batches = math.ceil(num_samples / sample_batch_size)
        pbar = tqdm(range(total_batches), desc='Sampling (random labels)')
        torch.manual_seed(seed)

        for batch_idx in pbar:
            remaining = num_samples - generated_count
            n = min(sample_batch_size, remaining)

            if class_ids is not None:
                c = torch.tensor(class_ids[:n], device=device, dtype=torch.long)
                if len(c) < n:
                    c = c.repeat(math.ceil(n / len(c)))[:n]
            else:
                c = torch.randint(0, num_classes, (n,), device=device)

            sampled_seqs = ar_model.sample(
                c=c,
                cfg_scale=cfg_scale,
                cfg_interval=cfg_interval,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            if hasattr(ar_model, 'reset_caches'):
                ar_model.reset_caches()

            sampled_batch = decode_batch(vae, sampled_seqs, num_codebooks)

            if sampled_batch.shape[2] >= 10:
                sample_feats = fvd_calculator.get_feature_stats_for_batch(
                    sampled_batch, sample_feats)

            if save_videos:
                sb_u8 = (sampled_batch * 255.).to(torch.uint8)
                sb_u8 = einops.rearrange(sb_u8, 'b c t h w -> b t h w c').cpu()
                cls_ids = c.cpu().tolist()
                for i in range(n):
                    idx = generated_count + i
                    cls_id = cls_ids[i]
                    fname = f'sample_{idx:05d}_cls{cls_id}'
                    save_futures.append(executor.submit(
                        save_frame_strip,
                        sb_u8[i],
                        os.path.join(image_dir, f'{fname}.png'),
                    ))

            generated_count += n
            pbar.set_postfix(generated=generated_count)
            del sampled_batch, sampled_seqs

    if executor is not None:
        executor.shutdown(wait=True)
        for future in save_futures:
            future.result()
    if save_videos:
        print(f'Generated {generated_count} samples -> {image_dir}')
    else:
        print(f'Generated {generated_count} samples (frame strips not saved)')

    stats_path = os.path.join(output_dir, 'sample_fvd_stats.pkl')
    if sample_feats is None:
        print('WARNING: generated videos have < 10 frames, skip FVD stats saving')
        return None, None

    sample_feats.save(stats_path)
    print(f'Saved generated FVD feature stats to {stats_path}')

    gfvd = None
    if compute_gfvd:
        print('\nComputing gFVD ...')
        if gt_feats is not None:
            if gt_stats is None:
                gt_stats = os.path.join(output_dir, f'gt_fvd_stats_n{num_samples}.pkl')
            gt_feats.save(gt_stats)
            print(f'Saved real video stats to {gt_stats}')
            print(f'Loaded generated stats: {sample_feats.num_items} samples')
            print(f'Loaded real stats: {gt_feats.num_items} samples')
            gfvd = compute_gfvd_score(sample_feats, gt_feats, device=device)
            save_gfvd_result(
                output_dir=output_dir,
                gen_stats_path=stats_path,
                gt_stats_path=gt_stats,
                gen_stats=sample_feats,
                gt_stats=gt_feats,
                dataset_csv=dataset_csv,
                dataset_split_seed=dataset_split_seed,
                gfvd=gfvd,
            )
            print(f'\n========== gFVD Result ==========')
            print(f'gFVD: {gfvd:.4f}')
            print(f'=================================')
        else:
            if gt_stats is None:
                gt_stats = os.path.join(output_dir, f'gt_fvd_stats_n{num_samples}.pkl')
            gfvd = compute_and_report_gfvd(
                gen_stats=sample_feats,
                output_dir=output_dir,
                gen_stats_path=stats_path,
                gt_stats_path=gt_stats,
                dataset_csv=dataset_csv,
                num_samples=num_samples,
                dataset_split_seed=dataset_split_seed,
                frame_num=frame_num,
                crop_size=crop_size,
                batch_size=gfvd_batch_size,
                num_workers=gfvd_num_workers,
                force_recompute_gt=force_recompute_gt,
                device=device,
            )

    return sample_feats, gfvd


def main():
    args = parse_args()

    if not isinstance(args.dtype, torch.dtype):
        dtype = getattr(torch, args.dtype)
    else:
        dtype = args.dtype

    print(f'Loading AR model from {args.ar_model} ...')
    ar_model = LARP_AR.from_checkpoint(args.ar_model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ar_model = ar_model.cuda()
    ar_model = ar_model.to(dtype)
    ar_model.eval()

    print(f'Loading VAE from {args.vae} ...')
    vae = load_vae_from_checkpoint(args.vae, version=args.vae_version)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    vae = vae.cuda()
    vae.eval()

    class_ids = None
    if args.class_ids is not None:
        class_ids = [int(x) for x in args.class_ids.split(',')]
        print(f'Sampling specific classes: {class_ids}')

    sample_videos(
        ar_model=ar_model,
        vae=vae,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        sample_batch_size=args.sample_batch_size,
        num_classes=args.num_classes,
        cfg_scale=args.cfg_scale,
        cfg_interval=args.cfg_interval,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        class_ids=class_ids,
        seed=args.seed,
        use_dataset_labels=not args.random_labels,
        compute_gfvd=not args.skip_gfvd,
        dataset_csv=args.dataset_csv,
        dataset_split_seed=args.dataset_split_seed,
        frame_num=args.frame_num,
        crop_size=args.crop_size,
        gfvd_batch_size=args.gfvd_batch_size,
        gfvd_num_workers=args.gfvd_num_workers,
        gt_stats=args.gt_stats,
        force_recompute_gt=args.force_recompute_gt,
    )


if __name__ == '__main__':
    main()
