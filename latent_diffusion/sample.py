"""
Generate videos from a trained latent diffusion model.
Supports both CFG and non-CFG sampling, DDPM and DDIM.

Usage (non-CFG, unconditional):
    python latent_diffusion/sample.py \
        --dit-ckpt save/latent_diffusion/000-DiT1D-L-cosine/checkpoints/0100000.pt \
        --ae-ckpt /path/to/ae_checkpoint.pth \
        --num-samples 16

Usage (CFG, class-conditional):
    python latent_diffusion/sample.py \
        --dit-ckpt save/latent_diffusion/000-DiT1D-L-cosine/checkpoints/0100000.pt \
        --ae-ckpt /path/to/ae_checkpoint.pth \
        --num-samples 16 \
        --cfg-scale 4.0 \
        --class-label 0

Usage (FVD evaluation against dataset ground truth):
    python latent_diffusion/sample.py \
        --dit-ckpt save/latent_diffusion/000-DiT1D-L-cosine/checkpoints/0100000.pt \
        --ae-ckpt /path/to/ae_checkpoint.pth \
        --eval-fvd \
        --num-samples 2048 \
        --cfg-scale 4.0 \
        --data-path data/metadata \
        --csv-file ucf101_train.csv
"""

import argparse
import os
import random
import sys

import imageio
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from modelling.diffusion import create_diffusion
from latent_diffusion.dit import DiT1D_models
from latent_diffusion.ae_loader import load_autoencoder
from datasets.video_dataset import VideoDataset
from utils import FVDCalculator


def load_dit_and_diffusion(args, device):
    print(f"Loading DiT checkpoint: {args.dit_ckpt}")
    dit_ckpt = torch.load(args.dit_ckpt, map_location='cpu')

    if 'ema' not in dit_ckpt and 'model' not in dit_ckpt:
        if 'mean' in dit_ckpt and 'std' in dit_ckpt:
            raise ValueError(
                f"{args.dit_ckpt} only contains latent mean/std (latent_stats.pt), "
                "not DiT weights. Point --dit-ckpt to a training checkpoint under "
                "e.g. save/latent_diffusion1/*/checkpoints/XXXXXXX.pt"
            )
        raise ValueError(
            f"Invalid DiT checkpoint: {args.dit_ckpt} (expected keys 'ema' or 'model')"
        )

    ckpt_args = dit_ckpt.get('args', {})
    model_name = ckpt_args.get('model', args.model)
    learn_sigma = ckpt_args.get('learn_sigma', args.learn_sigma)
    num_classes = ckpt_args.get('num_classes', args.num_classes)
    noise_schedule = ckpt_args.get('noise_schedule', args.noise_schedule)
    diffusion_steps = ckpt_args.get('diffusion_steps', args.diffusion_steps)

    num_tokens = dit_ckpt.get('num_tokens', 512)
    latent_dim = dit_ckpt.get('latent_dim', 32)
    print(f"Latent shape: ({num_tokens}, {latent_dim})")

    latent_mean = dit_ckpt.get('latent_mean', None)
    latent_std = dit_ckpt.get('latent_std', None)
    if latent_mean is None or latent_std is None:
        stats_path = args.latent_stats
        if stats_path is None:
            stats_path = os.path.join(os.path.dirname(args.dit_ckpt), '..', '..', 'latent_stats.pt')
            stats_path = os.path.normpath(stats_path)
        if os.path.isfile(stats_path):
            print(f"Loading latent stats from {stats_path}")
            stats = torch.load(stats_path, map_location='cpu')
            latent_mean = stats['mean']
            latent_std = stats['std']
        else:
            latent_mean = torch.zeros(latent_dim)
            latent_std = torch.ones(latent_dim)
    latent_mean = latent_mean.to(device)
    latent_std = latent_std.to(device).clamp(min=1e-6)

    model = DiT1D_models[model_name](
        num_tokens=num_tokens,
        in_channels=latent_dim,
        learn_sigma=learn_sigma,
        num_classes=num_classes,
    )

    if 'ema' in dit_ckpt:
        model.load_state_dict(dit_ckpt['ema'])
        print("Loaded EMA weights")
    else:
        model.load_state_dict(dit_ckpt['model'])
        print("Loaded model weights (no EMA found)")

    model = model.to(device).eval()
    print(f"DiT: {model_name}, params: {sum(p.numel() for p in model.parameters()):,}")

    diffusion = create_diffusion(
        str(args.sampling_steps),
        diffusion_steps=diffusion_steps,
        noise_schedule=noise_schedule,
        learn_sigma=learn_sigma,
        vae_1d=True,
    )

    return model, diffusion, num_tokens, latent_dim, latent_mean, latent_std, num_classes


@torch.no_grad()
def diffusion_sample_batch(
    model,
    diffusion,
    ae,
    device,
    ptdtype,
    n,
    num_tokens,
    latent_dim,
    latent_mean,
    latent_std,
    num_classes,
    cfg_scale,
    sample_method,
    class_labels=None,
    class_label_fixed=None,
    show_progress=False,
):
    """Run diffusion sampling and decode to videos in [0, 1], shape (n, 3, T, H, W)."""
    using_cfg = cfg_scale > 1.0 and num_classes > 0
    z = torch.randn(n, num_tokens, latent_dim, device=device)

    if using_cfg:
        z = torch.cat([z, z], dim=0)
        if class_labels is not None:
            y_cond = class_labels.to(device=device, dtype=torch.long)
        elif class_label_fixed is not None:
            y_cond = torch.full((n,), class_label_fixed, dtype=torch.long, device=device)
        else:
            y_cond = torch.randint(0, num_classes, (n,), device=device)
        y_null = torch.full((n,), num_classes, dtype=torch.long, device=device)
        y = torch.cat([y_cond, y_null], dim=0)
        model_kwargs = {'y': y, 'cfg_scale': cfg_scale}
        sample_fn = model.forward_with_cfg
    else:
        model_kwargs = {}
        if num_classes > 0:
            if class_labels is not None:
                model_kwargs['y'] = class_labels.to(device=device, dtype=torch.long)
            elif class_label_fixed is not None:
                model_kwargs['y'] = torch.full(
                    (n,), class_label_fixed, dtype=torch.long, device=device
                )
            else:
                model_kwargs['y'] = torch.randint(0, num_classes, (n,), device=device)
        sample_fn = model.forward

    with torch.cuda.amp.autocast(dtype=ptdtype):
        if sample_method == 'ddim':
            samples = diffusion.ddim_sample_loop(
                sample_fn, z.shape, z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=show_progress,
                device=device,
            )
        else:
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=show_progress,
                device=device,
            )

    if using_cfg:
        samples, _ = samples.chunk(2, dim=0)

    samples = samples.float() * latent_std + latent_mean

    with torch.cuda.amp.autocast(dtype=ptdtype):
        videos = ae.decode(samples)

    return torch.clamp(videos, 0.0, 1.0)


@torch.no_grad()
def generate_unconditional(
    args,
    model,
    diffusion,
    ae,
    device,
    ptdtype,
    num_tokens,
    latent_dim,
    latent_mean,
    latent_std,
    num_classes,
):
    using_cfg = args.cfg_scale > 1.0 and num_classes > 0
    sample_method = args.sample_method

    print(
        f"Generating {args.num_samples} samples with {args.sampling_steps} steps "
        f"({sample_method}), CFG={'%.1f' % args.cfg_scale if using_cfg else 'off'}"
    )

    batch_size = min(args.batch_size, args.num_samples)
    all_videos = []
    num_generated = 0

    while num_generated < args.num_samples:
        n = min(batch_size, args.num_samples - num_generated)
        videos = diffusion_sample_batch(
            model, diffusion, ae, device, ptdtype,
            n, num_tokens, latent_dim, latent_mean, latent_std,
            num_classes, args.cfg_scale, sample_method,
            class_label_fixed=args.class_label,
            show_progress=(num_generated == 0),
        )
        all_videos.append(videos.cpu())
        num_generated += n
        print(f"  Generated {num_generated}/{args.num_samples}")

    all_videos = torch.cat(all_videos, dim=0)[:args.num_samples]

    for i in range(all_videos.shape[0]):
        video = all_videos[i]
        video_np = (video * 255).add_(0.5).clamp_(0, 255).to(torch.uint8)
        video_np = video_np.permute(1, 2, 3, 0).numpy()
        video_path = os.path.join(args.output_dir, f'video_{i:04d}.mp4')
        imageio.mimwrite(video_path, video_np, fps=8, quality=9)

    first_frames = all_videos[:, :, 0, :, :]
    from torchvision.utils import save_image
    nrow = int(np.ceil(np.sqrt(args.num_samples)))
    save_image(
        first_frames,
        os.path.join(args.output_dir, 'grid_first_frames.png'),
        nrow=nrow,
    )

    print(f"Done! Videos saved to {args.output_dir}")


@torch.no_grad()
def eval_fvd(
    args,
    model,
    diffusion,
    ae,
    device,
    ptdtype,
    num_tokens,
    latent_dim,
    latent_mean,
    latent_std,
    num_classes,
):
    print(f"FVD evaluation on {args.csv_file} ({args.data_path})")
    print(f"  num_samples={args.num_samples}, cfg_scale={args.cfg_scale}")

    dataset = VideoDataset(
        root_path=args.data_path,
        split='train',
        frame_num=args.frame_num,
        csv_file=args.csv_file,
        crop_size=args.image_size,
        scale=1.0,
        aspect_ratio=1.0,
        rand_flip='no',
        frame_rate='native',
        use_all_frames=True,
        rand_augment='no',
        cls_vid_num='-1_-1',
        pre_load=False,
    )

    dataset_idx_seq = random.Random(args.dataset_split_seed).sample(
        range(len(dataset)), min(args.num_samples, len(dataset))
    )
    dataset = Subset(dataset, dataset_idx_seq)
    num_eval = len(dataset)

    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, num_eval),
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    fvd_calculator = FVDCalculator(device=device)
    sample_i3d_feats = None
    orig_i3d_feats = None

    gt_fvd_stats_path = os.path.join(args.output_dir, 'gt_fvd_stats.pkl')
    generated_fvd_stats_path = os.path.join(args.output_dir, 'generated_fvd_stats.pkl')

    if not args.stats_only:
        sample_video_dir = os.path.join(args.output_dir, 'sampled_videos')
        os.makedirs(sample_video_dir, exist_ok=True)

    num_processed = 0
    pbar = tqdm(loader, desc='FVD eval')

    for data in pbar:
        orig_video = data['gt'].clamp(0., 1.).to(device)
        labels = data['label'].to(device)
        n = orig_video.shape[0]

        videos = diffusion_sample_batch(
            model, diffusion, ae, device, ptdtype,
            n, num_tokens, latent_dim, latent_mean, latent_std,
            num_classes, args.cfg_scale, args.sample_method,
            class_labels=labels,
            show_progress=False,
        )

        if videos.shape[2] >= 10:
            # I3D expects float32 contiguous tensors in [0, 1]
            sample_i3d_feats = fvd_calculator.get_feature_stats_for_batch(
                videos.float().contiguous(), sample_i3d_feats
            )
            orig_i3d_feats = fvd_calculator.get_feature_stats_for_batch(
                orig_video.float().contiguous(), orig_i3d_feats
            )

        if not args.stats_only:
            videos_cpu = videos.cpu()
            for j in range(n):
                idx = num_processed + j
                video_np = (videos_cpu[j] * 255).add_(0.5).clamp_(0, 255).to(torch.uint8)
                video_np = video_np.permute(1, 2, 3, 0).numpy()
                video_path = os.path.join(sample_video_dir, f'{idx:05d}.mp4')
                imageio.mimwrite(video_path, video_np, fps=8, quality=9)

        num_processed += n
        pbar.set_postfix(processed=num_processed)

    if sample_i3d_feats is None or orig_i3d_feats is None:
        raise RuntimeError(
            f'Not enough frames for FVD (need >= 10, got T={videos.shape[2] if num_processed else "?"})'
        )

    sample_i3d_feats.save(generated_fvd_stats_path)
    orig_i3d_feats.save(gt_fvd_stats_path)
    print(f'Saved generated FVD stats to {generated_fvd_stats_path}')
    print(f'Saved gt FVD stats to {gt_fvd_stats_path}')

    fvd = fvd_calculator.calculate_fvd(sample_i3d_feats, orig_i3d_feats)
    if isinstance(fvd, torch.Tensor):
        fvd = fvd.item()

    print(f'==> FVD = {fvd:.4f}  (n={sample_i3d_feats.num_items})')
    return fvd


@torch.no_grad()
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ptdtype = torch.bfloat16 if args.bf16 else torch.float32

    os.makedirs(args.output_dir, exist_ok=True)

    model, diffusion, num_tokens, latent_dim, latent_mean, latent_std, num_classes = (
        load_dit_and_diffusion(args, device)
    )

    print(f"Loading autoencoder: {args.ae_ckpt}")
    ae = load_autoencoder(args.ae_ckpt, device, model_name=args.ae_model)

    if args.eval_fvd:
        eval_fvd(
            args, model, diffusion, ae, device, ptdtype,
            num_tokens, latent_dim, latent_mean, latent_std, num_classes,
        )
    else:
        generate_unconditional(
            args, model, diffusion, ae, device, ptdtype,
            num_tokens, latent_dim, latent_mean, latent_std, num_classes,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate videos from trained latent diffusion model")

    parser.add_argument("--dit-ckpt", type=str, required=True,
                        help="Path to DiT checkpoint (checkpoints/XXXXXXX.pt, not latent_stats.pt)")
    parser.add_argument("--latent-stats", type=str, default=None,
                        help="Optional latent_stats.pt if not embedded in dit-ckpt")
    parser.add_argument("--ae-ckpt", type=str, required=True, help="Path to AE checkpoint")
    parser.add_argument("--ae-model", type=str, default=None,
                        help="Registered AE model name (auto-detected from checkpoint if omitted)")
    parser.add_argument("--output-dir", type=str, default="generated_videos/")

    parser.add_argument("--model", type=str, default="DiT1D-L")
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--learn-sigma", action="store_true", default=False)
    parser.add_argument("--noise-schedule", type=str, default="cosine")
    parser.add_argument("--diffusion-steps", type=int, default=1000)

    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int, default=250)
    parser.add_argument("--cfg-scale", type=float, default=1.0,
                        help="CFG scale. 1.0 = no CFG. >1.0 enables CFG.")
    parser.add_argument("--class-label", type=int, default=None,
                        help="Specify class label. None = random.")
    parser.add_argument("--sample-method", type=str, default="ddpm",
                        choices=["ddpm", "ddim"], help="Sampling method.")
    parser.add_argument("--bf16", action="store_true", default=True)

    # FVD evaluation (reference: sample.py)
    parser.add_argument("--eval-fvd", action="store_true",
                        help="Evaluate FVD against dataset ground-truth videos.")
    parser.add_argument("--data-path", type=str, default="data/metadata",
                        help="Dataset root path for FVD evaluation.")
    parser.add_argument("--csv-file", type=str, default="ucf101_train.csv",
                        help="Dataset CSV for FVD evaluation.")
    parser.add_argument("--frame-num", type=int, default=16,
                        help="Number of frames per video (match training).")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Spatial resolution (match training).")
    parser.add_argument("--dataset-split-seed", type=int, default=42,
                        help="Seed for random subset of dataset indices.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers for FVD evaluation.")
    parser.add_argument("--stats-only", action="store_true",
                        help="Only compute FVD stats, do not save sampled mp4 files.")

    args = parser.parse_args()
    main(args)
