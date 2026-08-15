"""
Latent Diffusion training script for AutoEncoderC32Repa.
Trains a DiT model on the latent space of the frozen autoencoder.

Usage:
    torchrun --nproc_per_node=8 latent_diffusion/train.py \
        --ae-ckpt /path/to/ae_checkpoint.pth \
        --data-path data/metadata \
        --csv-file kinetics_dataset.csv \
        --model DiT1D-L \
        --global-batch-size 256 \
        --mixed-precision bf16
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import sys


from modelling.diffusion import create_diffusion
from latent_diffusion.dit import DiT1D_models
from latent_diffusion.ae_loader import load_autoencoder, get_ae_latent_info
from datasets.video_dataset import VideoDataset


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    if dist.get_rank() == 0 and logging_dir is not None:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def manage_checkpoints(save_dir, keep_last_n=10):
    checkpoints = [f for f in os.listdir(save_dir) if f.endswith('.pt')]
    checkpoints.sort(key=lambda f: int(f.split('.')[0]))
    if len(checkpoints) > keep_last_n + 1:
        for ckpt_file in checkpoints[:-keep_last_n - 1]:
            ckpt_path = os.path.join(save_dir, ckpt_file)
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)


@torch.no_grad()
def compute_latent_stats(ae, loader, device, num_tokens, latent_dim,
                         max_batches=200, ptdtype=torch.float32):
    """Compute mean and std of latent codes over a subset of the dataset."""
    all_latents = []
    count = 0
    for batch in loader:
        if count >= max_batches:
            break
        if isinstance(batch, dict):
            x = batch['gt']
        elif isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        x = x.to(device)
        with torch.cuda.amp.autocast(dtype=ptdtype):
            z = ae.encode(x)  # (B, num_tokens, latent_dim)
        all_latents.append(z.float())
        count += 1

    all_latents = torch.cat(all_latents, dim=0)
    latent_mean = all_latents.mean(dim=(0, 1))    # (latent_dim,)
    latent_std = all_latents.std(dim=(0, 1))       # (latent_dim,)

    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(latent_mean, op=dist.ReduceOp.SUM)
        latent_mean /= dist.get_world_size()
        dist.all_reduce(latent_std, op=dist.ReduceOp.SUM)
        latent_std /= dist.get_world_size()

    return latent_mean, latent_std


def main(args):
    assert torch.cuda.is_available(), "Training requires at least one GPU."

    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    experiment_dir = None
    checkpoint_dir = None
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        model_string_name = args.model.replace("/", "-")

        if args.resume is not None:
            experiment_dir = '/'.join(args.resume.split('/')[:-2])
        elif args.exp_index is not None:
            experiment_dir = f"{args.results_dir}/{int(args.exp_index):03d}-{model_string_name}-{args.noise_schedule}"
        else:
            # Auto-detect: find existing experiment dir matching model+schedule
            exp_pattern = f"*-{model_string_name}-{args.noise_schedule}"
            existing = sorted(glob(f"{args.results_dir}/{exp_pattern}"))
            if existing:
                candidate = existing[-1]
                candidate_ckpts = sorted(glob(os.path.join(candidate, 'checkpoints', '*.pt')))
                if candidate_ckpts:
                    experiment_dir = candidate
                    print(f"Auto-resume: reusing existing experiment dir: {experiment_dir}")
            if experiment_dir is None:
                experiment_index = len(glob(f"{args.results_dir}/*"))
                experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-{args.noise_schedule}"

        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory: {experiment_dir}")
    else:
        logger = create_logger(None)

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    # Load frozen autoencoder
    logger.info(f"Loading autoencoder from: {args.ae_ckpt}")
    ae = load_autoencoder(args.ae_ckpt, device, model_name=args.ae_model)
    logger.info(f"AE parameters: {sum(p.numel() for p in ae.parameters()):,} (frozen)")

    # Read latent shape from AE instead of hardcoding
    num_tokens, latent_dim = get_ae_latent_info(ae)
    logger.info(f"AE latent shape: ({num_tokens}, {latent_dim})")

    # Create DiT model
    model = DiT1D_models[args.model](
        num_tokens=num_tokens,
        in_channels=latent_dim,
        learn_sigma=args.learn_sigma,
        num_classes=args.num_classes,
        class_dropout_prob=args.class_dropout_prob,
    )
    logger.info(f"DiT model: {args.model}, parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Class-conditional: num_classes={args.num_classes}, "
                f"class_dropout_prob={args.class_dropout_prob}")

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[device])

    # Diffusion (vae_1d=True for 1D latent token sequences)
    diffusion = create_diffusion(
        timestep_respacing="",
        diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        learn_sigma=args.learn_sigma,
        vae_1d=True,
    )
    gen_diffusion = create_diffusion(
        str(args.sampling_steps),
        diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        learn_sigma=args.learn_sigma,
        vae_1d=True,
    )

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == 'fp16'))

    # Dataset
    dataset = VideoDataset(
        root_path=args.data_path,
        split='train',
        frame_num=args.frame_num,
        csv_file=args.csv_file,
        crop_size=args.image_size,
        scale=1.0,
        aspect_ratio=1.0,
        rand_flip='yes',
        frame_rate='native',
        use_all_frames=False,
        rand_augment='no',
        cls_vid_num='-1_-1',
        pre_load=False,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset: {len(dataset):,} videos ({args.data_path})")

    # Compute latent normalization statistics
    os.makedirs(args.results_dir, exist_ok=True)
    latent_stats_path = os.path.join(args.results_dir, 'latent_stats.pt')
    if os.path.exists(latent_stats_path):
        stats = torch.load(latent_stats_path, map_location=f'cuda:{device}')
        latent_mean = stats['mean'].to(device)
        latent_std = stats['std'].to(device)
        logger.info(f"Loaded latent stats from {latent_stats_path}")
    else:
        logger.info("Computing latent statistics over training set...")
        latent_mean, latent_std = compute_latent_stats(
            ae, loader, device, num_tokens, latent_dim,
            max_batches=200, ptdtype=ptdtype
        )
        if rank == 0:
            torch.save({'mean': latent_mean.cpu(), 'std': latent_std.cpu()}, latent_stats_path)
        logger.info(f"Latent mean norm: {latent_mean.norm():.4f}, std mean: {latent_std.mean():.4f}")

    latent_mean = latent_mean.to(device)
    latent_std = latent_std.to(device).clamp(min=1e-6)

    logger.info(f"Latent stats: mean range=[{latent_mean.min():.4f}, {latent_mean.max():.4f}], "
                f"std range=[{latent_std.min():.4f}, {latent_std.max():.4f}], "
                f"mean_norm={latent_mean.norm():.4f}, std_mean={latent_std.mean():.4f}")

    # Prepare training
    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    # Resume: auto-detect latest checkpoint if --resume is not specified
    start_epoch = 0
    train_steps = 0
    resume_path = args.resume
    if resume_path is None and rank == 0 and checkpoint_dir is not None and os.path.isdir(checkpoint_dir):
        existing_ckpts = sorted(glob(os.path.join(checkpoint_dir, '*.pt')))
        if existing_ckpts:
            resume_path = existing_ckpts[-1]
            logger.info(f"Auto-resume: found {len(existing_ckpts)} checkpoint(s), "
                        f"resuming from latest: {resume_path}")

    # Broadcast resume path from rank 0 to all ranks
    if dist.get_world_size() > 1:
        resume_info = [resume_path] if rank == 0 else [None]
        dist.broadcast_object_list(resume_info, src=0)
        resume_path = resume_info[0]

    if resume_path is not None:
        logger.info(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        model.module.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        opt.load_state_dict(ckpt['opt'])
        if 'scaler' in ckpt and scaler.is_enabled():
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt.get('epoch', 0)
        train_steps = ckpt.get('train_steps', 0)
        logger.info(f"Resumed at epoch={start_epoch}, train_steps={train_steps}")

    # Wandb
    if rank == 0 and args.wandb:
        import wandb
        wandb_run = wandb.init(project='LARP-DiT', config=vars(args),
                               name=os.path.basename(experiment_dir))

    # Training loop
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        for batch in loader:
            if isinstance(batch, dict):
                x = batch['gt']
                label = batch.get('label', None)
            elif isinstance(batch, (list, tuple)):
                x = batch[0]
                label = batch[1] if len(batch) > 1 else None
            else:
                x = batch
                label = None
            x = x.to(device, non_blocking=True)

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=ptdtype):
                    z = ae.encode(x)

            with torch.cuda.amp.autocast(dtype=ptdtype):
                z = (z.float() - latent_mean) / latent_std

                t = torch.randint(0, diffusion.num_timesteps, (z.shape[0],), device=device)
                model_kwargs = {}
                if args.num_classes > 0 and label is not None:
                    y = label.to(device, non_blocking=True) if isinstance(label, torch.Tensor) else torch.tensor(label, device=device)
                    model_kwargs['y'] = y

                loss_dict = diffusion.training_losses(model, z, t, model_kwargs)
                loss = loss_dict["loss"].mean()

            opt.zero_grad()
            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(opt)
            scaler.update()

            update_ema(ema, model.module)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Loss: {avg_loss:.4f}, Steps/Sec: {steps_per_sec:.2f}")

                if rank == 0 and args.wandb:
                    wandb_run.log({'train_loss': avg_loss, 'steps_per_sec': steps_per_sec}, step=train_steps)

                running_loss = 0
                log_steps = 0
                start_time = time()

            # Checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    ckpt = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(),
                        "args": vars(args),
                        "train_steps": train_steps,
                        "epoch": epoch,
                        "latent_mean": latent_mean.cpu(),
                        "latent_std": latent_std.cpu(),
                        "num_tokens": num_tokens,
                        "latent_dim": latent_dim,
                    }
                    ckpt_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(ckpt, ckpt_path)
                    logger.info(f"Saved checkpoint: {ckpt_path}")
                    manage_checkpoints(checkpoint_dir, keep_last_n=10)
                dist.barrier()

            # Visualization sampling
            if train_steps % args.sample_every == 0 and train_steps > 0:
                model.eval()
                n = min(4, int(args.global_batch_size // dist.get_world_size()))

                # === Part 1: AE reconstruction (verify encoder-decoder works) ===
                recon_frames = None
                if rank == 0:
                    try:
                        recon_batch = next(iter(loader))
                        if isinstance(recon_batch, dict):
                            recon_x = recon_batch['gt'][:n]
                        else:
                            recon_x = recon_batch[0][:n]
                        recon_x = recon_x.to(device)
                        with torch.no_grad(), torch.cuda.amp.autocast(dtype=ptdtype):
                            recon_z = ae.encode(recon_x)
                            recon_vid = ae.decode(recon_z)
                        orig_frames = recon_x[:, :, 0, :, :].float().clamp(0, 1)
                        recon_frames_out = recon_vid[:, :, 0, :, :].float().clamp(0, 1)
                        recon_frames = torch.cat([orig_frames, recon_frames_out], dim=0)
                    except Exception as e:
                        logger.info(f"Reconstruction visualization skipped: {e}")

                # === Part 2: Unconditional sampling (no CFG) ===
                z_noise_nocfg = torch.randn(n, num_tokens, latent_dim, device=device)
                nocfg_kwargs = {}
                if args.num_classes > 0:
                    nocfg_kwargs['y'] = torch.randint(0, args.num_classes, (n,), device=device)

                with torch.cuda.amp.autocast(dtype=ptdtype):
                    nocfg_samples = gen_diffusion.p_sample_loop(
                        ema.forward, z_noise_nocfg.shape, z_noise_nocfg,
                        clip_denoised=False,
                        model_kwargs=nocfg_kwargs,
                        progress=False,
                        device=device,
                    )

                nocfg_samples = nocfg_samples.float() * latent_std + latent_mean
                with torch.cuda.amp.autocast(dtype=ptdtype):
                    nocfg_videos = ae.decode(nocfg_samples)

                # === Part 3: CFG sampling (if enabled) ===
                using_cfg = args.cfg_scale > 1.0 and args.num_classes > 0
                cfg_videos = None
                if using_cfg:
                    z_noise = torch.randn(n, num_tokens, latent_dim, device=device)
                    z_noise = torch.cat([z_noise, z_noise], dim=0)
                    y_cond = torch.randint(0, args.num_classes, (n,), device=device)
                    y_null = torch.full((n,), args.num_classes, dtype=torch.long, device=device)
                    y_cfg = torch.cat([y_cond, y_null], dim=0)
                    sample_kwargs = {'y': y_cfg, 'cfg_scale': args.cfg_scale}

                    with torch.cuda.amp.autocast(dtype=ptdtype):
                        cfg_samples = gen_diffusion.p_sample_loop(
                            ema.forward_with_cfg, z_noise.shape, z_noise,
                            clip_denoised=False,
                            model_kwargs=sample_kwargs,
                            progress=False,
                            device=device,
                        )

                    cfg_samples, _ = cfg_samples.chunk(2, dim=0)
                    cfg_samples = cfg_samples.float() * latent_std + latent_mean
                    with torch.cuda.amp.autocast(dtype=ptdtype):
                        cfg_videos = ae.decode(cfg_samples)

                # === Save all samples ===
                if rank == 0:
                    from torchvision.utils import save_image

                    all_frames = []

                    if recon_frames is not None:
                        all_frames.append(recon_frames)

                    nocfg_frames = torch.clamp(nocfg_videos, 0.0, 1.0)[:, :, 0, :, :]
                    all_frames.append(nocfg_frames)

                    if cfg_videos is not None:
                        cfg_frames = torch.clamp(cfg_videos, 0.0, 1.0)[:, :, 0, :, :]
                        all_frames.append(cfg_frames)

                    grid = torch.cat(all_frames, dim=0)
                    save_path = f"{experiment_dir}/samples_step{train_steps:07d}.png"
                    save_image(grid, save_path, nrow=n, padding=2)
                    logger.info(f"Saved samples: {save_path} "
                                f"(rows: {'recon_orig+recon, ' if recon_frames is not None else ''}"
                                f"no-cfg{', cfg' if cfg_videos is not None else ''})")

                    if args.wandb:
                        import wandb
                        from PIL import Image
                        img = Image.open(save_path)
                        wandb_run.log({'samples': [wandb.Image(img)]}, step=train_steps)

                model.train()
                dist.barrier()

    logger.info("Training complete!")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train latent diffusion on AutoEncoderC32Repa latents")

    parser.add_argument("--data-path", type=str, default="data/metadata")
    parser.add_argument("--csv-file", type=str, default="kinetics_dataset.csv")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--frame-num", type=int, default=16)

    parser.add_argument("--ae-ckpt", type=str, required=True, help="Path to frozen AE checkpoint")
    parser.add_argument("--ae-model", type=str, default=None,
                        help="Registered AE model name (auto-detected from checkpoint if omitted)")

    parser.add_argument("--model", type=str, default="DiT1D-L", choices=list(DiT1D_models.keys()))
    parser.add_argument("--num-classes", type=int, default=0, help="0 = unconditional")
    parser.add_argument("--class-dropout-prob", type=float, default=0.1)
    parser.add_argument("--learn-sigma", action="store_true", default=False)

    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--noise-schedule", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--sampling-steps", type=int, default=250)
    parser.add_argument("--cfg-scale", type=float, default=1.0,
                        help="CFG scale for visualization sampling (1.0 = no CFG)")

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["none", "bf16", "fp16"])

    parser.add_argument("--results-dir", type=str, default="save/latent_diffusion")
    parser.add_argument("--exp-index", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=25000)
    parser.add_argument("--sample-every", type=int, default=10000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--wandb", action="store_true", default=False)

    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()
    main(args)
