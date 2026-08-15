"""
Compute generation FVD (gFVD) from saved generated feature stats and UCF101 real videos.

Usage:
    python compute_gfvd.py \
        --gen_stats output_vfm/vfm_ar_repa_samples/sample_fvd_stats.pkl \
        --gt_stats output_vfm/vfm_ar_repa_samples/gt_fvd_stats.pkl \
        --num_samples 10000
"""
import argparse
import os
import random
import sys

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import datasets
from utils import FeatureStats, FVDCalculator


def build_real_dataset_subset(
    dataset_csv='ucf101_train.csv',
    num_samples=10000,
    dataset_split_seed=42,
    frame_num=16,
    crop_size=256,
):
    dataset_cfg = {
        'name': 'video_dataset',
        'args': {
            'root_path': 'data/metadata',
            'frame_num': frame_num,
            'cls_vid_num': '-1_-1',
            'crop_size': crop_size,
            'csv_file': dataset_csv,
            'frame_rate': 'native',
            'use_all_frames': True,
            'pre_load': False,
        },
    }
    dataset = datasets.make(dataset_cfg)
    assert len(dataset) >= num_samples, (
        f'Dataset has {len(dataset)} clips, but {num_samples} requested'
    )

    idx_seq = random.Random(dataset_split_seed).sample(
        range(len(dataset)), num_samples
    )
    return Subset(dataset, idx_seq)


def get_eval_labels(
    dataset_csv='ucf101_train.csv',
    num_samples=10000,
    dataset_split_seed=42,
    frame_num=16,
    crop_size=256,
):
    """Return LARP-aligned class labels for the eval subset without loading videos."""
    dataset_cfg = {
        'name': 'video_dataset',
        'args': {
            'root_path': 'data/metadata',
            'frame_num': frame_num,
            'cls_vid_num': '-1_-1',
            'crop_size': crop_size,
            'csv_file': dataset_csv,
            'frame_rate': 'native',
            'use_all_frames': True,
            'pre_load': False,
        },
    }
    dataset = datasets.make(dataset_cfg)
    assert len(dataset) >= num_samples, (
        f'Dataset has {len(dataset)} clips, but {num_samples} requested'
    )
    idx_seq = random.Random(dataset_split_seed).sample(
        range(len(dataset)), num_samples
    )
    return [dataset.idx2label[i] for i in idx_seq]


@torch.inference_mode()
def extract_gt_stats(
    dataset_csv='ucf101_train.csv',
    num_samples=10000,
    dataset_split_seed=42,
    frame_num=16,
    crop_size=256,
    batch_size=32,
    num_workers=0,
    gt_stats_path=None,
    force_recompute=False,
    device='cuda',
):
    if gt_stats_path and os.path.exists(gt_stats_path) and not force_recompute:
        cached = FeatureStats.load(gt_stats_path)
        if cached.num_items != num_samples:
            print(f'Cached gt stats has {cached.num_items} samples, '
                  f'but num_samples={num_samples}; will recompute.')
        else:
            print(f'Loading cached real video stats from {gt_stats_path}')
            return cached

    print(f'Extracting real video stats from {dataset_csv} '
          f'({num_samples} clips, seed={dataset_split_seed}) ...')
    dataset = build_real_dataset_subset(
        dataset_csv=dataset_csv,
        num_samples=num_samples,
        dataset_split_seed=dataset_split_seed,
        frame_num=frame_num,
        crop_size=crop_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    fvd_calculator = FVDCalculator(device=device)
    gt_feats = None
    for batch in tqdm(loader, desc='Extracting real video features'):
        gt_feats = fvd_calculator.get_feature_stats_for_batch(batch, gt_feats)

    if gt_stats_path:
        os.makedirs(os.path.dirname(os.path.abspath(gt_stats_path)), exist_ok=True)
        gt_feats.save(gt_stats_path)
        print(f'Saved real video stats to {gt_stats_path}')

    return gt_feats


def compute_gfvd_score(gen_stats, gt_stats, device='cuda'):
    fvd_calculator = FVDCalculator(device=device)
    fvd = fvd_calculator.calculate_fvd(gen_stats, gt_stats)
    if isinstance(fvd, torch.Tensor):
        fvd = fvd.item()
    return fvd


def save_gfvd_result(
    output_dir,
    gen_stats_path,
    gt_stats_path,
    gen_stats,
    gt_stats,
    dataset_csv,
    dataset_split_seed,
    gfvd,
):
    result_path = os.path.join(output_dir, 'gfvd_result.txt')
    with open(result_path, 'w') as f:
        f.write(f'gen_stats={gen_stats_path}\n')
        f.write(f'gt_stats={gt_stats_path}\n')
        f.write(f'num_gen={gen_stats.num_items}\n')
        f.write(f'num_real={gt_stats.num_items}\n')
        f.write(f'dataset_csv={dataset_csv}\n')
        f.write(f'dataset_split_seed={dataset_split_seed}\n')
        f.write(f'gfvd={gfvd:.4f}\n')
    print(f'Saved result to {result_path}')
    return result_path


def compute_and_report_gfvd(
    gen_stats,
    output_dir,
    gen_stats_path,
    gt_stats_path=None,
    dataset_csv='ucf101_train.csv',
    num_samples=10000,
    dataset_split_seed=42,
    frame_num=16,
    crop_size=256,
    batch_size=32,
    num_workers=0,
    force_recompute_gt=False,
    device='cuda',
):
    if gt_stats_path is None:
        gt_stats_path = os.path.join(output_dir, 'gt_fvd_stats.pkl')

    gt_stats = extract_gt_stats(
        dataset_csv=dataset_csv,
        num_samples=num_samples,
        dataset_split_seed=dataset_split_seed,
        frame_num=frame_num,
        crop_size=crop_size,
        batch_size=batch_size,
        num_workers=num_workers,
        gt_stats_path=gt_stats_path,
        force_recompute=force_recompute_gt,
        device=device,
    )

    print(f'Loaded generated stats: {gen_stats.num_items} samples, '
          f'{gen_stats.num_features} features')
    print(f'Loaded real stats: {gt_stats.num_items} samples, '
          f'{gt_stats.num_features} features')

    gfvd = compute_gfvd_score(gen_stats, gt_stats, device=device)

    print(f'\n========== gFVD Result ==========')
    print(f'Generated stats : {gen_stats_path}')
    print(f'Real stats      : {gt_stats_path}')
    print(f'Num samples     : gen={gen_stats.num_items}, real={gt_stats.num_items}')
    print(f'gFVD            : {gfvd:.4f}')
    print(f'=================================')

    save_gfvd_result(
        output_dir=output_dir,
        gen_stats_path=gen_stats_path,
        gt_stats_path=gt_stats_path,
        gen_stats=gen_stats,
        gt_stats=gt_stats,
        dataset_csv=dataset_csv,
        dataset_split_seed=dataset_split_seed,
        gfvd=gfvd,
    )
    return gfvd


def parse_args():
    parser = argparse.ArgumentParser(description='Compute gFVD from generated and real video stats')
    parser.add_argument('--gen_stats', type=str, required=True,
                        help='Path to generated feature stats pkl (e.g. sample_fvd_stats.pkl)')
    parser.add_argument('--gt_stats', type=str, default=None,
                        help='Path to save/load real video feature stats pkl')
    parser.add_argument('--dataset_csv', type=str, default='ucf101_train.csv',
                        help='UCF101 csv for real video reference set')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of real videos to use for FVD reference')
    parser.add_argument('--dataset_split_seed', type=int, default=42,
                        help='Random seed for selecting real video subset (match sample.py)')
    parser.add_argument('--frame_num', type=int, default=16)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers (use 0 with decord after CUDA init)')
    parser.add_argument('--force_recompute_gt', action='store_true',
                        help='Recompute real video stats even if gt_stats exists')
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.gen_stats):
        print(f'ERROR: generated stats not found: {args.gen_stats}')
        sys.exit(1)

    gen_stats = FeatureStats.load(args.gen_stats)
    output_dir = os.path.dirname(os.path.abspath(args.gen_stats))
    compute_and_report_gfvd(
        gen_stats=gen_stats,
        output_dir=output_dir,
        gen_stats_path=args.gen_stats,
        gt_stats_path=args.gt_stats,
        dataset_csv=args.dataset_csv,
        num_samples=args.num_samples,
        dataset_split_seed=args.dataset_split_seed,
        frame_num=args.frame_num,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        force_recompute_gt=args.force_recompute_gt,
    )


if __name__ == '__main__':
    main()
