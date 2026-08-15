"""Load frozen autoencoder checkpoints for latent diffusion training/sampling."""

import os

import torch

import models as _models_pkg  # noqa: F401 — trigger model registration
from models.models import models as AE_MODEL_REGISTRY


def _load_checkpoint(ckpt_path):
    try:
        return torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location='cpu', weights_only=False)


def _to_plain_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, '__dict__'):
        return dict(obj)
    return {}


def _get_nested(cfg, *keys, default=None):
    cur = cfg
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)
    return cur if cur is not None else default


def detect_ae_model_name(ckpt):
    """Infer registered AE model name from a LARP training checkpoint."""
    name = _get_nested(ckpt, 'model', 'name')
    if name in AE_MODEL_REGISTRY:
        return name

    name = _get_nested(ckpt, 'cfg', 'model', 'name')
    if name in AE_MODEL_REGISTRY:
        return name

    args = _get_nested(ckpt, 'model', 'args', default={})
    args = _to_plain_dict(args)
    if args.get('use_videomae_loss') or args.get('videomae_encoder_ckpt'):
        for candidate in (
            'autoencoder_videomae_ae_c32_repa_huge',
            'autoencoder_videomae_ae_c32_repa',
            'autoencoder_videomae_vq_repa_huge',
            'autoencoder_videomae_vq_repa',
        ):
            if candidate in AE_MODEL_REGISTRY:
                return candidate
    if args.get('use_vjepa_loss') or args.get('vjepa2_encoder_ckpt'):
        for candidate in (
            'autoencoder_vfm_ae_c32_repa_huge',
            'autoencoder_vfm_ae_c32_repa_vith',
            'autoencoder_vfm_ae_c32_repa',
        ):
            if candidate in AE_MODEL_REGISTRY:
                return candidate

    return None


def _count_non_teacher_missing(msg):
    return [
        k for k in msg.missing_keys
        if 'teacher_model' not in k and 'feature_aligner' not in k
    ]


def _try_from_checkpoint(cls, ckpt):
    if not hasattr(cls, 'from_checkpoint'):
        return None, None
    ae = cls.from_checkpoint(ckpt)
    return ae, None


def _try_legacy_load(cls, ckpt):
    if 'model' in ckpt and 'sd' in ckpt['model']:
        kwargs = _to_plain_dict(ckpt['model'].get('args', {}))
        ae = cls(**kwargs)
        msg = ae.load_state_dict(ckpt['model']['sd'], strict=False)
        return ae, msg
    if 'state_dict' in ckpt:
        ae = cls()
        state_dict = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}
        msg = ae.load_state_dict(state_dict, strict=False)
        return ae, msg
    ae = cls()
    msg = ae.load_state_dict(ckpt, strict=False)
    return ae, msg


def load_autoencoder(ckpt_path, device, model_name=None):
    """
    Load a frozen autoencoder from a LARP checkpoint.

    Model selection priority:
      1. explicit model_name argument / --ae-model CLI flag
      2. ckpt['model']['name'] or ckpt['cfg']['model']['name']
      3. heuristic from checkpoint args (videomae vs vjepa)
      4. fallback trial: V-JEPA variants first, then VideoMAE variants
    """
    assert os.path.exists(ckpt_path), f"AE checkpoint not found: {ckpt_path}"
    ckpt = _load_checkpoint(ckpt_path)

    candidates = []
    if model_name is not None:
        candidates.append(model_name)
    detected = detect_ae_model_name(ckpt)
    if detected is not None and detected not in candidates:
        candidates.append(detected)

    for name in (
        'autoencoder_vfm_ae_c32_repa',
        'autoencoder_vfm_ae_c32_repa_vith',
        'autoencoder_vfm_ae_c32_repa_huge',
        'autoencoder_videomae_ae_c32_repa',
        'autoencoder_videomae_ae_c32_repa_huge',
    ):
        if name not in candidates and name in AE_MODEL_REGISTRY:
            candidates.append(name)

    last_error = None
    for name in candidates:
        if name not in AE_MODEL_REGISTRY:
            continue
        cls = AE_MODEL_REGISTRY[name]
        try:
            ae, msg = _try_from_checkpoint(cls, ckpt)
            if ae is None:
                ae, msg = _try_legacy_load(cls, ckpt)
            non_teacher_missing = _count_non_teacher_missing(msg) if msg is not None else []
            if non_teacher_missing:
                print(f"[AE] {name}: skipped, missing non-teacher keys: {non_teacher_missing[:5]}"
                      f"{'...' if len(non_teacher_missing) > 5 else ''}")
                continue
            if msg is not None:
                print(f"[AE] Loaded {name}: missing={len(msg.missing_keys)}, "
                      f"unexpected={len(msg.unexpected_keys)}")
            else:
                print(f"[AE] Loaded {name} via from_checkpoint")
            ae = ae.to(device).eval()
            for p in ae.parameters():
                p.requires_grad = False
            return ae
        except Exception as exc:
            last_error = exc
            print(f"[AE] Failed to load as {name}: {exc}")

    raise RuntimeError(
        f"Could not load autoencoder from {ckpt_path}. "
        f"Tried: {candidates}. Last error: {last_error}"
    )


def get_ae_latent_info(ae):
    """Extract (num_tokens, latent_dim) from a loaded autoencoder."""
    num_tokens = getattr(ae, 'tokenizer_encoder', None)
    if num_tokens is not None and hasattr(ae.tokenizer_encoder, 'out_tokens'):
        num_tokens = ae.tokenizer_encoder.out_tokens
    else:
        num_tokens = getattr(ae, 'num_latent_tokens', 512)
    latent_dim = getattr(ae, 'latent_dim', 32)
    return num_tokens, latent_dim
