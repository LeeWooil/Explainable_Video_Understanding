from __future__ import annotations

import torch


def cos_similarity_cubed_single_concept(clip_feats, target_feats):
    clip_feats = clip_feats.float()
    clip_feats = clip_feats - torch.mean(clip_feats, dim=0, keepdim=True)
    target_feats = target_feats - torch.mean(target_feats, dim=0, keepdim=True)
    clip_feats = clip_feats**3
    target_feats = target_feats**3
    clip_feats = clip_feats / torch.norm(clip_feats, p=2, dim=0, keepdim=True)
    target_feats = target_feats / torch.norm(target_feats, p=2, dim=0, keepdim=True)
    return torch.sum(target_feats * clip_feats, dim=0)


def cos_similarity_cubed_single_firstpower_concept(clip_feats, target_feats):
    clip_feats = clip_feats.float()
    clip_feats = clip_feats - torch.mean(clip_feats, dim=0, keepdim=True)
    target_feats = target_feats - torch.mean(target_feats, dim=0, keepdim=True)
    clip_feats = clip_feats / torch.norm(clip_feats, p=2, dim=0, keepdim=True)
    target_feats = target_feats / torch.norm(target_feats, p=2, dim=0, keepdim=True)
    return torch.sum(target_feats * clip_feats, dim=0)


def cos_similarity_cubed_single_firstpower_sample(clip_feats, target_feats):
    clip_feats = clip_feats.float()
    clip_feats = clip_feats - torch.mean(clip_feats, dim=1, keepdim=True)
    target_feats = target_feats - torch.mean(target_feats, dim=1, keepdim=True)
    clip_feats = clip_feats / torch.norm(clip_feats, p=2, dim=1, keepdim=True)
    target_feats = target_feats / torch.norm(target_feats, p=2, dim=1, keepdim=True)
    return torch.sum(target_feats * clip_feats, dim=1)


def cos_similarity_cubed_single_secondpower(clip_feats, target_feats):
    clip_feats = clip_feats.float()
    clip_feats = clip_feats - torch.mean(clip_feats, dim=0, keepdim=True)
    target_feats = target_feats - torch.mean(target_feats, dim=0, keepdim=True)
    clip_feats = clip_feats**2
    target_feats = target_feats**2
    clip_feats = clip_feats / torch.norm(clip_feats, p=2, dim=0, keepdim=True)
    target_feats = target_feats / torch.norm(target_feats, p=2, dim=0, keepdim=True)
    return torch.sum(target_feats * clip_feats, dim=0)


def cos_similarity_cubed_single_sample(clip_feats, target_feats):
    clip_feats = clip_feats.float()
    clip_feats = clip_feats - torch.mean(clip_feats, dim=1, keepdim=True)
    target_feats = target_feats - torch.mean(target_feats, dim=1, keepdim=True)
    clip_feats = clip_feats**3
    target_feats = target_feats**3
    clip_feats = clip_feats / torch.norm(clip_feats, p=2, dim=1, keepdim=True)
    target_feats = target_feats / torch.norm(target_feats, p=2, dim=1, keepdim=True)
    return torch.sum(target_feats * clip_feats, dim=1)
