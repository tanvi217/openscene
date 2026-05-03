"""Train a lightweight feature adapter on frozen OpenScene fused features.

Loss: J(theta, psi) = L_sup + lambda2 * L_ent

  L_sup -- cross-entropy on the labeled 5% of points per scene
  L_ent -- masked entropy on unlabeled points (H2 regulariser by default:
           only points with H(y_i) < tau; use entropy_mode inverted_mask for H > tau)

The adapter is a 2-layer residual MLP that transforms 768-dim OpenSeg/CLIP
features before a linear classifier head.  Frozen fused features are never
updated.

Usage (from the openscene/ root directory):
  # Supervised only (ablation Run 1):
  python run/train_adapter.py --config config/matterport/adapter_sup_only.yaml

  # With confidence-masked entropy (Run 2, H2 method):
  python run/train_adapter.py --config config/matterport/adapter_with_entropy.yaml
"""
import os
import random
import logging
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data
from tensorboardX import SummaryWriter
from tqdm import tqdm

from util import config
from util.util import AverageMeter, save_checkpoint
from dataset.feature_loader import FusedFeatureLoader, collation_fn_adapter
from models.adapter import (
    FeatureAdapter,
    confidence_masked_entropy,
    inverted_masked_entropy,
    vanilla_entropy_minimization,
    predictive_entropy,
    soft_weighted_entropy,
    pseudo_label_loss,
    temperature_sharpening_loss,
)


best_train_loss = float('inf')


def get_parser():
    parser = argparse.ArgumentParser(description='H2 Adapter Training')
    parser.add_argument('--config', type=str,
                        default='config/matterport/adapter_with_entropy.yaml',
                        help='path to yaml config file')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='optional key=value overrides, e.g. lambda2 0.1 entropy_unmasked True')
    args_in = parser.parse_args()
    assert args_in.config is not None
    cfg = config.load_cfg_from_cfg_file(args_in.config)
    if args_in.opts:
        cfg = config.merge_cfg_from_list(cfg, args_in.opts)
    os.makedirs(cfg.save_path, exist_ok=True)
    os.makedirs(os.path.join(cfg.save_path, 'model'), exist_ok=True)
    return cfg


def get_logger():
    logger_name = 'adapter-logger'
    log = logging.getLogger(logger_name)
    log.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    fmt = '[%(asctime)s %(filename)s line %(lineno)d] %(message)s'
    handler.setFormatter(logging.Formatter(fmt))
    log.addHandler(handler)
    return log


def main():
    global args, logger, writer, best_train_loss

    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    logger.info(args)

    # Model + optimizer
    adapter = FeatureAdapter(
        feat_dim=args.feat_dim, num_classes=args.classes
    ).cuda()

    optimizer = torch.optim.Adam(
        adapter.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    if hasattr(args, 'resume') and args.resume and os.path.isfile(str(args.resume)):
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        ckpt = torch.load(args.resume, map_location='cuda')
        args.start_epoch = ckpt['epoch']
        adapter.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        best_train_loss = ckpt.get('best_train_loss', float('inf'))
        logger.info("=> resumed from epoch {}".format(args.start_epoch))

    # Data loader
    use_shm = getattr(args, 'use_shm', False)
    use_aug = getattr(args, 'aug', False)

    train_data = FusedFeatureLoader(
        datapath_prefix=args.data_root,
        datapath_prefix_feat=args.data_root_2d_fused_feature,
        voxel_size=args.voxel_size,
        split='train',
        aug=use_aug,
        memcache_init=use_shm,
        loop=getattr(args, 'loop', 1),
        input_color=args.input_color,
        return_inds_reconstruct=True,   # needed to detect labeled voxels
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1,           # must be 1: labeled indices are per-scene
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collation_fn_adapter,
    )

    model_dir = os.path.join(args.save_path, 'model')

    # Training loop (model_best = lowest epoch-averaged training loss)
    for epoch in range(args.start_epoch, args.epochs):
        metrics = train_one_epoch(train_loader, adapter, optimizer, epoch)
        epoch_log = epoch + 1

        writer.add_scalar('loss/total', metrics['loss'],     epoch_log)
        writer.add_scalar('loss/L_sup', metrics['l_sup'],    epoch_log)
        writer.add_scalar('loss/L_ent', metrics['l_ent'],    epoch_log)
        writer.add_scalar('frac_confident', metrics['frac_conf'], epoch_log)

        scheduler.step()

        is_best = metrics['loss'] < best_train_loss
        if is_best:
            best_train_loss = metrics['loss']

        state = {
            'epoch': epoch_log,
            'state_dict': adapter.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_train_loss': best_train_loss,
        }
        save_checkpoint(
            state,
            is_best=is_best,
            sav_path=model_dir,
            filename='model_last.pth.tar',
        )
        if is_best:
            logger.info('=> new best train loss {:.4f}  (saved model_best.pth.tar)'
                        .format(best_train_loss))

        if epoch_log % args.save_freq == 0:
            save_checkpoint(
                state,
                is_best=False,
                sav_path=model_dir,
                filename='checkpoint_epoch_{}.pth.tar'.format(epoch_log),
            )
            logger.info('Checkpoint saved  epoch={}'.format(epoch_log))

    save_checkpoint(
        {
            'epoch': args.epochs,
            'state_dict': adapter.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_train_loss': best_train_loss,
        },
        is_best=False,
        sav_path=model_dir,
        filename='model_final.pth.tar',
    )
    writer.close()
    logger.info('==> Training done.')


def train_one_epoch(train_loader, adapter, optimizer, epoch):
    """One full pass over the training set."""

    adapter.train()
    loss_meter = AverageMeter()
    sup_meter  = AverageMeter()
    ent_meter  = AverageMeter()
    conf_meter = AverageMeter()

    lambda2     = getattr(args, 'lambda2', 0.0)
    entropy_tau = getattr(args, 'entropy_tau', None)
    entropy_unmasked = getattr(args, 'entropy_unmasked', False)
    entropy_mode = getattr(args, 'entropy_mode', 'hard_mask')
    labeled_frac = float(getattr(args, 'labeled_frac', 0.05))
    soft_temp = float(getattr(args, 'soft_entropy_temperature', 0.5))
    pseudo_thr = float(getattr(args, 'pseudo_label_threshold', 0.9))
    sharp_temp = float(getattr(args, 'sharpening_temperature', 0.5))
    warmup_epochs = int(getattr(args, 'entropy_warmup_epochs', 0))
    if warmup_epochs > 0:
        lambda2_effective = lambda2 * min(1.0, float(epoch + 1) / float(warmup_epochs))
    else:
        lambda2_effective = lambda2

    for i, (coords, feat, labels, feat_3d, mask, scene_names, inds_list) in enumerate(
            tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}')):

        scene_name      = scene_names[0]
        inds_reconstruct = inds_list[0].numpy()  # (N_orig,): orig point idx -> voxel idx

        # Labeled indices for this scene (original point indices)
        labeled_path = os.path.join(args.labeled_indices_dir, f'{scene_name}.npy')
        labeled_arr = None
        if os.path.exists(labeled_path):
            labeled_arr = np.load(labeled_path)  # original-point indices of labeled points

        # Map labeled original points to voxel indices
        N_vox = coords.shape[0]
        N_orig = len(inds_reconstruct)
        is_labeled_vox = np.zeros(N_vox, dtype=bool)
        if labeled_arr is not None:
            orig_labeled_mask = np.zeros(N_orig, dtype=bool)
            valid_labeled = labeled_arr[labeled_arr < N_orig]  # safety clip
            orig_labeled_mask[valid_labeled] = True
            labeled_vox_idxs = inds_reconstruct[orig_labeled_mask]
            valid_vox = labeled_vox_idxs[labeled_vox_idxs < N_vox]  # safety clip
            is_labeled_vox[valid_vox] = True

        # Align to visible points (those with fused features)
        # mask: (N_vox,) bool which voxelized points have fused 3D features
        # feat_3d: (N_vis, 768) where N_vis = mask.sum()
        # labels: (N_vox,)  voxelized GT labels
        mask_bool = mask.bool()
        labels_vis      = labels[mask_bool]                         # (N_vis,)
        is_labeled_vis  = torch.from_numpy(is_labeled_vox)[mask_bool]  # (N_vis,)
        # Fallback when no pre-generated labeled split exists for the scene:
        # stable per-scene class-balanced sampling on visible valid points.
        if labeled_arr is None:
            valid_vis = (labels_vis != args.ignore_label)
            vis_idx = valid_vis.nonzero(as_tuple=True)[0]
            if vis_idx.numel() > 0:
                content_seed = int(labels_vis[valid_vis].sum().item() % (2**31))
                content_seed += int(labels_vis.shape[0])
                gen = torch.Generator()
                base_seed = int(getattr(args, 'manual_seed', 0) or 0)
                gen.manual_seed(base_seed + content_seed)
                sampled = []
                valid_labels = labels_vis[valid_vis]
                for cls in valid_labels.unique():
                    cls_vis_idx = vis_idx[(valid_labels == cls)]
                    k = max(1, int(len(cls_vis_idx) * labeled_frac))
                    order = torch.randperm(len(cls_vis_idx), generator=gen)
                    sampled.append(cls_vis_idx[order[:k]])
                if sampled:
                    sampled_idx = torch.cat(sampled, dim=0)
                    is_labeled_vis = torch.zeros_like(valid_vis)
                    is_labeled_vis[sampled_idx] = True
            else:
                is_labeled_vis = torch.zeros_like(valid_vis)
            logger.warning(f'No labeled split for {scene_name}, using stable class-balanced fallback.')

        feat_3d_cuda = feat_3d.float().cuda()
        logits = adapter(feat_3d_cuda)  # (N_vis, C)

        if is_labeled_vis.sum() > 0:
            L_sup = F.cross_entropy(
                logits[is_labeled_vis],
                labels_vis[is_labeled_vis].cuda(),
                ignore_index=args.ignore_label,
            )
        else:
            L_sup = torch.tensor(0.0, device='cuda', requires_grad=True)

        # L_ent: masked entropy (H2) or vanilla entropy (TENT / Run 3)
        is_unlabeled_vis = ~is_labeled_vis
        frac_conf = 0.0
        if lambda2_effective > 0.0 and is_unlabeled_vis.sum() > 0:
            logits_u = logits[is_unlabeled_vis]
            if entropy_mode == 'pseudo_label':
                L_ent, frac_conf = pseudo_label_loss(logits_u, threshold=pseudo_thr)
            elif entropy_mode == 'temp_sharpen':
                L_ent = temperature_sharpening_loss(logits_u, sharpening_temp=sharp_temp)
                with torch.no_grad():
                    H_u = predictive_entropy(logits_u)
                    frac_conf = (
                        (H_u < float(entropy_tau)).float().mean().item()
                        if entropy_tau is not None else 0.0
                    )
            elif entropy_mode == 'soft_weight':
                L_ent = soft_weighted_entropy(logits_u, temperature=soft_temp)
                with torch.no_grad():
                    H_u = predictive_entropy(logits_u)
                    frac_conf = (
                        torch.exp(-H_u / soft_temp).mean().item()
                    )
            elif entropy_unmasked or entropy_mode == 'unmasked':
                L_ent = vanilla_entropy_minimization(logits_u)
                with torch.no_grad():
                    H_u = predictive_entropy(logits_u)
                    frac_conf = (
                        (H_u < float(entropy_tau)).float().mean().item()
                        if entropy_tau is not None else 0.0
                    )
            elif entropy_mode == 'inverted_mask' and entropy_tau is not None:
                L_ent = inverted_masked_entropy(logits_u, tau=float(entropy_tau))
                with torch.no_grad():
                    H_u = predictive_entropy(logits_u)
                    frac_conf = (H_u > float(entropy_tau)).float().mean().item()
            elif entropy_tau is not None:
                L_ent = confidence_masked_entropy(logits_u, tau=float(entropy_tau))
                with torch.no_grad():
                    H_u = predictive_entropy(logits_u)
                    frac_conf = (H_u < float(entropy_tau)).float().mean().item()
            else:
                L_ent = torch.tensor(0.0, device='cuda')
        else:
            L_ent = torch.tensor(0.0, device='cuda')

        loss = L_sup + lambda2_effective * L_ent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item())
        sup_meter.update(L_sup.item())
        ent_meter.update(L_ent.item())
        conf_meter.update(frac_conf)

        if (i + 1) % args.print_freq == 0:
            logger.info(
                'Epoch [{}/{}][{}/{}]  loss {:.4f}  L_sup {:.4f}  '
                'L_ent {:.4f}  frac_conf {:.3f}'.format(
                    epoch + 1, args.epochs, i + 1, len(train_loader),
                    loss_meter.avg, sup_meter.avg, ent_meter.avg, conf_meter.avg,
                )
            )

    return {
        'loss':      loss_meter.avg,
        'l_sup':     sup_meter.avg,
        'l_ent':     ent_meter.avg,
        'frac_conf': conf_meter.avg,
    }


if __name__ == '__main__':
    main()
