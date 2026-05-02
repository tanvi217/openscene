"""Evaluate a trained feature adapter and run the H2 diagnostic.

Computes:
  1. Overall mIoU (21 classes) on the test split.
  2. H2 diagnostic: per-quartile point accuracy using baseline confidence
     scores saved by run/evaluate.py --save_confidence.

The adapter replaces the CLIP nearest-neighbour step: instead of computing
  pred = feat_3d @ text_features.T  ->  argmax
we do:
  logits = adapter(feat_3d)  ->  argmax

Usage (from the openscene/ root directory):
  python run/evaluate_adapter.py \\
      --config config/matterport/adapter_with_entropy.yaml \\
      --model_path exp/matterport/adapter_with_entropy/model/model_final.pth.tar \\
      --save_folder exp/matterport/adapter_with_entropy/eval
"""
import os
import logging
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from tqdm import tqdm

from util import config, metric
from dataset.feature_loader import FusedFeatureLoader, collation_fn_eval_all
from models.adapter import FeatureAdapter


def get_parser():
    parser = argparse.ArgumentParser(description='H2 Adapter Evaluation')
    parser.add_argument('--config', type=str,
                        default='config/matterport/adapter_with_entropy.yaml',
                        help='path to yaml config file')
    parser.add_argument('--model_path', type=str, default=None,
                        help='override checkpoint path (else TEST.model_path in yaml)')
    parser.add_argument('--save_folder', type=str, default=None,
                        help='override output folder for logs/artifacts')
    parser.add_argument('--confidence_dir', type=str, default=None,
                        help='directory with Run-0 per-scene confidence .npy files')
    parser.add_argument('--confidence_load_dir', type=str, default=None,
                        help='alias for --confidence_dir (matches older docs)')
    parser.add_argument('--split', type=str, default=None,
                        help='override eval split (e.g. test)')
    parser.add_argument('--no_h2', action='store_true',
                        help='disable H2 quartile diagnostic')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='optional key=value overrides')
    args_in = parser.parse_args()
    assert args_in.config is not None
    cfg = config.load_cfg_from_cfg_file(args_in.config)
    if args_in.opts:
        cfg = config.merge_cfg_from_list(cfg, args_in.opts)
    if args_in.model_path:
        cfg.model_path = args_in.model_path
    if args_in.save_folder:
        cfg.save_folder = args_in.save_folder
    conf_override = args_in.confidence_dir or args_in.confidence_load_dir
    if conf_override:
        cfg.confidence_dir = conf_override
    if args_in.split:
        cfg.split = args_in.split
    if args_in.no_h2:
        cfg.do_h2_diagnostic = False
    return cfg


def get_logger():
    logger_name = 'adapter-eval-logger'
    log = logging.getLogger(logger_name)
    log.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    fmt = '[%(asctime)s %(filename)s line %(lineno)d] %(message)s'
    handler.setFormatter(logging.Formatter(fmt))
    log.addHandler(handler)
    return log


def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    logger.info(args)

    cudnn.benchmark = True

    os.makedirs(args.save_folder, exist_ok=True)

    # Load adapter
    adapter = FeatureAdapter(feat_dim=args.feat_dim, num_classes=args.classes).cuda()

    if not args.model_path or not os.path.isfile(str(args.model_path)):
        raise FileNotFoundError(
            "Adapter checkpoint not found at '{}'. "
            "Set model_path in the config or as a CLI override.".format(args.model_path)
        )
    logger.info("=> loading checkpoint '{}'".format(args.model_path))
    ckpt = torch.load(args.model_path, map_location='cuda')
    adapter.load_state_dict(ckpt['state_dict'])
    logger.info("=> checkpoint loaded (epoch {})".format(ckpt.get('epoch', '?')))
    adapter.eval()

    # Validation loader
    use_shm = getattr(args, 'use_shm', False)
    val_data = FusedFeatureLoader(
        datapath_prefix=args.data_root,
        datapath_prefix_feat=args.data_root_2d_fused_feature,
        voxel_size=args.voxel_size,
        split=args.split,
        aug=False,
        memcache_init=use_shm,
        eval_all=True,
        identifier=6799,
        input_color=args.input_color,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,           # must be 1 for per-scene H2 diagnostic
        shuffle=False,
        num_workers=args.test_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collation_fn_eval_all,
    )

    labelset_name = args.data_root.split('/')[-1]
    evaluate(adapter, val_loader, labelset_name)


def evaluate(adapter, val_loader, labelset_name='matterport_3d'):
    """Run inference and compute mIoU + H2 diagnostic."""

    torch.backends.cudnn.enabled = False
    do_h2 = getattr(args, 'do_h2_diagnostic', False)
    confidence_dir = getattr(args, 'confidence_dir', None)

    all_preds = []
    all_gts   = []

    # Per-scene buffers for H2 diagnostic
    h2_preds_all = []
    h2_gts_all   = []
    h2_conf_all  = []

    with torch.no_grad():
        for i, (coords, feat, label, feat_3d, mask, inds_reverse) in enumerate(
                tqdm(val_loader, desc='Evaluating')):

            # feat_3d: (N_vox, 768) — full voxel grid; zeros where no fused feature
            # Forward adapter only on visible voxels (plan §8), then scatter back.
            feat_3d_cuda = feat_3d.float().cuda()
            mask_bool = mask.bool().cuda()
            num_classes = int(args.classes)
            logits_vox = feat_3d_cuda.new_zeros(
                feat_3d_cuda.shape[0], num_classes)
            if mask_bool.any():
                logits_vox[mask_bool] = adapter(feat_3d_cuda[mask_bool])
            logits_orig = logits_vox[inds_reverse, :]   # (N_orig, C)
            preds = logits_orig.argmax(dim=-1).cpu()    # (N_orig,)

            # Mark points without fused features as 'no prediction'
            no_feat_mask = ~mask[inds_reverse].bool()
            preds[no_feat_mask] = 256                   # NO_FEATURE_ID handled by metric.confusion_matrix

            all_preds.append(preds)
            all_gts.append(label.cpu())

            # H2 diagnostic: collect per-scene
            if do_h2 and confidence_dir:
                data_path  = val_loader.dataset.data_paths[i]
                scene_name = os.path.splitext(os.path.basename(data_path))[0]
                conf_path  = os.path.join(confidence_dir, f'{scene_name}.npy')
                if os.path.exists(conf_path):
                    conf_scores = np.load(conf_path)    # (N_orig,)
                    h2_preds_all.append(preds.numpy())
                    h2_gts_all.append(label.cpu().numpy())
                    h2_conf_all.append(conf_scores)
                else:
                    logger.warning(
                        'No confidence file for {} (H2 diagnostic skipped for this scene)'
                        .format(scene_name)
                    )

    # Overall mIoU
    gt_all   = torch.cat(all_gts).numpy()
    pred_all = torch.cat(all_preds).numpy()

    logger.info('Computing mIoU on {} points...'.format(gt_all.size))
    miou = metric.evaluate(pred_all, gt_all, dataset=labelset_name, stdout=True)
    logger.info('mIoU: {:.4f}'.format(miou))

    # H2 diagnostic (quartile accuracy)
    if do_h2 and h2_conf_all:
        logger.info('\n===== H2 Diagnostic: Per-Quartile Accuracy =====')
        logger.info('(Q1 = least confident / suspected unreliable features)')
        logger.info('(H2 confirmed if  gain(Run1?Run2) is larger in Q1 than Q4)\n')

        h2_preds = np.concatenate(h2_preds_all)
        h2_gts   = np.concatenate(h2_gts_all)
        h2_conf  = np.concatenate(h2_conf_all)

        results = metric.evaluate_by_confidence_bin(
            h2_preds, h2_gts, h2_conf, n_bins=4,
            dataset=labelset_name, stdout=True,
        )
        logger.info('H2 results: {}'.format(results))

    return miou


if __name__ == '__main__':
    main()
