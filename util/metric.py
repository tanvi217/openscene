'''IoU'''
import numpy as np
from dataset.label_constants import *

UNKNOWN_ID = 255
NO_FEATURE_ID = 256


def confusion_matrix(pred_ids, gt_ids, num_classes):
    '''calculate the confusion matrix.'''

    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
    idxs = gt_ids != UNKNOWN_ID
    if NO_FEATURE_ID in pred_ids: # some points have no feature assigned for prediction
        pred_ids[pred_ids==NO_FEATURE_ID] = num_classes
        confusion = np.bincount(
            pred_ids[idxs] * (num_classes+1) + gt_ids[idxs],
            minlength=(num_classes+1)**2).reshape((
            num_classes+1, num_classes+1)).astype(np.ulonglong)
        return confusion[:num_classes, :num_classes]

    return np.bincount(
        pred_ids[idxs] * num_classes + gt_ids[idxs],
        minlength=num_classes**2).reshape((
        num_classes, num_classes)).astype(np.ulonglong)


def get_iou(label_id, confusion):
    '''calculate IoU.'''

    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false positives
    fp = np.longlong(confusion[label_id, :].sum()) - tp
    # false negatives
    fn = np.longlong(confusion[:, label_id].sum()) - tp

    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')
    return float(tp) / denom, tp, denom


def evaluate_by_confidence_bin(pred_ids, gt_ids, conf_scores,
                                n_bins=4, dataset='matterport_3d', stdout=True):
    '''Per-quartile point accuracy for the H2 diagnostic.

    Bins all points by their confidence score (max-softmax from the baseline
    fusion model) and reports per-bin accuracy.  H2 is confirmed when the
    accuracy *gain* from the supervised-only run to the H2 run is larger in
    the lowest-confidence quartile (Q1) than in the highest (Q4).

    Args:
        pred_ids:    (N,) int  --- predicted class indices
        gt_ids:      (N,) int  --- ground-truth class indices (255 = ignore)
        conf_scores: (N,) float --- per-point confidence proxy (higher = more reliable)
        n_bins:      number of quantile bins (default 4 = quartiles)
        dataset:     dataset name (used for display only)
        stdout:      whether to print results

    Returns:
        dict mapping 'Q1'..'Qn' to per-bin accuracy floats
    '''
    assert pred_ids.shape == gt_ids.shape == conf_scores.shape

    # Quartiles for n_bins=4: edges at 0.25, 0.5, 0.75 (plan §6)
    edges = np.quantile(conf_scores, np.linspace(0, 1, n_bins + 1)[1:-1])
    bins = np.digitize(conf_scores, edges)

    if stdout:
        print(f'\nH2 Diagnostic --- Per-quartile accuracy  (dataset: {dataset})')
        print('  Q1 = least confident  (suspected low view count / unreliable features)')
        print('  Q{} = most  confident  (suspected high view count / reliable features)'.format(n_bins))
        print('  ---------------------------------------------------------------')

    results = {}
    for q in range(n_bins):
        mask_q   = (bins == q)
        valid_q  = mask_q & (gt_ids != UNKNOWN_ID)
        n_valid  = int(valid_q.sum())
        if n_valid == 0:
            continue
        acc_q = float((pred_ids[valid_q] == gt_ids[valid_q]).mean())
        results[f'Q{q + 1}'] = acc_q
        if stdout:
            print(f'  Q{q + 1}: acc = {acc_q:.4f}  (n = {n_valid})')

    if stdout and len(results) >= 2:
        q1_key = 'Q1'
        q4_key = f'Q{n_bins}'
        if q1_key in results and q4_key in results:
            print(f'\n  Q1 acc = {results[q1_key]:.4f}   '
                  f'Q{n_bins} acc = {results[q4_key]:.4f}   '
                  f'diff (Q1-Q{n_bins}) = {results[q1_key] - results[q4_key]:+.4f}')

    return results


def evaluate(pred_ids, gt_ids, stdout=False, dataset='scannet_3d'):
    if stdout:
        print('evaluating', gt_ids.size, 'points...')
    if 'scannet_3d' in dataset:
        CLASS_LABELS = SCANNET_LABELS_20
    elif 'matterport_3d_40' in dataset:
        CLASS_LABELS = MATTERPORT_LABELS_40
    elif 'matterport_3d_80' in dataset:
        CLASS_LABELS = MATTERPORT_LABELS_80
    elif 'matterport_3d_160' in dataset:
        CLASS_LABELS = MATTERPORT_LABELS_160
    elif 'matterport_3d' in dataset:
        CLASS_LABELS = MATTERPORT_LABELS_21
    elif 'nuscenes_3d' in dataset:
        CLASS_LABELS = NUSCENES_LABELS_16
    else:
        raise NotImplementedError

    N_CLASSES = len(CLASS_LABELS)
    confusion = confusion_matrix(pred_ids, gt_ids, N_CLASSES)
    class_ious = {}
    class_accs = {}
    mean_iou = 0
    mean_acc = 0

    count = 0
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        if (gt_ids==i).sum() == 0: # at least 1 point needs to be in the evaluation for this class
            continue

        class_ious[label_name] = get_iou(i, confusion)
        class_accs[label_name] = class_ious[label_name][1] / (gt_ids==i).sum()
        count+=1

        mean_iou += class_ious[label_name][0]
        mean_acc += class_accs[label_name]

    mean_iou /= N_CLASSES
    mean_acc /= N_CLASSES
    if stdout:
        print('classes          IoU')
        print('----------------------------')
        for i in range(N_CLASSES):
            label_name = CLASS_LABELS[i]
            try:
                if 'matterport' in dataset:
                    print('{0:<14s}: {1:>5.3f}'.format(label_name, class_accs[label_name]))

                else:
                    print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(
                        label_name,
                        class_ious[label_name][0],
                        class_ious[label_name][1],
                        class_ious[label_name][2]))
            except:
                print(label_name + ' error!')
                continue
        print('Mean IoU', mean_iou)
        print('Mean Acc', mean_acc)
    return mean_iou
