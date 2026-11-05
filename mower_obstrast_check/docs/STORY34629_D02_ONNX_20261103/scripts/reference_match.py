"""Unmodified match() archived from .p2-stage/analyze_detect_d00.py."""
import numpy as np

def match(gt, gc, pred, pc):
    if len(gt) == 0 or len(pred) == 0:
        return np.zeros((0, 2), dtype=int)
    inter = np.maximum(0, np.minimum(gt[:, None, 2:], pred[None, :, 2:]) - np.maximum(gt[:, None, :2], pred[None, :, :2])).prod(2)
    area_gt = (gt[:, 2:] - gt[:, :2]).prod(1)
    area_pred = (pred[:, 2:] - pred[:, :2]).prod(1)
    iou = inter / (area_gt[:, None] + area_pred[None, :] - inter + 1e-7)
    pairs = np.argwhere((iou >= .5) & (gc[:, None] == pc[None, :]))
    if len(pairs) > 1:
        pairs = pairs[np.argsort(iou[pairs[:, 0], pairs[:, 1]])[::-1]]
        pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
        pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
    return pairs
