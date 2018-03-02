"""
from the RPN, features produce logits and rpn_cls_prob [1, 9 * 2, H, W]

"""

import numpy as np 
import torch

from utils_.box_utils import bbox_overlaps, bbox_transform, _unmap
from utils_.anchor_utils import generate_anchors, get_anchor

def rpn_target(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info, _feat_stride=[16, ]):
    _allowed_border = 0

    anchors = generate_anchors() # [9, 4]
    _num_anchors = anchors.shape[0] # 9
    height, width = rpn_cls_score.shape[2:4] # H, W

    all_anchors = get_anchor(rpn_cls_score, anchors)   # [H * W * 9, 4]
    total_anchors = all_anchors.shape[0] #[H * W * 9]


    # only keep anchors inside the image
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) & 
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] >= -_allowed_border) &
        (all_anchors[:, 3] >= -_allowed_border)
    )[0]

    # keep only inside anchors
    inside_anchors = all_anchors[inds_inside, :]

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)

    overlaps = bbox_overlaps(inside_anchors, gt_boxes[:, :-1]).cpu().numpy()
    argmax_overlaps = overlaps.argmax(axis = 1)
    #Use np.arange will choose each row's maximum
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]

    gt_argmax_overlaps = overlaps.argmax(axis = 0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]                 
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    # assign bg labels first so that positive labels can clobber them
    labels[max_overlaps < args.neg_threshold] = 0

    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1

    # fg label: above threshold IOU
    labels[max_overlaps >= args.pos_threshold] = 1


    #preclude hard samples that are highly occlusioned, truncated or difficult to see
    gt_ishard.shape[0] = gt_boxes.shape[0]
    gt_ishard = gt_ishard.astype(int)
    gt_hardboxes = gt_boxes[gt_ishard == 1, :]
    if gt_hardboxes.shape[0] > 0:
        # H * A
        hard_overlaps = bbox_overlaps(inside_anchors, gt_hardboxes[:, :-1]).cpu().numpy
        hard_max_overlaps = hard_overlaps.max(axis=0)  # (A)
        labels[hard_max_overlaps >= args.pos_threshold] = -1
        max_intersec_label_inds = hard_overlaps.argmax(axis=1)  # H x 1
        labels[max_intersec_label_inds] = -1  #
    

    # subsample positive labels if we have too many
    num_fg = int(0.5 * args.rpn_batch_size)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = np.random.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = args.rpn_batch_size - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = np.random.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
    

    # transform boxes to deltas boxes
    bbox_targets = bbox_transform(inside_anchors, gt_boxes[argmax_overlaps, :-1])
    #for the use of loss
    bbox_inside_weights = np.zeros((bbox_targets.shape[0], 4), dtype=np.float32)

    mask = np.where(labels == 1)[0]
    bbox_inside_weights[mask, :] = [1.0, 1.0, 1.0, 1.0]
    #print(bbox_targets.shape, bbox_inside_weights.shape, labels.shape)
    # map up to original set of anchors
    # inds_inside 는 data로 채우고 나머지는 fill의 값으로 채움. 즉 backround인 box의 target을 채워준다.
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)


    return labels, bbox_targets, bbox_inside_weights










    