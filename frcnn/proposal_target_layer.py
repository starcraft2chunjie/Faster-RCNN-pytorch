import numpy as np
import torch

from utils_.box_utils import bbox_overlaps, bbox_transform, _unmap, _jitter_gt_boxes

def frcnn_targets(prop_boxes, gt_boxes_c, args, num_classes):
    """
    Arguments:
        prop_boxes (Tensor) : (# proposal boxes , 4)
        gt_boxes_c (Ndarray) : (# gt boxes, 5) [x, y, x`, y`, class]
        test (Bool) : True or False
        args (argparse.Namespace) : global arguments
    Return:
        labels (Ndarray) : (256,)
        roi_boxes_c[:, :-1] : (256, 4)
        targets (Ndarray) : (256, num_classes * 4)
        bbox_inside_weights (Ndarray) : (256, num_classes * 4)
    """

    gt_labels = gt_boxes_c[:, -1]
    gt_boxes = gt_boxes_c[:, -1]
    jitter_gt_boxes = _jitter_gt_boxes(gt_boxes)

    all_boxes = np.vstack((prop_boxes, jitter_gt_boxes, gt_boxes))
    zeros = np.zeros((all_boxes.shape[0], 1), dtype=all_boxes.dtype)
    all_boxes_c = np.hstack((all_boxes, zeros))

    num_images = 1

    # number of roi_boxes_c each per image
    rois_per_image = int(args.frcnn_batch_size / num_images)
    # number of foreground roi_boxes_c per image
    fg_rois_per_image = int(np.round(rois_per_image * args.fg_fraction))

    #sample rois

    #compute each overlaps with each ground truth
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_boxes_c[:, :-1], dtype=np.float)
        np.ascontiguousarray(gt_boxes, dtype=np.float)
    )

    # overlaps (iou, index of class)
    overlaps = overlaps.cpu().numpy()

    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)  
    labels = gt_labels[gt_assignment]

    # Select foreground ROIs as those with >= fg_threshold 
    fg_indices = np.where(max_overlaps >= args.fg_threshold)[0]

    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, len(fg_indices))

    #Sample foreground regions without replacement
    if len(fg_indices) > 0:
        
        fg_indices = np.random.choice(fg_indices, size = fg_rois_per_image, replace=False)

    #select background RoIs as those within [0, bg_threshold[1]]
    bg_indices = np.where((max_overlaps < args.bg_threshold[1]) &
                          (max_overlaps >= 0))[0]

    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, len(bg_indices))

    # Sample background regions without replacement
    if len(bg_indices) > 0:
        bg_indices = np.random.choice(bg_indices, size= bg_rois_per_this_image, replace=False)
    
    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_indices, bg_indices)

    # Select sampled values from various arrays
    labels = labels[keep_inds]
    roi_boxes_c = all_boxes_c[keep_inds]

    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0

    # _compute_target
    # all_boxes_c 를 delta_boxes 로 바꿔준다.

    # a = np.array([[j] for j in range(10)])
    # a[[0,0,0]]  : [[0],[0],[0]]
    # index array라서 해당 index에 해당하는 array의 객체가 반복되어 연산된다.
    #a = gt_boxes[gt_assignment[keep_inds], :]
    delta_boxes = bbox_transform(roi_boxes_c[:, :-1], gt_boxes[gt_assignment[keep_inds], :])

    if args.target_normalization:
        delta_boxes = ((delta_boxes - np.array((0.0, 0.0, 0.0, 0.0))) / np.array((0.1, 0.1, 0.2, 0.2)))

        #_get_bbox_regression_labels
        targets = np.zeros((len(labels), 4 * num_classes), dtype = np.float32)

        bbox_inside_weights = np.zeros(targets.shape, dtype= np.float32)

        #foreground object index
        indices = np.where(labels > 0)[0]

        for index in indices:
            cls = int(labels[index])
            start = 4 * cls
            end = start + 4
            targets[index, start:end] = delta_boxes[index, :]
            bbox_inside_weights[index, start:end] = [1, 1, 1, 1]

        return labels, roi_boxes_c[:, :-1], targets, bbox_inside_weights
    





    
