import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F  
from utils_.utils import to_var, to_tensor

def rpn_loss(rpn_cls_prob, rpn_logits, rpn_bbox_pred, rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights):
    """
    Arguments:
        rpn_cls_prob (Tensor): (1, 2*9, H/16, W/16)
        rpn_logits (Tensor): (H/16 * W/16 , 2) object or non-object rpn_logits
        rpn_bbox_pred (Tensor): (1, 4*9, H/16, W/16) predicted boxes
        rpn_labels (Ndarray) : (H/16 * W/16 * 9 ,)
        rpn_bbox_targets (Ndarray) : (H/16 * W/16 * 9, 4)
        rpn_bbox_inside_weights (Ndarray) : (H/16 * W/16 * 9, 4) masking for only positive box loss
    Return:
        cls_loss (Scalar) : classfication loss
        reg_loss * 10 (Scalar) : regression loss
        log (Tuple) : for logging
    """

    height, width = rpn_cls_prob[-2:]

    rpn_cls_prob = rpn_cls_prob.squeeze(0).permute(1, 2, 0).contiguous() #(H/16, W/16, 2 * 9)
    rpn_cls_prob = rpn_cls_prob.view(-1, 2) #(H/16 * W/16 * 9, 2)

    rpn_labels = to_tensor(rpn_labels).long() #(H/16 * W/16 * 9)
    
