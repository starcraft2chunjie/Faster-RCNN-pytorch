import torch
import torch.nn as nn
import numpy as np 
import math
import yaml
from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch
from model.nms.nms_wrapper import soft_nms

import pdb
"""
Transform the anchors according to the bounding box regression coefficients to 
generate transformed anchors. Then prune the number of anchors by applying non-maximum
suppression (see Appendix) using the probability of an anchor being a foreground region
"""
"""
1.生成所有的anchor，对anchor进行4个坐标变换生成新的坐标变成proposals（按照老方法先在最后一层feature map的每个像素点上滑动生成所有的anchor，
然后将所有的anchor坐标乘以16，即映射到原图就得到所有的region proposal，接着再用boundingbox regression对每个region proposal进行坐标变换
生成更优的region proposal坐标，也是最终的region proposal坐标）　　
2.处理掉所有坐标超过了图像边界的proposal　
3.处理掉所有长度宽度小于min_size的proposal　　
4.把所有的proposal按score高低进行排序　　
5.选择得分前pre_nms_topN的proposal，这是在进行nms前进行一次选择　 
6.进行nms处理　　
7.选择得分前post_nms_topN的proposal，这是在进行nms后进行的一次选择　　最终就得到了需要传入fast rcnn网络的region proposal。
"""
class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """
    def __init__(self, feat_stride, scales, ratios):
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feat_stride #映射因子
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)
        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        # top[0].reshape(1, 5)
        #
        # # scores blob: holds scores for R regions of interest
        # if len(top) > 1:
        #     top[1].reshape(1, 1, 1, 1)
    
    def forward(self, input):
        # Algorithm:
        #
        # for each (H, W) location i
        # generate A anchor boxes centered on cell i
        # apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)


        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs
        scores = input[0][:, self._num_anchors:, :, :] #(B, C/2(9), H, W)
        bbox_deltas = input[1] #(B, C(4 * 9), H, W)
        im_info = input[2]
        cfg_key = input[3]

        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
        min_size      = cfg[cfg_key].RPN_MIN_SIZE

        batch_size = bbox_deltas.size(0)

        feat_height, feat_width = scores.size(2), scores.size(3)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose())

        shifts = shifts.contiguous().type_as(scores).float()
        
        A = self._num_anchors
        K = shifts.size(0)

        self._anchors = self._anchors.type_as(scores)

        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchor

        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)

        #same story for the scores:
        scores = scores.permute(0, 2, 3, 1).contiguous()
        scores = scores.view(batch, -1) #(batch, H * W * 9)

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info, batch_size)
        # proposals = clip_boxes_batch(proposals, im_info, batch_size)

        scores_keep = scores
        proposals_keep = proposals
        _, order = torch.sort(scores_keep, 1, True)
        
        output = scores.new(batch_size, post_nms_topN, 5).zero_()

        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (Note: convert min_size to input image scale stored in im_info[2])
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)

            order_single = order[i]

            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]
            
            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1, 1)

            # 6. apply nms (soft_nms)
            # 7. take post_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)

            keep_idx_i = soft_nms(torch.cat((proposals_single, scores_single), 1))
            keep_idx_i = keep_idx_i.long().view(-1)

            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            # padding 0 at the end 
            num_proposal = proposals_single.size(0)
            output[i,:,0] = i
            output[i,:num_proposal,1:] = proposals_single

        return output
    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradient"""
        pass
    
    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""

    def _filter_boxes(self, boxes, min_size):
        """Remove all the boxes with any side smaller than min_size"""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size.view(-1, 1).expand_as(ws)) & (hs >= min_size.view(-1, 1).expand_as(hs)))

        return keep


        


